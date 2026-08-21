from __future__ import annotations

from dataclasses import replace
import subprocess
import sys

import pytest

from ac_jobs import ArtifactDigest, ArtifactSourceRef
from ac_llm import (
    DEFAULT_MAX_PARALLEL_PROVIDER_CALLS,
    ExecutionLimits,
    InvalidRequestError,
    JsonOutput,
    LLMExecutionOptions,
    LLMExecutionProfile,
    LLMInputArtifact,
    LLMRequest,
    ModelSelection,
    ProviderGateOptions,
    ResumeAction,
    ResumeInput,
    decode_request,
    decode_resume_input,
    request_to_document,
    resume_input_to_document,
)
from ac_llm.identity import (
    execution_document,
    execution_fingerprint,
    semantic_document,
    semantic_key,
)


def test_request_and_resume_codecs_are_closed_round_trips() -> None:
    request = LLMRequest(
        "review-1",
        "Review this.",
        JsonOutput({"type": "object", "required": ["ok"]}),
        ModelSelection("codex"),
    )
    assert decode_request(request_to_document(request)) == request
    document = request_to_document(request)
    document["unknown"] = True
    with pytest.raises(InvalidRequestError):
        decode_request(document)

    resume = ResumeInput("resume-3", ResumeAction.REPLACE, reason="new evidence")
    assert decode_resume_input(resume_input_to_document(resume)) == resume
    legacy = resume_input_to_document(resume)
    legacy["schema_version"] = "ac.llm.resume_input.v1"
    with pytest.raises(InvalidRequestError, match="Unsupported resume input schema_version"):
        decode_resume_input(legacy)


def _input(
    input_id: str,
    digest: str,
    *,
    run_id: str = "source-run",
    artifact_id: str = "paper/source",
    media_type: str = "text/markdown",
) -> LLMInputArtifact:
    return LLMInputArtifact(
        input_id,
        ArtifactSourceRef(
            run_id,
            artifact_id,
            ArtifactDigest("sha256", digest, 3),
        ),
        media_type,
    )


def test_request_v4_input_codec_and_content_identity() -> None:
    first = _input("paper", "a" * 64, media_type=" Text/Markdown ")
    request = LLMRequest(
        "with-input",
        "Review.",
        JsonOutput({"type": "object"}),
        inputs=(first,),
    )
    document = request_to_document(request)
    assert document["schema_version"] == "ac.llm.request.v4"
    assert document["inputs"][0]["media_type"] == "text/markdown"
    assert decode_request(document) == request

    relocated = replace(
        request,
        inputs=(
            _input(
                "paper",
                "a" * 64,
                run_id="other-run",
                artifact_id="other/artifact",
            ),
        ),
    )
    changed = replace(request, inputs=(_input("paper", "b" * 64),))
    assert semantic_key(relocated) == semantic_key(request)
    assert semantic_key(changed) != semantic_key(request)


@pytest.mark.parametrize(
    "digest",
    ("A" * 64, "g" * 64, "a" * 63),
)
def test_request_v4_rejects_invalid_artifact_digests(digest: str) -> None:
    with pytest.raises(InvalidRequestError):
        _input("paper", digest)

    request = LLMRequest(
        "with-input",
        "Review.",
        JsonOutput({"type": "object"}),
        inputs=(_input("paper", "a" * 64),),
    )
    document = request_to_document(request)
    document["inputs"][0]["source"]["expected_digest"]["value"] = digest
    with pytest.raises(InvalidRequestError):
        decode_request(document)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_run_id", "../source"),
        ("source_run_id", ["source"]),
        ("source_artifact_id", "../paper"),
        ("source_artifact_id", {"artifact": "paper"}),
    ),
)
def test_request_v2_rejects_invalid_source_locator_identifiers(
    field: str,
    value: object,
) -> None:
    request = LLMRequest(
        "with-input",
        "Review.",
        JsonOutput({"type": "object"}),
        inputs=(_input("paper", "a" * 64),),
    )
    document = request_to_document(request)
    document["inputs"][0]["source"][field] = value

    with pytest.raises(InvalidRequestError):
        decode_request(document)


def test_input_order_and_ids_are_semantic_and_duplicate_ids_are_invalid() -> None:
    left = _input("left", "a" * 64)
    right = _input("right", "b" * 64)
    request = LLMRequest(
        "ordered",
        "Review.",
        JsonOutput({"type": "object"}),
        inputs=(left, right),
    )
    assert semantic_key(request) != semantic_key(replace(request, inputs=(right, left)))
    with pytest.raises(InvalidRequestError):
        replace(request, inputs=(left, left))


def test_model_constraints_and_json_booleans_are_strict() -> None:
    with pytest.raises(InvalidRequestError):
        ModelSelection(model="exact")
    with pytest.raises(InvalidRequestError):
        ModelSelection("codex", "exact", "high")
    with pytest.raises(InvalidRequestError):
        LLMExecutionOptions(internet="false")  # type: ignore[arg-type]
    with pytest.raises(InvalidRequestError):
        LLMExecutionOptions(profile="bounded")  # type: ignore[arg-type]
    assert LLMExecutionOptions(
        profile=LLMExecutionProfile.BOUNDED
    ).profile is LLMExecutionProfile.BOUNDED


def test_operational_limits_do_not_change_semantic_key() -> None:
    request = LLMRequest("task", "prompt", JsonOutput({"type": "object"}))
    first = LLMExecutionOptions(ExecutionLimits(idle_timeout_seconds=1))
    second = replace(first, limits=ExecutionLimits(idle_timeout_seconds=900))
    assert semantic_key(request) == semantic_key(request)
    assert first != second


def test_provider_gate_defaults_and_overrides_are_typed_operational_policy() -> None:
    defaults = ProviderGateOptions()
    assert defaults.enabled
    assert defaults.global_limit == DEFAULT_MAX_PARALLEL_PROVIDER_CALLS == 100
    assert defaults.minimum_available_memory_fraction == 0.10
    assert defaults.memory_poll_interval_seconds == 1.0
    assert defaults.memory_launch_interval_seconds == 0.25
    configured = ProviderGateOptions(
        global_limit=8,
        provider_limits={"codex": 2},
        circuit_failure_threshold=2,
        circuit_cooldown_seconds=30,
        minimum_available_memory_fraction=None,
    )
    assert configured.provider_limits["codex"] == 2
    with pytest.raises(InvalidRequestError):
        ProviderGateOptions(global_limit=2, provider_limits={"codex": 3})
    for invalid in (0, -0.1, 1.1, True):
        with pytest.raises(InvalidRequestError):
            ProviderGateOptions(  # type: ignore[arg-type]
                minimum_available_memory_fraction=invalid
            )
    for field in (
        "memory_poll_interval_seconds",
        "memory_launch_interval_seconds",
    ):
        with pytest.raises(InvalidRequestError):
            ProviderGateOptions(**{field: 0})


def test_runtime_internet_is_not_semantic_and_prompt_changes_are_detected() -> None:
    request = LLMRequest("task", "prompt", JsonOutput({"type": "object"}))
    assert semantic_key(request) != semantic_key(replace(request, prompt="other"))
    assert LLMExecutionOptions(internet=False) != LLMExecutionOptions(internet=True)


def test_semantic_identity_uses_explicit_task_vocabulary() -> None:
    document = semantic_document(
        LLMRequest("task", "prompt", JsonOutput({"type": "object"}))
    )
    assert document["task_id"] == "task"
    assert "logical_key" not in document


def test_public_api_import_is_lightweight_and_excludes_retired_surfaces() -> None:
    script = (
        "import json,sys,ac_llm; "
        "print(json.dumps({'exports': sorted(ac_llm.__all__), "
        "'loaded': sorted(k for k in sys.modules if k.startswith('ac_'))}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    document = __import__("json").loads(result.stdout)
    assert "LLMClient" in document["exports"]
    assert "LLMTaskService" in document["exports"]
    assert not any(name.startswith("ac_proposer_reviewer") for name in document["loaded"])
    assert not {
        "run_json",
        "run_text",
        "schema_format",
        "ProposerReviewerRunner",
    } & set(document["exports"])


def test_semantic_execution_and_operational_identity_matrix() -> None:
    request = LLMRequest(
        "identity",
        "prompt",
        JsonOutput({"type": "object"}),
        ModelSelection("codex", "model-a"),
    )
    base_semantic = semantic_key(request)
    assert semantic_key(replace(request, prompt="changed")) != base_semantic
    assert semantic_key(request) == base_semantic
    recipe = execution_document(
        provider="codex",
        model="model-a",
        capabilities={"internet": False, "effective_config": {"reasoning": "high"}},
        adapter_compatibility_version="v1",
    )
    fingerprint = execution_fingerprint(recipe)
    assert execution_fingerprint({**recipe, "model": "model-b"}) != fingerprint
    assert execution_fingerprint(
        {**recipe, "adapter_compatibility_version": "v2"}
    ) != fingerprint
    assert execution_fingerprint(
        {
            **recipe,
            "capabilities": {
                "internet": False,
                "effective_config": {"reasoning": "low"},
            },
        }
    ) != fingerprint

    options = LLMExecutionOptions()
    changed_policy = replace(
        options,
        limits=ExecutionLimits(idle_timeout_seconds=1),
        gate=ProviderGateOptions(global_limit=1),
    )
    assert changed_policy != options
    assert semantic_key(request) == base_semantic


def test_idle_timeout_is_opt_in_and_remains_operational_policy() -> None:
    assert ExecutionLimits().idle_timeout_seconds is None
    assert ExecutionLimits(idle_timeout_seconds=1).idle_timeout_seconds == 1
    with pytest.raises(InvalidRequestError):
        ExecutionLimits(idle_timeout_seconds=0)
