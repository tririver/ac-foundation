"""Cross-cutting tests for the current public and durable LLM contracts."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from arc_llm import (
    CorruptTaskStateError,
    ExecutionLimits,
    HostAuthority,
    InvalidRequestError,
    JsonOutput,
    LLMClient,
    LLMCompleted,
    LLMExecutionOptions,
    LLMPaused,
    LLMRequest,
    ModelSelection,
    NativeResumeHandle,
    ProviderExecution,
    ProviderGateOptions,
    ProviderTerminalKind,
    ProviderUsage,
    ResumeAction,
    ResumeInput,
)
from arc_llm.config import DEFAULT_MODELS, detect_host, resolve_model_selection
from arc_llm.errors import CandidateConflictError, OutputInvalidError
from arc_llm.executor import LLMTaskExecutor
from arc_llm.identity import (
    canonical_json_bytes,
    execution_document,
    execution_fingerprint,
    semantic_document,
    semantic_key,
)
from arc_llm.output import CandidateMaterial, candidate_digest, select_output
from arc_llm.providers._cli import EventAccumulator
from arc_llm.providers.claude import _parse_event as parse_claude_event
from arc_llm.providers.codex import _parse_event as parse_codex_event


def test_config_resolution_matrix_is_explicit_and_host_deterministic() -> None:
    explicit = detect_host(
        env={
            "ARC_AGENT_HOST": "kimi",
            "ARC_LLM_PROVIDER": "ignored",
            "CODEX_THREAD_ID": "ignored",
        },
        process_chain=("claude --print",),
    )
    assert (explicit.host, explicit.evidence) == ("kimi", "ARC_AGENT_HOST")

    env_cases = (
        ({"CODEX_THREAD_ID": "thread"}, "codex", "CODEX_THREAD_ID"),
        ({"CLAUDE_CODE": "1"}, "claude", "CLAUDE_CODE"),
        ({"CLAUDECODE": "1"}, "claude", "CLAUDECODE"),
        ({"KIMI_CODE": "1"}, "kimi", "KIMI_CODE"),
    )
    for env, host, evidence in env_cases:
        detected_host = detect_host(env=env)
        assert (detected_host.host, detected_host.evidence) == (host, evidence)
    for command, host in (
        ("openai-codex exec", "codex"),
        ("claude --print", "claude"),
        ("kimi -p task", "kimi"),
    ):
        assert detect_host(env={}, process_chain=(command,)).host == host

    # Auto resolution is constrained by the caller's available-provider set.
    detected = resolve_model_selection(
        ModelSelection(),
        env={"CLAUDE_CODE": "1"},
        available=("codex", "claude"),
    )
    assert (detected.provider, detected.model, detected.tier) == (
        "claude",
        DEFAULT_MODELS["claude"]["medium"],
        "medium",
    )
    fallback = resolve_model_selection(
        ModelSelection(tier="high"),
        env={},
        available=("kimi",),
    )
    assert (fallback.provider, fallback.model, fallback.tier) == (
        "kimi",
        DEFAULT_MODELS["kimi"]["high"],
        "high",
    )

    # Model-selection environment variables are deliberately not an implicit
    # configuration surface. Exact requirements and tiers are request data.
    exact = resolve_model_selection(
        ModelSelection("codex", model="pinned"),
        env={"ARC_LLM_MODEL": "ignored", "ARC_CODEX_MODEL": "ignored"},
    )
    assert (exact.provider, exact.model, exact.tier) == (
        "codex",
        "pinned",
        "medium",
    )
    assert resolve_model_selection(ModelSelection("codex", tier="xhigh")).model == (
        DEFAULT_MODELS["codex"]["xhigh"]
    )
    assert DEFAULT_MODELS == {
        "codex": {
            "low": "gpt-5.6-luna",
            "medium": "gpt-5.6-luna",
            "high": "gpt-5.6-sol",
            "xhigh": "gpt-5.6-sol",
        },
        "claude": {
            "low": "haiku",
            "medium": "sonnet",
            "high": "opus",
            "xhigh": "opus",
        },
        "kimi": {
            "low": "default",
            "medium": "default",
            "high": "default",
            "xhigh": "default",
        },
    }
    with pytest.raises(InvalidRequestError, match="low, medium, high, or xhigh"):
        ModelSelection(tier="max")  # type: ignore[arg-type]
    with pytest.raises(InvalidRequestError, match="unavailable"):
        resolve_model_selection(ModelSelection("claude"), available=("codex",))


def test_public_import_exposes_only_the_supported_facades_and_lazy_adapters() -> None:
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(root / "packages/arc-jobs/src"),
            str(root / "packages/arc-llm/src"),
        )
    )
    source = """
import json
import sys
import arc_llm
print(json.dumps({
    "exports": sorted(arc_llm.__all__),
    "modules": sorted(name for name in sys.modules if name.startswith("arc_")),
    "version": arc_llm.__version__,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    observed = json.loads(completed.stdout)
    exports = set(observed["exports"])
    assert observed["version"] == "1.0.1"
    assert {"LLMClient", "LLMTaskService", "LLMRequest"} <= exports
    assert not {
        "run_json",
        "run_text",
        "schema_format",
        "ProposerReviewerRunner",
        "resolve_llm_config",
    } & exports
    assert not any(
        name.startswith(
            (
                "arc_domain",
                "arc_mcp",
                "arc_paper",
                "arc_proposer_reviewer",
            )
        )
        for name in observed["modules"]
    )
    assert not {
        "arc_llm.providers.codex",
        "arc_llm.providers.claude",
        "arc_llm.providers.kimi",
    } & set(observed["modules"])


def test_explicit_run_root_is_the_only_llm_storage_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, adapter: Any, registry: Any
) -> None:
    shadow = tmp_path / "implicit-home"
    monkeypatch.setenv("ARC_HOME", str(shadow))
    monkeypatch.setenv("ARC_LLM_CACHE", str(shadow / "cache"))
    monkeypatch.setenv("ARC_LLM_TMP_DIR", str(shadow / "tmp"))
    adapter.steps.append(
        ProviderExecution(
            ProviderTerminalKind.COMPLETED,
            (CandidateMaterial(value={"answer": 1}, terminal=True),),
            NativeResumeHandle("codex", "thread-1"),
        )
    )
    run_root = tmp_path / "explicit-repository"
    result = LLMClient(registry=registry).generate(
        LLMRequest(
            "path-contract",
            "Return an answer.",
            JsonOutput(
                {
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                }
            ),
            ModelSelection("codex"),
        ),
        run_root=run_root,
    )

    assert isinstance(result.outcome, LLMCompleted)
    assert not shadow.exists()
    run_dir = run_root / "runs" / result.snapshot.run_id
    assert run_dir.is_dir()
    assert result.snapshot.result_ref is not None
    assert (run_dir / result.snapshot.result_ref.relative_path).is_file()
    assert {path.name for path in run_root.iterdir()} <= {"operational", "runs"}
    assert not any(
        candidate.exists()
        for candidate in (
            run_root / "cache",
            run_root / "tmp",
            run_root / "schemas",
        )
    )


class _EventObserver:
    def __init__(self) -> None:
        self.handles: list[NativeResumeHandle] = []
        self.raw: list[Any] = []

    def native_handle(self, handle: NativeResumeHandle) -> None:
        self.handles.append(handle)

    def raw_event(self, event: Any) -> None:
        self.raw.append(event)


def test_provider_event_normalization_matrix_preserves_terminal_material() -> None:
    cases = (
        (
            "claude",
            parse_claude_event,
            (
                {
                    "type": "result",
                    "session_id": "claude-session",
                    "structured_output": {"ok": True},
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 2,
                        "cache_read_input_tokens": 1,
                    },
                },
            ),
            "claude-session",
            {"ok": True},
            ProviderUsage(5, 2, 1),
        ),
    )
    for provider, parser, events, handle, value, usage in cases:
        observer = _EventObserver()
        accumulator = EventAccumulator(provider, observer, parser)
        encoded = b"\n".join(
            json.dumps(event, separators=(",", ":")).encode() for event in events
        )
        midpoint = len(encoded) // 2
        accumulator.feed(encoded[:midpoint])
        accumulator.feed(encoded[midpoint:])
        accumulator.finish()

        assert observer.handles == [NativeResumeHandle(provider, handle)]
        assert len(accumulator.candidates) == 1
        candidate = accumulator.candidates[0]
        assert candidate.terminal
        assert (candidate.value if candidate.has_value else candidate.text) == value
        assert accumulator.usage == usage
        assert len(accumulator.raw_events) == len(events)


def test_codex_jsonl_events_preserve_diagnostics_without_candidates() -> None:
    observer = _EventObserver()
    accumulator = EventAccumulator("codex", observer, parse_codex_event)
    events = (
        {"type": "thread.started", "thread_id": "codex-session"},
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": '{"ok":true}'},
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 7,
                "output_tokens": 3,
                "cached_input_tokens": 2,
            },
        },
    )
    accumulator.feed(b"\n".join(json.dumps(event).encode() for event in events))
    accumulator.finish(validate_terminal=False)

    assert observer.handles == [NativeResumeHandle("codex", "codex-session")]
    assert accumulator.candidates == []
    assert accumulator.usage == ProviderUsage(7, 3, 2)
    assert len(accumulator.raw_events) == len(events)


def test_output_candidate_matrix_is_complete_schema_only_and_conflict_safe() -> None:
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "string"}}},
        "required": ["items"],
        "additionalProperties": False,
    }
    local = JsonOutput(schema, repair="local")
    strict = JsonOutput(schema, repair="strict")

    # Equivalent material collapses by canonical digest. A unique terminal
    # candidate is authoritative even when a different draft is also valid.
    assert select_output(
        (
            CandidateMaterial(value={"items": ["same"]}),
            CandidateMaterial(text='{"items":["same"]}'),
        ),
        strict,
    ) == {"items": ["same"]}
    assert select_output(
        (
            CandidateMaterial(value={"items": ["draft"]}),
            CandidateMaterial(value={"items": ["final"]}, terminal=True),
        ),
        strict,
    ) == {"items": ["final"]}

    conflicting = (
        CandidateMaterial(value={"items": ["a"]}),
        CandidateMaterial(value={"items": ["b"]}),
    )
    with pytest.raises(CandidateConflictError) as caught:
        select_output(conflicting, strict)
    expected_digests = tuple(
        sorted((candidate_digest({"items": ["a"]}), candidate_digest({"items": ["b"]})))
    )
    assert caught.value.candidate_digests == expected_digests
    assert caught.value.details == {"candidate_digests": list(expected_digests)}
    selected = select_output(
        conflicting,
        strict,
        selected_digest=expected_digests[1],
    )
    assert candidate_digest(selected) == expected_digests[1]

    # Only complete values enter selection. Quoted braces are prose, and a
    # balanced but invalid top-level object does not hide a later valid one.
    text = (
        'quoted "{\\"items\\":[\\"fake\\"]}" '
        'broken {"items": invalid} then {"items":["kept"]}'
    )
    assert select_output((CandidateMaterial(text=text),), strict) == {
        "items": ["kept"]
    }
    outer = {"items": ["outer"], "nested": {"items": ["inner"]}}
    outer_contract = JsonOutput(
        {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
                "nested": {"type": "object"},
            },
            "required": ["items"],
        },
        repair="strict",
    )
    assert select_output(
        (CandidateMaterial(text=json.dumps(outer)),),
        outer_contract,
    ) == outer
    with pytest.raises(OutputInvalidError):
        select_output(
            (CandidateMaterial(text='{"items": NaN}'),),
            JsonOutput({"type": "object"}),
        )

    # Local recovery repairs syntax only. It neither adds required business
    # fields nor removes forbidden fields to manufacture validity.
    assert select_output(
        (CandidateMaterial(text='```json\n{"answer":3,}\n```'),),
        JsonOutput(
            {
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
            repair="local",
        ),
    ) == {"answer": 3}
    extensible = JsonOutput(
        {
            "type": "object",
            "properties": {"known": {"type": "string"}},
            "additionalProperties": True,
        }
    )
    assert select_output(
        (CandidateMaterial(value={"known": "x", "preserved": 3}),),
        extensible,
    ) == {"known": "x", "preserved": 3}
    for material in (
        CandidateMaterial(text='{"items":["fixed"],}'),
        CandidateMaterial(value={}),
        CandidateMaterial(value={"items": [], "invented": True}),
        CandidateMaterial(text='"items":["rootless"]}'),
    ):
        contract = strict if material.text is not None else local
        with pytest.raises(OutputInvalidError):
            select_output((material,), contract)

    # JSON atoms are substantive candidates rather than sentinel values.
    assert select_output(
        (CandidateMaterial(value=False, terminal=True),),
        JsonOutput({"type": "boolean"}),
    ) is False
    assert select_output(
        (CandidateMaterial(value=0, terminal=True),),
        JsonOutput({"type": "integer"}),
    ) == 0


def test_candidate_conflict_pause_selects_saved_value_without_provider_replay(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    first_value = {"answer": 1}
    second_value = {"answer": 2}
    adapter.steps.append(
        ProviderExecution(
            ProviderTerminalKind.COMPLETED,
            (
                CandidateMaterial(value=first_value),
                CandidateMaterial(value=second_value),
            ),
            NativeResumeHandle("codex", "thread-conflict"),
        )
    )
    client = LLMClient(registry=registry)
    request = LLMRequest(
        "candidate-conflict",
        "Return one answer.",
        JsonOutput(
            {
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"],
                "additionalProperties": False,
            }
        ),
        ModelSelection("codex"),
    )
    direct = LLMExecutionOptions(host_authority=HostAuthority.UNRESTRICTED)
    paused = client.generate(request, run_root=tmp_path, options=direct)
    assert isinstance(paused.outcome, LLMPaused)
    assert paused.outcome.input_required
    assert paused.outcome.details["code"] == "candidate_selection_required"
    assert paused.outcome.request_ref is not None
    assert adapter.start_calls == 1

    chosen_digest = candidate_digest(second_value)
    accepted = client.resume(
        run_root=tmp_path,
        run_id=paused.snapshot.run_id,
        input=ResumeInput(
            paused.outcome.resume_key,
            ResumeAction.ACCEPT_CANDIDATE,
            candidate_digest=chosen_digest,
        ),
        options=direct,
    )
    assert isinstance(accepted.outcome, LLMCompleted)
    assert accepted.outcome.value == second_value
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 0

    replayed = client.generate(request, run_root=tmp_path, options=direct)
    assert isinstance(replayed.outcome, LLMCompleted)
    assert replayed.outcome.value == second_value
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 0


def test_text_candidate_artifact_maps_each_value_digest_for_resume(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    second_value = {"answer": 2}
    raw_text = (
        "draft\n"
        "{\n"
        '  "answer": 1\n'
        "}\n"
        "revised\n"
        "{\n"
        '  "answer": 2\n'
        "}\n"
    )
    adapter.steps.append(
        ProviderExecution(
            ProviderTerminalKind.COMPLETED,
            (CandidateMaterial(text=raw_text, terminal=True),),
            NativeResumeHandle("codex", "thread-text-conflict"),
        )
    )
    client = LLMClient(registry=registry)
    request = LLMRequest(
        "text-candidate-conflict",
        "Return one answer.",
        JsonOutput(
            {
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"],
                "additionalProperties": False,
            }
        ),
        ModelSelection("codex"),
    )

    direct = LLMExecutionOptions(host_authority=HostAuthority.UNRESTRICTED)
    paused = client.generate(request, run_root=tmp_path, options=direct)

    assert isinstance(paused.outcome, LLMPaused)
    assert paused.outcome.request_ref is not None
    candidate_path = (
        tmp_path
        / "runs"
        / paused.snapshot.run_id
        / paused.outcome.request_ref.relative_path
    )
    candidate_doc = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_entries = candidate_doc["candidates"]
    entry_digests = [entry["digest"] for entry in candidate_entries]
    assert candidate_doc["candidate_digests"] == entry_digests
    assert {entry["value"]["answer"] for entry in candidate_entries} == {1, 2}
    assert all(
        entry["digest"] == candidate_digest(entry["value"])
        for entry in candidate_entries
    )
    assert hashlib.sha256(raw_text.encode()).hexdigest() not in entry_digests

    selected_entry = next(
        entry for entry in candidate_entries if entry["value"] == second_value
    )
    accepted = client.resume(
        run_root=tmp_path,
        run_id=paused.snapshot.run_id,
        input=ResumeInput(
            paused.outcome.resume_key,
            ResumeAction.ACCEPT_CANDIDATE,
            candidate_digest=selected_entry["digest"],
        ),
        options=direct,
    )

    assert isinstance(accepted.outcome, LLMCompleted)
    assert accepted.outcome.value == second_value
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 0


def test_provider_material_round_trips_the_closed_current_shape() -> None:
    value = {"answer": 7}
    execution = ProviderExecution(
        ProviderTerminalKind.COMPLETED,
        (
            CandidateMaterial(value=value, terminal=True),
            CandidateMaterial(text='{"answer":8}', terminal=False),
        ),
        NativeResumeHandle("codex", "thread-current"),
        ProviderUsage(11, 5, 3),
        diagnostics={"source": "fixture"},
    )
    document = LLMTaskExecutor._execution_document_value(execution)

    assert "digest" not in document["candidates"][0]

    context = SimpleNamespace(
        artifacts=SimpleNamespace(
            read_bytes=lambda ref: json.dumps(document).encode("utf-8")
        )
    )
    recovered = LLMTaskExecutor()._execution_from_raw(context, object(), "codex")

    assert recovered == execution


@pytest.mark.parametrize(
    "mutation",
    (
        lambda document: document.update({"unknown": True}),
        lambda document: document.pop("diagnostics"),
        lambda document: document["candidates"][0].update(
            {"digest": "removed-field"}
        ),
        lambda document: document["candidates"][0].pop("text"),
        lambda document: document.update({"schema_version": "unknown"}),
        lambda document: document["usage"].update({"unknown": 1}),
    ),
)
def test_provider_material_rejects_old_or_open_shapes(mutation: Any) -> None:
    document = LLMTaskExecutor._execution_document_value(
        ProviderExecution(
            ProviderTerminalKind.COMPLETED,
            (CandidateMaterial(value={"answer": 7}, terminal=True),),
            usage=ProviderUsage(1, 2, 0),
        )
    )
    mutation(document)
    context = SimpleNamespace(
        artifacts=SimpleNamespace(
            read_bytes=lambda ref: json.dumps(document).encode("utf-8")
        )
    )

    with pytest.raises(CorruptTaskStateError) as caught:
        LLMTaskExecutor()._execution_from_raw(context, object(), "codex")

    assert caught.value.details["code"] == "provider_material_corrupt"


def test_identity_matrix_separates_semantics_execution_and_operations() -> None:
    output = JsonOutput(
        {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )
    base = LLMRequest(
        "identity-task",
        "Return one answer.",
        output,
        ModelSelection(),
    )
    semantic_mutants = (
        replace(base, task_id="other-task"),
        replace(base, prompt="Return a different answer."),
        replace(base, output=JsonOutput({"type": "array"})),
        replace(base, model=ModelSelection(tier="high")),
        replace(base, model=ModelSelection("codex")),
    )
    base_key = semantic_key(base)
    assert all(semantic_key(mutant) != base_key for mutant in semantic_mutants)
    semantic = semantic_document(base)
    assert semantic["model_requirement"] == {
        "provider": "auto",
        "model": None,
        "tier": "medium",
    }
    assert not {
        "provider_resolved",
        "run_id",
        "run_root",
        "timeout",
        "created_at",
        "request_sha256",
    } & semantic.keys()

    recipe = execution_document(
        provider="codex",
        model="gpt",
        capabilities={"internet": False, "effective_host_mode": "brokered"},
        adapter_compatibility_version="codex-jsonl.v3",
        session_compatibility={"native": True},
    )
    recipe_fingerprint = execution_fingerprint(recipe)
    execution_mutants = (
        {**recipe, "provider": "claude"},
        {**recipe, "model": "other"},
        {**recipe, "capabilities": {"internet": True, "effective_host_mode": "brokered"}},
        {**recipe, "adapter_compatibility_version": "codex-jsonl.v2"},
        {**recipe, "session_compatibility": {"native": False}},
    )
    assert all(
        execution_fingerprint(mutant) != recipe_fingerprint
        for mutant in execution_mutants
    )
    reordered = {
        "session_compatibility": {"native": True},
        "adapter_compatibility_version": "codex-jsonl.v3",
        "capabilities": {"effective_host_mode": "brokered", "internet": False},
        "model": "gpt",
        "provider": "codex",
        "schema_version": recipe["schema_version"],
    }
    assert execution_fingerprint(reordered) == recipe_fingerprint

    first_options = LLMExecutionOptions(
        limits=ExecutionLimits(idle_timeout_seconds=1),
        gate=ProviderGateOptions(global_limit=1, circuit_failure_threshold=1),
    )
    second_options = LLMExecutionOptions(
        limits=ExecutionLimits(idle_timeout_seconds=900),
        gate=ProviderGateOptions(
            global_limit=24,
            provider_limits={"codex": 2},
            circuit_failure_threshold=9,
            circuit_cooldown_seconds=30,
        ),
    )
    assert first_options != second_options
    assert semantic_key(base) == base_key
    assert execution_fingerprint(recipe) == recipe_fingerprint
    assert canonical_json_bytes(recipe) == canonical_json_bytes(reordered)


def test_usage_matrix_preserves_unknown_partial_and_cached_diagnostics() -> None:
    _, _, codex = parse_codex_event(
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "cached_input_tokens": 3,
            },
        }
    )
    _, _, claude = parse_claude_event(
        {
            "type": "result",
            "result": "done",
            "usage": {
                "input_tokens": 13,
                "cache_read_input_tokens": 2,
            },
        }
    )
    _, _, malformed = parse_codex_event(
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": True,
                "output_tokens": "7",
                "cached_input_tokens": -1,
            },
        }
    )
    assert codex == ProviderUsage(11, 7, 3)
    assert claude == ProviderUsage(13, None, 2)
    assert malformed == ProviderUsage(None, None, None)
    assert ProviderUsage() == ProviderUsage(None, None, None)
