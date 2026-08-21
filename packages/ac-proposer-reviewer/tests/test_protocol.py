from __future__ import annotations

import copy

import pytest

from ac_jobs import ArtifactDigest, ArtifactSourceRef
from ac_llm import LLMInputArtifact, ModelSelection
from ac_proposer_reviewer.models import (
    BATCH_SCHEMA_VERSION,
    BatchFailurePolicy,
    BatchRequest,
    LoopSpec,
    ProposerFailurePolicy,
    RevisionContextMode,
    WorkerSpec,
)
from ac_proposer_reviewer.protocol import decode_batch_request, encode_batch_request
from ac_proposer_reviewer.validation import RequestValidationError


SCHEMA = {
    "type": "object",
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
    "additionalProperties": False,
}


def request() -> BatchRequest:
    return BatchRequest(
        schema_version=BATCH_SCHEMA_VERSION,
        batch_id="batch-1",
        loops=(
            LoopSpec(
                loop_id="loop-a",
                context={"question": "Q"},
                proposers=(
                    WorkerSpec(
                        worker_id="proposer-a",
                        instructions="Propose.",
                        output_schema=SCHEMA,
                        model=ModelSelection(provider="codex", model="gpt-test"),
                    ),
                ),
                reviewer=WorkerSpec("reviewer", "Review.", SCHEMA),
                max_rounds=2,
                allow_early_stop=False,
                on_proposer_failure=ProposerFailurePolicy.CONTINUE_IF_ANY,
            ),
        ),
        failure_policy=BatchFailurePolicy.FAIL_FAST,
    )


def test_batch_request_round_trips_with_closed_worker_shape() -> None:
    original = request()
    encoded = encode_batch_request(original)
    proposer = encoded["loops"][0]["proposers"][0]  # type: ignore[index]
    assert set(proposer) == {"worker_id", "instructions", "output_schema", "model"}
    assert encoded["loops"][0]["revision_context_mode"] == "feedback_only"  # type: ignore[index]
    assert encoded["loops"][0]["input_ids"] is None  # type: ignore[index]
    assert decode_batch_request(encoded) == original


@pytest.mark.parametrize("version", ("v4", "v5", "v6"))
def test_obsolete_batch_schemas_are_rejected(version: str) -> None:
    document = encode_batch_request(request())
    document["schema_version"] = f"ac.proposer_reviewer.batch.{version}"

    with pytest.raises(RequestValidationError, match="batch.v7"):
        decode_batch_request(document)


def test_full_review_envelope_revision_context_round_trips() -> None:
    original = request()
    loop = original.loops[0]
    original = BatchRequest(
        original.schema_version,
        original.batch_id,
        (LoopSpec(
            loop.loop_id,
            loop.context,
            loop.proposers,
            loop.reviewer,
            loop.max_rounds,
            loop.allow_early_stop,
            loop.on_proposer_failure,
            loop.review_final_round,
            RevisionContextMode.FULL_REVIEW_ENVELOPE,
        ),),
        original.failure_policy,
    )

    assert decode_batch_request(encode_batch_request(original)) == original


@pytest.mark.parametrize("value", ("unknown", True, None))
def test_revision_context_mode_is_closed_enum(value: object) -> None:
    document = encode_batch_request(request())
    document["loops"][0]["revision_context_mode"] = value  # type: ignore[index]

    with pytest.raises(RequestValidationError, match="unknown enum value|must be a string"):
        decode_batch_request(document)


def test_batch_request_round_trips_verified_input_references_without_content() -> None:
    original = request()
    input_artifact = LLMInputArtifact(
        "domain-markdown-001",
        ArtifactSourceRef("batch-1", "proposer-reviewer/inputs/source/0000-domain-markdown-001", ArtifactDigest("sha256", "a" * 64, 12)),
        "text/markdown",
    )
    original = BatchRequest(
        original.schema_version,
        original.batch_id,
        original.loops,
        original.failure_policy,
        (input_artifact,),
    )
    encoded = encode_batch_request(original)
    assert encoded["inputs"] == [{
        "input_id": "domain-markdown-001",
        "source": {
            "source_run_id": "batch-1",
            "source_artifact_id": "proposer-reviewer/inputs/source/0000-domain-markdown-001",
            "expected_digest": {"algorithm": "sha256", "value": "a" * 64, "size_bytes": 12},
        },
        "media_type": "text/markdown",
    }]
    assert "content" not in str(encoded)
    assert decode_batch_request(encoded) == original


def test_loop_input_ids_round_trip_and_must_reference_batch_inputs() -> None:
    original = request()
    input_artifact = LLMInputArtifact(
        "chapter-001",
        ArtifactSourceRef(
            "batch-1",
            "proposer-reviewer/inputs/source/0000-chapter-001",
            ArtifactDigest("sha256", "b" * 64, 7),
        ),
        "text/markdown",
    )
    loop = original.loops[0]
    selected_loop = LoopSpec(
        loop_id=loop.loop_id,
        context=loop.context,
        proposers=loop.proposers,
        reviewer=loop.reviewer,
        max_rounds=loop.max_rounds,
        allow_early_stop=loop.allow_early_stop,
        on_proposer_failure=loop.on_proposer_failure,
        review_final_round=loop.review_final_round,
        revision_context_mode=loop.revision_context_mode,
        input_ids=("chapter-001",),
    )
    selected = BatchRequest(
        original.schema_version,
        original.batch_id,
        (selected_loop,),
        original.failure_policy,
        (input_artifact,),
    )
    assert decode_batch_request(encode_batch_request(selected)) == selected

    invalid = encode_batch_request(selected)
    invalid["loops"][0]["input_ids"] = ["unknown"]  # type: ignore[index]
    with pytest.raises(RequestValidationError, match="refer to a batch input"):
        decode_batch_request(invalid)


@pytest.mark.parametrize("field", ("surprise", "unexpected_capability"))
def test_unknown_worker_fields_are_rejected(field: str) -> None:
    document = encode_batch_request(request())
    worker = document["loops"][0]["proposers"][0]  # type: ignore[index]
    worker[field] = True
    with pytest.raises(RequestValidationError, match="unknown field"):
        decode_batch_request(document)


def test_worker_fields_remain_required_and_request_validation_is_strict() -> None:
    document = encode_batch_request(request())
    worker = document["loops"][0]["proposers"][0]  # type: ignore[index]
    del worker["model"]
    with pytest.raises(RequestValidationError, match="required field is missing"):
        decode_batch_request(document)
    for mutation, message in (
        (lambda value: value["loops"].append(copy.deepcopy(value["loops"][0])), "duplicate loop_id"),
        (lambda value: value["loops"][0].update({"proposers": []}), "at least one proposer"),
    ):
        document = encode_batch_request(request())
        mutation(document)
        with pytest.raises(RequestValidationError, match=message):
            decode_batch_request(document)
