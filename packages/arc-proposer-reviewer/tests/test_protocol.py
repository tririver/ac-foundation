from __future__ import annotations

import copy

import pytest

from arc_jobs import ArtifactDigest, ArtifactSourceRef
from arc_llm import LLMInputArtifact, ModelSelection
from arc_proposer_reviewer.models import (
    BATCH_SCHEMA_VERSION,
    BatchFailurePolicy,
    BatchRequest,
    LoopSpec,
    ProposerFailurePolicy,
    WorkerSpec,
)
from arc_proposer_reviewer.protocol import decode_batch_request, encode_batch_request
from arc_proposer_reviewer.validation import RequestValidationError


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
    assert decode_batch_request(encoded) == original


def test_v4_request_decodes_with_final_reviewer_enabled_by_default() -> None:
    legacy = encode_batch_request(request())
    legacy["schema_version"] = "arc.proposer_reviewer.batch.v4"
    del legacy["loops"][0]["review_final_round"]  # type: ignore[index]

    decoded = decode_batch_request(legacy)

    assert decoded.schema_version == BATCH_SCHEMA_VERSION
    assert decoded.loops[0].review_final_round is True
    assert encode_batch_request(decoded)["schema_version"] == BATCH_SCHEMA_VERSION
    assert encode_batch_request(decoded)["loops"][0]["review_final_round"] is True  # type: ignore[index]


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
