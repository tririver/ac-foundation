from __future__ import annotations

import copy

import pytest

from arc_llm import ModelSelection
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
