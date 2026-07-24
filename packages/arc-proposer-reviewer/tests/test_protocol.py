from __future__ import annotations

import copy

import pytest

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
from arc_llm import CapabilityPolicy, ModelSelection


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
                        capabilities=CapabilityPolicy(
                            internet=True, allowed_tools=("search",)
                        ),
                    ),
                ),
                reviewer=WorkerSpec(
                    worker_id="reviewer",
                    instructions="Review.",
                    output_schema=SCHEMA,
                ),
                max_rounds=2,
                allow_early_stop=False,
                on_proposer_failure=ProposerFailurePolicy.CONTINUE_IF_ANY,
            ),
        ),
        failure_policy=BatchFailurePolicy.FAIL_FAST,
    )


def test_batch_request_round_trips_without_losing_typed_llm_options() -> None:
    original = request()
    decoded = decode_batch_request(encode_batch_request(original))
    assert decoded == original
    proposer = decoded.loops[0].proposers[0]
    assert proposer.model.provider == "codex"
    assert proposer.model.model == "gpt-test"
    assert proposer.capabilities.internet is True
    assert proposer.capabilities.allowed_tools == ("search",)


def test_unknown_request_field_is_rejected() -> None:
    for field in ("surprise", "total_timeout"):
        document = encode_batch_request(request())
        document[field] = True
        with pytest.raises(RequestValidationError, match="unknown field"):
            decode_batch_request(document)


def test_string_boolean_is_rejected() -> None:
    document = encode_batch_request(request())
    document["loops"][0]["allow_early_stop"] = "false"  # type: ignore[index]
    with pytest.raises(RequestValidationError, match="must be a boolean"):
        decode_batch_request(document)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["loops"].append(copy.deepcopy(value["loops"][0])),
            "duplicate loop_id",
        ),
        (
            lambda value: value["loops"][0].update({"proposers": []}),
            "at least one proposer",
        ),
        (
            lambda value: value["loops"][0]["reviewer"].update(
                {"worker_id": "proposer-a"}
            ),
            "unique within the loop",
        ),
    ],
)
def test_batch_identity_and_worker_cardinality_validation(mutation, message: str) -> None:
    document = encode_batch_request(request())
    mutation(document)
    with pytest.raises(RequestValidationError, match=message):
        decode_batch_request(document)
