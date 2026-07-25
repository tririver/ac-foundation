from __future__ import annotations

import copy
from dataclasses import replace

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
from arc_llm import CapabilityPolicy, ModelSelection, OperationContract


SCHEMA = {
    "type": "object",
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
    "additionalProperties": False,
}

LOOKUP_OPERATION = OperationContract(
    arguments_schema={
        "type": "object",
        "required": ["query"],
        "properties": {"query": {"type": "string"}},
        "additionalProperties": False,
    },
    response_schema={
        "type": "object",
        "required": ["value"],
        "properties": {"value": {"type": "string"}},
        "additionalProperties": False,
    },
)


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


def test_worker_interaction_contract_round_trips_as_closed_protocol() -> None:
    original = request()
    loop = original.loops[0]
    proposer = replace(
        loop.proposers[0],
        interaction_operations={"lookup": LOOKUP_OPERATION},
        # The arc-llm contract counts automatic resolved interaction turns;
        # two means turns one and two are automatic and a third pauses.
        max_interaction_turns=2,
    )
    reviewer = replace(
        loop.reviewer,
        interaction_operations={"lookup": LOOKUP_OPERATION},
        max_interaction_turns=3,
    )
    configured = BatchRequest(
        schema_version=original.schema_version,
        batch_id=original.batch_id,
        loops=(
            LoopSpec(
                loop_id=loop.loop_id,
                context=loop.context,
                proposers=(proposer,),
                reviewer=reviewer,
                max_rounds=loop.max_rounds,
                allow_early_stop=loop.allow_early_stop,
                on_proposer_failure=loop.on_proposer_failure,
            ),
        ),
        failure_policy=original.failure_policy,
    )

    encoded = encode_batch_request(configured)
    proposer_document = encoded["loops"][0]["proposers"][0]  # type: ignore[index]
    assert set(proposer_document) == {
        "worker_id",
        "instructions",
        "output_schema",
        "model",
        "capabilities",
        "interaction_operations",
        "max_interaction_turns",
    }
    assert proposer_document["interaction_operations"] == {  # type: ignore[index]
        "lookup": {
            "arguments_schema": dict(LOOKUP_OPERATION.arguments_schema),
            "response_schema": dict(LOOKUP_OPERATION.response_schema),
        }
    }
    assert proposer_document["max_interaction_turns"] == 2  # type: ignore[index]
    assert decode_batch_request(encoded) == configured

    proposer_document["interaction_operations"]["lookup"]["surprise"] = True  # type: ignore[index]
    with pytest.raises(RequestValidationError, match="unknown field"):
        decode_batch_request(encoded)


def test_default_worker_protocol_uses_the_complete_current_shape() -> None:
    encoded = encode_batch_request(request())
    proposer_document = encoded["loops"][0]["proposers"][0]  # type: ignore[index]
    assert set(proposer_document) == {
        "worker_id",
        "instructions",
        "output_schema",
        "model",
        "capabilities",
        "interaction_operations",
        "max_interaction_turns",
    }
    assert proposer_document["interaction_operations"] == {}
    assert proposer_document["max_interaction_turns"] == 2
    assert decode_batch_request(encoded) == request()


@pytest.mark.parametrize(
    "missing_field",
    ("interaction_operations", "max_interaction_turns"),
)
def test_worker_protocol_requires_the_complete_current_shape(
    missing_field: str,
) -> None:
    document = encode_batch_request(request())
    worker = document["loops"][0]["proposers"][0]  # type: ignore[index]
    del worker[missing_field]

    with pytest.raises(RequestValidationError, match="required field is missing"):
        decode_batch_request(document)


@pytest.mark.parametrize(
    "interaction_operations,max_turns",
    [
        ({"": LOOKUP_OPERATION}, 2),
        ({"lookup": object()}, 2),
        ({"lookup": LOOKUP_OPERATION}, 0),
    ],
)
def test_invalid_worker_interaction_contract_is_rejected(
    interaction_operations, max_turns: int
) -> None:
    original = request()
    loop = original.loops[0]
    invalid_proposer = replace(
        loop.proposers[0],
        interaction_operations=interaction_operations,
        max_interaction_turns=max_turns,
    )
    invalid = BatchRequest(
        schema_version=original.schema_version,
        batch_id=original.batch_id,
        loops=(
            LoopSpec(
                loop_id=loop.loop_id,
                context=loop.context,
                proposers=(invalid_proposer,),
                reviewer=loop.reviewer,
                max_rounds=loop.max_rounds,
                allow_early_stop=loop.allow_early_stop,
                on_proposer_failure=loop.on_proposer_failure,
            ),
        ),
        failure_policy=original.failure_policy,
    )
    with pytest.raises(RequestValidationError):
        encode_batch_request(invalid)


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
