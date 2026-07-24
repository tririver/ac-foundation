from __future__ import annotations

import pytest

from arc_llm import (
    CandidateConflictError,
    InteractionResponse,
    InteractiveJsonOutput,
    InvalidRequestError,
    JsonOutput,
    OperationContract,
)
from arc_llm.interaction import decode_interactive_turn, validate_responses
from arc_llm.output import CandidateMaterial, candidate_digest, select_output


def test_json_output_repairs_syntax_but_not_missing_business_fields() -> None:
    contract = JsonOutput(
        {
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "integer"}},
            "additionalProperties": False,
        }
    )
    assert select_output((CandidateMaterial(text='{"answer": 3,}', terminal=True),), contract) == {
        "answer": 3
    }
    with pytest.raises(Exception):
        select_output((CandidateMaterial(text="{}", terminal=True),), contract)


def test_non_equivalent_valid_candidates_require_explicit_selection() -> None:
    contract = JsonOutput({"type": "object"})
    candidates = (
        CandidateMaterial(value={"x": 1}),
        CandidateMaterial(value={"x": 2}),
    )
    with pytest.raises(CandidateConflictError):
        select_output(candidates, contract)
    selected = candidate_digest({"x": 2})
    assert select_output(candidates, contract, selected_digest=selected) == {"x": 2}


def test_nested_object_is_not_misclassified_as_a_second_candidate() -> None:
    contract = JsonOutput({"type": "object"})
    assert select_output(
        (CandidateMaterial(text='prefix {"outer":{"inner":1}} suffix'),),
        contract,
    ) == {"outer": {"inner": 1}}


def test_structured_null_is_a_real_candidate_not_missing_material() -> None:
    assert select_output(
        (CandidateMaterial(value=None, terminal=True),),
        JsonOutput({"type": "null"}),
    ) is None


def _interactive_contract() -> InteractiveJsonOutput:
    return InteractiveJsonOutput(
        {"type": "object", "required": ["answer"]},
        {
            "lookup": OperationContract(
                {"type": "object", "required": ["query"]},
                {"type": "object", "required": ["value"]},
            )
        },
    )


def test_interaction_is_operation_opaque_and_binds_exact_response_ids() -> None:
    turn = decode_interactive_turn(
        {
            "schema_version": "arc.llm.interactive_turn.v1",
            "state": "interact",
            "result": None,
            "requests": [
                {
                    "request_id": "req-1",
                    "operation": "lookup",
                    "arguments": {"query": "q"},
                }
            ],
        },
        _interactive_contract(),
    )
    responses = (InteractionResponse("req-1", result={"value": "v"}),)
    assert validate_responses(turn, responses, _interactive_contract()) == responses
    with pytest.raises(InvalidRequestError):
        validate_responses(
            turn,
            (InteractionResponse("wrong", result={"value": "v"}),),
            _interactive_contract(),
        )
