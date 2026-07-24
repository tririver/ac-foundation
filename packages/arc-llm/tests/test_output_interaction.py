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
    assert select_output(
        (CandidateMaterial(text='prefix {"text":"escaped \\" } [ {"} suffix'),),
        contract,
    ) == {"text": 'escaped " } [ {'}


def test_structured_null_is_a_real_candidate_not_missing_material() -> None:
    assert select_output(
        (CandidateMaterial(value=None, terminal=True),),
        JsonOutput({"type": "null"}),
    ) is None


@pytest.mark.parametrize(
    ("text", "schema", "expected"),
    [
        ("false", {"type": "boolean"}, False),
        ("0", {"type": "integer"}, 0),
        ('noise {"a": 1} tail', {"type": "object"}, {"a": 1}),
        ('```json\n{"a": 1,}\n```', {"type": "object"}, {"a": 1}),
    ],
)
def test_candidate_extraction_and_local_repair_matrix(text, schema, expected) -> None:
    assert select_output(
        (CandidateMaterial(text=text, terminal=True),),
        JsonOutput(schema),
    ) == expected


def test_candidate_selection_rejects_nonfinite_and_schema_invalid_values() -> None:
    contract = JsonOutput({"type": "object", "required": ["answer"]})
    with pytest.raises(Exception):
        select_output(
            (CandidateMaterial(text='{"answer": NaN}', terminal=True),),
            contract,
        )
    with pytest.raises(Exception):
        select_output(
            (CandidateMaterial(value={"other": 1}, terminal=True),),
            contract,
        )


def test_local_repair_prefers_repaired_root_over_nested_array_candidate() -> None:
    contract = JsonOutput(
        {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"],
            "additionalProperties": False,
        },
        repair="local",
    )
    assert select_output(
        (CandidateMaterial(text='{"items":["fixed"],}', terminal=True),),
        contract,
    ) == {"items": ["fixed"]}
    with pytest.raises(Exception):
        select_output(
            (CandidateMaterial(text='"items":["rootless"]}', terminal=True),),
            contract,
        )


def test_invalid_root_never_promotes_a_valid_nested_object() -> None:
    contract = JsonOutput(
        {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        repair="local",
    )
    for text in (
        '{"wrapper":{"answer":7},',
        'noise {"wrapper":{"answer":7},',
        'noise {"x": invalid, "nested":{"answer":7}}',
    ):
        with pytest.raises(Exception):
            select_output(
                (CandidateMaterial(text=text, terminal=True),),
                contract,
            )


def test_prose_may_contain_multiple_complete_top_level_json_values() -> None:
    contract = JsonOutput(
        {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )
    with pytest.raises(CandidateConflictError):
        select_output(
            (
                CandidateMaterial(
                    text='first {"answer":1} then {"answer":2} end',
                    terminal=True,
                ),
            ),
            contract,
        )


def test_json_openers_inside_double_quoted_prose_are_ignored() -> None:
    contract = JsonOutput(
        {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )
    assert select_output(
        (
            CandidateMaterial(
                text='quoted "{\\"answer\\":99}" then {"answer":1}',
                terminal=True,
            ),
        ),
        contract,
    ) == {"answer": 1}
    with pytest.raises(CandidateConflictError):
        select_output(
            (
                CandidateMaterial(
                    text='apostrophe \'{"answer":99}\' then {"answer":1}',
                    terminal=True,
                ),
            ),
            contract,
        )


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
