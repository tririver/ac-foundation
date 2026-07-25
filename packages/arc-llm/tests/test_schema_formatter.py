from __future__ import annotations

import pytest

from arc_llm import JsonOutput
from arc_llm.output import CandidateMaterial
from arc_llm.schema_formatter import (
    FormattingSource,
    SchemaFormatterError,
    decode_formatting_decision,
    formatter_task_id,
    select_formatting_source,
)


def test_json_output_defaults_to_format_recovery() -> None:
    assert JsonOutput({"type": "object"}).repair == "format"


def test_formatting_source_prefers_one_substantive_terminal_candidate() -> None:
    source = select_formatting_source(
        (
            CandidateMaterial(text='{"draft":"old material"}'),
            CandidateMaterial(text='{"answer_text":"final material"}', terminal=True),
        )
    )

    assert source is not None
    assert source.text == '{"answer_text":"final material"}'


def test_formatting_source_rejects_low_content_and_ambiguous_candidates() -> None:
    assert select_formatting_source((CandidateMaterial(value={}),)) is None
    assert (
        select_formatting_source(
            (
                CandidateMaterial(text='{"answer_text":"first material"}'),
                CandidateMaterial(text='{"answer_text":"second material"}'),
            )
        )
        is None
    )


def test_formatter_decision_validates_target_and_rejects_fabricated_numbers() -> None:
    source = FormattingSource("Score 92 and explanation present.", "0" * 64)
    schema = {
        "type": "object",
        "properties": {"score": {"type": "number"}},
        "required": ["score"],
        "additionalProperties": False,
    }

    decision = decode_formatting_decision(
        {
            "action": "format",
            "reason": "content present",
            "formatted_output": {"score": 92},
        },
        source=source,
        target_schema=schema,
    )
    assert decision.value == {"score": 92}

    with pytest.raises(SchemaFormatterError, match="fabricated_numbers"):
        decode_formatting_decision(
            {
                "action": "format",
                "reason": "invented",
                "formatted_output": {"score": 88},
            },
            source=source,
            target_schema=schema,
        )


def test_formatter_task_identity_is_deterministic_and_generation_bound() -> None:
    first = formatter_task_id(
        outer_semantic_key="a" * 64,
        generation=1,
        source_sha256="b" * 64,
    )
    assert first == formatter_task_id(
        outer_semantic_key="a" * 64,
        generation=1,
        source_sha256="b" * 64,
    )
    assert first != formatter_task_id(
        outer_semantic_key="a" * 64,
        generation=2,
        source_sha256="b" * 64,
    )
