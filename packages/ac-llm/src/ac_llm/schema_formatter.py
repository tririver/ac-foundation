"""Private, content-preserving schema-format recovery helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

from jsonschema import Draft202012Validator

from .identity import canonical_json_bytes
from .output import CandidateMaterial


FORMATTER_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["format", "insufficient"]},
        "reason": {"type": "string"},
        "formatted_output": {},
    },
    "required": ["action", "reason", "formatted_output"],
    "additionalProperties": False,
}

_LOW_CONTENT_CHARACTER_THRESHOLD = 10


class SchemaFormatterError(ValueError):
    """The formatter result cannot safely be adopted."""


@dataclass(frozen=True)
class FormattingSource:
    text: str
    sha256: str


@dataclass(frozen=True)
class FormattingDecision:
    action: Literal["format", "insufficient"]
    reason: str
    value: Any = None


def select_formatting_source(
    materials: Iterable[CandidateMaterial],
) -> FormattingSource | None:
    """Select one unambiguous, substantive provider candidate."""

    unique: dict[str, tuple[str, bool]] = {}
    for material in materials:
        text = _material_text(material)
        if text is None or not _substantive(text):
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        previous = unique.get(digest)
        unique[digest] = (
            text,
            material.terminal or (previous is not None and previous[1]),
        )
    if not unique:
        return None
    terminal = [item for item in unique.values() if item[1]]
    selected: tuple[str, bool] | None
    if len(terminal) == 1:
        selected = terminal[0]
    elif not terminal and len(unique) == 1:
        selected = next(iter(unique.values()))
    else:
        return None
    return FormattingSource(selected[0], hashlib.sha256(selected[0].encode()).hexdigest())


def formatter_task_id(
    *,
    outer_semantic_key: str,
    generation: int,
    source_sha256: str,
) -> str:
    identity = (
        f"ac.llm.schema_formatter.v1\0{outer_semantic_key}\0"
        f"{generation}\0{source_sha256}"
    )
    return f"format-v1-{hashlib.sha256(identity.encode()).hexdigest()[:32]}"


def formatter_prompt(
    source: FormattingSource,
    *,
    schema: Mapping[str, Any],
) -> str:
    return (
        "You are a schema formatter. Reorganize only the information already "
        "present in the source into the requested JSON shape. Do not solve the "
        "original task again. Do not search, use tools, add evidence, invent "
        "scientific claims, scores, numbers, or judgments, or change the meaning.\n"
        "Return action=\"format\" only when every required content field can be "
        "filled from the source. Otherwise return action=\"insufficient\" with "
        "formatted_output=null. For action=\"format\", put the reformatted value "
        "in formatted_output. Return exactly one JSON object matching the supplied "
        "formatter decision schema.\n\n"
        "## Source\n"
        f"{source.text}\n\n"
        "## Target JSON Schema\n"
        f"{json.dumps(dict(schema), ensure_ascii=False, sort_keys=True)}"
    )


def decode_formatting_decision(
    value: Any,
    *,
    source: FormattingSource,
    target_schema: Mapping[str, Any],
) -> FormattingDecision:
    errors = tuple(Draft202012Validator(FORMATTER_DECISION_SCHEMA).iter_errors(value))
    if errors or not isinstance(value, dict):
        raise SchemaFormatterError("schema_formatter_invalid_decision")
    action = value["action"]
    reason = value["reason"]
    formatted = value["formatted_output"]
    if action == "insufficient":
        if formatted is not None:
            raise SchemaFormatterError("schema_formatter_invalid_insufficient_decision")
        return FormattingDecision("insufficient", reason)
    validation_errors = tuple(
        Draft202012Validator(dict(target_schema)).iter_errors(formatted)
    )
    if validation_errors:
        raise SchemaFormatterError("schema_formatter_target_validation_failed")
    missing_numbers = _numeric_values_not_in_source(formatted, source_text=source.text)
    if missing_numbers:
        raise SchemaFormatterError(
            "schema_formatter_fabricated_numbers:" + ",".join(missing_numbers)
        )
    return FormattingDecision("format", reason, formatted)


def _material_text(material: CandidateMaterial) -> str | None:
    if material.text is not None:
        return material.text.strip()
    try:
        return canonical_json_bytes(material.value).decode("utf-8")
    except (TypeError, ValueError, UnicodeDecodeError):
        return None


def _substantive(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped in {"{}", "[]"}:
        return False
    return sum(character.isalnum() for character in stripped) >= _LOW_CONTENT_CHARACTER_THRESHOLD


def _numeric_values_not_in_source(value: Any, *, source_text: str) -> list[str]:
    source_numbers = _source_numbers(source_text)
    missing: list[str] = []
    _collect_missing_numeric(value, source_numbers=source_numbers, path="", missing=missing)
    return missing


def _collect_missing_numeric(
    value: Any,
    *,
    source_numbers: set[str],
    path: str,
    missing: list[str],
) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not _number_in_source(value, source_numbers):
            missing.append(path or "$")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            _collect_missing_numeric(
                child,
                source_numbers=source_numbers,
                path=child_path,
                missing=missing,
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _collect_missing_numeric(
                child,
                source_numbers=source_numbers,
                path=f"{path}[{index}]",
                missing=missing,
            )


def _source_numbers(text: str) -> set[str]:
    numbers: set[str] = set()
    for match in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?", text):
        raw = match.group(0)
        numbers.add(raw)
        try:
            number = float(raw)
        except ValueError:
            continue
        numbers.add(str(int(number)) if number.is_integer() else str(number))
    return numbers


def _number_in_source(value: Any, source_numbers: set[str]) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return str(value) in source_numbers
    if isinstance(value, float):
        if value.is_integer() and str(int(value)) in source_numbers:
            return True
        return str(value) in source_numbers
    return False
