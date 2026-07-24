"""Shared text/JSON candidate extraction, repair, and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from json_repair import repair_json
from jsonschema import Draft202012Validator

from .errors import CandidateConflictError, OutputInvalidError
from .identity import canonical_json_bytes
from .request import InteractiveJsonOutput, JsonOutput, OutputContract, TextOutput


_MISSING = object()


@dataclass(frozen=True)
class CandidateMaterial:
    value: Any = field(default=_MISSING, repr=False)
    text: str | None = None
    terminal: bool = False

    def __post_init__(self) -> None:
        if self.has_value == (self.text is not None):
            raise ValueError("Candidate material has exactly one of value or text.")

    @property
    def has_value(self) -> bool:
        return self.value is not _MISSING


@dataclass(frozen=True)
class ValidCandidate:
    value: Any
    digest: str
    terminal: bool


def candidate_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def select_output(
    materials: Iterable[CandidateMaterial],
    contract: OutputContract,
    *,
    selected_digest: str | None = None,
) -> Any:
    material_list = tuple(materials)
    if isinstance(contract, TextOutput):
        return _select_text(material_list)
    valid: list[ValidCandidate] = []
    for material in material_list:
        for value in _values(
            material,
            allow_repair=(
                isinstance(contract, JsonOutput) and contract.repair == "local"
            ),
        ):
            if _valid_json_value(value, contract):
                valid.append(ValidCandidate(value, candidate_digest(value), material.terminal))
    if not valid:
        raise OutputInvalidError()
    by_digest: dict[str, ValidCandidate] = {}
    for item in valid:
        by_digest[item.digest] = item
    if selected_digest is not None:
        try:
            return by_digest[selected_digest].value
        except KeyError as exc:
            raise OutputInvalidError("The selected candidate is not a saved valid candidate.") from exc
    if len(by_digest) > 1:
        terminal = [item for item in valid if item.terminal]
        terminal_digests = {item.digest for item in terminal}
        if len(terminal_digests) == 1:
            return terminal[-1].value
        raise CandidateConflictError(tuple(sorted(by_digest)))
    return valid[-1].value


def validate_value(value: Any, contract: OutputContract) -> None:
    if isinstance(contract, TextOutput):
        if not isinstance(value, str) or not value:
            raise OutputInvalidError("Adopted text output must be a non-empty string.")
        return
    if not _valid_json_value(value, contract):
        raise OutputInvalidError("The value does not satisfy the output contract.")


def provider_schema(contract: OutputContract) -> dict[str, Any] | None:
    if isinstance(contract, TextOutput):
        return None
    if isinstance(contract, JsonOutput):
        return dict(contract.schema)
    return _interactive_schema(contract)


def _select_text(materials: tuple[CandidateMaterial, ...]) -> str:
    values = [
        item.text if item.text is not None else item.value
        for item in materials
        if (item.text is not None and item.text.strip())
        or (isinstance(item.value, str) and item.value.strip())
    ]
    if not values:
        raise OutputInvalidError("Provider returned no substantive text.")
    terminal = [
        item.text if item.text is not None else item.value
        for item in materials
        if item.terminal
        and ((item.text is not None and item.text.strip()) or isinstance(item.value, str))
    ]
    return terminal[-1] if terminal else values[-1]


def _values(material: CandidateMaterial, *, allow_repair: bool) -> tuple[Any, ...]:
    if material.has_value:
        return (material.value,)
    assert material.text is not None
    result: list[Any] = []
    parsed_complete = False
    try:
        result.append(json.loads(material.text))
        parsed_complete = True
    except (json.JSONDecodeError, ValueError):
        pass
    has_json_root = _has_json_root(material.text)
    if not parsed_complete and not has_json_root:
        result.extend(_complete_json_values(material.text))
    if not parsed_complete and allow_repair and has_json_root:
        try:
            repaired = repair_json(material.text, return_objects=True)
        except Exception:
            repaired = None
        if isinstance(repaired, (dict, list, str, int, float, bool)):
            if repaired != "":
                result.append(repaired)
    unique: dict[str, Any] = {}
    for value in result:
        try:
            unique[candidate_digest(value)] = value
        except (TypeError, ValueError):
            continue
    return tuple(unique.values())


def _has_json_root(text: str) -> bool:
    candidate = text.lstrip()
    if candidate.startswith("```"):
        newline = candidate.find("\n")
        if newline < 0:
            return False
        candidate = candidate[newline + 1 :].lstrip()
    return candidate.startswith(("{", "["))


def _complete_json_values(text: str) -> tuple[Any, ...]:
    values: list[Any] = []
    start: int | None = None
    stack: list[str] = []
    in_string = False
    escaped = False
    in_prose_string = False
    prose_escaped = False
    for index, char in enumerate(text):
        if start is None:
            if in_prose_string:
                if prose_escaped:
                    prose_escaped = False
                elif char == "\\":
                    prose_escaped = True
                elif char == '"':
                    in_prose_string = False
                continue
            if char == '"':
                in_prose_string = True
                continue
            if char not in "[{":
                continue
            start = index
            stack.append(char)
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "[{":
            stack.append(char)
            continue
        if char not in "]}":
            continue
        expected = "[" if char == "]" else "{"
        if not stack or stack[-1] != expected:
            # A mismatched closer does not make a nested opener top-level.
            # Keep the conservative outer boundary until its typed stack
            # actually closes, or discard it as unfinished at end of input.
            continue
        stack.pop()
        if stack:
            continue
        try:
            value = json.loads(text[start : index + 1])
        except (json.JSONDecodeError, ValueError):
            pass
        else:
            values.append(value)
        start = None
    return tuple(values)


def _valid_json_value(value: Any, contract: JsonOutput | InteractiveJsonOutput) -> bool:
    schema = contract.schema if isinstance(contract, JsonOutput) else _interactive_schema(contract)
    return not tuple(Draft202012Validator(schema).iter_errors(value))


def _interactive_schema(contract: InteractiveJsonOutput) -> dict[str, Any]:
    request_variants = []
    for operation, operation_contract in contract.operations.items():
        request_variants.append(
            {
                "type": "object",
                "properties": {
                    "request_id": {"type": "string"},
                    "operation": {"const": operation},
                    "arguments": dict(operation_contract.arguments_schema),
                },
                "required": ["request_id", "operation", "arguments"],
                "additionalProperties": False,
            }
        )
    request_schema: dict[str, Any] = {"oneOf": request_variants} if request_variants else False
    return {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "schema_version": {"const": "arc.llm.interactive_turn.v1"},
                    "state": {"const": "complete"},
                    "result": dict(contract.result_schema),
                    "requests": {"type": "array", "maxItems": 0},
                },
                "required": ["schema_version", "state", "result", "requests"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "schema_version": {"const": "arc.llm.interactive_turn.v1"},
                    "state": {"const": "interact"},
                    "result": {"type": "null"},
                    "requests": {
                        "type": "array",
                        "minItems": 1,
                        "items": request_schema,
                    },
                },
                "required": ["schema_version", "state", "result", "requests"],
                "additionalProperties": False,
            },
        ]
    }
