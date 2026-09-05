"""Versioned, local diagnostics for optional RichDocument projections."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

DOCUMENT_DIAGNOSTICS_METADATA_KEY = "document_diagnostics"
DOCUMENT_DIAGNOSTICS_SCHEMA = "ac.document.document_diagnostics.v1"

_ENVELOPE_FIELDS = {"schema_version", "projections", "visible_content"}
_PROJECTION_FIELDS = {
    "category",
    "scope",
    "locator",
    "status",
    "fallback",
    "evidence",
}
_VISIBLE_CONTENT_FIELDS = {
    "visible_units",
    "emitted",
    "documented_exclusions",
    "opaque",
    "unaccounted",
}
_LOCATOR_FIELDS = {
    "source_format",
    "line_start",
    "column_start",
    "line_end",
    "column_end",
    "selector",
    "source_id",
}
_STATUS = {"exact", "neutral", "unavailable"}
_FALLBACK = {
    "none",
    "neutral_projection",
    "omit_projection",
    "plain_block",
    "documented_exclusion",
}
_CATEGORY_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")


def document_diagnostics(document: Any) -> Mapping[str, Any] | None:
    """Return validated optional-projection and visible-content diagnostics."""

    if DOCUMENT_DIAGNOSTICS_METADATA_KEY not in document.metadata:
        return None
    value = document.metadata[DOCUMENT_DIAGNOSTICS_METADATA_KEY]
    validate_document_diagnostics(value, source=document.source)
    return value


def validate_document_diagnostics(value: Any, *, source: Any) -> None:
    """Validate the small, serializable P0 degradation ledger."""

    envelope = _mapping(value, "document diagnostics must be an object")
    if set(envelope) != _ENVELOPE_FIELDS:
        _invalid("document diagnostics has invalid fields")
    if envelope.get("schema_version") != DOCUMENT_DIAGNOSTICS_SCHEMA:
        _invalid("document diagnostics has an unsupported schema")
    projections = _sequence(
        envelope.get("projections"), "document diagnostics projections must be a list"
    )
    previous = None
    for raw in projections:
        projection = _mapping(raw, "document diagnostic must be an object")
        if set(projection) != _PROJECTION_FIELDS:
            _invalid("document diagnostic has invalid fields")
        category = _string(projection, "category")
        if _CATEGORY_RE.fullmatch(category) is None:
            _invalid("document diagnostic category is invalid")
        scope = _string(projection, "scope")
        if not scope:
            _invalid("document diagnostic scope is empty")
        _validate_locator(projection.get("locator"), source.source_format.value)
        status = _string(projection, "status")
        fallback = _string(projection, "fallback")
        if status not in _STATUS or fallback not in _FALLBACK:
            _invalid("document diagnostic status or fallback is unknown")
        if status == "exact" and fallback != "none":
            _invalid("exact document diagnostic has a fallback")
        if status != "exact" and fallback == "none":
            _invalid("degraded document diagnostic lacks a fallback")
        evidence = _sequence(
            projection.get("evidence"), "document diagnostic evidence must be a list"
        )
        if not evidence or len(set(evidence)) != len(evidence):
            _invalid("document diagnostic evidence is empty or duplicate")
        if any(
            not isinstance(item, str) or not item or len(item) > 160
            for item in evidence
        ):
            _invalid("document diagnostic evidence is invalid")
        current = (category, scope)
        if previous is not None and current < previous:
            _invalid("document diagnostics are not deterministically ordered")
        previous = current
    visible = _mapping(
        envelope.get("visible_content"), "visible content accounting must be an object"
    )
    if set(visible) != _VISIBLE_CONTENT_FIELDS:
        _invalid("visible content accounting has invalid fields")
    values = {key: _integer(visible, key) for key in _VISIBLE_CONTENT_FIELDS}
    if any(value < 0 for value in values.values()):
        _invalid("visible content accounting is negative")
    if values["visible_units"] != sum(
        values[key]
        for key in ("emitted", "documented_exclusions", "opaque", "unaccounted")
    ):
        _invalid("visible content accounting does not reconcile")
    if values["unaccounted"] != 0:
        _invalid("visible content accounting contains unaccounted units")


def _validate_locator(value: Any, source_format: str) -> None:
    locator = _mapping(value, "document diagnostic locator must be an object")
    if set(locator) != _LOCATOR_FIELDS:
        _invalid("document diagnostic locator has invalid fields")
    if locator.get("source_format") != source_format:
        _invalid("document diagnostic locator format differs from source")
    positions = [
        locator.get(key)
        for key in ("line_start", "column_start", "line_end", "column_end")
    ]
    if any(
        value is not None and (not isinstance(value, int) or isinstance(value, bool))
        for value in positions
    ):
        _invalid("document diagnostic locator positions are invalid")
    if any(value is not None and value < 1 for value in positions):
        _invalid("document diagnostic locator positions are invalid")
    if (positions[0] is None) != (positions[2] is None):
        _invalid("document diagnostic locator line range is incomplete")
    if (positions[1] is None) != (positions[3] is None):
        _invalid("document diagnostic locator column range is incomplete")
    if positions[1] is not None and positions[0] is None:
        _invalid("document diagnostic locator columns lack lines")
    if any(not isinstance(locator.get(key), str) for key in ("selector", "source_id")):
        _invalid("document diagnostic locator identity is invalid")


def _mapping(value: Any, message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _invalid(message)
    return value


def _sequence(value: Any, message: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        _invalid(message)
    return value


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        _invalid(f"document diagnostic {key} must be a string")
    return item


def _integer(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        _invalid(f"visible content {key} must be an integer")
    return item


def _invalid(message: str) -> None:
    raise ValueError(message)


__all__ = [
    "DOCUMENT_DIAGNOSTICS_METADATA_KEY",
    "DOCUMENT_DIAGNOSTICS_SCHEMA",
    "document_diagnostics",
    "validate_document_diagnostics",
]
