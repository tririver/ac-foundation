from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from ..sources import SourceArtifact, SourceFormat, SourceOrigin, SourceOriginKind


PARSED_DOCUMENT_SCHEMA = "ac.document.parsed_document.v2"


class MathSpanKind(str, Enum):
    INLINE = "inline"
    DISPLAY = "display"


@dataclass(frozen=True)
class MathSpan:
    """One source-positioned mathematical span.

    ``span_id`` commits to source content, position, kind, and normalized TeX.
    It intentionally excludes local paths and repository locations.
    """

    span_id: str
    kind: MathSpanKind
    source_line_start: int | None
    source_column_start: int | None
    source_line_end: int | None
    source_column_end: int | None
    normalized_tex: str
    context_before: str = ""
    context_after: str = ""
    source_label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", MathSpanKind(self.kind))
        lines = (self.source_line_start, self.source_line_end)
        columns = (self.source_column_start, self.source_column_end)
        if (lines[0] is None) != (lines[1] is None):
            raise ValueError("math span line positions must be paired")
        if (columns[0] is None) != (columns[1] is None):
            raise ValueError("math span column positions must be paired")
        if lines[0] is None and columns[0] is not None:
            raise ValueError("math span columns require line positions")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in (*lines, *columns)
            if value is not None
        ):
            raise ValueError("math span positions must be positive integers")
        if lines[0] is not None and lines[1] < lines[0]:
            raise ValueError("math span end line cannot precede its start")
        if (
            lines[0] is not None
            and lines[0] == lines[1]
            and columns[0] is not None
            and columns[1] < columns[0]
        ):
            raise ValueError("math span end column cannot precede its start")
        if not self.span_id or not self.normalized_tex:
            raise ValueError("math span requires an ID and normalized TeX")

    @property
    def context(self) -> str:
        return "\n".join(
            value for value in (self.context_before, self.context_after) if value
        )


@dataclass(frozen=True)
class ParsedSection:
    section_id: str
    title: str
    level: int
    text: str
    ordinal: int
    page_start: int | None = None
    page_end: int | None = None

    def __post_init__(self) -> None:
        if not self.section_id or self.level < 1 or self.ordinal < 0:
            raise ValueError("parsed section metadata is invalid")


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("parsed page number must be positive")


@dataclass(frozen=True)
class ParsedDocument:
    """Format-neutral deterministic parse result."""

    source: SourceArtifact
    sections: tuple[ParsedSection, ...] = ()
    math_spans: tuple[MathSpan, ...] = ()
    pages: tuple[ParsedPage, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PARSED_DOCUMENT_SCHEMA
    document_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != PARSED_DOCUMENT_SCHEMA:
            raise ValueError("unsupported parsed document schema")
        sections = tuple(self.sections)
        spans = tuple(self.math_spans)
        pages = tuple(self.pages)
        if len({item.section_id for item in sections}) != len(sections):
            raise ValueError("parsed document contains duplicate section IDs")
        if len({item.span_id for item in spans}) != len(spans):
            raise ValueError("parsed document contains duplicate math span IDs")
        if tuple(item.ordinal for item in sections) != tuple(range(len(sections))):
            raise ValueError("parsed section ordinals must be contiguous")
        if tuple(item.page_number for item in pages) != tuple(range(1, len(pages) + 1)):
            raise ValueError("parsed page numbers must be contiguous")
        metadata = MappingProxyType(dict(self.metadata))
        material = {
            "schema": self.schema_version,
            "source": self.source.content_identity,
            "sections": [
                (item.section_id, item.title, item.level, item.text, item.ordinal)
                for item in sections
            ],
            "math_spans": [
                (
                    item.span_id,
                    item.kind.value,
                    item.source_line_start,
                    item.source_column_start,
                    item.source_line_end,
                    item.source_column_end,
                    item.normalized_tex,
                    item.context_before,
                    item.context_after,
                    item.source_label,
                )
                for item in spans
            ],
            "pages": [(item.page_number, item.text) for item in pages],
        }
        digest = hashlib.sha256(
            json.dumps(material, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "math_spans", spans)
        object.__setattr__(self, "pages", pages)
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "document_digest", digest)

    @property
    def source_format(self) -> SourceFormat:
        return self.source.source_format

_DOCUMENT_FIELDS = {
    "schema_version",
    "document_digest",
    "source",
    "sections",
    "math_spans",
    "pages",
    "warnings",
    "metadata",
}
_SOURCE_FIELDS = {"source_format", "artifact_digest", "size", "media_type"}
_SECTION_FIELDS = {
    "section_id",
    "title",
    "level",
    "text",
    "ordinal",
    "page_start",
    "page_end",
}
_MATH_SPAN_FIELDS = {
    "span_id",
    "kind",
    "source_line_start",
    "source_column_start",
    "source_line_end",
    "source_column_end",
    "normalized_tex",
    "context_before",
    "context_after",
    "source_label",
}
_PAGE_FIELDS = {"page_number", "text"}


def parsed_document_to_document(parsed: ParsedDocument) -> dict[str, Any]:
    """Encode a strict, path-free artifact document for ac-jobs publication."""

    return {
        "schema_version": parsed.schema_version,
        "document_digest": parsed.document_digest,
        "source": {
            "source_format": parsed.source.source_format.value,
            "artifact_digest": parsed.source.artifact_digest,
            "size": parsed.source.size,
            "media_type": parsed.source.media_type,
        },
        "sections": [
            {
                "section_id": item.section_id,
                "title": item.title,
                "level": item.level,
                "text": item.text,
                "ordinal": item.ordinal,
                "page_start": item.page_start,
                "page_end": item.page_end,
            }
            for item in parsed.sections
        ],
        "math_spans": [
            {
                "span_id": item.span_id,
                "kind": item.kind.value,
                "source_line_start": item.source_line_start,
                "source_column_start": item.source_column_start,
                "source_line_end": item.source_line_end,
                "source_column_end": item.source_column_end,
                "normalized_tex": item.normalized_tex,
                "context_before": item.context_before,
                "context_after": item.context_after,
                "source_label": item.source_label,
            }
            for item in parsed.math_spans
        ],
        "pages": [
            {"page_number": item.page_number, "text": item.text}
            for item in parsed.pages
        ],
        "warnings": list(parsed.warnings),
        "metadata": dict(parsed.metadata),
    }


def parsed_document_from_document(value: Mapping[str, Any]) -> ParsedDocument:
    """Decode the strict v2 artifact contract and revalidate its digest."""

    _require_fields(value, _DOCUMENT_FIELDS, "parsed document")
    if value.get("schema_version") != PARSED_DOCUMENT_SCHEMA:
        raise ValueError("unsupported parsed document schema")
    source_value = _mapping(value.get("source"), "parsed document source")
    _require_fields(source_value, _SOURCE_FIELDS, "parsed document source")
    source = SourceArtifact(
        source_format=SourceFormat(_string(source_value, "source_format")),
        artifact_digest=_string(source_value, "artifact_digest"),
        size=_integer(source_value, "size"),
        media_type=_string(source_value, "media_type"),
        origin=SourceOrigin(
            SourceOriginKind.REPOSITORY,
            locator=(
                f"{source_value['source_format']}/sha256/"
                f"{source_value['artifact_digest']}"
            ),
        ),
    )
    sections = []
    for raw in _list(value, "sections"):
        item = _mapping(raw, "parsed section")
        _require_fields(item, _SECTION_FIELDS, "parsed section")
        sections.append(
            ParsedSection(
                section_id=_string(item, "section_id"),
                title=_string(item, "title"),
                level=_integer(item, "level"),
                text=_string(item, "text"),
                ordinal=_integer(item, "ordinal"),
                page_start=_optional_integer(item, "page_start"),
                page_end=_optional_integer(item, "page_end"),
            )
        )
    spans = []
    for raw in _list(value, "math_spans"):
        item = _mapping(raw, "math span")
        _require_fields(item, _MATH_SPAN_FIELDS, "math span")
        spans.append(
            MathSpan(
                span_id=_string(item, "span_id"),
                kind=MathSpanKind(_string(item, "kind")),
                source_line_start=_optional_integer(item, "source_line_start"),
                source_column_start=_optional_integer(item, "source_column_start"),
                source_line_end=_optional_integer(item, "source_line_end"),
                source_column_end=_optional_integer(item, "source_column_end"),
                normalized_tex=_string(item, "normalized_tex"),
                context_before=_string(item, "context_before"),
                context_after=_string(item, "context_after"),
                source_label=_string(item, "source_label"),
            )
        )
    pages = []
    for raw in _list(value, "pages"):
        item = _mapping(raw, "parsed page")
        _require_fields(item, _PAGE_FIELDS, "parsed page")
        pages.append(
            ParsedPage(
                page_number=_integer(item, "page_number"),
                text=_string(item, "text"),
            )
        )
    warnings = _list(value, "warnings")
    if any(not isinstance(item, str) for item in warnings):
        raise ValueError("parsed document warnings must be strings")
    metadata = _mapping(value.get("metadata"), "parsed document metadata")
    parsed = ParsedDocument(
        source=source,
        sections=tuple(sections),
        math_spans=tuple(spans),
        pages=tuple(pages),
        warnings=tuple(warnings),
        metadata=metadata,
    )
    if parsed.document_digest != _string(value, "document_digest"):
        raise ValueError("parsed document digest does not match its content")
    return parsed


def _require_fields(
    value: Mapping[str, Any], expected: set[str], description: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{description} has invalid fields")


def _mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _list(value: Mapping[str, Any], key: str) -> list[Any]:
    item = value.get(key)
    if not isinstance(item, list):
        raise ValueError(f"parsed document {key} must be a list")
    return item


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise ValueError(f"{key} must be a string")
    return item


def _integer(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise ValueError(f"{key} must be an integer")
    return item


def _optional_integer(value: Mapping[str, Any], key: str) -> int | None:
    item = value.get(key)
    if item is None:
        return None
    return _integer(value, key)


@dataclass(frozen=True)
class PDFTextLayer:
    pages: tuple[str, ...]
    warning: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "pages", tuple(str(page) for page in self.pages))

    @property
    def has_text(self) -> bool:
        return any(page.strip() for page in self.pages)


@dataclass(frozen=True)
class VisualPageReviewInput:
    """Pure page scheduling descriptor for the in-run visual review service."""

    primary: SourceArtifact
    pdf_validator: SourceArtifact
    page_number: int
    math_spans: tuple[MathSpan, ...]

    def __post_init__(self) -> None:
        if self.pdf_validator.source_format is not SourceFormat.PDF:
            raise ValueError("visual page review requires a PDF validator")
        if self.page_number < 1:
            raise ValueError("visual page number must be positive")
        object.__setattr__(self, "math_spans", tuple(self.math_spans))

    @property
    def math_span_ids(self) -> tuple[str, ...]:
        return tuple(item.span_id for item in self.math_spans)


__all__ = [
    "MathSpan",
    "MathSpanKind",
    "PARSED_DOCUMENT_SCHEMA",
    "PDFTextLayer",
    "ParsedDocument",
    "ParsedPage",
    "ParsedSection",
    "VisualPageReviewInput",
    "parsed_document_from_document",
    "parsed_document_to_document",
]
