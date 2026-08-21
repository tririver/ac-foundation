"""Verified, cache-only access to one content-addressed document.

The public reference in this module is a logical identity, not a filesystem
path.  Every operation reopens the source through :class:`SourceRepository`
and rematerializes the deterministic parsed projection when its derived cache
is absent or damaged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .document_search import TableOfContentsEntry
from .sources import SourceFormat


CACHED_DOCUMENT_REF_SCHEMA = "ac.document.cached_document_ref.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REF_FIELDS = {
    "source_format",
    "source_sha256",
    "source_size",
    "media_type",
    "parser_contract",
    "parsed_document_sha256",
}


class CachedDocumentError(RuntimeError):
    """A stable cache-only document failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class CachedDocumentRef:
    """Immutable logical handle for one verified parsed source."""

    source_format: SourceFormat | str
    source_sha256: str
    source_size: int
    media_type: str
    parser_contract: str
    parsed_document_sha256: str

    def __post_init__(self) -> None:
        try:
            source_format = SourceFormat(self.source_format)
        except (TypeError, ValueError) as exc:
            raise ValueError("source_format is unsupported") from exc
        source_sha256 = str(self.source_sha256).casefold()
        parsed_sha256 = str(self.parsed_document_sha256).casefold()
        media_type = str(self.media_type).strip().casefold()
        parser_contract = str(self.parser_contract).strip()
        if _SHA256_RE.fullmatch(source_sha256) is None:
            raise ValueError("source_sha256 must be a SHA-256 digest")
        if _SHA256_RE.fullmatch(parsed_sha256) is None:
            raise ValueError("parsed_document_sha256 must be a SHA-256 digest")
        if (
            not isinstance(self.source_size, int)
            or isinstance(self.source_size, bool)
            or self.source_size < 0
        ):
            raise ValueError("source_size cannot be negative")
        if not media_type or "/" not in media_type or ";" in media_type:
            raise ValueError("media_type must be a normalized MIME type")
        if not parser_contract:
            raise ValueError("parser_contract is required")
        object.__setattr__(self, "source_format", source_format)
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "parser_contract", parser_contract)
        object.__setattr__(self, "parsed_document_sha256", parsed_sha256)


@dataclass(frozen=True)
class CachedTableOfContents:
    document: CachedDocumentRef
    entries: tuple[TableOfContentsEntry, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CachedSection:
    document: CachedDocumentRef
    section_id: str
    title: str
    text: str
    level: int
    ordinal: int
    page_start: int | None
    page_end: int | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CachedSourceRange:
    document: CachedDocumentRef
    start_line: int
    end_line: int
    total_lines: int
    text: str


def cached_document_ref_to_document(value: CachedDocumentRef) -> dict[str, Any]:
    if not isinstance(value, CachedDocumentRef):
        raise TypeError("value must be a CachedDocumentRef")
    return {
        "source_format": value.source_format.value,
        "source_sha256": value.source_sha256,
        "source_size": value.source_size,
        "media_type": value.media_type,
        "parser_contract": value.parser_contract,
        "parsed_document_sha256": value.parsed_document_sha256,
    }


def cached_document_ref_from_document(
    value: Mapping[str, Any],
) -> CachedDocumentRef:
    if not isinstance(value, Mapping) or set(value) != _REF_FIELDS:
        raise ValueError("cached document reference has invalid fields")
    try:
        return CachedDocumentRef(
            source_format=value["source_format"],
            source_sha256=value["source_sha256"],
            source_size=value["source_size"],
            media_type=value["media_type"],
            parser_contract=value["parser_contract"],
            parsed_document_sha256=value["parsed_document_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid cached document reference: {exc}") from exc


__all__ = [
    "CACHED_DOCUMENT_REF_SCHEMA",
    "CachedDocumentError",
    "CachedDocumentRef",
    "CachedSection",
    "CachedSourceRange",
    "CachedTableOfContents",
    "cached_document_ref_from_document",
    "cached_document_ref_to_document",
]
