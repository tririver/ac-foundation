"""Content-addressed structural overlays for cached text documents.

An overlay reconciles a text source whose heading levels are unreliable with
the outline of an independently cached PDF.  It never mutates either source
or changes :class:`~arc_document.cached_document.CachedDocumentRef` identity.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ._durable_io import atomic_write_bytes
from ._file_lock import exclusive_file_lock
from .cached_document import (
    CachedDocumentError,
    CachedDocumentRef,
    cached_document_ref_from_document,
    cached_document_ref_to_document,
)
from .parse import PDFOutlineExtractionError, QpdfOutlineExtractor


DOCUMENT_STRUCTURE_OVERLAY_SCHEMA = "arc.document.document_structure_overlay.v1"
CACHED_DOCUMENT_STRUCTURE_REF_SCHEMA = (
    "arc.document.cached_document_structure_ref.v1"
)
DOCUMENT_STRUCTURE_CACHE_SCHEMA = "arc.document.document_structure_cache.v1"
DOCUMENT_STRUCTURE_CONTRACT = (
    "arc.document.document_structure.qpdf_outline_markdown_alignment.v2"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ATX_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_STRUCTURE_REF_FIELDS = {
    "document",
    "outline_document",
    "structure_contract",
    "structure_sha256",
}
_OVERLAY_FIELDS = {
    "schema_version",
    "document",
    "outline_document",
    "structure_contract",
    "structure_sha256",
    "entries",
    "warnings",
}
_ENTRY_FIELDS = {
    "section_id",
    "title",
    "level",
    "parent_id",
    "ordinal",
    "heading_line",
    "source_line_start",
    "source_line_end",
    "pdf_page_start",
    "pdf_page_end",
    "kind",
    "matching_method",
    "provenance",
}


class DocumentStructureError(CachedDocumentError):
    """Stable failure while reconstructing or reopening an overlay."""


class DocumentStructureNodeKind(str, Enum):
    CONTAINER = "container"
    CONTENT = "content"
    INTERNAL = "internal"
    STRUCTURAL = "structural"


@dataclass(frozen=True)
class DocumentStructureEntry:
    section_id: str
    title: str
    level: int
    parent_id: str | None
    ordinal: int
    heading_line: int
    source_line_start: int
    source_line_end: int
    pdf_page_start: int | None
    pdf_page_end: int | None
    kind: DocumentStructureNodeKind | str
    matching_method: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = DocumentStructureNodeKind(self.kind)
        if (
            not self.section_id
            or not self.title
            or self.level < 1
            or self.ordinal < 0
            or self.heading_line < 1
            or self.source_line_start != self.heading_line
            or self.source_line_end < self.source_line_start
        ):
            raise ValueError("document structure entry metadata is invalid")
        for value in (self.pdf_page_start, self.pdf_page_end):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError("document structure PDF pages must be positive")
        if (
            self.pdf_page_start is not None
            and self.pdf_page_end is not None
            and self.pdf_page_end < self.pdf_page_start
        ):
            raise ValueError("document structure PDF page range is inverted")
        if not self.matching_method:
            raise ValueError("document structure matching_method is required")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "provenance", MappingProxyType(dict(self.provenance))
        )


@dataclass(frozen=True)
class CachedDocumentStructureRef:
    """Logical handle tying one overlay to both immutable source documents."""

    document: CachedDocumentRef
    outline_document: CachedDocumentRef
    structure_contract: str
    structure_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.document, CachedDocumentRef):
            raise TypeError("document must be a CachedDocumentRef")
        if not isinstance(self.outline_document, CachedDocumentRef):
            raise TypeError("outline_document must be a CachedDocumentRef")
        contract = str(self.structure_contract).strip()
        digest = str(self.structure_sha256).casefold()
        if not contract:
            raise ValueError("structure_contract is required")
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("structure_sha256 must be a SHA-256 digest")
        object.__setattr__(self, "structure_contract", contract)
        object.__setattr__(self, "structure_sha256", digest)


@dataclass(frozen=True)
class DocumentStructureOverlay:
    document: CachedDocumentRef
    outline_document: CachedDocumentRef
    entries: tuple[DocumentStructureEntry, ...]
    warnings: tuple[str, ...] = ()
    structure_contract: str = DOCUMENT_STRUCTURE_CONTRACT
    schema_version: str = DOCUMENT_STRUCTURE_OVERLAY_SCHEMA
    structure_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != DOCUMENT_STRUCTURE_OVERLAY_SCHEMA:
            raise ValueError("unsupported document structure overlay schema")
        if not isinstance(self.document, CachedDocumentRef):
            raise TypeError("document must be a CachedDocumentRef")
        if not isinstance(self.outline_document, CachedDocumentRef):
            raise TypeError("outline_document must be a CachedDocumentRef")
        if self.document.source_format.value != "markdown":
            raise ValueError("document structure source must be Markdown")
        if self.outline_document.source_format.value != "pdf":
            raise ValueError("document structure outline source must be PDF")
        entries = tuple(self.entries)
        if not entries:
            raise ValueError("document structure overlay requires entries")
        if tuple(item.ordinal for item in entries) != tuple(range(len(entries))):
            raise ValueError("document structure ordinals must be contiguous")
        ids = {item.section_id for item in entries}
        if len(ids) != len(entries):
            raise ValueError("document structure section IDs must be unique")
        for item in entries:
            if item.parent_id is not None and item.parent_id not in ids:
                raise ValueError("document structure parent is absent")
        warnings = tuple(str(item) for item in self.warnings)
        material = {
            "schema_version": self.schema_version,
            "document": cached_document_ref_to_document(self.document),
            "outline_document": cached_document_ref_to_document(
                self.outline_document
            ),
            "structure_contract": self.structure_contract,
            "entries": [_entry_to_document(item) for item in entries],
            "warnings": list(warnings),
        }
        digest = hashlib.sha256(_json_bytes(material)).hexdigest()
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "structure_sha256", digest)

    @property
    def reference(self) -> CachedDocumentStructureRef:
        return CachedDocumentStructureRef(
            self.document,
            self.outline_document,
            self.structure_contract,
            self.structure_sha256,
        )


class DocumentStructureCache:
    """Atomic overlay storage with an exact two-document identity index."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def lookup(
        self,
        document: CachedDocumentRef,
        outline_document: CachedDocumentRef,
        *,
        structure_contract: str = DOCUMENT_STRUCTURE_CONTRACT,
    ) -> DocumentStructureOverlay | None:
        key = _identity_key(document, outline_document, structure_contract)
        path = self._identity_path(key)
        with exclusive_file_lock(self._identity_lock(key)):
            if not path.is_file():
                return None
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                raw_ref = value["structure"]
                if (
                    value.get("schema_version")
                    != DOCUMENT_STRUCTURE_CACHE_SCHEMA
                    or set(value) != {"schema_version", "structure"}
                    or not isinstance(raw_ref, Mapping)
                ):
                    raise ValueError("invalid identity manifest")
                reference = cached_document_structure_ref_from_document(raw_ref)
                if (
                    reference.document != document
                    or reference.outline_document != outline_document
                    or reference.structure_contract != structure_contract
                ):
                    raise ValueError("identity manifest does not match inputs")
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                return None
        try:
            return self.read(reference)
        except DocumentStructureError:
            return None

    def store(
        self, overlay: DocumentStructureOverlay
    ) -> CachedDocumentStructureRef:
        reference = overlay.reference
        key = _identity_key(
            overlay.document,
            overlay.outline_document,
            overlay.structure_contract,
        )
        with exclusive_file_lock(self._object_lock(overlay.structure_sha256)):
            atomic_write_bytes(
                self._object_path(overlay.structure_sha256),
                _json_bytes(document_structure_overlay_to_document(overlay)),
            )
        with exclusive_file_lock(self._identity_lock(key)):
            atomic_write_bytes(
                self._identity_path(key),
                _json_bytes(
                    {
                        "schema_version": DOCUMENT_STRUCTURE_CACHE_SCHEMA,
                        "structure": cached_document_structure_ref_to_document(
                            reference
                        ),
                    }
                ),
            )
        return reference

    def read(
        self, reference: CachedDocumentStructureRef
    ) -> DocumentStructureOverlay:
        if not isinstance(reference, CachedDocumentStructureRef):
            raise TypeError("reference must be a CachedDocumentStructureRef")
        path = self._object_path(reference.structure_sha256)
        with exclusive_file_lock(self._object_lock(reference.structure_sha256)):
            try:
                payload = path.read_bytes()
                value = json.loads(payload.decode("utf-8"))
                overlay = document_structure_overlay_from_document(value)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                raise DocumentStructureError(
                    "document_structure_corrupt",
                    "cached document structure overlay is unreadable or invalid",
                ) from exc
        if overlay.reference != reference:
            raise DocumentStructureError(
                "document_structure_mismatch",
                "cached document structure does not match its logical reference",
            )
        return overlay

    def _base(self) -> Path:
        return self.root / "document-structure" / "v1"

    def _object_path(self, digest: str) -> Path:
        return self._base() / "objects" / digest[:2] / digest / "overlay.json"

    def _identity_path(self, key: str) -> Path:
        return self._base() / "identities" / key[:2] / key / "manifest.json"

    def _object_lock(self, digest: str) -> Path:
        return self._base() / "locks" / "objects" / f"{digest}.lock"

    def _identity_lock(self, key: str) -> Path:
        return self._base() / "locks" / "identities" / f"{key}.lock"


def reconstruct_document_structure(
    document: CachedDocumentRef,
    outline_document: CachedDocumentRef,
    *,
    markdown_payload: bytes,
    pdf_payload: bytes,
    pdf_pages: Sequence[str],
) -> DocumentStructureOverlay:
    """Strictly align a Markdown heading stream with a PDF bookmark tree."""

    if document.source_format.value != "markdown":
        raise DocumentStructureError(
            "document_structure_source_not_markdown",
            "document structure reconstruction requires a Markdown source",
        )
    if outline_document.source_format.value != "pdf":
        raise DocumentStructureError(
            "document_structure_outline_not_pdf",
            "document structure reconstruction requires a PDF outline source",
        )
    try:
        markdown = markdown_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentStructureError(
            "document_structure_source_not_utf8",
            "Markdown structure source is not valid UTF-8",
        ) from exc
    headings = _markdown_headings(markdown)
    if not headings:
        raise DocumentStructureError(
            "document_structure_no_headings",
            "Markdown structure source contains no ATX headings",
        )
    outlines = _qpdf_outlines(pdf_payload)
    if not outlines:
        raise DocumentStructureError(
            "document_structure_no_outline",
            "PDF contains no usable bookmarks",
        )
    lines = markdown.splitlines()
    matched, warnings = _align_outline(outlines, headings, lines, pdf_pages)
    if not matched:
        raise DocumentStructureError(
            "document_structure_alignment_failed",
            "no PDF bookmark could be aligned to a Markdown heading",
        )
    entries = _build_entries(
        document,
        outlines,
        headings,
        matched,
        total_lines=len(lines),
    )
    return DocumentStructureOverlay(
        document=document,
        outline_document=outline_document,
        entries=entries,
        warnings=warnings,
    )


def cached_document_structure_ref_to_document(
    value: CachedDocumentStructureRef,
) -> dict[str, Any]:
    if not isinstance(value, CachedDocumentStructureRef):
        raise TypeError("value must be a CachedDocumentStructureRef")
    return {
        "document": cached_document_ref_to_document(value.document),
        "outline_document": cached_document_ref_to_document(
            value.outline_document
        ),
        "structure_contract": value.structure_contract,
        "structure_sha256": value.structure_sha256,
    }


def cached_document_structure_ref_from_document(
    value: Mapping[str, Any],
) -> CachedDocumentStructureRef:
    if not isinstance(value, Mapping) or set(value) != _STRUCTURE_REF_FIELDS:
        raise ValueError("cached document structure reference has invalid fields")
    try:
        return CachedDocumentStructureRef(
            document=cached_document_ref_from_document(value["document"]),
            outline_document=cached_document_ref_from_document(
                value["outline_document"]
            ),
            structure_contract=value["structure_contract"],
            structure_sha256=value["structure_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid cached document structure reference: {exc}"
        ) from exc


def document_structure_overlay_to_document(
    value: DocumentStructureOverlay,
) -> dict[str, Any]:
    if not isinstance(value, DocumentStructureOverlay):
        raise TypeError("value must be a DocumentStructureOverlay")
    return {
        "schema_version": value.schema_version,
        "document": cached_document_ref_to_document(value.document),
        "outline_document": cached_document_ref_to_document(
            value.outline_document
        ),
        "structure_contract": value.structure_contract,
        "structure_sha256": value.structure_sha256,
        "entries": [_entry_to_document(item) for item in value.entries],
        "warnings": list(value.warnings),
    }


def document_structure_overlay_from_document(
    value: Mapping[str, Any],
) -> DocumentStructureOverlay:
    if not isinstance(value, Mapping) or set(value) != _OVERLAY_FIELDS:
        raise ValueError("document structure overlay has invalid fields")
    raw_entries = value.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("document structure entries must be an array")
    overlay = DocumentStructureOverlay(
        document=cached_document_ref_from_document(value["document"]),
        outline_document=cached_document_ref_from_document(
            value["outline_document"]
        ),
        structure_contract=value["structure_contract"],
        schema_version=value["schema_version"],
        entries=tuple(_entry_from_document(item) for item in raw_entries),
        warnings=tuple(value.get("warnings") or ()),
    )
    if overlay.structure_sha256 != value.get("structure_sha256"):
        raise ValueError("document structure digest does not match content")
    return overlay


@dataclass(frozen=True)
class _Heading:
    title: str
    line: int
    source_level: int


@dataclass(frozen=True)
class _Outline:
    key: str
    title: str
    level: int
    parent_key: str | None
    page: int
    has_children: bool


def _qpdf_outlines(payload: bytes) -> tuple[_Outline, ...]:
    try:
        value = QpdfOutlineExtractor().extract(payload)
        roots = value["outlines"]
    except PDFOutlineExtractionError as exc:
        raise DocumentStructureError(
            "document_structure_qpdf_failed", exc.message
        ) from exc
    except (KeyError, TypeError) as exc:
        raise DocumentStructureError(
            "document_structure_qpdf_invalid",
            "qpdf returned malformed bookmark JSON",
        ) from exc
    result: list[_Outline] = []

    def visit(
        values: Any, level: int, parent_key: str | None, path: tuple[int, ...]
    ) -> None:
        if not isinstance(values, list):
            raise DocumentStructureError(
                "document_structure_qpdf_invalid",
                "qpdf bookmark children are malformed",
            )
        for index, raw in enumerate(values):
            if not isinstance(raw, Mapping):
                raise DocumentStructureError(
                    "document_structure_qpdf_invalid",
                    "qpdf bookmark entry is malformed",
                )
            title = _clean_title(raw.get("title"))
            page = raw.get("destpageposfrom1")
            kids = raw.get("kids", [])
            if not title or isinstance(page, bool) or not isinstance(page, int) or page < 1:
                continue
            key_path = (*path, index)
            key = ".".join(str(item) for item in key_path)
            result.append(_Outline(key, title, level, parent_key, page, bool(kids)))
            visit(kids, level + 1, key, key_path)

    visit(roots, 1, None, ())
    return tuple(result)


def _markdown_headings(text: str) -> tuple[_Heading, ...]:
    values = []
    in_fence = False
    fence = ""
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if match := re.match(r"(```+|~~~+)", stripped):
            marker = match.group(1)
            if not in_fence:
                in_fence, fence = True, marker[0]
            elif marker[0] == fence:
                in_fence, fence = False, ""
            continue
        if in_fence:
            continue
        match = _ATX_HEADING_RE.match(line)
        if match:
            title = _clean_title(match.group(2))
            if title:
                values.append(_Heading(title, line_number, len(match.group(1))))
    return tuple(values)


def _align_outline(
    outlines: Sequence[_Outline],
    headings: Sequence[_Heading],
    lines: Sequence[str],
    pdf_pages: Sequence[str],
) -> tuple[dict[str, int], tuple[str, ...]]:
    candidate_sets = [
        tuple(
            index
            for index, heading in enumerate(headings)
            if _normalized_title(heading.title)
            == _normalized_title(outline.title)
        )
        for outline in outlines
    ]
    evidence: dict[tuple[int, int], int] = {}
    for outline_index, (outline, candidates) in enumerate(
        zip(outlines, candidate_sets, strict=True)
    ):
        if len(candidates) > 1:
            page_text = (
                pdf_pages[outline.page - 1]
                if outline.page <= len(pdf_pages)
                else ""
            )
            for candidate in candidates:
                evidence[(outline_index, candidate)] = _body_overlap(
                    headings[candidate],
                    (
                        headings[candidate + 1]
                        if candidate + 1 < len(headings)
                        else None
                    ),
                    lines,
                    page_text,
                )

    @lru_cache(maxsize=None)
    def solve(
        outline_index: int, previous_heading: int
    ) -> tuple[tuple[int, int], tuple[tuple[tuple[int, int], ...], ...]]:
        if outline_index == len(outlines):
            return (0, 0), ((),)
        options: list[
            tuple[tuple[int, int], tuple[tuple[int, int], ...]]
        ] = []
        skipped_score, skipped_paths = solve(
            outline_index + 1, previous_heading
        )
        options.extend((skipped_score, path) for path in skipped_paths)
        for candidate in candidate_sets[outline_index]:
            if candidate <= previous_heading:
                continue
            child_score, child_paths = solve(outline_index + 1, candidate)
            score = (
                child_score[0] + 1,
                child_score[1] + evidence.get((outline_index, candidate), 0),
            )
            options.extend(
                (score, ((outline_index, candidate), *path))
                for path in child_paths
            )
        best_score = max(score for score, _ in options)
        best_paths: list[tuple[tuple[int, int], ...]] = []
        for score, path in options:
            if score == best_score and path not in best_paths:
                best_paths.append(path)
                if len(best_paths) == 2:
                    break
        return best_score, tuple(best_paths)

    _, paths = solve(0, -1)
    if len(paths) != 1:
        differing = sorted(
            {
                heading_index
                for path in paths
                for _, heading_index in path
            }
        )
        detail = ",".join(str(headings[index].line) for index in differing[:8])
        raise DocumentStructureError(
            "document_structure_alignment_ambiguous",
            "PDF bookmarks have multiple equally supported monotonic Markdown "
            f"alignments; candidate heading lines: {detail}",
        )
    matched = {
        outlines[outline_index].key: heading_index
        for outline_index, heading_index in paths[0]
    }
    warnings = [
        f"outline_unmatched:{outline.key}:{outline.title}"
        for outline in outlines
        if outline.key not in matched
    ]
    return matched, tuple(warnings)


def _build_entries(
    document: CachedDocumentRef,
    outlines: Sequence[_Outline],
    headings: Sequence[_Heading],
    matched: Mapping[str, int],
    *,
    total_lines: int,
) -> tuple[DocumentStructureEntry, ...]:
    outline_by_key = {item.key: item for item in outlines}
    matched_by_heading = {heading: key for key, heading in matched.items()}
    draft: list[dict[str, Any]] = []
    id_by_key: dict[str, str] = {}
    matched_heading_indexes = sorted(matched_by_heading)
    for heading_index, heading in enumerate(headings):
        if heading_index in matched_by_heading:
            key = matched_by_heading[heading_index]
            outline = outline_by_key[key]
            section_id = _section_id(document, heading, key)
            id_by_key[key] = section_id
            candidate_lines = [
                candidate.line
                for candidate in headings
                if _normalized_title(candidate.title)
                == _normalized_title(outline.title)
            ]
            draft.append(
                {
                    "section_id": section_id,
                    "title": heading.title,
                    "level": outline.level,
                    "parent_key": outline.parent_key,
                    "heading": heading,
                    "page_start": outline.page,
                    "kind": (
                        DocumentStructureNodeKind.CONTAINER
                        if outline.has_children
                        else DocumentStructureNodeKind.CONTENT
                    ),
                    "matching_method": (
                        "normalized_title_destination_body_global"
                        if len(candidate_lines) > 1
                        else "normalized_title_monotonic"
                    ),
                    "provenance": {
                        "outline_key": key,
                        "outline_title": outline.title,
                        "destination_page": outline.page,
                        "candidate_heading_lines": candidate_lines,
                    },
                }
            )
            continue
        preceding = [
            index for index in matched_heading_indexes if index < heading_index
        ]
        parent_key = (
            matched_by_heading[preceding[-1]] if preceding else None
        )
        parent_outline = (
            outline_by_key[parent_key] if parent_key is not None else None
        )
        level = (parent_outline.level + 1) if parent_outline is not None else 1
        structural = _is_structural_heading(heading.title)
        draft.append(
            {
                "section_id": _section_id(
                    document, heading, f"internal:{heading.line}"
                ),
                "title": heading.title,
                "level": level,
                "parent_key": parent_key,
                "heading": heading,
                "page_start": None,
                "kind": (
                    DocumentStructureNodeKind.STRUCTURAL
                    if structural
                    else DocumentStructureNodeKind.INTERNAL
                ),
                "matching_method": "inferred_between_outline_nodes",
                "provenance": {
                    "source_heading_level": heading.source_level,
                    "preceding_outline_key": parent_key or "",
                },
            }
        )
    for item in draft:
        parent_key = item.pop("parent_key")
        item["parent_id"] = id_by_key.get(parent_key)
    for index, item in enumerate(draft):
        level = item["level"]
        next_line = total_lines + 1
        for later in draft[index + 1 :]:
            if later["level"] <= level:
                next_line = later["heading"].line
                break
        next_pdf = None
        for later in draft[index + 1 :]:
            if later["level"] <= level and later["page_start"] is not None:
                next_pdf = later["page_start"]
                break
        item["line_end"] = max(item["heading"].line, next_line - 1)
        item["page_end"] = (
            max(item["page_start"], next_pdf - 1)
            if item["page_start"] is not None and next_pdf is not None
            else item["page_start"]
        )
    return tuple(
        DocumentStructureEntry(
            section_id=item["section_id"],
            title=item["title"],
            level=item["level"],
            parent_id=item["parent_id"],
            ordinal=ordinal,
            heading_line=item["heading"].line,
            source_line_start=item["heading"].line,
            source_line_end=item["line_end"],
            pdf_page_start=item["page_start"],
            pdf_page_end=item["page_end"],
            kind=item["kind"],
            matching_method=item["matching_method"],
            provenance=item["provenance"],
        )
        for ordinal, item in enumerate(draft)
    )


def _entry_to_document(value: DocumentStructureEntry) -> dict[str, Any]:
    return {
        "section_id": value.section_id,
        "title": value.title,
        "level": value.level,
        "parent_id": value.parent_id,
        "ordinal": value.ordinal,
        "heading_line": value.heading_line,
        "source_line_start": value.source_line_start,
        "source_line_end": value.source_line_end,
        "pdf_page_start": value.pdf_page_start,
        "pdf_page_end": value.pdf_page_end,
        "kind": value.kind.value,
        "matching_method": value.matching_method,
        "provenance": dict(value.provenance),
    }


def _entry_from_document(value: Mapping[str, Any]) -> DocumentStructureEntry:
    if not isinstance(value, Mapping) or set(value) != _ENTRY_FIELDS:
        raise ValueError("document structure entry has invalid fields")
    return DocumentStructureEntry(**dict(value))


def _identity_key(
    document: CachedDocumentRef,
    outline_document: CachedDocumentRef,
    contract: str,
) -> str:
    return hashlib.sha256(
        _json_bytes(
            {
                "document": cached_document_ref_to_document(document),
                "outline_document": cached_document_ref_to_document(
                    outline_document
                ),
                "structure_contract": contract,
            }
        )
    ).hexdigest()


def _section_id(
    document: CachedDocumentRef, heading: _Heading, identity: str
) -> str:
    digest = hashlib.sha256(
        f"{document.source_sha256}\0{heading.line}\0{heading.title}\0{identity}".encode(
            "utf-8"
        )
    ).hexdigest()
    return f"struct-{digest[:24]}"


def _body_overlap(
    heading: _Heading,
    following: _Heading | None,
    lines: Sequence[str],
    page_text: str,
) -> int:
    end = following.line - 1 if following is not None else min(len(lines), heading.line + 80)
    body = "\n".join(lines[heading.line : min(end, heading.line + 80)])
    source_tokens = set(_evidence_tokens(body))
    page_tokens = set(_evidence_tokens(page_text))
    return len(source_tokens.intersection(page_tokens))


def _evidence_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(
        token
        for token in re.findall(r"[^\W_]{4,}", normalized, flags=re.UNICODE)
        if token not in {"this", "that", "with", "from", "have", "were", "which"}
    )


def _clean_title(value: Any) -> str:
    return " ".join(str(value or "").replace("\x00", "").split()).strip()


def _normalized_title(value: str) -> str:
    text = unicodedata.normalize("NFKC", _clean_title(value)).casefold()
    text = re.sub(r"^(?:chapter\s+)?\d+\s*[.:\-]\s*", "", text)
    text = re.sub(r"^[ivxlcdm]+\s+", "", text)
    text = re.sub(r"\s+\d+\s*$", "", text)
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def _is_structural_heading(title: str) -> bool:
    normalized = _clean_title(title)
    return bool(re.fullmatch(r"(?:[IVXLCDM]+|\d+)", normalized, re.IGNORECASE))


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


__all__ = [
    "CACHED_DOCUMENT_STRUCTURE_REF_SCHEMA",
    "DOCUMENT_STRUCTURE_CACHE_SCHEMA",
    "DOCUMENT_STRUCTURE_CONTRACT",
    "DOCUMENT_STRUCTURE_OVERLAY_SCHEMA",
    "CachedDocumentStructureRef",
    "DocumentStructureCache",
    "DocumentStructureEntry",
    "DocumentStructureError",
    "DocumentStructureNodeKind",
    "DocumentStructureOverlay",
    "cached_document_structure_ref_from_document",
    "cached_document_structure_ref_to_document",
    "document_structure_overlay_from_document",
    "document_structure_overlay_to_document",
    "reconstruct_document_structure",
]
