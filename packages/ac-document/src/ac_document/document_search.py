"""Deterministic full-text and mathematical-span search.

Search operates only on already parsed, typed documents.  It does not discover
cache files, accept local paths, fetch providers, or own workflow state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections.abc import Sequence
import re
from typing import Iterable

from .parse.models import MathSpanKind, ParsedDocument, ParsedSection


class DocumentSearchError(ValueError):
    """A typed invalid-request error raised before search work starts."""

    code = "invalid_search_request"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class TextMatchLocation(str, Enum):
    SECTION = "section"
    PAGE = "page"


class SectionSelectionError(LookupError):
    """A typed failure to select one section from a parsed document."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class TableOfContentsEntry:
    section_id: str
    title: str
    level: int
    ordinal: int
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class FullTextMatch:
    document_digest: str
    source_digest: str
    location: TextMatchLocation
    location_id: str
    title: str
    ordinal: int
    page_number: int | None
    matched_in: str
    snippet: str


@dataclass(frozen=True)
class FullTextSearchResult:
    query: str
    matches: tuple[FullTextMatch, ...]
    searched_documents: int
    limit: int
    context_lines: int
    case_sensitive: bool
    truncated: bool


@dataclass(frozen=True)
class EquationMatch:
    document_digest: str
    source_digest: str
    span_id: str
    kind: MathSpanKind
    normalized_tex: str
    source_label: str
    source_line_start: int | None
    source_column_start: int | None
    source_line_end: int | None
    source_column_end: int | None
    context_before: str
    context_after: str
    matched_in: str
    matched_terms: tuple[str, ...] = ()
    matched_fields: tuple[str, ...] = ()
    page_candidates: tuple[int, ...] = ()
    source_excerpt: str = ""


@dataclass(frozen=True)
class EquationSearchResult:
    query: str
    matches: tuple[EquationMatch, ...]
    searched_documents: int
    limit: int
    case_sensitive: bool
    truncated: bool


@dataclass(frozen=True)
class EquationTermsSearchResult:
    terms: tuple[str, ...]
    matches: tuple[EquationMatch, ...]
    searched_documents: int
    limit: int
    context_lines: int
    case_sensitive: bool
    truncated: bool


def table_of_contents(document: ParsedDocument) -> tuple[TableOfContentsEntry, ...]:
    """Project the typed section inventory without reconstructing parser data."""

    if not isinstance(document, ParsedDocument):
        raise SectionSelectionError(
            "invalid_document", "document must be a ParsedDocument"
        )
    return tuple(
        TableOfContentsEntry(
            section_id=section.section_id,
            title=section.title,
            level=section.level,
            ordinal=section.ordinal,
            page_start=section.page_start,
            page_end=section.page_end,
        )
        for section in document.sections
    )


def select_section(
    document: ParsedDocument, selector: str | int
) -> ParsedSection:
    """Select one section by ordinal, exact ID/title, or unique title fragment."""

    if not isinstance(document, ParsedDocument):
        raise SectionSelectionError(
            "invalid_document", "document must be a ParsedDocument"
        )
    if isinstance(selector, bool) or not isinstance(selector, (str, int)):
        raise SectionSelectionError(
            "invalid_section_selector",
            "section selector must be a string or integer ordinal",
        )
    if isinstance(selector, int):
        matches = tuple(
            section for section in document.sections if section.ordinal == selector
        )
    else:
        needle = _normalize(selector, case_sensitive=False)
        if not needle:
            raise SectionSelectionError(
                "invalid_section_selector", "section selector is required"
            )
        matches = tuple(
            section
            for section in document.sections
            if _normalize(section.section_id, case_sensitive=False) == needle
            or _normalize(section.title, case_sensitive=False) == needle
        )
        if not matches:
            matches = tuple(
                section
                for section in document.sections
                if needle in _normalize(section.title, case_sensitive=False)
            )
    if not matches:
        raise SectionSelectionError(
            "section_not_found", f"section not found: {selector}"
        )
    if len(matches) > 1:
        raise SectionSelectionError(
            "section_ambiguous", f"section selector is ambiguous: {selector}"
        )
    return matches[0]


def search_full_text(
    documents: ParsedDocument | Iterable[ParsedDocument],
    query: str,
    *,
    limit: int = 20,
    context_lines: int = 1,
    case_sensitive: bool = False,
) -> FullTextSearchResult:
    """Search section text, falling back to pages for sectionless documents.

    One hit is returned per matching section or page.  Documents and their
    locations are visited in caller-provided order, so limiting is stable and
    reproducible.
    """

    normalized_query, normalized_limit = _validate_request(query, limit)
    if not isinstance(context_lines, int) or isinstance(context_lines, bool):
        raise DocumentSearchError("context_lines must be an integer")
    if not 0 <= context_lines <= 5:
        raise DocumentSearchError("context_lines must be between 0 and 5")
    items = _normalize_documents(documents)
    needle = _normalize(normalized_query, case_sensitive=case_sensitive)
    matches: list[FullTextMatch] = []
    truncated = False

    for document in items:
        locations = (
            (
                TextMatchLocation.SECTION,
                section.section_id,
                section.title,
                section.ordinal,
                section.page_start,
                section.text,
            )
            for section in document.sections
        )
        if not document.sections:
            locations = (
                (
                    TextMatchLocation.PAGE,
                    f"page-{page.page_number}",
                    f"Page {page.page_number}",
                    page.page_number - 1,
                    page.page_number,
                    page.text,
                )
                for page in document.pages
            )
        for location, location_id, title, ordinal, page_number, text in locations:
            matched_in = ""
            snippet = ""
            if needle in _normalize(title, case_sensitive=case_sensitive):
                matched_in = "title"
                snippet = _title_snippet(title, text)
            elif needle in _normalize(text, case_sensitive=case_sensitive):
                matched_in = "text"
                snippet = _text_snippet(
                    text,
                    normalized_query,
                    context_lines=context_lines,
                    case_sensitive=case_sensitive,
                )
            if not matched_in:
                continue
            if len(matches) >= normalized_limit:
                truncated = True
                break
            matches.append(
                FullTextMatch(
                    document_digest=document.document_digest,
                    source_digest=document.source.artifact_digest,
                    location=location,
                    location_id=location_id,
                    title=title,
                    ordinal=ordinal,
                    page_number=page_number,
                    matched_in=matched_in,
                    snippet=snippet,
                )
            )
        if truncated:
            break

    return FullTextSearchResult(
        query=normalized_query,
        matches=tuple(matches),
        searched_documents=len(items),
        limit=normalized_limit,
        context_lines=context_lines,
        case_sensitive=case_sensitive,
        truncated=truncated,
    )


def search_equations(
    documents: ParsedDocument | Iterable[ParsedDocument],
    query: str,
    *,
    limit: int = 20,
    case_sensitive: bool = False,
) -> EquationSearchResult:
    """Search every inline and display ``MathSpan`` and its source context."""

    normalized_query, _ = _validate_request(query, limit)
    result = search_equation_terms(
        documents,
        (normalized_query,),
        limit=limit,
        context_lines=8,
        case_sensitive=case_sensitive,
    )
    return EquationSearchResult(
        query=normalized_query,
        matches=result.matches,
        searched_documents=result.searched_documents,
        limit=result.limit,
        case_sensitive=result.case_sensitive,
        truncated=result.truncated,
    )


def search_equation_terms(
    documents: ParsedDocument | Iterable[ParsedDocument],
    terms: Sequence[str],
    *,
    limit: int = 20,
    context_lines: int = 8,
    case_sensitive: bool = False,
) -> EquationTermsSearchResult:
    """Search equation labels, math, and context with literal-OR terms."""

    normalized_terms = _validate_terms(terms)
    _, normalized_limit = _validate_request(normalized_terms[0], limit)
    if (
        not isinstance(context_lines, int)
        or isinstance(context_lines, bool)
        or not 0 <= context_lines <= 20
    ):
        raise DocumentSearchError("context_lines must be between 0 and 20")
    items = _normalize_documents(documents)
    needles = tuple(
        _normalize(term, case_sensitive=case_sensitive) for term in normalized_terms
    )
    ranked: list[tuple[int, int, int, EquationMatch]] = []

    for document_index, document in enumerate(items):
        for span_index, span in enumerate(document.math_spans):
            term_matches: list[str] = []
            field_matches: list[str] = []
            ranks: list[int] = []
            normalized_label = _normalize(
                span.source_label, case_sensitive=case_sensitive
            )
            normalized_span_id = _normalize(
                span.span_id, case_sensitive=case_sensitive
            )
            fields = (
                ("math", span.normalized_tex, 2),
                ("context_before", span.context_before, 3),
                ("context_after", span.context_after, 3),
            )
            for term, needle in zip(normalized_terms, needles, strict=True):
                matched_fields: list[tuple[str, int]] = []
                if normalized_label and needle == normalized_label:
                    matched_fields.append(("source_label", 0))
                elif normalized_label and needle in normalized_label:
                    matched_fields.append(("source_label", 1))
                if needle == normalized_span_id:
                    matched_fields.append(("span_id", 0))
                for name, value, rank in fields:
                    if value and needle in _normalize(
                        value, case_sensitive=case_sensitive
                    ):
                        matched_fields.append((name, rank))
                if not matched_fields:
                    continue
                term_matches.append(term)
                for field, rank in matched_fields:
                    if field not in field_matches:
                        field_matches.append(field)
                    ranks.append(rank)
            if not term_matches:
                continue
            page_candidates, excerpt = _pdf_equation_evidence(
                document, span, context_lines=context_lines
            )
            match = (
                EquationMatch(
                    document_digest=document.document_digest,
                    source_digest=document.source.artifact_digest,
                    span_id=span.span_id,
                    kind=span.kind,
                    normalized_tex=span.normalized_tex,
                    source_label=span.source_label,
                    source_line_start=span.source_line_start,
                    source_column_start=span.source_column_start,
                    source_line_end=span.source_line_end,
                    source_column_end=span.source_column_end,
                    context_before=span.context_before,
                    context_after=span.context_after,
                    matched_in=field_matches[0],
                    matched_terms=tuple(term_matches),
                    matched_fields=tuple(field_matches),
                    page_candidates=page_candidates,
                    source_excerpt=excerpt,
                )
            )
            ranked.append((min(ranks), document_index, span_index, match))

    ranked.sort(key=lambda item: item[:3])
    matches = tuple(item[3] for item in ranked[:normalized_limit])

    return EquationTermsSearchResult(
        terms=normalized_terms,
        matches=matches,
        searched_documents=len(items),
        limit=normalized_limit,
        context_lines=context_lines,
        case_sensitive=case_sensitive,
        truncated=len(ranked) > normalized_limit,
    )


def _validate_terms(terms: Sequence[str]) -> tuple[str, ...]:
    if isinstance(terms, (str, bytes)):
        terms = (str(terms),)
    normalized: list[str] = []
    seen: set[str] = set()
    for value in terms:
        term = " ".join(str(value or "").split())
        if not term:
            raise DocumentSearchError("search terms cannot be empty")
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            normalized.append(term)
    if not normalized:
        raise DocumentSearchError("at least one search term is required")
    return tuple(normalized)


def _pdf_equation_evidence(document, span, *, context_lines: int):
    if (
        document.source.source_format.value != "pdf"
        or span.source_line_start is None
        or not span.source_label
    ):
        return (), ""
    candidates: list[tuple[int, str]] = []
    label_pattern = re.compile(rf"\(\s*{re.escape(span.source_label)}\s*\)")
    for page in document.pages:
        lines = page.text.splitlines()
        index = span.source_line_start - 1
        if not 0 <= index < len(lines):
            continue
        line = " ".join(lines[index].split())
        if label_pattern.search(line) is None:
            continue
        first = max(0, index - context_lines)
        last = min(len(lines), index + context_lines + 1)
        candidates.append((page.page_number, "\n".join(lines[first:last])))
    return (
        tuple(item[0] for item in candidates),
        candidates[0][1] if candidates else "",
    )


def _normalize_documents(
    documents: ParsedDocument | Iterable[ParsedDocument],
) -> tuple[ParsedDocument, ...]:
    if isinstance(documents, ParsedDocument):
        items = (documents,)
    else:
        try:
            items = tuple(documents)
        except TypeError as exc:
            raise DocumentSearchError(
                "documents must be a ParsedDocument or iterable of ParsedDocument"
            ) from exc
    if any(not isinstance(item, ParsedDocument) for item in items):
        raise DocumentSearchError("documents must contain only ParsedDocument values")
    digests = [item.document_digest for item in items]
    if len(set(digests)) != len(digests):
        raise DocumentSearchError("documents must not contain duplicate content")
    return items


def _validate_request(query: str, limit: int) -> tuple[str, int]:
    if not isinstance(query, str) or not query.strip():
        raise DocumentSearchError("query is required")
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise DocumentSearchError("limit must be an integer")
    if not 1 <= limit <= 200:
        raise DocumentSearchError("limit must be between 1 and 200")
    return query.strip(), limit


def _normalize(value: str, *, case_sensitive: bool) -> str:
    compact = " ".join(value.split())
    return compact if case_sensitive else compact.casefold()


def _title_snippet(title: str, text: str) -> str:
    lines = _clean_lines(text)
    return "\n".join((title, lines[0])) if lines else title


def _text_snippet(
    text: str,
    query: str,
    *,
    context_lines: int,
    case_sensitive: bool,
) -> str:
    lines = _clean_lines(text)
    needle = _normalize(query, case_sensitive=case_sensitive)
    for index, line in enumerate(lines):
        if needle in _normalize(line, case_sensitive=case_sensitive):
            start = max(0, index - context_lines)
            end = min(len(lines), index + context_lines + 1)
            return "\n".join(lines[start:end])

    compact = " ".join(lines)
    normalized = _normalize(compact, case_sensitive=case_sensitive)
    match_at = normalized.find(needle)
    if match_at < 0:
        return compact[:900]
    start = max(0, match_at - 300)
    end = min(len(compact), match_at + len(needle) + 300)
    prefix = "…" if start else ""
    suffix = "…" if end < len(compact) else ""
    return f"{prefix}{compact[start:end].strip()}{suffix}"


def _clean_lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


__all__ = [
    "DocumentSearchError",
    "EquationMatch",
    "EquationSearchResult",
    "EquationTermsSearchResult",
    "FullTextMatch",
    "FullTextSearchResult",
    "SectionSelectionError",
    "TableOfContentsEntry",
    "TextMatchLocation",
    "select_section",
    "search_equations",
    "search_equation_terms",
    "search_full_text",
    "table_of_contents",
]
