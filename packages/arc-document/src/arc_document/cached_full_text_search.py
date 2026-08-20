"""Read-only search over current, catalog-selected parsed full text."""

from __future__ import annotations

import heapq
import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Protocol

from ._cache_root import resolve_cache_root
from ._full_text_catalog import (
    FullTextCatalog,
    FullTextRepresentation,
)
from ._parsed_document_cache import PARSER_CONTRACT, ParsedDocumentCache
from ._ripgrep import RipgrepCandidateSelector
from .cached_document import CachedDocumentRef
from .parse.models import ParsedDocument
from .source_repository import SourceRepositoryError
from .sources import SourceFormat


class CachedFullTextSearchMode(str, Enum):
    OCCURRENCES = "occurrences"
    REFINEMENT_REQUIRED = "refinement_required"


class CachedFullTextContextStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    INCLUDED = "included"
    OMITTED_TOO_BROAD = "omitted_too_broad"
    OMITTED_REFINEMENT_REQUIRED = "omitted_refinement_required"


class CachedFullTextLocation(str, Enum):
    SECTION = "section"
    PAGE = "page"


class CachedFullTextSearchError(ValueError):
    """Typed invalid cached-search request."""

    code = "invalid_search_request"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class CachedFullTextOccurrence:
    source_kind: str
    document_ids: tuple[str, ...]
    source_format: str
    source_digest: str
    document_digest: str
    location: CachedFullTextLocation
    location_id: str
    title: str
    page_number: int | None
    line: int
    column: int
    matched_terms: tuple[str, ...]
    context: str = ""


@dataclass(frozen=True)
class CachedFullTextSearchResult:
    mode: CachedFullTextSearchMode
    terms: tuple[str, ...]
    limit: int
    context_lines: int
    case_sensitive: bool
    total_occurrences: int
    matched_document_count: int
    occurrences: tuple[CachedFullTextOccurrence, ...]
    top_document_titles: tuple[str, ...]
    context_status: CachedFullTextContextStatus
    message: str
    warnings: tuple[str, ...] = ()
    documents: tuple["CachedFullTextDocument", ...] = ()


@dataclass(frozen=True)
class CachedFullTextDocument:
    source_kind: str
    document_ids: tuple[str, ...]
    document: CachedDocumentRef


class CandidateSelector(Protocol):
    def ensure_available(self) -> None: ...

    def files_with_matches(
        self,
        patterns: Sequence[str],
        paths: Sequence[Path],
        *,
        case_sensitive: bool,
    ) -> tuple[Path, ...]: ...


class _CatalogEntry(Protocol):
    kind: str
    representations: tuple[FullTextRepresentation, ...]


class _CatalogReader(Protocol):
    def current_entries(self) -> tuple[_CatalogEntry, ...]: ...


@dataclass(frozen=True)
class _SelectedDocument:
    source_kind: str
    identifiers: tuple[str, ...]
    identified: bool
    representation: FullTextRepresentation
    path: Path

    @property
    def stable_identity(self) -> tuple[str, ...]:
        if self.identified:
            return (self.source_kind, *self.identifiers)
        identity = self.representation.source_identity
        return (
            self.source_kind,
            str(identity["source_format"]),
            str(identity["artifact_digest"]),
            self.representation.document_digest,
        )


@dataclass(frozen=True)
class _LocatedOccurrence:
    occurrence: CachedFullTextOccurrence
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _SearchLocation:
    kind: CachedFullTextLocation
    location_id: str
    title: str
    page_number: int | None
    text: str
    excluded_ranges: tuple[tuple[int, int], ...] = ()


_CORRUPT_CANDIDATE_WARNING = (
    "a catalog-selected cached full-text document failed verification and was skipped"
)
_STALE_PARSER_CONTRACT_WARNING = (
    "a catalog-selected cached full-text document uses a stale parser contract and was skipped"
)
_JSON_WHITESPACE_PATTERN = (
    r"(?:\p{White_Space}|\\[nrtf]|\\u(?:000[bB]|001[c-fC-F]))+"
)


class CachedFullTextSearcher:
    """Search current catalog locators without scanning other cache data."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        candidate_selector: CandidateSelector | None = None,
        _catalog: _CatalogReader | None = None,
        _identified_kind: str = "identified",
        _entry_identifiers: Callable[[_CatalogEntry], tuple[str, ...]] | None = None,
    ) -> None:
        self.root = resolve_cache_root(root)
        self.catalog = _catalog or FullTextCatalog(self.root)
        self.candidate_selector = candidate_selector or RipgrepCandidateSelector()
        self._identified_kind = _identified_kind
        self._entry_identifiers = _entry_identifiers or (
            lambda entry: entry.document_ids  # type: ignore[attr-defined]
        )

    def search(
        self,
        terms: Sequence[str],
        *,
        limit: int = 100,
        context_lines: int = 0,
        case_sensitive: bool = False,
    ) -> CachedFullTextSearchResult:
        normalized_terms = _validate_request(
            terms,
            limit=limit,
            context_lines=context_lines,
            case_sensitive=case_sensitive,
        )
        self.candidate_selector.ensure_available()
        selected, selection_warnings = self._selected_documents()
        if not selected:
            return _zero_result(
                normalized_terms,
                limit=limit,
                context_lines=context_lines,
                case_sensitive=case_sensitive,
                warnings=selection_warnings,
            )

        by_path: dict[Path, list[_SelectedDocument]] = {}
        for item in selected:
            by_path.setdefault(item.path, []).append(item)
        matched_paths = self.candidate_selector.files_with_matches(
            tuple(
                _rg_pattern(term, case_sensitive=case_sensitive)
                for term in normalized_terms
            ),
            tuple(by_path),
            case_sensitive=case_sensitive,
        )

        warnings = list(selection_warnings)
        located: list[_LocatedOccurrence] = []
        occurrence_counts: dict[tuple[str, ...], int] = {}
        matched_documents: dict[tuple[str, ...], CachedFullTextDocument] = {}
        display_titles: dict[tuple[str, ...], str] = {}
        total = 0
        retaining_occurrences = True
        for path in matched_paths:
            for selected_document in by_path[path]:
                document = self._read_verified(selected_document)
                if document is None:
                    warnings.append(_CORRUPT_CANDIDATE_WARNING)
                    continue
                document_occurrence_count = 0
                for match in _document_occurrences(
                    selected_document,
                    document,
                    normalized_terms,
                    case_sensitive=case_sensitive,
                ):
                    document_occurrence_count += 1
                    total += 1
                    if retaining_occurrences and total <= limit:
                        located.append(match)
                    elif retaining_occurrences:
                        located.clear()
                        retaining_occurrences = False
                if not document_occurrence_count:
                    continue
                identity = selected_document.stable_identity
                occurrence_counts[identity] = document_occurrence_count
                matched_documents[identity] = _cached_document(selected_document)
                display_titles[identity] = _display_title(
                    selected_document, document
                )

        warnings_tuple = tuple(dict.fromkeys(warnings))
        matched_document_count = len(occurrence_counts)
        if total == 0:
            return _zero_result(
                normalized_terms,
                limit=limit,
                context_lines=context_lines,
                case_sensitive=case_sensitive,
                warnings=warnings_tuple,
            )
        if total > limit:
            ranked = sorted(
                occurrence_counts,
                key=lambda identity: (
                    -occurrence_counts[identity],
                    display_titles[identity].casefold(),
                    identity,
                ),
            )
            context_status = (
                CachedFullTextContextStatus.OMITTED_REFINEMENT_REQUIRED
                if context_lines
                else CachedFullTextContextStatus.NOT_REQUESTED
            )
            return CachedFullTextSearchResult(
                mode=CachedFullTextSearchMode.REFINEMENT_REQUIRED,
                terms=normalized_terms,
                limit=limit,
                context_lines=context_lines,
                case_sensitive=case_sensitive,
                total_occurrences=total,
                matched_document_count=matched_document_count,
                occurrences=(),
                top_document_titles=tuple(
                    display_titles[identity] for identity in ranked[:50]
                ),
                context_status=context_status,
                message=(
                    "The cached full-text query is too broad; use more specific "
                    "multi-word terms and include synonymous expressions in one request."
                ),
                warnings=warnings_tuple,
                documents=tuple(
                    matched_documents[identity] for identity in ranked
                ),
            )

        if context_lines == 0:
            context_status = CachedFullTextContextStatus.NOT_REQUESTED
        elif total > 20:
            context_status = CachedFullTextContextStatus.OMITTED_TOO_BROAD
        else:
            context_status = CachedFullTextContextStatus.INCLUDED
            located = [
                replace(
                    item,
                    occurrence=replace(
                        item.occurrence,
                        context=_context(
                            item.text,
                            item.start,
                            item.end,
                            context_lines=context_lines,
                        ),
                    ),
                )
                for item in located
            ]
        return CachedFullTextSearchResult(
            mode=CachedFullTextSearchMode.OCCURRENCES,
            terms=normalized_terms,
            limit=limit,
            context_lines=context_lines,
            case_sensitive=case_sensitive,
            total_occurrences=total,
            matched_document_count=matched_document_count,
            occurrences=tuple(item.occurrence for item in located),
            top_document_titles=(),
            context_status=context_status,
            message=(
                f"Found {total} cached full-text occurrence"
                f"{'' if total == 1 else 's'} in {matched_document_count} document"
                f"{'' if matched_document_count == 1 else 's'}."
            ),
            warnings=warnings_tuple,
            documents=tuple(
                matched_documents[identity]
                for identity in sorted(matched_documents)
            ),
        )

    def _selected_documents(
        self,
    ) -> tuple[tuple[_SelectedDocument, ...], tuple[str, ...]]:
        selected: list[_SelectedDocument] = []
        warnings: list[str] = []
        for entry in self.catalog.current_entries():
            representation = _preferred_current_representation(
                entry,
                identified_kind=self._identified_kind,
            )
            if representation is None:
                warnings.append(_STALE_PARSER_CONTRACT_WARNING)
                continue
            try:
                cache = ParsedDocumentCache(
                    self.root,
                    parser_contract=representation.parser_contract,
                )
                path = cache.candidate_document_path_by_key(
                    representation.parsed_cache_key,
                    expected_source_identity=representation.source_identity,
                    expected_parser_contract=representation.parser_contract,
                ).resolve(strict=False)
            except (TypeError, ValueError):
                warnings.append(_CORRUPT_CANDIDATE_WARNING)
                continue
            if not path.is_file():
                warnings.append(_CORRUPT_CANDIDATE_WARNING)
                continue
            selected.append(
                _SelectedDocument(
                    source_kind=entry.kind,
                    identifiers=self._entry_identifiers(entry),
                    identified=entry.kind == self._identified_kind,
                    representation=representation,
                    path=path,
                )
            )

        merged: dict[tuple[str, str], _SelectedDocument] = {}
        local: list[_SelectedDocument] = []
        for item in selected:
            if not item.identified:
                local.append(item)
                continue
            key = (
                item.representation.source_format,
                item.representation.document_digest,
            )
            previous = merged.get(key)
            if previous is None:
                merged[key] = item
            else:
                merged[key] = replace(
                    previous,
                    identifiers=tuple(
                        sorted(
                            set(previous.identifiers).union(item.identifiers),
                            key=str.casefold,
                        )
                    ),
                )
        output = tuple(merged.values()) + tuple(local)
        return (
            tuple(sorted(output, key=lambda item: item.stable_identity)),
            tuple(dict.fromkeys(warnings)),
        )

    def _read_verified(
        self, selected: _SelectedDocument
    ) -> ParsedDocument | None:
        representation = selected.representation
        try:
            cache = ParsedDocumentCache(
                self.root,
                parser_contract=representation.parser_contract,
            )
            return cache.read_verified_by_key(
                representation.parsed_cache_key,
                expected_source_identity=representation.source_identity,
                expected_parser_contract=representation.parser_contract,
                expected_document_digest=representation.document_digest,
            )
        except (OSError, SourceRepositoryError, TypeError, ValueError):
            return None


def _preferred_current_representation(
    entry: _CatalogEntry,
    *,
    identified_kind: str = "identified",
) -> FullTextRepresentation | None:
    """Choose the current HTML/PDF projection without mutating stale locators."""

    current = tuple(
        item for item in entry.representations if _has_current_parser_contract(item)
    )
    if not current:
        return None
    if entry.kind == identified_kind:
        by_format = {item.source_format: item for item in current}
        if "html" in by_format:
            return by_format["html"]
        if "pdf" in by_format:
            return by_format["pdf"]
    return current[0]


def _has_current_parser_contract(representation: FullTextRepresentation) -> bool:
    """Keep cache-wide search read-only across parsed-cache generations."""

    if representation.source_format == SourceFormat.PDF.value:
        return representation.parser_contract.startswith(
            f"{PARSER_CONTRACT}+pdf-extractor:"
        )
    return representation.parser_contract == PARSER_CONTRACT


def _document_occurrences(
    selected: _SelectedDocument,
    document: ParsedDocument,
    terms: tuple[str, ...],
    *,
    case_sensitive: bool,
) -> Iterator[_LocatedOccurrence]:
    yield from _located_document_occurrences(
        document,
        terms,
        source_kind=selected.source_kind,
        document_ids=selected.identifiers,
        case_sensitive=case_sensitive,
    )


def _located_document_occurrences(
    document: ParsedDocument,
    terms: tuple[str, ...],
    *,
    source_kind: str,
    document_ids: tuple[str, ...],
    case_sensitive: bool,
) -> Iterator[_LocatedOccurrence]:
    patterns = tuple(
        re.compile(
            r"\s+".join(re.escape(part) for part in term.split()),
            0 if case_sensitive else re.IGNORECASE,
        )
        for term in terms
    )
    for location in _search_locations(document):
        for start, end, matched_terms in _iter_term_spans(
            location.text, terms, patterns
        ):
            if _contained_in_any(start, end, location.excluded_ranges):
                continue
            line, column = _line_and_column(location.text, start)
            yield _LocatedOccurrence(
                occurrence=CachedFullTextOccurrence(
                    source_kind=source_kind,
                    document_ids=document_ids,
                    source_format=document.source.source_format.value,
                    source_digest=document.source.artifact_digest,
                    document_digest=document.document_digest,
                    location=location.kind,
                    location_id=location.location_id,
                    title=location.title,
                    page_number=location.page_number,
                    line=line,
                    column=column,
                    matched_terms=tuple(matched_terms),
                ),
                text=location.text,
                start=start,
                end=end,
            )


def search_document_occurrences(
    document: ParsedDocument,
    terms: Sequence[str],
    *,
    context_lines: int = 0,
    case_sensitive: bool = False,
) -> tuple[CachedFullTextOccurrence, ...]:
    """Return occurrence-level literal-OR matches for one parsed document."""

    normalized = _validate_request(
        terms,
        limit=1,
        context_lines=context_lines,
        case_sensitive=case_sensitive,
    )
    located = _located_document_occurrences(
        document,
        normalized,
        source_kind="target",
        document_ids=(),
        case_sensitive=case_sensitive,
    )
    return tuple(
        replace(
            item.occurrence,
            context=(
                _context(
                    item.text,
                    item.start,
                    item.end,
                    context_lines=context_lines,
                )
                if context_lines
                else ""
            ),
        )
        for item in located
    )


def _search_locations(document: ParsedDocument) -> tuple[_SearchLocation, ...]:
    if not document.sections:
        return tuple(
            _SearchLocation(
                kind=CachedFullTextLocation.PAGE,
                location_id=f"page-{page.page_number}",
                title=f"Page {page.page_number}",
                page_number=page.page_number,
                text=page.text,
            )
            for page in document.pages
        )

    texts = tuple(
        _section_search_text(section.title, section.text)
        for section in document.sections
    )
    children_by_parent: dict[int, list[int]] = {}
    if document.source.source_format is SourceFormat.HTML:
        parent_indices = _section_parent_indices(document)
        for child_index, parent_index in enumerate(parent_indices):
            if parent_index is not None:
                children_by_parent.setdefault(parent_index, []).append(
                    child_index
                )

    locations: list[_SearchLocation] = []
    for section_index, section in enumerate(document.sections):
        exclusions: list[tuple[int, int]] = []
        child_indices = children_by_parent.get(section_index, ())
        parent_matches = (
            tuple(re.finditer(r"\S+", texts[section_index]))
            if child_indices
            else ()
        )
        parent_tokens = tuple(
            match.group(0) for match in parent_matches
        )
        cursor = len(texts[section_index])
        for child_index in reversed(child_indices):
            child = document.sections[child_index]
            child_projection = (
                f"{child.title}\n{child.text}"
                if child.text
                else child.title
            )
            child_range = _whitespace_equivalent_range(
                parent_matches,
                parent_tokens,
                child_projection,
                end=cursor,
            )
            if child_range is not None:
                exclusions.append(child_range)
                cursor = child_range[0]
        locations.append(
            _SearchLocation(
                kind=CachedFullTextLocation.SECTION,
                location_id=section.section_id,
                title=section.title,
                page_number=section.page_start,
                text=texts[section_index],
                excluded_ranges=tuple(reversed(exclusions)),
            )
        )
    return tuple(locations)


def _section_parent_indices(
    document: ParsedDocument,
) -> tuple[int | None, ...]:
    parents: list[int | None] = []
    active: list[int] = []
    for index, section in enumerate(document.sections):
        while (
            active
            and document.sections[active[-1]].level >= section.level
        ):
            active.pop()
        parents.append(active[-1] if active else None)
        active.append(index)
    return tuple(parents)


def _whitespace_equivalent_range(
    matches: tuple[re.Match[str], ...],
    tokens: tuple[str, ...],
    projection: str,
    *,
    end: int,
) -> tuple[int, int] | None:
    """Locate a direct child projection embedded in a parent projection.

    Hierarchical HTML/ar5iv sections may contain the complete projection of a
    nested child section. Matching complete whitespace-delimited token
    sequences keeps duplicate suppression narrow: equal prose in unrelated or
    sibling sections remains independently searchable.
    """

    projection_tokens = tuple(projection.split())
    if not tokens or not projection_tokens:
        return None
    width = len(projection_tokens)
    for index in range(len(tokens) - width, -1, -1):
        if matches[index + width - 1].end() > end:
            continue
        if tokens[index : index + width] == projection_tokens:
            return (
                matches[index].start(),
                matches[index + width - 1].end(),
            )
    return None


def _contained_in_any(
    start: int,
    end: int,
    ranges: tuple[tuple[int, int], ...],
) -> bool:
    return any(
        range_start <= start and end <= range_end
        for range_start, range_end in ranges
    )


def _overlapping_matches(
    pattern: re.Pattern[str],
    text: str,
) -> Iterator[re.Match[str]]:
    position = 0
    while match := pattern.search(text, position):
        yield match
        position = match.start() + 1


def _iter_term_spans(
    text: str,
    terms: tuple[str, ...],
    patterns: tuple[re.Pattern[str], ...],
) -> Iterator[tuple[int, int, tuple[str, ...]]]:
    iterators = tuple(_overlapping_matches(pattern, text) for pattern in patterns)
    pending: list[
        tuple[int, int, int, re.Match[str], Iterator[re.Match[str]]]
    ] = []
    for term_index, iterator in enumerate(iterators):
        if match := next(iterator, None):
            heapq.heappush(
                pending,
                (match.start(), match.end(), term_index, match, iterator),
            )
    while pending:
        start, end, term_index, _, iterator = heapq.heappop(pending)
        grouped = [(term_index, iterator)]
        while pending and pending[0][:2] == (start, end):
            _, _, grouped_term_index, _, grouped_iterator = heapq.heappop(
                pending
            )
            grouped.append((grouped_term_index, grouped_iterator))
        grouped.sort(key=lambda item: item[0])
        yield start, end, tuple(terms[index] for index, _ in grouped)
        for index, grouped_iterator in grouped:
            if match := next(grouped_iterator, None):
                heapq.heappush(
                    pending,
                    (
                        match.start(),
                        match.end(),
                        index,
                        match,
                        grouped_iterator,
                    ),
                )


def _validate_request(
    terms: Sequence[str],
    *,
    limit: int,
    context_lines: int,
    case_sensitive: bool,
) -> tuple[str, ...]:
    if isinstance(terms, (str, bytes)) or not isinstance(terms, Sequence):
        raise CachedFullTextSearchError("terms must be a sequence of strings")
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= 500
    ):
        raise CachedFullTextSearchError("limit must be between 1 and 500")
    if (
        not isinstance(context_lines, int)
        or isinstance(context_lines, bool)
        or not 0 <= context_lines <= 2
    ):
        raise CachedFullTextSearchError("context_lines must be between 0 and 2")
    if not isinstance(case_sensitive, bool):
        raise CachedFullTextSearchError("case_sensitive must be a boolean")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in terms:
        if not isinstance(item, str) or not item.strip():
            raise CachedFullTextSearchError("each term must be a non-empty string")
        term = " ".join(item.split())
        if term not in seen:
            normalized.append(term)
            seen.add(term)
    if not normalized:
        raise CachedFullTextSearchError("at least one term is required")
    return tuple(normalized)


def _rg_pattern(term: str, *, case_sensitive: bool) -> str:
    encoded = (
        json.dumps(part, ensure_ascii=False)[1:-1]
        for part in term.split()
    )
    return _JSON_WHITESPACE_PATTERN.join(
        _rg_literal(part, case_sensitive=case_sensitive)
        for part in encoded
    )


def _rg_literal(value: str, *, case_sensitive: bool) -> str:
    if case_sensitive:
        return re.escape(value)
    output: list[str] = []
    literal: list[str] = []
    for character in value:
        if character not in "iIİı":
            literal.append(character)
            continue
        if literal:
            output.append(re.escape("".join(literal)))
            literal.clear()
        output.append("[iIİı]")
    if literal:
        output.append(re.escape("".join(literal)))
    return "".join(output)


def _section_search_text(title: str, text: str) -> str:
    normalized_title = " ".join(title.split())
    first_line = next(
        (" ".join(line.split()) for line in text.splitlines() if line.strip()),
        "",
    )
    if (
        normalized_title
        and normalized_title.casefold() not in first_line.casefold()
    ):
        return f"{title}\n{text}" if text else title
    return text


def _line_and_column(text: str, start: int) -> tuple[int, int]:
    line = text.count("\n", 0, start) + 1
    previous_newline = text.rfind("\n", 0, start)
    return line, start - previous_newline


def _context(
    text: str,
    start: int,
    end: int,
    *,
    context_lines: int,
) -> str:
    lines = text.splitlines()
    start_line_index = text.count("\n", 0, start)
    end_line_index = text.count("\n", 0, max(start, end - 1))
    first = max(0, start_line_index - context_lines)
    last = min(len(lines), end_line_index + context_lines + 1)
    value = "\n".join(lines[first:last])
    if len(value) <= 400:
        return value
    before = "\n".join(lines[first:start_line_index])
    offset = len(before) + (1 if before else 0)
    column = start - (text.rfind("\n", 0, start) + 1)
    center = min(len(value), offset + column)
    clip_start = max(0, min(center - 190, len(value) - 398))
    clip_end = min(len(value), clip_start + 398)
    prefix = "…" if clip_start else ""
    suffix = "…" if clip_end < len(value) else ""
    return f"{prefix}{value[clip_start:clip_end]}{suffix}"


def _display_title(
    selected: _SelectedDocument, document: ParsedDocument
) -> str:
    for section in document.sections:
        title = " ".join(section.title.split())
        if title and title.casefold() != "document":
            return title
    for page in document.pages:
        for line in page.text.splitlines():
            title = " ".join(line.split())
            if title:
                return title
    if selected.identifiers:
        return ", ".join(selected.identifiers)
    representation = selected.representation
    return (
        f"local {representation.source_format} "
        f"{representation.source_identity['artifact_digest'][:12]}"
    )


def _cached_document(selected: _SelectedDocument) -> CachedFullTextDocument:
    identity = selected.representation.source_identity
    return CachedFullTextDocument(
        source_kind=selected.source_kind,
        document_ids=selected.identifiers,
        document=CachedDocumentRef(
            source_format=identity["source_format"],
            source_sha256=identity["artifact_digest"],
            source_size=identity["size"],
            media_type=identity["media_type"],
            parser_contract=selected.representation.parser_contract,
            parsed_document_sha256=selected.representation.document_digest,
        ),
    )


def _zero_result(
    terms: tuple[str, ...],
    *,
    limit: int,
    context_lines: int,
    case_sensitive: bool,
    warnings: tuple[str, ...],
) -> CachedFullTextSearchResult:
    return CachedFullTextSearchResult(
        mode=CachedFullTextSearchMode.OCCURRENCES,
        terms=terms,
        limit=limit,
        context_lines=context_lines,
        case_sensitive=case_sensitive,
        total_occurrences=0,
        matched_document_count=0,
        occurrences=(),
        top_document_titles=(),
        context_status=(
            CachedFullTextContextStatus.INCLUDED
            if context_lines
            else CachedFullTextContextStatus.NOT_REQUESTED
        ),
        message=(
            "No cached full-text occurrences were found; add synonymous, "
            "abbreviated, or alternative multi-word terms in one request."
        ),
        warnings=warnings,
        documents=(),
    )


__all__ = [
    "CachedFullTextContextStatus",
    "CachedFullTextDocument",
    "CachedFullTextLocation",
    "CachedFullTextOccurrence",
    "CachedFullTextSearchError",
    "CachedFullTextSearchMode",
    "CachedFullTextSearchResult",
    "CachedFullTextSearcher",
    "CandidateSelector",
    "search_document_occurrences",
]
