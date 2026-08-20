from __future__ import annotations

import re
import unicodedata
from typing import Any

from ..sources import (
    ReconciliationEntry,
    ReconciliationStatus,
    SourceArtifact,
    SourceFormat,
)
from .models import MathSpan, ParsedDocument, VisualPageReviewInput


RICH_FORMATS = {SourceFormat.HTML, SourceFormat.MARKDOWN, SourceFormat.TEX}


def reconcile_validator(
    primary: ParsedDocument,
    validator: ParsedDocument,
) -> tuple[tuple[ReconciliationEntry, ...], tuple[str, ...]]:
    """Compare one validator independently without modifying the primary."""

    if validator.source.source_format is SourceFormat.PDF:
        return _reconcile_pdf(primary, validator)
    if validator.source.source_format in RICH_FORMATS:
        return _reconcile_rich(primary, validator)
    return (
        (
            _entry(
                validator.source,
                ReconciliationStatus.UNREVIEWED,
                "validator",
                "validator format has no deterministic reconciler",
            ),
        ),
        (f"{validator.source.source_format.value} validator was not reviewed",),
    )


def build_visual_page_review_inputs(
    primary: ParsedDocument,
    pdf_validator: ParsedDocument,
) -> tuple[VisualPageReviewInput, ...]:
    """Build one visual-review descriptor per PDF page, including empty pages."""

    if pdf_validator.source.source_format is not SourceFormat.PDF:
        raise ValueError("visual review input requires a parsed PDF validator")
    return tuple(
        VisualPageReviewInput(
            primary=primary.source,
            pdf_validator=pdf_validator.source,
            page_number=page.page_number,
            math_spans=primary.math_spans,
        )
        for page in pdf_validator.pages
    )


def _reconcile_rich(
    primary: ParsedDocument, validator: ParsedDocument
) -> tuple[tuple[ReconciliationEntry, ...], tuple[str, ...]]:
    entries: list[ReconciliationEntry] = []
    warnings: list[str] = []
    primary_titles = [_fingerprint(item.title) for item in primary.sections]
    validator_titles = [_fingerprint(item.title) for item in validator.sections]
    if primary_titles == validator_titles:
        entries.append(
            _entry(
                validator.source,
                ReconciliationStatus.VERIFIED,
                "structure",
                "validator section order and titles agree with the primary",
                section_count=len(primary_titles),
            )
        )
    else:
        entries.append(
            _entry(
                validator.source,
                ReconciliationStatus.MISMATCH,
                "structure",
                "validator section structure differs from the primary",
                primary_titles=primary_titles,
                observed_titles=validator_titles,
            )
        )
        warnings.append(
            f"{validator.source.source_format.value} validator structure differs from primary"
        )

    unmatched = set(range(len(validator.math_spans)))
    equal_span_counts = len(primary.math_spans) == len(validator.math_spans)
    for ordinal, span in enumerate(primary.math_spans):
        entry, matched = _match_rich_span(
            span,
            ordinal,
            validator.math_spans,
            unmatched,
            validator.source,
            equal_span_counts=equal_span_counts,
        )
        entries.append(entry)
        unmatched.difference_update(matched)
        if entry.status is not ReconciliationStatus.VERIFIED:
            warnings.append(f"validator math conflict for {span.span_id}: {entry.status.value}")
    for index in sorted(unmatched):
        observed = validator.math_spans[index]
        entries.append(
            _entry(
                validator.source,
                ReconciliationStatus.MISMATCH,
                f"validator:{observed.span_id}",
                "validator contains mathematical content not matched to the primary",
                observed_tex=observed.normalized_tex,
                observed_kind=observed.kind.value,
                observed_position=_position(observed),
            )
        )
        warnings.append(f"validator contains unmatched math {observed.span_id}")
    return tuple(entries), tuple(warnings)


def _match_rich_span(
    primary: MathSpan,
    ordinal: int,
    candidates: tuple[MathSpan, ...],
    unmatched: set[int],
    validator: SourceArtifact,
    *,
    equal_span_counts: bool,
) -> tuple[ReconciliationEntry, set[int]]:
    same_label = {
        index
        for index in unmatched
        if primary.source_label
        and candidates[index].source_label
        and candidates[index].source_label == primary.source_label
    }
    same_tex = {
        index
        for index in unmatched
        if _math_fingerprint(candidates[index].normalized_tex)
        == _math_fingerprint(primary.normalized_tex)
    }
    same_kind_tex = {
        index for index in same_tex if candidates[index].kind is primary.kind
    }
    selected = same_label or same_kind_tex or same_tex
    method = "source_label" if same_label else ("kind_and_math" if same_kind_tex else "math")
    if len(selected) > 1:
        context_matches = {
            index
            for index in selected
            if _context_fingerprint(candidates[index])
            and _context_fingerprint(candidates[index]) == _context_fingerprint(primary)
        }
        if len(context_matches) == 1:
            selected = context_matches
            method += "_and_context"
    if len(selected) > 1:
        return (
            _entry(
                validator,
                ReconciliationStatus.AMBIGUOUS,
                primary.span_id,
                "multiple validator math spans match the primary",
                primary_tex=primary.normalized_tex,
                candidate_span_ids=[candidates[index].span_id for index in sorted(selected)],
                matching_method=method,
            ),
            set(),
        )
    if len(selected) == 1:
        index = next(iter(selected))
        observed = candidates[index]
        if _math_fingerprint(observed.normalized_tex) == _math_fingerprint(
            primary.normalized_tex
        ):
            return (
                _entry(
                    validator,
                    ReconciliationStatus.VERIFIED,
                    primary.span_id,
                    "validator mathematical content agrees with the primary",
                    observed_span_id=observed.span_id,
                    observed_tex=observed.normalized_tex,
                    observed_position=_position(observed),
                    matching_method=method,
                ),
                {index},
            )
        return (
            _entry(
                validator,
                ReconciliationStatus.MISMATCH,
                primary.span_id,
                "validator label matches but mathematical content differs",
                primary_tex=primary.normalized_tex,
                observed_span_id=observed.span_id,
                observed_tex=observed.normalized_tex,
                matching_method=method,
            ),
            {index},
        )
    # Stable order is evidence only when both sources expose the same span
    # count. It may identify a disagreement, but never overwrites the primary.
    if equal_span_counts and ordinal < len(candidates) and ordinal in unmatched:
        observed = candidates[ordinal]
        return (
            _entry(
                validator,
                ReconciliationStatus.MISMATCH,
                primary.span_id,
                "validator span at the same sequence position differs",
                primary_tex=primary.normalized_tex,
                observed_span_id=observed.span_id,
                observed_tex=observed.normalized_tex,
                matching_method="sequence",
            ),
            {ordinal},
        )
    return (
        _entry(
            validator,
            ReconciliationStatus.MISSING,
            primary.span_id,
            "validator contains no deterministic match for primary math",
            primary_tex=primary.normalized_tex,
            primary_position=_position(primary),
        ),
        set(),
    )


def _reconcile_pdf(
    primary: ParsedDocument, validator: ParsedDocument
) -> tuple[tuple[ReconciliationEntry, ...], tuple[str, ...]]:
    if not bool(validator.metadata.get("text_layer")):
        message = (
            "PDF validator has no extractable text layer; deterministic validation is partial"
        )
        return (
            (
                _entry(
                    validator.source,
                    ReconciliationStatus.UNREVIEWED,
                    "pdf-text-layer",
                    message,
                    page_count=len(validator.pages),
                ),
            ),
            (message,),
        )

    entries: list[ReconciliationEntry] = []
    warnings: list[str] = []
    raw_pages = [page.text for page in validator.pages]
    for section in primary.sections:
        title = _fingerprint(section.title)
        exact_matching_pages = _pages_for_exact_section_title(
            raw_pages, section.title
        )
        joined_matching_pages = _pages_for_joined_section_title(
            raw_pages, section.title
        )
        body_anchor, body_matching_pages = _pages_for_section_body_anchor(
            raw_pages, section.title, section.text
        )
        substring_matching_pages = _pages_for_section_title_substrings(
            raw_pages,
            _fingerprint(
                _without_conventional_pdf_section_prefix(section.title)
            )
            or title,
        )
        # A unique exact title is strongest.  A title can also occur in a TOC,
        # however, so a bounded joined-heading or body anchor may safely break
        # that tie before the deliberately conservative substring fallback.
        if len(exact_matching_pages) == 1:
            matching_pages, method = exact_matching_pages, "normalized_exact_line"
        elif len(joined_matching_pages) == 1:
            matching_pages, method = joined_matching_pages, "joined_heading_lines"
        elif len(body_matching_pages) == 1:
            matching_pages, method = body_matching_pages, "content_anchor"
        elif exact_matching_pages:
            matching_pages, method = exact_matching_pages, "normalized_exact_line"
        elif joined_matching_pages:
            matching_pages, method = joined_matching_pages, "joined_heading_lines"
        elif body_matching_pages:
            matching_pages, method = body_matching_pages, "content_anchor"
        else:
            matching_pages = substring_matching_pages
            method = "normalized_page_substring" if matching_pages else "none"
        if len(matching_pages) == 1:
            status = ReconciliationStatus.VERIFIED
            message = "primary section title maps to one PDF page"
        elif not matching_pages:
            status = ReconciliationStatus.MISSING
            message = "primary section title was not found in the PDF text layer"
        else:
            status = ReconciliationStatus.AMBIGUOUS
            message = "primary section title maps to multiple PDF pages"
        entries.append(
            _entry(
                validator.source,
                status,
                f"section:{section.section_id}",
                message,
                page_candidates=matching_pages,
                title=section.title,
                matching_method=method,
                body_anchor=body_anchor,
            )
        )
        if status is not ReconciliationStatus.VERIFIED:
            warnings.append(
                f"PDF section evidence {status.value} for {section.section_id}"
            )

    for span in primary.math_spans:
        pages_by_label = _pages_for_printed_label(raw_pages, span.source_label)
        pages_by_math = _pages_for_math(raw_pages, span.normalized_tex)
        # A printed number locates an equation, but it does not independently
        # prove that the equation's mathematical content agrees.  Preserve it
        # as provenance only; content verification requires math evidence.
        matching_pages = sorted(set(pages_by_math))
        method = "normalized_math" if pages_by_math else "none"
        if len(matching_pages) == 1:
            status = ReconciliationStatus.VERIFIED
            message = "PDF text layer provides deterministic math evidence"
        elif len(matching_pages) > 1:
            status = ReconciliationStatus.AMBIGUOUS
            message = "PDF math evidence occurs on multiple pages"
        else:
            status = ReconciliationStatus.UNREVIEWED
            message = "PDF text layer does not provide deterministic evidence for this span"
        provenance: dict[str, Any] = {
            "page_candidates": matching_pages,
            "matching_method": method,
        }
        if pages_by_label:
            provenance["printed_label_page_candidates"] = pages_by_label
        printed = _printed_number(span.source_label)
        if printed:
            provenance["printed_equation_number"] = printed
        entries.append(
            _entry(
                validator.source,
                status,
                span.span_id,
                message,
                **provenance,
            )
        )
        if status is not ReconciliationStatus.VERIFIED:
            warnings.append(f"PDF math evidence {status.value} for {span.span_id}")
    return tuple(entries), tuple(warnings)


def _pages_for_exact_section_title(pages: list[str], title: str) -> list[int]:
    """Return pages containing the title as one normalized text-layer line.

    PDF tables of contents normally retain a page number or leader after a
    section title.  Treating the whole extracted line as evidence therefore
    prefers a rendered heading over a TOC mention without requiring layout
    metadata from the PDF extractor.  A source title is authoritative: only
    after an exact source-title match fails may a conventional PDF-only
    section number be ignored.
    """

    needle = _fingerprint(title)
    if not needle:
        return []
    exact_matching_pages = [
        page_number
        for page_number, page in enumerate(pages, 1)
        if any(_fingerprint(line) == needle for line in page.splitlines())
    ]
    if exact_matching_pages:
        return exact_matching_pages
    semantic_needle = _fingerprint(_without_conventional_pdf_section_prefix(title))
    if semantic_needle and semantic_needle != needle:
        semantic_matches = [
            page_number
            for page_number, page in enumerate(pages, 1)
            if any(
                _fingerprint(line) == semantic_needle
                or _fingerprint(_without_conventional_pdf_section_prefix(line))
                == semantic_needle
                for line in page.splitlines()
            )
        ]
        if semantic_matches:
            return semantic_matches
    return [
        page_number
        for page_number, page in enumerate(pages, 1)
        if any(
            _fingerprint(_without_conventional_pdf_section_prefix(line)) == needle
            for line in page.splitlines()
        )
    ]


_PDF_CONVENTIONAL_SECTION_PREFIX = re.compile(
    r"^\s*(?:"
    r"(?:\d{1,2}|[IVXLCDM]+)(?:\s*\.\s*\d{1,2})+(?:\s*[.)])?"
    r"|\d{1,2}\s*[.)]"
    r"|\d{1,2}"
    r"|[IVXLCDM]+\s*[.)]"
    r"|[IVXLCDM]+"
    r"|[A-Z]\s*[.)]"
    r")\s+(?=\S)"
)
_PDF_SECTION_LIKE_PREFIX = re.compile(
    r"^\s*(?:(?:\d+|[IVXLCDM]+)(?:\s*\.\s*\d+)*|[IVXLCDM]+)\s*[.)]?\s+(?=\S)"
)


def _pages_for_section_title_substrings(pages: list[str], title: str) -> list[int]:
    """Find line-level prose evidence without accepting title-like prefixes."""

    if not title:
        return []
    return [
        page_number
        for page_number, page in enumerate(pages, 1)
        if any(
            _line_has_section_title_substring(line, title)
            for line in page.splitlines()
        )
    ]


def _pages_for_joined_section_title(pages: list[str], title: str) -> list[int]:
    """Find an exact heading split across at most three extracted lines."""

    needle = _fingerprint(_without_conventional_pdf_section_prefix(title))
    if not needle:
        return []
    matches: list[int] = []
    for page_number, page in enumerate(pages, 1):
        lines = page.splitlines()
        for index in range(len(lines)):
            for width in (2, 3):
                joined = " ".join(lines[index : index + width])
                if len(lines[index : index + width]) != width:
                    continue
                normalized = _fingerprint(joined)
                stripped = _fingerprint(_without_conventional_pdf_section_prefix(joined))
                if normalized == needle or stripped == needle:
                    matches.append(page_number)
                    break
            else:
                continue
            break
    return matches


def _pages_for_section_body_anchor(
    pages: list[str], title: str, text: str
) -> tuple[list[str], list[int]]:
    """Return the first unique eight-token body anchor within the first 128 tokens."""

    tokens = _fingerprint(text).split()
    title_tokens = _fingerprint(title).split()
    if title_tokens and tokens[: len(title_tokens)] == title_tokens:
        tokens = tokens[len(title_tokens) :]
    tokens = tokens[:128]
    if len(tokens) < 8:
        return [], []
    page_tokens = [_fingerprint(page).split() for page in pages]
    fallback: tuple[list[str], list[int]] = ([], [])
    for start in range(len(tokens) - 7):
        anchor = tokens[start : start + 8]
        candidates = [
            page_number
            for page_number, observed in enumerate(page_tokens, 1)
            if _contains_token_run(observed, anchor)
        ]
        if len(candidates) == 1:
            return anchor, candidates
        if candidates and not fallback[1]:
            fallback = (anchor, candidates)
    return fallback


def _contains_token_run(values: list[str], needle: list[str]) -> bool:
    width = len(needle)
    return any(
        values[index : index + width] == needle
        for index in range(len(values) - width + 1)
    )


def _line_has_section_title_substring(line: str, title: str) -> bool:
    """Return whether one non-heading line provides page-level title evidence."""

    # A one-letter leading token is indistinguishable from prose (notably the
    # article in ``A Model``) in a substring search.  Exact and conventional
    # prefixed-heading matching have already run before this fallback, so do
    # not let weak body prose establish a section page for such titles.
    if re.match(r"^[A-Za-z]\s+", title):
        return False
    normalized = _fingerprint(line)
    if f" {title} " not in f" {normalized} ":
        return False
    if _title_with_only_trailing_page_number(normalized, title):
        return False
    remainder = _PDF_SECTION_LIKE_PREFIX.sub("", line, count=1)
    if remainder != line:
        normalized_remainder = _fingerprint(remainder)
        if normalized_remainder == title or _title_with_only_trailing_page_number(
            normalized_remainder, title
        ):
            return False
    return True


def _title_with_only_trailing_page_number(value: str, title: str) -> bool:
    return re.fullmatch(rf"{re.escape(title)} \d+", value) is not None


def _without_conventional_pdf_section_prefix(value: str) -> str:
    """Remove one conventional decimal or uppercase-Roman PDF section label."""

    return _PDF_CONVENTIONAL_SECTION_PREFIX.sub("", value, count=1)


def _pages_for_printed_label(pages: list[str], label: str) -> list[int]:
    printed = _printed_number(label)
    if not printed:
        return []
    pattern = re.compile(rf"\(\s*{re.escape(printed)}\s*\)")
    return [index for index, page in enumerate(pages, 1) if pattern.search(page)]


def _printed_number(label: str) -> str:
    match = re.search(r"(?:^|[^\d])(\d+(?:\.\d+)+|\d+)(?:$|[^\d])", label)
    return match.group(1) if match else ""


def _pages_for_math(pages: list[str], tex: str) -> list[int]:
    needle = _math_text_fingerprint(tex)
    if len(needle) < 3:
        return []
    return [
        index
        for index, page in enumerate(pages, 1)
        if needle in _math_text_fingerprint(page)
    ]


def _math_text_fingerprint(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"\\(?:left|right|mathrm|mathbf|mathcal|operatorname)", "", text)
    text = re.sub(r"\\([a-zA-Z]+)", r"\1", text)
    return "".join(re.findall(r"[\w=+\-*/^]", text, flags=re.UNICODE))


def _math_fingerprint(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def _fingerprint(value: str) -> str:
    return " ".join(
        re.findall(
            r"[^\W_]+",
            unicodedata.normalize("NFKC", value).casefold(),
            flags=re.UNICODE,
        )
    )


def _context_fingerprint(span: MathSpan) -> str:
    return _fingerprint(f"{span.context_before} {span.context_after}")


def _position(span: MathSpan) -> dict[str, int | None]:
    return {
        "line_start": span.source_line_start,
        "column_start": span.source_column_start,
        "line_end": span.source_line_end,
        "column_end": span.source_column_end,
    }


def _entry(
    validator: SourceArtifact,
    status: ReconciliationStatus,
    subject_id: str,
    message: str,
    **provenance: Any,
) -> ReconciliationEntry:
    return ReconciliationEntry(
        validator=validator,
        status=status,
        subject_id=subject_id,
        message=message,
        provenance=provenance,
    )


__all__ = ["build_visual_page_review_inputs", "reconcile_validator"]
