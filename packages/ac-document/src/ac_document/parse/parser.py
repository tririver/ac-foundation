from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from bs4 import BeautifulSoup, Tag

from .._parsing import ParseError, normalize_tex
from .._parsing.html_source import (
    html_heading_is_document_metadata,
    html_roots,
    html_source_position,
)
from .._parsing.html_equations import (
    html_displayed_equation_label,
    html_equation_table_units,
    html_math_tex as _html_math_tex,
)
from .._parsing.markdown_lex import (
    markdown_column_width as _markdown_column_width,
    markdown_front_matter_end as _markdown_front_matter_end,
    markdown_indent_width as _markdown_indent_width,
    markdown_math_end as _markdown_math_end,
    markdown_quote_content as _markdown_quote_content,
    match_atx_heading,
    match_fence,
    match_setext_heading,
)
from .._parsing.tex_lex import (
    scan_tex_heading as _scan_tex_heading,
    tex_structural_text as _tex_structural_text,
    unwrap_texorpdfstring as _unwrap_texorpdfstring,
)
from ..sources import SourceArtifact, SourceFormat
from .models import (
    MathSpan,
    MathSpanKind,
    PDFTextLayer,
    ParsedDocument,
    ParsedPage,
    ParsedSection,
)


class PDFTextExtractionError(RuntimeError):
    """A deterministic PDF extraction failure, distinct from a missing text layer."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class PDFOutlineExtractionError(RuntimeError):
    """A deterministic PDF bookmark extraction failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class PDFTextExtractor(Protocol):
    contract_id: str

    def extract(self, payload: bytes) -> PDFTextLayer: ...


@dataclass(frozen=True)
class _PDFEquationUnit:
    """One math-like PDF text line recognized by the shared PDF parser rule."""

    raw: str
    source_label: str


class PdftotextExtractor:
    """Narrow, replaceable adapter for deterministic PDF text extraction."""

    contract_id = "ac.document.pdf_text.pdftotext.layout_utf8.v1"

    def __init__(self, *, executable: str = "pdftotext", timeout_seconds: float = 30):
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def extract(self, payload: bytes) -> PDFTextLayer:
        try:
            with tempfile.TemporaryDirectory(prefix="ac-document-pdf-") as directory:
                path = Path(directory) / "source.pdf"
                path.write_bytes(payload)
                completed = subprocess.run(
                    [
                        self.executable,
                        "-layout",
                        "-enc",
                        "UTF-8",
                        str(path),
                        "-",
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout_seconds,
                )
        except FileNotFoundError as exc:
            raise PDFTextExtractionError(
                "pdf_text_extractor_unavailable",
                "pdftotext is unavailable; install it before parsing PDF full text",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise PDFTextExtractionError(
                "pdf_text_extraction_timeout",
                "pdftotext timed out while extracting PDF full text",
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise PDFTextExtractionError(
                "pdf_invalid",
                "pdftotext rejected the PDF document",
            ) from exc
        except (subprocess.SubprocessError, OSError, UnicodeError) as exc:
            raise PDFTextExtractionError(
                "pdf_extraction_failed",
                "pdftotext could not extract the PDF document",
            ) from exc
        pages = completed.stdout.split("\f")
        if pages and not pages[-1]:
            pages.pop()
        if not pages:
            return PDFTextLayer((), "PDF contains no extractable text layer")
        if not any(page.strip() for page in pages):
            return PDFTextLayer(
                tuple(pages), "PDF contains no extractable text layer; partial parse retained"
            )
        return PDFTextLayer(tuple(page.strip() for page in pages))


class QpdfOutlineExtractor:
    """Narrow qpdf adapter returning the versioned bookmark JSON object."""

    contract_id = "ac.document.pdf_outline.qpdf_json.v1"

    def __init__(self, *, executable: str = "qpdf", timeout_seconds: float = 60):
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def extract(self, payload: bytes) -> dict[str, object]:
        try:
            with tempfile.TemporaryDirectory(
                prefix="ac-document-outline-"
            ) as directory:
                path = Path(directory) / "source.pdf"
                path.write_bytes(payload)
                completed = subprocess.run(
                    [
                        self.executable,
                        "--json",
                        "--json-key=outlines",
                        str(path),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self.timeout_seconds,
                )
        except FileNotFoundError as exc:
            raise PDFOutlineExtractionError(
                "pdf_outline_extractor_unavailable",
                "qpdf is unavailable; install it before reconstructing PDF structure",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise PDFOutlineExtractionError(
                "pdf_outline_extraction_timeout",
                "qpdf timed out while extracting PDF bookmarks",
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise PDFOutlineExtractionError(
                "pdf_outline_invalid", "qpdf rejected the PDF document"
            ) from exc
        except (subprocess.SubprocessError, OSError) as exc:
            raise PDFOutlineExtractionError(
                "pdf_outline_extraction_failed",
                "qpdf could not extract PDF bookmarks",
            ) from exc
        try:
            value = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PDFOutlineExtractionError(
                "pdf_outline_extraction_invalid",
                "qpdf returned malformed bookmark JSON",
            ) from exc
        if not isinstance(value, dict):
            raise PDFOutlineExtractionError(
                "pdf_outline_extraction_invalid",
                "qpdf bookmark output is not an object",
            )
        return value


def parse_artifact_bytes(
    artifact: SourceArtifact,
    payload: bytes,
    *,
    pdf_text_extractor: PDFTextExtractor | None = None,
) -> ParsedDocument:
    if len(payload) != artifact.size or hashlib.sha256(payload).hexdigest() != artifact.artifact_digest:
        raise ParseError(
            "source_artifact_mismatch",
            "source bytes do not match the supplied artifact",
            artifact=artifact,
        )
    if artifact.source_format is SourceFormat.PDF:
        return _parse_pdf(
            artifact,
            payload,
            extractor=pdf_text_extractor or PdftotextExtractor(),
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ParseError(
            "source_encoding_invalid",
            f"{artifact.source_format.value} source must be UTF-8",
            artifact=artifact,
        ) from exc
    if artifact.source_format is SourceFormat.HTML:
        return _parse_html(artifact, text)
    if artifact.source_format is SourceFormat.MARKDOWN:
        return _parse_markdown(artifact, text)
    if artifact.source_format is SourceFormat.TEX:
        return _parse_tex(artifact, text)
    raise ParseError("unsupported_source", "unsupported source format", artifact=artifact)


def _span_id(
    artifact: SourceArtifact,
    *,
    kind: MathSpanKind,
    identity_scope: str,
    start_line: int | None,
    start_column: int | None,
    end_line: int | None,
    end_column: int | None,
    tex: str,
) -> str:
    parts = [
        artifact.artifact_digest,
        kind.value,
        str(start_line),
        str(start_column),
        str(end_line),
        str(end_column),
        tex,
    ]
    if identity_scope:
        parts.insert(2, identity_scope)
    material = "\0".join(parts)
    return f"math-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:24]}"


def _make_span(
    artifact: SourceArtifact,
    lines: list[str],
    *,
    kind: MathSpanKind,
    start_line: int,
    start_column: int,
    end_line: int,
    end_column: int,
    raw: str,
    precise_columns: bool = True,
    identity_scope: str = "",
) -> MathSpan | None:
    tex = normalize_tex(raw)
    if not tex:
        return None
    before = ""
    after = ""
    if kind is MathSpanKind.INLINE and start_line == end_line:
        source_line = lines[start_line - 1]
        before = re.sub(r"\s+", " ", source_line[: start_column - 1]).strip()
        after = re.sub(r"\s+", " ", source_line[end_column:]).strip()
    before = before or _nearest_context(lines, start_line - 2, -1)
    after = after or _nearest_context(lines, end_line, 1)
    label_match = re.search(r"\\(?:label|tag)\s*\{([^{}]+)\}", raw)
    source_column_start = start_column if precise_columns else None
    source_column_end = end_column if precise_columns else None
    return MathSpan(
        span_id=_span_id(
            artifact,
            kind=kind,
            identity_scope=identity_scope,
            start_line=start_line,
            start_column=start_column,
            end_line=end_line,
            end_column=end_column,
            tex=tex,
        ),
        kind=kind,
        source_line_start=start_line,
        source_column_start=source_column_start,
        source_line_end=end_line,
        source_column_end=source_column_end,
        normalized_tex=tex,
        context_before=before,
        context_after=after,
        source_label=label_match.group(1) if label_match else "",
    )


def _nearest_context(lines: list[str], index: int, direction: int) -> str:
    while 0 <= index < len(lines):
        candidate = lines[index].strip()
        if candidate and not _is_math_delimiter_line(candidate):
            return re.sub(r"\s+", " ", candidate)
        index += direction
    return ""


def _is_math_delimiter_line(value: str) -> bool:
    return value in {"$$", r"\[", r"\]", r"\(", r"\)"} or bool(
        re.fullmatch(r"\\(?:begin|end)\{[^{}]+\*?\}", value)
    )


def _sections_from_headings(
    headings: Iterable[tuple[int, int, str]], lines: list[str], artifact: SourceArtifact
) -> tuple[ParsedSection, ...]:
    values = list(headings)
    if not values and any(line.strip() for line in lines):
        values = [(1, 1, "Document")]
    output: list[ParsedSection] = []
    for ordinal, (line_number, level, title) in enumerate(values):
        end = values[ordinal + 1][0] - 1 if ordinal + 1 < len(values) else len(lines)
        text = "\n".join(lines[line_number - 1 : end]).strip()
        section_material = (
            f"{artifact.artifact_digest}\0{line_number}\0{level}\0{title}"
        )
        output.append(
            ParsedSection(
                section_id=f"sec-{hashlib.sha256(section_material.encode()).hexdigest()[:20]}",
                title=" ".join(title.split()),
                level=level,
                text=text,
                ordinal=ordinal,
            )
        )
    return tuple(output)


def _parse_markdown(artifact: SourceArtifact, text: str) -> ParsedDocument:
    lines = text.splitlines()
    headings: list[tuple[int, int, str]] = []
    fenced_lines: set[int] = set()
    active_fence: tuple[int, str] | None = None
    front_matter_end = _markdown_front_matter_end(lines)
    for index, line in enumerate(lines, 1):
        content, quote_depth = _markdown_quote_content(line)
        if front_matter_end and index <= front_matter_end:
            continue
        fence_match = match_fence(content)
        if active_fence is not None and quote_depth < active_fence[0]:
            active_fence = None
        if active_fence is not None:
            fenced_lines.add(index)
            if (
                quote_depth == active_fence[0]
                and fence_match
                and fence_match.group(1)[0] == active_fence[1][0]
                and len(fence_match.group(1)) >= len(active_fence[1])
                and not fence_match.group(2).strip()
            ):
                active_fence = None
            continue
        if fence_match:
            marker = fence_match.group(1)
            active_fence = (quote_depth, marker)
            fenced_lines.add(index)
            continue
        heading = match_atx_heading(content)
        if heading:
            headings.append((index, len(heading.group(1)), heading.group(2)))
            continue
        setext = match_setext_heading(content)
        if (
            setext
            and index > 1
            and index - 1 not in fenced_lines
        ):
            previous, previous_depth = _markdown_quote_content(lines[index - 2])
            if (
                previous_depth == quote_depth
                and previous.strip()
                and _markdown_indent_width(previous) < 4
                and match_atx_heading(previous) is None
                and match_fence(previous) is None
            ):
                headings.append(
                    (index - 1, 1 if setext.group(1)[0] == "=" else 2, previous.strip())
                )
    indented_code_lines = _markdown_indented_code_lines(lines, fenced_lines)
    front_matter_lines = (
        set(range(1, front_matter_end + 1)) if front_matter_end else set()
    )
    spans = _scan_delimited_math(
        artifact,
        lines,
        excluded_lines=fenced_lines | indented_code_lines | front_matter_lines,
        include_tex_environments=True,
        precise_columns=False,
    )
    metadata: dict[str, object] = {"format": "markdown"}
    explicit_fields = _markdown_explicit_term_fields(text)
    if explicit_fields:
        metadata["explicit_term_fields"] = explicit_fields
    return ParsedDocument(
        source=artifact,
        sections=_sections_from_headings(
            headings,
            [
                "" if line_number in front_matter_lines else line
                for line_number, line in enumerate(lines, 1)
            ],
            artifact,
        ),
        math_spans=spans,
        metadata=metadata,
    )


def _markdown_indented_code_lines(
    lines: list[str], fenced_lines: set[int]
) -> set[int]:
    """Identify indented code blocks without hiding ordinary indented content."""

    excluded: set[int] = set()
    code_active = False
    paragraph_active = False
    list_content_indent: int | None = None
    math_end: str | None = None
    quote_depth = 0

    for line_number, line in enumerate(lines, 1):
        content, current_quote_depth = _markdown_quote_content(line)
        if current_quote_depth != quote_depth:
            code_active = False
            paragraph_active = False
            list_content_indent = None
            math_end = None
            quote_depth = current_quote_depth
        if line_number in fenced_lines:
            code_active = False
            paragraph_active = False
            math_end = None
            continue
        stripped = content.strip()
        if not stripped:
            paragraph_active = False
            continue
        if math_end is not None:
            if math_end in content:
                math_end = None
            paragraph_active = True
            continue

        indent = _markdown_indent_width(content)
        list_match = re.match(
            r"^( {0,3})(?:[-+*]|\d+[.)])([ \t]+)", content
        )
        if list_match:
            prefix = list_match.group(0)
            list_content_indent = _markdown_column_width(prefix)
            code_active = False
            paragraph_active = True
            math_end = _markdown_math_end(content)
            continue

        if list_content_indent is not None and indent < list_content_indent:
            list_content_indent = None

        code_threshold = (
            list_content_indent + 4 if list_content_indent is not None else 4
        )
        if indent >= code_threshold and (code_active or not paragraph_active):
            excluded.add(line_number)
            code_active = True
            paragraph_active = False
            continue

        code_active = False
        paragraph_active = not bool(
            re.match(r"^\s{0,3}(?:#{1,6})(?:\s+|$)", content)
            or re.match(r"^\s{0,3}(?:[-*_]\s*){3,}$", content)
        )
        math_end = _markdown_math_end(content)

    return excluded


def _scan_delimited_math(
    artifact: SourceArtifact,
    lines: list[str],
    *,
    excluded_lines: set[int],
    include_tex_environments: bool,
    precise_columns: bool = True,
) -> tuple[MathSpan, ...]:
    _validate_display_math(
        artifact,
        lines,
        excluded_lines=excluded_lines,
        include_tex_environments=include_tex_environments,
    )
    spans: list[MathSpan] = []
    occupied: set[tuple[int, int]] = set()
    environment_names = "equation|align|gather|multline|eqnarray"
    scan_lines = (
        [_mask_markdown_inline_code(line) for line in lines]
        if artifact.source_format is SourceFormat.MARKDOWN
        else lines
    )
    joined = "\n".join(scan_lines)
    bracket_joined = _mask_pattern_preserving_lines(
        joined,
        re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", re.DOTALL),
    )
    bracket_joined = _mask_pattern_preserving_lines(
        bracket_joined,
        re.compile(r"(?<!\\)(?<!\$)\$(?!\$)(.+?)(?<!\\)\$(?!\$)"),
    )
    offsets = _line_offsets(lines)
    patterns: list[tuple[MathSpanKind, re.Pattern[str], str]] = [
        (
            MathSpanKind.DISPLAY,
            re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
            joined,
        ),
    ]
    if include_tex_environments:
        patterns.append(
            (
                MathSpanKind.DISPLAY,
                re.compile(
                    rf"\\begin\{{(?P<env>{environment_names})\*?\}}.*?"
                    rf"\\end\{{(?P=env)\*?\}}",
                    re.DOTALL,
                ),
                joined,
            )
        )

    def record_span(
        kind: MathSpanKind, start: int, end: int, raw: str
    ) -> None:
        start_line, start_column = _offset_position(offsets, start)
        end_line, end_column = _offset_position(
            offsets, max(start, end - 1)
        )
        if any(
            line in excluded_lines
            for line in range(start_line, end_line + 1)
        ):
            return
        cells = {
            (line, column)
            for line in range(start_line, end_line + 1)
            for column in range(
                start_column if line == start_line else 1,
                (
                    end_column
                    if line == end_line
                    else len(lines[line - 1])
                )
                + 1,
            )
        }
        if cells.intersection(occupied):
            return
        span = _make_span(
            artifact,
            lines,
            kind=kind,
            start_line=start_line,
            start_column=start_column,
            end_line=end_line,
            end_column=end_column,
            raw=raw,
            precise_columns=precise_columns,
        )
        if span:
            spans.append(span)
            occupied.update(cells)

    for kind, pattern, search_text in patterns:
        for match in pattern.finditer(search_text):
            record_span(kind, match.start(), match.end(), match.group(0))
    for start, end in _active_tex_delimited_ranges(
        bracket_joined, r"\[", r"\]"
    ):
        record_span(
            MathSpanKind.DISPLAY,
            start,
            end,
            joined[start:end],
        )

    inline_pattern = re.compile(
        r"(?<!\\)(?<!\$)\$(?!\$)(.+?)(?<!\\)\$(?!\$)"
    )
    for line_number, line in enumerate(lines, 1):
        if line_number in excluded_lines:
            continue
        # Inline code is excluded without interpreting its contents.
        code_ranges = [
            range(match.start() + 1, match.end() + 1)
            for match in re.finditer(r"`+[^`]*`+", line)
        ]
        inline_ranges = [
            (match.start(), match.end(), match.group(0))
            for match in inline_pattern.finditer(line)
        ]
        inline_ranges.extend(
            (start, end, line[start:end])
            for start, end in _active_tex_delimited_ranges(
                line, r"\(", r"\)"
            )
        )
        for start, end, raw in inline_ranges:
            columns = range(start + 1, end + 1)
            if any(
                (line_number, column) in occupied
                or any(column in code_range for code_range in code_ranges)
                for column in columns
            ):
                continue
            span = _make_span(
                artifact,
                lines,
                kind=MathSpanKind.INLINE,
                start_line=line_number,
                start_column=start + 1,
                end_line=line_number,
                end_column=end,
                raw=raw,
                precise_columns=precise_columns,
            )
            if span:
                spans.append(span)
                occupied.update(
                    (line_number, column) for column in columns
                )
    return tuple(
        sorted(
            spans,
            key=lambda item: (
                item.source_line_start or 0,
                item.source_column_start or 0,
                item.source_line_end or 0,
                item.source_column_end or 0,
            ),
        )
    )


def _validate_display_math(
    artifact: SourceArtifact,
    lines: list[str],
    *,
    excluded_lines: set[int],
    include_tex_environments: bool,
) -> None:
    active = "\n".join(
        (
            ""
            if line_number in excluded_lines
            else (
                _mask_markdown_inline_code(line)
                if artifact.source_format is SourceFormat.MARKDOWN
                else line
            )
        )
        for line_number, line in enumerate(lines, 1)
    )
    if len(re.findall(r"(?<!\\)\$\$", active)) % 2:
        raise ParseError(
            "unclosed_rich_block",
            "unclosed display-math delimiter $$",
            artifact=artifact,
        )
    # Bracket delimiters are not active while another math delimiter owns the
    # same text.  Mask balanced dollar-math first so OCR text such as
    # ``$S\[\phi\]$`` does not look like an unclosed outer display block; the
    # span scanner already resolves these overlaps in favor of dollar math.
    bracket_active = _mask_pattern_preserving_lines(
        active,
        re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", re.DOTALL),
    )
    bracket_active = _mask_pattern_preserving_lines(
        bracket_active,
        re.compile(r"(?<!\\)(?<!\$)\$(?!\$)(.+?)(?<!\\)\$(?!\$)"),
    )
    bracket_depth = 0
    for _position, token in _active_tex_delimiter_tokens(
        bracket_active, r"\[", r"\]"
    ):
        if token == r"\[":
            bracket_depth += 1
        elif bracket_depth:
            bracket_depth -= 1
    if bracket_depth:
        raise ParseError(
            "unclosed_rich_block",
            r"unclosed display-math delimiter \[",
            artifact=artifact,
        )
    if not include_tex_environments:
        return
    stack: list[str] = []
    for token in re.finditer(
        r"\\(?P<kind>begin|end)\{(?P<env>equation|align|gather|multline|eqnarray)\*?\}",
        active,
    ):
        environment = token.group("env")
        if token.group("kind") == "begin":
            stack.append(environment)
        elif stack and stack[-1] == environment:
            stack.pop()
    if stack:
        raise ParseError(
            "unclosed_rich_block",
            f"unclosed {stack[-1]} environment",
            artifact=artifact,
        )


def _mask_pattern_preserving_lines(
    value: str, pattern: re.Pattern[str]
) -> str:
    return pattern.sub(
        lambda match: re.sub(r"[^\n]", " ", match.group(0)),
        value,
    )


def _active_tex_delimiter_tokens(
    value: str, opening: str, closing: str
) -> tuple[tuple[int, str], ...]:
    pattern = re.compile(
        rf"{re.escape(opening)}|{re.escape(closing)}"
    )
    values: list[tuple[int, str]] = []
    for match in pattern.finditer(value):
        preceding = 0
        cursor = match.start() - 1
        while cursor >= 0 and value[cursor] == "\\":
            preceding += 1
            cursor -= 1
        if preceding % 2 == 0:
            values.append((match.start(), match.group(0)))
    return tuple(values)


def _active_tex_delimited_ranges(
    value: str, opening: str, closing: str
) -> tuple[tuple[int, int], ...]:
    start: int | None = None
    values: list[tuple[int, int]] = []
    for position, token in _active_tex_delimiter_tokens(
        value, opening, closing
    ):
        if token == opening:
            if start is None:
                start = position
        elif start is not None:
            values.append((start, position + len(closing)))
            start = None
    return tuple(values)


def _mask_markdown_inline_code(value: str) -> str:
    return re.sub(
        r"`+[^`]*`+",
        lambda match: " " * len(match.group(0)),
        value,
    )


def _line_offsets(lines: list[str]) -> list[int]:
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line) + 1
    return offsets or [0]


def _offset_position(offsets: list[int], offset: int) -> tuple[int, int]:
    for index in range(len(offsets) - 1, -1, -1):
        if offset >= offsets[index]:
            return index + 1, offset - offsets[index] + 1
    return 1, 1


def _parse_tex(artifact: SourceArtifact, text: str) -> ParsedDocument:
    active = _tex_structural_text(text)
    if re.search(r"\\(?:input|include)(?![A-Za-z@])\s*(?:\{|[^\s])", active):
        raise ParseError(
            "unsupported_tex_project",
            "TeX parsing accepts one pre-flattened file; input/include is unsupported",
            artifact=artifact,
        )
    lines = active.splitlines()
    headings: list[tuple[int, int, str]] = []
    levels = {"section": 1, "subsection": 2, "subsubsection": 3}
    index = 0
    while index < len(lines):
        cursor = 0
        while True:
            heading = _scan_tex_heading(
                lines, index, artifact, cursor=cursor
            )
            if heading is None:
                break
            start = index
            end, cursor, command, title = heading
            headings.append(
                (start + 1, levels[command], _tex_heading_text(title))
            )
            index = end
        index += 1
    spans = _scan_delimited_math(
        artifact,
        lines,
        excluded_lines=set(),
        include_tex_environments=True,
        precise_columns=False,
    )
    metadata: dict[str, object] = {"format": "tex", "single_file": True}
    explicit_fields = _tex_explicit_term_fields(active)
    if explicit_fields:
        metadata["explicit_term_fields"] = explicit_fields
    return ParsedDocument(
        source=artifact,
        sections=_sections_from_headings(headings, lines, artifact),
        math_spans=spans,
        metadata=metadata,
    )


def _tex_heading_text(value: str) -> str:
    title = _unwrap_texorpdfstring(value)
    title = title.replace(r"\{", "\0OPEN\0").replace(r"\}", "\0CLOSE\0")
    title = re.sub(
        r"\\(?:textbf|textit|emph|mathrm|mathbf|mathcal)\{([^{}]*)\}",
        r"\1",
        title,
    )
    title = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", "", title)
    title = title.replace("{", "").replace("}", "")
    return " ".join(
        title.replace("\0OPEN\0", "{").replace("\0CLOSE\0", "}").split()
    )


def _parse_html(artifact: SourceArtifact, text: str) -> ParsedDocument:
    soup = BeautifulSoup(text, "html.parser")
    roots = html_roots(soup)
    headings = [
        tag
        for root in roots
        for tag in root.find_all(re.compile(r"^h[1-6]$"))
        if isinstance(tag, Tag) and not html_heading_is_document_metadata(tag)
    ]
    sections: list[ParsedSection] = []
    if headings:
        for ordinal, heading in enumerate(headings):
            title = heading.get_text(" ", strip=True)
            content: list[str] = []
            for sibling in heading.next_siblings:
                if isinstance(sibling, Tag) and sibling.name in {
                    f"h{level}" for level in range(1, int(heading.name[1:]) + 1)
                }:
                    break
                if isinstance(sibling, Tag):
                    value = sibling.get_text(" ", strip=True)
                    if value:
                        content.append(value)
            source_key = str(heading.get("id") or f"{ordinal}:{title}")
            sections.append(
                ParsedSection(
                    section_id=f"sec-{hashlib.sha256((artifact.artifact_digest + source_key).encode()).hexdigest()[:20]}",
                    title=title,
                    level=int(heading.name[1:]),
                    text="\n".join(content),
                    ordinal=ordinal,
                )
            )
    elif any(root.get_text(" ", strip=True) for root in roots):
        visible_text = "\n".join(
            root.get_text(" ", strip=True)
            for root in roots
            if root.get_text(" ", strip=True)
        )
        sections.append(
            ParsedSection(
                section_id=f"sec-{artifact.artifact_digest[:20]}",
                title="Document",
                level=1,
                text=visible_text,
                ordinal=0,
            )
        )
    spans: list[MathSpan] = []
    math_nodes = [
        node
        for root in roots
        for node in root.find_all("math")
        if isinstance(node, Tag) and node.find_parent("math") is None
    ]
    processed_equation_tables: set[int] = set()
    ordinal = 0

    def append_span(
        tex: str,
        *,
        node: Tag,
        display: bool,
        label: str = "",
    ) -> None:
        nonlocal ordinal
        if not tex:
            return
        line_start, column_start, line_end, column_end = html_source_position(node)
        source_key = ":".join(
            str(value)
            for value in (
                node.get("id") or "",
                line_start,
                column_start,
                ordinal,
            )
        )
        span_id = (
            f"math-{hashlib.sha256((artifact.artifact_digest + source_key + tex).encode()).hexdigest()[:24]}"
        )
        spans.append(
            MathSpan(
                span_id=span_id,
                kind=MathSpanKind.DISPLAY if display else MathSpanKind.INLINE,
                source_line_start=line_start,
                source_column_start=column_start,
                source_line_end=line_end,
                source_column_end=column_end,
                normalized_tex=tex,
                context_before=_html_neighbor_text(node, previous=True),
                context_after=_html_neighbor_text(node, previous=False),
                source_label=label if display else "",
            )
        )
        ordinal += 1

    for math in math_nodes:
        table = math.find_parent("table")
        if isinstance(table, Tag) and _html_is_equation_table(table):
            if id(table) in processed_equation_tables:
                continue
            processed_equation_tables.add(id(table))
            for unit in html_equation_table_units(table):
                append_span(
                    normalize_tex(
                        " ".join(_html_math_tex(fragment) for fragment in unit.math_nodes)
                    ),
                    node=unit.locator_node,
                    display=True,
                    label=unit.label,
                )
            continue
        tex = _html_math_tex(math)
        container = math.find_parent(
            class_=re.compile(r"(?:^|\s)ltx_equation(?:\s|$)")
        )
        display = isinstance(container, Tag) or (
            str(math.get("display") or "").casefold() == "block"
        )
        append_span(
            tex,
            node=math,
            display=display,
            label=html_displayed_equation_label(math) if display else "",
        )
    metadata: dict[str, object] = {"format": "html"}
    explicit_fields = _html_explicit_term_fields(soup)
    if explicit_fields:
        metadata["explicit_term_fields"] = explicit_fields
    return ParsedDocument(
        source=artifact,
        sections=tuple(sections),
        math_spans=tuple(spans),
        metadata=metadata,
    )


def _html_neighbor_text(node: Tag, *, previous: bool) -> str:
    method: Callable[..., Tag | None] = (
        node.find_previous if previous else node.find_next
    )
    candidate = method(["p", "li", "blockquote"])
    if not isinstance(candidate, Tag):
        return ""
    section = node.find_parent("section")
    if section is not None and candidate.find_parent("section") is not section:
        return ""
    return candidate.get_text(" ", strip=True)


def _html_is_equation_table(node: Tag) -> bool:
    return any(
        "equation" in str(class_name).casefold()
        for class_name in node.get("class") or ()
    )


def _markdown_explicit_term_fields(text: str) -> list[dict[str, object]]:
    """Extract only syntactically explicit front-matter term fields."""

    lines = text.splitlines()
    front_matter_end = _markdown_front_matter_end(lines)
    if not front_matter_end:
        return []
    end = front_matter_end - 1
    output: list[dict[str, object]] = []
    index = 1
    while index < end:
        match = re.match(
            r"^\s*(keywords?|key[-_ ]?terms?|index[-_ ]?terms?)\s*:\s*(.*)$",
            lines[index],
            re.IGNORECASE,
        )
        if match is None:
            index += 1
            continue
        label, remainder = match.group(1), match.group(2).strip()
        entries: list[str] = []
        if remainder:
            unwrapped = remainder.strip("[]")
            entries.extend(
                item.strip(" \t\"'")
                for item in re.split(r"[,;]", unwrapped)
                if _is_explicit_term_entry(item.strip(" \t\"'"))
            )
        cursor = index + 1
        while cursor < end:
            item_match = re.match(r"^\s*-\s+(.+?)\s*$", lines[cursor])
            if item_match is None:
                break
            item = item_match.group(1).strip(" \t\"'")
            if _is_explicit_term_entry(item):
                entries.append(item)
            cursor += 1
        if entries:
            output.append(
                {
                    "kind": (
                        "index" if "index" in label.casefold() else "keywords"
                    ),
                    "label": label,
                    "entries": entries,
                }
            )
        index = max(index + 1, cursor)
    return output


def _tex_explicit_term_fields(text: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for match in re.finditer(
        r"\\(?:keywords?|keyterms?)\*?\s*\{([^{}]*)\}",
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        entries = [
            item.strip()
            for item in re.split(r"[,;]", match.group(1))
            if _is_explicit_term_entry(item.strip())
        ]
        if entries:
            output.append(
                {
                    "kind": "keywords",
                    "label": "keywords",
                    "entries": entries,
                }
            )
    for match in re.finditer(
        r"\\begin\s*\{theindex\}(.*?)\\end\s*\{theindex\}",
        text,
        re.IGNORECASE | re.DOTALL,
    ):
        entries = [
            item.strip()
            for item in re.findall(
                r"\\item\s+(.+?)(?=\\item|\\end\s*\{theindex\}|$)",
                match.group(1),
                re.DOTALL,
            )
            if _is_explicit_term_entry(item.strip())
        ]
        if entries:
            output.append(
                {
                    "kind": "index",
                    "label": "theindex",
                    "entries": entries,
                }
            )
    return output


def _html_explicit_term_fields(
    soup: BeautifulSoup,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for meta in soup.find_all("meta"):
        if not isinstance(meta, Tag):
            continue
        name = str(meta.get("name") or meta.get("property") or "")
        if re.sub(r"[^a-z]", "", name.casefold()) not in {
            "keyword",
            "keywords",
            "citationkeywords",
        }:
            continue
        entries = [
            item.strip()
            for item in re.split(r"[,;]", str(meta.get("content") or ""))
            if _is_explicit_term_entry(item.strip())
        ]
        if entries:
            output.append(
                {
                    "kind": "keywords",
                    "label": name,
                    "entries": entries,
                }
            )
    selectors = (
        "[role='doc-index']",
        ".ltx_index",
        ".keywords",
        ".keyword-list",
        "[data-type='keywords']",
    )
    seen_nodes: set[int] = set()
    for selector in selectors:
        for node in soup.select(selector):
            if not isinstance(node, Tag) or id(node) in seen_nodes:
                continue
            seen_nodes.add(id(node))
            kind = (
                "index"
                if node.get("role") == "doc-index"
                or "index" in " ".join(node.get("class") or ()).casefold()
                else "keywords"
            )
            listed = [
                item.get_text(" ", strip=True)
                for item in node.find_all("li")
                if _is_explicit_term_entry(item.get_text(" ", strip=True))
            ]
            entries = listed or [
                item.strip()
                for item in re.split(
                    r"[,;\n]", node.get_text("\n", strip=True)
                )
                if _is_explicit_term_entry(item.strip())
            ]
            if entries:
                output.append(
                    {
                        "kind": kind,
                        "label": str(node.get("id") or kind),
                        "entries": entries,
                    }
                )
    return output


def _is_explicit_term_entry(value: str) -> bool:
    item = value.strip()
    return bool(item) and not all(
        unicodedata.category(character) == "Pd" for character in item
    )


def _parse_pdf(
    artifact: SourceArtifact,
    payload: bytes,
    *,
    extractor: PDFTextExtractor,
) -> ParsedDocument:
    if not payload.startswith(b"%PDF"):
        raise ParseError(
            "pdf_invalid",
            "PDF source does not contain a PDF header",
            artifact=artifact,
        )
    try:
        result = extractor.extract(payload)
    except PDFTextExtractionError as exc:
        raise ParseError(exc.code, exc.message, artifact=artifact) from exc
    pages = tuple(ParsedPage(index, text) for index, text in enumerate(result.pages, 1))
    sections = tuple(
        ParsedSection(
            section_id=f"pdf-page-{index:05d}",
            title=_first_nonempty_line(page) or f"Page {index}",
            level=1,
            text=page,
            ordinal=index - 1,
            page_start=index,
            page_end=index,
        )
        for index, page in enumerate(result.pages, 1)
    )
    spans: list[MathSpan] = []
    for page_number, page in enumerate(result.pages, 1):
        page_lines = page.splitlines()
        for line_number, line in enumerate(page_lines, 1):
            unit = _pdf_equation_unit(line)
            if unit is None:
                continue
            span = _make_span(
                artifact,
                page_lines,
                kind=MathSpanKind.DISPLAY,
                start_line=line_number,
                start_column=1,
                end_line=line_number,
                end_column=max(1, len(line)),
                raw=unit.raw,
                identity_scope=f"pdf-page:{page_number}",
            )
            if span:
                spans.append(
                    replace(
                        span,
                        source_label=unit.source_label,
                    )
                )
    warnings = (result.warning,) if result.warning else ()
    return ParsedDocument(
        source=artifact,
        sections=sections,
        math_spans=tuple(spans),
        pages=pages,
        warnings=warnings,
        metadata={
            "format": "pdf",
            "page_count": len(result.pages),
            "text_layer": result.has_text,
        },
    )


def _first_nonempty_line(value: str) -> str:
    return next((line.strip() for line in value.splitlines() if line.strip()), "")


def _looks_like_pdf_math(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    has_operator = bool(re.search(r"[=+\-*/^_≤≥∑∫√]", stripped))
    printed_number = bool(re.search(r"\([^()]*\d[^()]*\)\s*$", stripped))
    return has_operator and (printed_number or len(stripped.split()) <= 24)


def _pdf_equation_unit(value: str) -> _PDFEquationUnit | None:
    """Recognize one logical displayed-equation line in extracted PDF text.

    The PDF parser retains this narrow rule for generic math extraction and
    printed-label capture. It does not infer canonical labels from text layout.
    """

    if not _looks_like_pdf_math(value):
        return None
    raw = re.sub(
        r"\s*\(([^()]*(?:\d|[ivxlcdm])[^()]*)\)\s*$", "", value
    ).strip()
    tex = normalize_tex(raw)
    if not tex:
        return None
    return _PDFEquationUnit(
        raw=raw,
        source_label=_pdf_printed_equation_label(value),
    )


def _pdf_printed_equation_label(value: str) -> str:
    match = re.search(r"\(\s*([^()]+?)\s*\)\s*$", value)
    return match.group(1).strip() if match else ""


__all__ = [
    "PDFOutlineExtractionError",
    "PDFTextExtractionError",
    "PDFTextExtractor",
    "ParseError",
    "PdftotextExtractor",
    "QpdfOutlineExtractor",
    "normalize_tex",
    "parse_artifact_bytes",
]
