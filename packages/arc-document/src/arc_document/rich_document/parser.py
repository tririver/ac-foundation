from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from .._parsing import ParseError, normalize_tex
from .._parsing.html_source import (
    html_heading_is_document_metadata,
    html_roots,
    html_source_position,
    rich_html_selector,
)
from .._parsing.html_equations import (
    html_displayed_equation_label as _html_displayed_equation_label,
    html_equation_table_units,
    html_math_tex as _html_math_tex,
)
from .._parsing.markdown_lex import (
    markdown_front_matter_end,
    markdown_indent_width,
    match_atx_heading,
    match_fence,
    match_setext_heading,
)
from .._parsing.tex_lex import (
    scan_tex_heading as _tex_heading,
    scan_tex_balanced as _scan_tex_balanced,
    tex_without_comments as _tex_without_comments,
    unwrap_texorpdfstring as _unwrap_texorpdfstring,
)
from ..sources import SourceArtifact, SourceFormat
from .models import (
    RichAsset,
    RichBlock,
    RichBlockKind,
    RichDocument,
    RichPageMapEntry,
    RichSection,
    SourceLocator,
)


AssetImporter = Callable[[str], RichAsset | None]
_MarkdownImage = tuple[str, str, str]
_MARKDOWN_IMAGE_RE = re.compile(
    r'!\[([^\]]*)\]\((\S+?)(?:\s+["\'](.*?)["\'])?\)'
)
_MARKDOWN_EXTRACTION_SIDECAR_KINDS = frozenset(
    {
        "natural_image",
        "text_image",
        "flowchart",
        "chemical",
        "line",
    }
)
_MARKDOWN_EXTRACTION_SUMMARY_RE = re.compile(
    r"<summary>\s*("
    + "|".join(sorted(_MARKDOWN_EXTRACTION_SIDECAR_KINDS))
    + r")\s*</summary>"
)
SOURCE_PAGE_BOUNDARIES_SCHEMA = "arc.document.source_page_boundaries.v1"
SOURCE_PAGE_BOUNDARIES_METADATA_KEY = "source_page_boundaries"
DOCUMENT_NOTES_SCHEMA = "arc.document.document_notes.v1"
DOCUMENT_NOTES_METADATA_KEY = "document_notes"
_MARKDOWN_PAGE_MARKER_RE = re.compile(
    r"^\s*<!--\s*(?:Source PDF page\s+|PDF_PAGE:\s*)([1-9][0-9]*)\s*-->\s*$"
)


@dataclass(frozen=True)
class RichSourceParseResult:
    document: RichDocument
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _RawBlock:
    kind: RichBlockKind
    locator: SourceLocator
    payload: Mapping[str, Any]
    section_reset_to_level: int | None = None


@dataclass(frozen=True)
class _HTMLFallback:
    locator_node: Tag
    values: tuple[Tag | NavigableString, ...]


_HTML_BLOCK_NAMES = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "ul",
    "ol",
    "pre",
    "table",
    "figure",
    "img",
}
_HTML_IGNORED_NAMES = {
    "head",
    "nav",
    "noscript",
    "script",
    "style",
    "template",
}


def parse_rich_artifact_bytes(
    artifact: SourceArtifact,
    payload: bytes,
    *,
    asset_importer: AssetImporter | None = None,
) -> RichSourceParseResult:
    """Parse one rich primary source independently of the standard parser."""

    if artifact.source_format not in {
        SourceFormat.MARKDOWN,
        SourceFormat.HTML,
        SourceFormat.TEX,
    }:
        raise ParseError(
            "rich_source_required",
            "rich document parsing requires Markdown, HTML, or flattened TeX",
            artifact=artifact,
        )
    if (
        len(payload) != artifact.size
        or hashlib.sha256(payload).hexdigest() != artifact.artifact_digest
    ):
        raise ParseError(
            "source_artifact_mismatch",
            "source bytes do not match the supplied artifact",
            artifact=artifact,
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ParseError(
            "source_encoding_invalid",
            f"{artifact.source_format.value} source must be UTF-8",
            artifact=artifact,
        ) from exc
    assets: dict[str, RichAsset] = {}
    warnings: list[str] = []

    def import_asset(target: str) -> RichAsset | None:
        if not _is_local_asset_target(target):
            return None
        if asset_importer is None:
            warnings.append(f"local asset was not imported: {target}")
            return None
        asset = asset_importer(target)
        if asset is None:
            warnings.append(f"local asset was not found: {target}")
            return None
        assets.setdefault(asset.artifact_digest, asset)
        return asset

    if artifact.source_format is SourceFormat.MARKDOWN:
        raw = _parse_markdown(text, artifact, import_asset)
    elif artifact.source_format is SourceFormat.HTML:
        raw = _parse_html(text, artifact, import_asset)
    else:
        raw = _parse_tex(text, artifact, import_asset)
    metadata: dict[str, Any] = {
        "format": artifact.source_format.value,
        "single_file": artifact.source_format is SourceFormat.TEX,
    }
    # Keep explicit term fields in the format-neutral RichDocument so
    # downstream keyword workflows never need to reopen the source path.
    from ..parse.parser import (
        _html_explicit_term_fields,
        _markdown_explicit_term_fields,
        _tex_explicit_term_fields,
    )

    if artifact.source_format is SourceFormat.MARKDOWN:
        explicit_fields = _markdown_explicit_term_fields(text)
    elif artifact.source_format is SourceFormat.HTML:
        explicit_fields = _html_explicit_term_fields(
            BeautifulSoup(text, "lxml")
        )
    else:
        explicit_fields = _tex_explicit_term_fields(text)
    if explicit_fields:
        metadata["explicit_term_fields"] = explicit_fields
    document = _finalize_document(
        artifact,
        raw,
        assets=tuple(assets.values()),
        metadata=metadata,
    )
    if artifact.source_format is SourceFormat.MARKDOWN:
        document = _with_markdown_page_boundaries(document, text)
    return RichSourceParseResult(document=document, warnings=tuple(_dedupe(warnings)))


def _parse_markdown(
    text: str,
    artifact: SourceArtifact,
    import_asset: AssetImporter,
) -> list[_RawBlock]:
    lines = text.splitlines()
    output: list[_RawBlock] = []
    front_matter_end = markdown_front_matter_end(lines)
    index = front_matter_end
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        comment_end = _markdown_standalone_comment_end(lines, index)
        if comment_end is not None:
            index = comment_end
            continue
        heading = match_atx_heading(line)
        if heading:
            output.append(
                _raw(
                    artifact,
                    RichBlockKind.HEADING,
                    index + 1,
                    index + 1,
                    {
                        "text": _markdown_plain_text(heading.group(2)),
                        "level": len(heading.group(1)),
                    },
                )
            )
            index += 1
            continue
        fence = match_fence(line)
        if fence:
            marker = fence.group(1)
            language = fence.group(2).strip().split(maxsplit=1)[0] if fence.group(2).strip() else ""
            start = index
            index += 1
            code: list[str] = []
            while index < len(lines):
                if re.match(
                    rf"^\s{{0,3}}{re.escape(marker[0])}{{{len(marker)},}}\s*$",
                    lines[index],
                ):
                    break
                code.append(lines[index])
                index += 1
            line_end = index + 1 if index < len(lines) else len(lines)
            if index < len(lines):
                index += 1
            output.append(
                _raw(
                    artifact,
                    RichBlockKind.CODE,
                    start + 1,
                    line_end,
                    {"text": "\n".join(code), "language": language},
                )
            )
            continue
        if (
            index + 1 < len(lines)
            and markdown_indent_width(line) < 4
            and (setext := match_setext_heading(lines[index + 1]))
            and not _markdown_starts_block(lines, index, artifact)
        ):
            output.append(
                _raw(
                    artifact,
                    RichBlockKind.HEADING,
                    index + 1,
                    index + 2,
                    {
                        "text": _markdown_plain_text(line.strip()),
                        "level": 1 if setext.group(1)[0] == "=" else 2,
                    },
                )
            )
            index += 2
            continue
        equation = _markdown_display_equation(lines, index, artifact)
        if equation is not None:
            end, raw_tex = equation
            output.append(
                _raw(
                    artifact,
                    RichBlockKind.EQUATION,
                    index + 1,
                    end + 1,
                    {
                        "tex": normalize_tex(raw_tex),
                        "display": True,
                        "label": _tex_label(raw_tex),
                    },
                )
            )
            index = end + 1
            continue
        table = _markdown_table(lines, index)
        if table is not None:
            end, headers, rows = table
            _append_markdown_table_blocks(
                output,
                artifact,
                line_start=index + 1,
                line_end=end + 1,
                headers=headers,
                rows=rows,
                import_asset=import_asset,
            )
            index = end + 1
            continue
        list_match = re.match(r"^\s{0,3}([-+*]|\d+[.)])\s+(.+)$", line)
        if list_match:
            ordered = bool(re.match(r"\d", list_match.group(1)))
            start = index
            contents: list[str] = []
            while index < len(lines):
                item = re.match(r"^\s{0,3}([-+*]|\d+[.)])\s+(.+)$", lines[index])
                if not item or bool(re.match(r"\d", item.group(1))) != ordered:
                    break
                contents.append(item.group(2).strip())
                index += 1
            _append_markdown_list_blocks(
                output,
                artifact,
                line_start=start + 1,
                line_end=index,
                ordered=ordered,
                contents=contents,
                import_asset=import_asset,
            )
            continue
        figure = _markdown_figure(line)
        if figure is not None:
            alt_text, target, caption = figure
            asset = import_asset(target)
            sidecar_end = _markdown_extraction_sidecar_end(lines, index)
            output.append(
                _raw(
                    artifact,
                    RichBlockKind.FIGURE,
                    index + 1,
                    sidecar_end,
                    _figure_payload(
                        asset,
                        alt_text=alt_text,
                        caption=caption,
                        target=target,
                    ),
                )
            )
            index = sidecar_end
            continue
        start = index
        paragraph_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            candidate = lines[index]
            if index > start and _markdown_starts_block(lines, index, artifact):
                break
            paragraph_lines.append(candidate.strip())
            index += 1
        raw_text = "\n".join(paragraph_lines)
        _append_markdown_paragraph_blocks(
            output,
            artifact,
            line_start=start + 1,
            line_end=index,
            value=raw_text,
            import_asset=import_asset,
        )
    return output


def _markdown_extraction_sidecar_end(
    lines: list[str],
    figure_index: int,
) -> int:
    """Return the exclusive end of one recognized figure extraction sidecar.

    Extractors may append machine-readable image descriptions in a reserved
    ``details`` block.  They are figure metadata rather than authored prose.
    Recognition stays deliberately narrow so ordinary authored ``details``
    blocks continue through the regular Markdown paragraph path.
    """

    cursor = figure_index + 1
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    if cursor >= len(lines):
        return figure_index + 1

    first = lines[cursor].strip()
    summary_index: int
    if first == "<details>":
        summary_index = cursor + 1
        while (
            summary_index < len(lines)
            and not lines[summary_index].strip()
        ):
            summary_index += 1
        if summary_index >= len(lines):
            return figure_index + 1
        summary = lines[summary_index].strip()
    else:
        combined = re.fullmatch(
            r"<details>\s*(<summary>.*</summary>)",
            first,
        )
        if combined is None:
            return figure_index + 1
        summary_index = cursor
        summary = combined.group(1)

    if _MARKDOWN_EXTRACTION_SUMMARY_RE.fullmatch(summary) is None:
        return figure_index + 1

    cursor = summary_index + 1
    while cursor < len(lines):
        stripped = lines[cursor].strip()
        if stripped == "</details>":
            return cursor + 1
        if stripped.startswith("<details"):
            return figure_index + 1
        cursor += 1
    return figure_index + 1


def _markdown_starts_block(
    lines: list[str], index: int, artifact: SourceArtifact
) -> bool:
    line = lines[index]
    return bool(
        re.match(r"^\s{0,3}(?:#{1,6})\s+", line)
        or re.match(r"^\s{0,3}(?:`{3,}|~{3,})", line)
        or re.match(r"^\s{0,3}(?:[-+*]|\d+[.)])\s+", line)
        or _markdown_figure(line)
        or _markdown_display_equation(lines, index, artifact)
        or _markdown_table(lines, index)
        or _markdown_standalone_comment_end(lines, index) is not None
    )


def _markdown_standalone_comment_end(
    lines: list[str], index: int
) -> int | None:
    """Return the exclusive end of a standalone HTML comment.

    Markdown comments are metadata/non-visible authoring material.  Mixed
    comment-and-prose lines remain ordinary paragraph input.
    """

    stripped = lines[index].strip()
    if not stripped.startswith("<!--"):
        return None
    current = index
    while current < len(lines):
        close = lines[current].find("-->")
        if close >= 0:
            if lines[current][close + 3 :].strip():
                return None
            return current + 1
        current += 1
    return None


def _with_markdown_page_boundaries(
    document: RichDocument, text: str
) -> RichDocument:
    lines = text.splitlines()
    markers: list[tuple[int, int, int]] = []
    comments: list[tuple[int, int, str, int | None]] = []
    index = 0
    while index < len(lines):
        end = _markdown_standalone_comment_end(lines, index)
        if end is None:
            index += 1
            continue
        raw_comment = "\n".join(lines[index:end]).strip()
        match = (
            _MARKDOWN_PAGE_MARKER_RE.fullmatch(raw_comment)
            if end == index + 1
            else None
        )
        page_number = int(match.group(1)) if match is not None else None
        comments.append((index + 1, end, raw_comment, page_number))
        if page_number is not None:
            markers.append((index + 1, end, page_number))
        index = end
    if not comments:
        return document
    page_numbers = [item[2] for item in markers]
    if any(
        current <= previous
        for previous, current in zip(page_numbers, page_numbers[1:])
    ):
        raise ParseError(
            "rich_page_markers_invalid",
            "Markdown source page markers must be strictly increasing",
            artifact=document.source,
        )

    positioned = [
        block
        for block in document.blocks
        if block.locator.line_start is not None
    ]
    def following_block(end: int) -> RichBlock | None:
        return next(
            (
                block
                for block in positioned
                if int(block.locator.line_start or 0) > end
            ),
            None,
        )

    boundary_items: list[dict[str, Any]] = []
    for _start, end, page_number in markers:
        following = following_block(end)
        boundary_items.append(
            {
                "page_number": page_number,
                "before_block_id": (
                    following.block_id if following is not None else None
                ),
            }
        )

    note_items: list[dict[str, Any]] = []
    for _start, end, raw_comment, page_number in comments:
        following = following_block(end)
        note: dict[str, Any] = {
            "kind": "source_page" if page_number is not None else "metadata",
            "text": raw_comment,
            "before_block_id": (
                following.block_id if following is not None else None
            ),
        }
        if page_number is not None:
            note["page_number"] = page_number
        note_items.append(note)

    page_map: list[RichPageMapEntry] = []
    marker_index = 0
    current_page: int | None = None
    for block in positioned:
        line_start = int(block.locator.line_start or 0)
        while (
            marker_index < len(markers)
            and markers[marker_index][0] < line_start
        ):
            current_page = markers[marker_index][2]
            marker_index += 1
        if current_page is not None:
            page_map.append(
                RichPageMapEntry(
                    block_id=block.block_id,
                    page_number=current_page,
                )
            )

    metadata = dict(document.metadata)
    metadata[DOCUMENT_NOTES_METADATA_KEY] = {
        "schema_version": DOCUMENT_NOTES_SCHEMA,
        "items": note_items,
    }
    if boundary_items:
        metadata[SOURCE_PAGE_BOUNDARIES_METADATA_KEY] = {
            "schema_version": SOURCE_PAGE_BOUNDARIES_SCHEMA,
            "items": boundary_items,
        }
    return RichDocument(
        source=document.source,
        blocks=document.blocks,
        sections=document.sections,
        assets=document.assets,
        page_map=tuple(page_map),
        metadata=metadata,
    )


def _markdown_display_equation(
    lines: list[str], index: int, artifact: SourceArtifact
) -> tuple[int, str] | None:
    stripped = lines[index].strip()
    delimiters = {"$$": "$$", r"\[": r"\]"}
    for opening, closing in delimiters.items():
        if not stripped.startswith(opening):
            continue
        if stripped != opening and stripped.endswith(closing):
            return index, stripped
        values = [lines[index]]
        current = index + 1
        while current < len(lines):
            values.append(lines[current])
            if lines[current].strip().endswith(closing):
                return current, "\n".join(values)
            current += 1
        raise ParseError(
            "unclosed_rich_block",
            f"unclosed display-math delimiter {opening}",
            artifact=artifact,
        )
    environment = re.match(
        r"^\s*\\begin\{(equation|align|gather|multline|eqnarray)\*?\}",
        stripped,
    )
    if environment:
        values = [lines[index]]
        current = index
        closing = re.compile(
            rf"\\end\{{{re.escape(environment.group(1))}\*?\}}"
        )
        while current + 1 < len(lines) and not closing.search(values[-1]):
            current += 1
            values.append(lines[current])
        if not closing.search(values[-1]):
            raise ParseError(
                "unclosed_rich_block",
                f"unclosed {environment.group(1)} environment",
                artifact=artifact,
            )
        return current, "\n".join(values)
    return None


def _markdown_table(
    lines: list[str], index: int
) -> tuple[int, list[str], list[list[str]]] | None:
    if index + 1 >= len(lines) or "|" not in lines[index]:
        return None
    separator = _split_pipe_row(lines[index + 1])
    if not separator or any(
        re.fullmatch(r":?-{3,}:?", cell.strip()) is None for cell in separator
    ):
        return None
    headers = [cell.strip() for cell in _split_pipe_row(lines[index])]
    rows: list[list[str]] = []
    current = index + 2
    while current < len(lines) and "|" in lines[current] and lines[current].strip():
        rows.append([cell.strip() for cell in _split_pipe_row(lines[current])])
        current += 1
    return current - 1, headers, rows


def _split_pipe_row(value: str) -> list[str]:
    stripped = value.strip().strip("|")
    return [item.replace(r"\|", "|") for item in re.split(r"(?<!\\)\|", stripped)]


def _markdown_figure(value: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(
        r'\s*!\[([^\]]*)\]\((\S+?)(?:\s+["\'](.*?)["\'])?\)\s*',
        value,
    )
    return match.groups(default="") if match else None


def _markdown_image_segments(value: str) -> list[str | _MarkdownImage]:
    segments: list[str | _MarkdownImage] = []
    cursor = 0
    for match in _MARKDOWN_IMAGE_RE.finditer(value):
        segments.append(value[cursor : match.start()])
        segments.append(match.groups(default=""))
        cursor = match.end()
    segments.append(value[cursor:])
    return segments


def _append_markdown_paragraph_blocks(
    output: list[_RawBlock],
    artifact: SourceArtifact,
    *,
    line_start: int,
    line_end: int,
    value: str,
    import_asset: AssetImporter,
) -> None:
    for segment in _markdown_image_segments(value):
        if isinstance(segment, str):
            payload = _markdown_inline_payload(segment)
            if payload["text"]:
                output.append(
                    _raw(
                        artifact,
                        RichBlockKind.PARAGRAPH,
                        line_start,
                        line_end,
                        payload,
                    )
                )
            continue
        output.append(
            _markdown_image_block(
                artifact,
                line_start,
                line_end,
                segment,
                import_asset,
            )
        )


def _append_markdown_list_blocks(
    output: list[_RawBlock],
    artifact: SourceArtifact,
    *,
    line_start: int,
    line_end: int,
    ordered: bool,
    contents: list[str],
    import_asset: AssetImporter,
) -> None:
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending:
            return
        output.append(
            _raw(
                artifact,
                RichBlockKind.LIST,
                line_start,
                line_end,
                {"ordered": ordered, "items": list(pending)},
            )
        )
        pending.clear()

    for content in contents:
        for segment in _markdown_image_segments(content):
            if isinstance(segment, str):
                payload = _markdown_inline_payload(segment)
                if payload["text"]:
                    pending.append(payload)
                continue
            flush()
            output.append(
                _markdown_image_block(
                    artifact,
                    line_start,
                    line_end,
                    segment,
                    import_asset,
                )
            )
    flush()


def _append_markdown_table_blocks(
    output: list[_RawBlock],
    artifact: SourceArtifact,
    *,
    line_start: int,
    line_end: int,
    headers: list[str],
    rows: list[list[str]],
    import_asset: AssetImporter,
) -> None:
    shape = [headers, *rows]
    pending = [["" for _ in row] for row in shape]

    def flush() -> None:
        if not any(cell for row in pending for cell in row):
            return
        output.append(
            _raw(
                artifact,
                RichBlockKind.TABLE,
                line_start,
                line_end,
                {
                    "headers": list(pending[0]),
                    "rows": [list(row) for row in pending[1:]],
                    "caption": "",
                },
            )
        )
        for row in pending:
            for column in range(len(row)):
                row[column] = ""

    for row_index, row in enumerate(shape):
        for column, cell in enumerate(row):
            for segment in _markdown_image_segments(cell):
                if isinstance(segment, str):
                    text = _markdown_plain_text(segment)
                    if text:
                        pending[row_index][column] = " ".join(
                            value
                            for value in (pending[row_index][column], text)
                            if value
                        )
                    continue
                flush()
                output.append(
                    _markdown_image_block(
                        artifact,
                        line_start,
                        line_end,
                        segment,
                        import_asset,
                    )
                )
    flush()


def _markdown_image_block(
    artifact: SourceArtifact,
    line_start: int,
    line_end: int,
    image: _MarkdownImage,
    import_asset: AssetImporter,
) -> _RawBlock:
    alt_text, target, caption = image
    return _raw(
        artifact,
        RichBlockKind.FIGURE,
        line_start,
        line_end,
        _figure_payload(
            import_asset(target),
            alt_text=alt_text,
            caption=caption,
            target=target,
        ),
    )


def _markdown_inline_payload(value: str) -> dict[str, Any]:
    token = re.compile(
        r"(?P<link>(?<!!)\[([^\]]+)\]\(([^)\s]+)(?:\s+[^)]*)?\))"
        r"|(?P<dollar>(?<!\\)(?<!\$)\$(?!\$)(.+?)(?<!\\)\$(?!\$))"
        r"|(?P<paren>\\\((.+?)\\\))"
    )
    parts: list[dict[str, str]] = []
    cursor = 0
    for match in token.finditer(value):
        _append_inline_part(
            parts, "text", _markdown_text_segment(value[cursor : match.start()])
        )
        if match.lastgroup == "link":
            _append_inline_part(
                parts,
                "link",
                _markdown_plain_text(match.group(2)),
                target=match.group(3),
            )
        else:
            source = match.group(0)
            tex = normalize_tex(source)
            if tex:
                _append_inline_part(
                    parts, "math", source, tex=tex, source=source
                )
            else:
                _append_inline_part(parts, "text", source)
        cursor = match.end()
    _append_inline_part(parts, "text", _markdown_text_segment(value[cursor:]))
    return _inline_payload(parts)


def _markdown_text_segment(value: str) -> str:
    value = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"(?<!\\)(?:\*\*|__)(.+?)(?:\*\*|__)", r"\1", value)
    value = re.sub(r"(?<!\\)(?:\*|_)(.+?)(?:\*|_)", r"\1", value)
    value = re.sub(r"`([^`]+)`", r"\1", value)
    return re.sub(r"\s+", " ", value)


def _markdown_plain_text(value: str) -> str:
    value = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"(?<!\\)(?:\*\*|__)(.+?)(?:\*\*|__)", r"\1", value)
    value = re.sub(r"(?<!\\)(?:\*|_)(.+?)(?:\*|_)", r"\1", value)
    value = re.sub(r"`([^`]+)`", r"\1", value)
    return " ".join(value.split())


def _html_explicit_candidates(
    roots: list[Tag | BeautifulSoup],
) -> list[Tag]:
    candidate_names = [
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "ul",
        "ol",
        "pre",
        "table",
        "figure",
        "img",
        "math",
    ]
    candidates: list[Tag] = []
    for root in roots:
        candidates.extend(
            node
            for node in root.find_all(candidate_names)
            if isinstance(node, Tag)
        )
    eligible: list[Tag] = []
    block_containers = _HTML_BLOCK_NAMES - {"img"}
    for node in candidates:
        parent = node.find_parent(block_containers)
        if isinstance(parent, Tag):
            continue
        if node.name == "img" and isinstance(node.find_parent("figure"), Tag):
            continue
        eligible.append(node)
    return eligible


def _html_visible_body_events(
    root: Tag | BeautifulSoup,
) -> list[Tag | _HTMLFallback]:
    events: list[Tag | _HTMLFallback] = []

    def visit(container: Tag | BeautifulSoup) -> None:
        pending: list[Tag | NavigableString] = []

        def flush() -> None:
            if not _html_values_have_visible_content(pending):
                pending.clear()
                return
            locator_node = next(
                (value for value in pending if isinstance(value, Tag)),
                container,
            )
            events.append(
                _HTMLFallback(
                    locator_node=locator_node,
                    values=tuple(pending),
                )
            )
            pending.clear()

        for child in container.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                pending.append(child)
                continue
            if not isinstance(child, Tag):
                continue
            if child.name in _HTML_IGNORED_NAMES:
                flush()
                continue
            if _html_is_block_node(child):
                flush()
                events.append(child)
                continue
            if _html_contains_block_node(child):
                flush()
                visit(child)
                continue
            pending.append(child)
        flush()

    visit(root)
    return events


def _html_is_block_node(node: Tag) -> bool:
    if node.name == "math":
        return str(node.get("display") or "").casefold() == "block"
    return (node.name or "") in _HTML_BLOCK_NAMES


def _html_contains_block_node(node: Tag) -> bool:
    return any(
        isinstance(descendant, Tag)
        and (
            descendant.name in _HTML_IGNORED_NAMES
            or _html_is_block_node(descendant)
        )
        for descendant in node.descendants
    )


def _html_values_have_visible_content(
    values: list[Tag | NavigableString],
) -> bool:
    for value in values:
        if isinstance(value, NavigableString):
            if str(value).strip():
                return True
            continue
        if value.name in {"img", "math"}:
            if (
                value.get("src")
                or value.get("alt")
                or value.get("alttext")
                or value.get_text(" ", strip=True)
            ):
                return True
            continue
        if value.get_text(" ", strip=True):
            return True
    return False


def _parse_html(
    text: str,
    artifact: SourceArtifact,
    import_asset: AssetImporter,
) -> list[_RawBlock]:
    soup = BeautifulSoup(text, "html.parser")
    roots = html_roots(soup)
    if any(getattr(root, "name", "") == "article" for root in roots):
        events: list[Tag | _HTMLFallback] = _html_explicit_candidates(roots)
    else:
        events = _html_visible_body_events(roots[0])
    output: list[_RawBlock] = []
    pending_section_reset: int | None = None
    for ordinal, event in enumerate(events):
        node = event.locator_node if isinstance(event, _HTMLFallback) else event
        line_start, column_start, line_end, column_end = (
            html_source_position(node)
        )
        locator = SourceLocator(
            source_format=artifact.source_format,
            line_start=line_start,
            column_start=column_start,
            line_end=line_end,
            column_end=column_end,
            selector=rich_html_selector(node, ordinal),
            source_id=str(node.get("id") or ""),
        )
        event_output_start = len(output)
        if isinstance(event, _HTMLFallback):
            _append_html_fallback_blocks(
                output,
                locator=locator,
                values=event.values,
                import_asset=import_asset,
            )
            pending_section_reset = _apply_section_reset_to_first_new_block(
                output, event_output_start, pending_section_reset
            )
            continue
        if re.fullmatch(r"h[1-6]", node.name or ""):
            if html_heading_is_document_metadata(node):
                pending_section_reset = 1
                values = tuple(
                    value
                    for value in node.next_siblings
                    if isinstance(value, (Tag, NavigableString))
                )
                _append_html_fallback_blocks(
                    output,
                    locator=locator,
                    values=values,
                    import_asset=import_asset,
                )
                pending_section_reset = _apply_section_reset_to_first_new_block(
                    output, event_output_start, pending_section_reset
                )
                continue
            output.append(
                _RawBlock(
                    RichBlockKind.HEADING,
                    locator,
                    {"text": node.get_text(" ", strip=True), "level": int(node.name[1:])},
                )
            )
        elif node.name == "p":
            _append_html_paragraph_blocks(
                output,
                locator=locator,
                node=node,
                import_asset=import_asset,
            )
        elif node.name in {"ul", "ol"}:
            _append_html_list_blocks(
                output,
                locator=locator,
                node=node,
                import_asset=import_asset,
            )
        elif node.name == "pre":
            code = node.find("code")
            language = ""
            if isinstance(code, Tag):
                for class_name in code.get("class") or ():
                    if str(class_name).startswith("language-"):
                        language = str(class_name)[9:]
                        break
            output.append(
                _RawBlock(
                    RichBlockKind.CODE,
                    locator,
                    {
                        "text": (code or node).get_text("", strip=False),
                        "language": language,
                    },
                )
            )
        elif node.name == "table":
            if _html_is_equation_table(node):
                _append_html_equation_table_blocks(
                    output,
                    locator=locator,
                    node=node,
                    import_asset=import_asset,
                )
                pending_section_reset = _apply_section_reset_to_first_new_block(
                    output, event_output_start, pending_section_reset
                )
                continue
            caption = node.find("caption")
            rows = node.find_all("tr")
            header_cells = rows[0].find_all(["th", "td"]) if rows else []
            data_start = 1 if rows and rows[0].find("th") else 0
            _append_html_table_blocks(
                output,
                locator=locator,
                headers=header_cells if data_start else [],
                rows=[
                    row.find_all(["th", "td"])
                    for row in rows[data_start:]
                ],
                caption=(
                    caption.get_text(" ", strip=True)
                    if isinstance(caption, Tag)
                    else ""
                ),
                import_asset=import_asset,
            )
        elif node.name in {"figure", "img"}:
            image = node.find("img") if node.name == "figure" else node
            if not isinstance(image, Tag):
                continue
            target = str(image.get("src") or "")
            asset = import_asset(target)
            caption = node.find("figcaption") if node.name == "figure" else None
            output.append(
                _RawBlock(
                    RichBlockKind.FIGURE,
                    locator,
                    _figure_payload(
                        asset,
                        alt_text=str(image.get("alt") or ""),
                        caption=(
                            caption.get_text(" ", strip=True)
                            if isinstance(caption, Tag)
                            else ""
                        ),
                        target=target,
                    ),
                )
            )
        elif node.name == "math" and str(node.get("display") or "").casefold() == "block":
            tex = _html_math_tex(node)
            if tex:
                output.append(
                    _RawBlock(
                        RichBlockKind.EQUATION,
                        locator,
                        {
                            "tex": tex,
                            "display": True,
                            "label": _html_displayed_equation_label(node),
                        },
                    )
                )
        pending_section_reset = _apply_section_reset_to_first_new_block(
            output, event_output_start, pending_section_reset
        )
    return output


def _apply_section_reset_to_first_new_block(
    output: list[_RawBlock],
    start: int,
    reset_to_level: int | None,
) -> int | None:
    if reset_to_level is None or len(output) <= start:
        return reset_to_level
    block = output[start]
    output[start] = _RawBlock(
        kind=block.kind,
        locator=block.locator,
        payload=block.payload,
        section_reset_to_level=reset_to_level,
    )
    return None


def _append_html_paragraph_blocks(
    output: list[_RawBlock],
    *,
    locator: SourceLocator,
    node: Tag,
    import_asset: AssetImporter,
) -> None:
    _append_html_flow_blocks(
        output,
        locator=locator,
        segments=_html_inline_segments(node),
        import_asset=import_asset,
    )


def _append_html_fallback_blocks(
    output: list[_RawBlock],
    *,
    locator: SourceLocator,
    values: tuple[Tag | NavigableString, ...],
    import_asset: AssetImporter,
) -> None:
    _append_html_flow_blocks(
        output,
        locator=locator,
        segments=_html_inline_segments_from_values(values),
        import_asset=import_asset,
    )


def _append_html_flow_blocks(
    output: list[_RawBlock],
    *,
    locator: SourceLocator,
    segments: list[dict[str, Any] | Tag],
    import_asset: AssetImporter,
) -> None:
    for segment in segments:
        if isinstance(segment, Tag):
            embedded = _html_embedded_block(
                locator,
                segment,
                import_asset,
            )
            if embedded is not None:
                output.append(embedded)
        elif segment["text"]:
            output.append(
                _RawBlock(RichBlockKind.PARAGRAPH, locator, segment)
            )


def _append_html_list_blocks(
    output: list[_RawBlock],
    *,
    locator: SourceLocator,
    node: Tag,
    import_asset: AssetImporter,
) -> None:
    ordered = node.name == "ol"
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending:
            return
        output.append(
            _RawBlock(
                RichBlockKind.LIST,
                locator,
                {"ordered": ordered, "items": list(pending)},
            )
        )
        pending.clear()

    for item in node.find_all("li", recursive=False):
        for segment in _html_inline_segments(item):
            if isinstance(segment, Tag):
                flush()
                embedded = _html_embedded_block(
                    locator,
                    segment,
                    import_asset,
                )
                if embedded is not None:
                    output.append(embedded)
            elif segment["text"]:
                pending.append(segment)
    flush()


def _html_is_equation_table(node: Tag) -> bool:
    return any(
        "equation" in str(class_name).casefold()
        for class_name in node.get("class") or ()
    )


def _append_html_equation_table_blocks(
    output: list[_RawBlock],
    *,
    locator: SourceLocator,
    node: Tag,
    import_asset: AssetImporter,
) -> None:
    """Emit one block per visibly numbered equation, not per MathML fragment."""

    for unit in html_equation_table_units(node):
        tex = normalize_tex(
            " ".join(_html_math_tex(math) for math in unit.math_nodes)
        )
        if not tex:
            continue
        output.append(
            _RawBlock(
                RichBlockKind.EQUATION,
                locator,
                {"tex": tex, "display": True, "label": unit.label},
            )
        )


def _append_html_table_blocks(
    output: list[_RawBlock],
    *,
    locator: SourceLocator,
    headers: list[Tag],
    rows: list[list[Tag]],
    caption: str,
    import_asset: AssetImporter,
) -> None:
    shape = [headers, *rows]
    pending = [["" for _ in row] for row in shape]
    emitted_table = False
    emitted_block_math = False

    def flush(*, force: bool = False) -> None:
        nonlocal emitted_table
        if not force and not any(cell for row in pending for cell in row):
            return
        output.append(
            _RawBlock(
                RichBlockKind.TABLE,
                locator,
                {
                    "headers": list(pending[0]),
                    "rows": [list(row) for row in pending[1:]],
                    "caption": caption,
                },
            )
        )
        emitted_table = True
        for row in pending:
            for column in range(len(row)):
                row[column] = ""

    for row_index, row in enumerate(shape):
        for column, cell in enumerate(row):
            for segment in _html_inline_segments(cell):
                if isinstance(segment, Tag):
                    flush()
                    embedded = _html_embedded_block(
                        locator,
                        segment,
                        import_asset,
                    )
                    if embedded is not None:
                        output.append(embedded)
                        if segment.name == "math":
                            emitted_block_math = True
                elif segment["text"]:
                    pending[row_index][column] = " ".join(
                        value
                        for value in (
                            pending[row_index][column],
                            segment["text"],
                        )
                        if value
                    )
    flush(force=not emitted_table and not emitted_block_math)


def _html_embedded_block(
    locator: SourceLocator,
    node: Tag,
    import_asset: AssetImporter,
    *,
    equation_label: str = "",
) -> _RawBlock | None:
    if node.name == "math":
        tex = _html_math_tex(node)
        if not tex:
            return None
        return _RawBlock(
            RichBlockKind.EQUATION,
            locator,
            {
                "tex": tex,
                "display": True,
                "label": equation_label,
            },
        )
    target = str(node.get("src") or "")
    return _RawBlock(
        RichBlockKind.FIGURE,
        locator,
        _figure_payload(
            import_asset(target),
            alt_text=str(node.get("alt") or ""),
            caption="",
            target=target,
        ),
    )


def _html_inline_segments(
    node: Tag,
) -> list[dict[str, Any] | Tag]:
    return _html_inline_segments_from_values((node,))


def _html_inline_segments_from_values(
    values: tuple[Tag | NavigableString, ...],
) -> list[dict[str, Any] | Tag]:
    segments: list[dict[str, Any] | Tag] = []
    parts: list[dict[str, str]] = []

    def flush() -> None:
        payload = _inline_payload(parts)
        if payload["text"]:
            segments.append(payload)
        parts.clear()

    def visit(
        value: Tag | NavigableString,
        *,
        link_target: str | None = None,
    ) -> None:
        if isinstance(value, NavigableString):
            text = re.sub(r"\s+", " ", str(value))
            if link_target is None:
                _append_inline_part(parts, "text", text)
            else:
                _append_inline_part(
                    parts, "link", text, target=link_target
                )
            return
        if value.name == "img":
            flush()
            segments.append(value)
            return
        if value.name == "math":
            if (
                str(value.get("display") or "").casefold() == "block"
            ):
                flush()
                segments.append(value)
                return
            tex = _html_math_tex(value)
            if tex:
                source = value.get_text(" ", strip=True) or tex
                _append_inline_part(
                    parts, "math", source, tex=tex, source=source
                )
            return
        nested_link = (
            str(value.get("href") or "")
            if value.name == "a"
            else link_target
        )
        for child in value.children:
            if isinstance(child, (Tag, NavigableString)):
                visit(child, link_target=nested_link)

    for value in values:
        visit(value)
    flush()
    return segments


def _parse_tex(
    text: str,
    artifact: SourceArtifact,
    import_asset: AssetImporter,
) -> list[_RawBlock]:
    active = _tex_without_comments(text)
    if re.search(
        r"\\(?:input|include)(?![A-Za-z@])\s*(?:\{|[^\s])", active
    ):
        raise ParseError(
            "unsupported_tex_project",
            "rich TeX parsing accepts one pre-flattened file; input/include is unsupported",
            artifact=artifact,
        )
    lines = active.splitlines()
    output: list[_RawBlock] = []
    levels = {"section": 1, "subsection": 2, "subsubsection": 3}
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        heading = _tex_heading(lines, index, artifact, cursor=0)
        if heading:
            while heading is not None:
                start = index
                end, cursor, command, title = heading
                output.append(
                    _raw(
                        artifact,
                        RichBlockKind.HEADING,
                        start + 1,
                        end + 1,
                        {
                            "text": _tex_heading_text(title),
                            "level": levels[command],
                        },
                    )
                )
                index = end
                heading = _tex_heading(
                    lines, index, artifact, cursor=cursor
                )
            index += 1
            continue
        figure_environment = re.search(r"\\begin\{figure\*?\}", line)
        if figure_environment:
            start = index
            values = [line]
            while index + 1 < len(lines) and not re.search(
                r"\\end\{figure\*?\}", values[-1]
            ):
                index += 1
                values.append(lines[index])
            joined = "\n".join(values)
            image = re.search(
                r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", joined
            )
            target = image.group(1) if image else ""
            asset = import_asset(target) if target else None
            output.append(
                _raw(
                    artifact,
                    RichBlockKind.FIGURE,
                    start + 1,
                    index + 1,
                    _figure_payload(
                        asset,
                        alt_text="",
                        caption=_tex_caption(joined),
                        target=target,
                    ),
                )
            )
            index += 1
            continue
        environment = re.search(
            r"\\begin\{(equation|align|gather|multline|eqnarray)\*?\}", line
        )
        if environment:
            start = index
            values = [line]
            closing = re.compile(
                rf"\\end\{{{re.escape(environment.group(1))}\*?\}}"
            )
            while index + 1 < len(lines) and not closing.search(values[-1]):
                index += 1
                values.append(lines[index])
            if not closing.search(values[-1]):
                raise ParseError(
                    "unclosed_rich_block",
                    f"unclosed {environment.group(1)} environment",
                    artifact=artifact,
                )
            raw_tex = "\n".join(values)
            output.append(
                _raw(
                    artifact,
                    RichBlockKind.EQUATION,
                    start + 1,
                    index + 1,
                    {
                        "tex": normalize_tex(raw_tex),
                        "display": True,
                        "label": _tex_label(raw_tex),
                    },
                )
            )
            index += 1
            continue
        display = _markdown_display_equation(lines, index, artifact)
        if display is not None:
            end, raw_tex = display
            output.append(
                _raw(
                    artifact,
                    RichBlockKind.EQUATION,
                    index + 1,
                    end + 1,
                    {
                        "tex": normalize_tex(raw_tex),
                        "display": True,
                        "label": _tex_label(raw_tex),
                    },
                )
            )
            index = end + 1
            continue
        verbatim = re.search(r"\\begin\{(verbatim|lstlisting)\}", line)
        if verbatim:
            start = index
            values: list[str] = []
            closing = re.compile(rf"\\end\{{{re.escape(verbatim.group(1))}\}}")
            index += 1
            while index < len(lines) and not closing.search(lines[index]):
                values.append(lines[index])
                index += 1
            output.append(
                _raw(
                    artifact,
                    RichBlockKind.CODE,
                    start + 1,
                    min(index + 1, len(lines)),
                    {"text": "\n".join(values), "language": "tex"},
                )
            )
            index += 1
            continue
        list_environment = re.search(r"\\begin\{(itemize|enumerate)\}", line)
        if list_environment:
            start = index
            values: list[str] = []
            index += 1
            while index < len(lines) and not re.search(
                rf"\\end\{{{list_environment.group(1)}\}}", lines[index]
            ):
                values.append(lines[index])
                index += 1
            joined = "\n".join(values)
            items = [
                _tex_inline_payload(value.strip())
                for value in re.split(r"\\item(?:\[[^\]]*\])?", joined)
                if value.strip()
            ]
            output.append(
                _raw(
                    artifact,
                    RichBlockKind.LIST,
                    start + 1,
                    min(index + 1, len(lines)),
                    {
                        "ordered": list_environment.group(1) == "enumerate",
                        "items": items,
                    },
                )
            )
            index += 1
            continue
        tabular = re.search(r"\\begin\{tabular\}", line)
        if tabular:
            start = index
            values = [line]
            while index + 1 < len(lines) and r"\end{tabular}" not in values[-1]:
                index += 1
                values.append(lines[index])
            body = "\n".join(values)
            body = re.sub(r"\\begin\{tabular\}\{[^{}]*\}", "", body)
            body = body.replace(r"\end{tabular}", "")
            rows = [
                [_tex_plain_text(cell.strip()) for cell in row.split("&")]
                for row in re.split(r"\\\\", body)
                if _tex_plain_text(row.strip())
            ]
            output.append(
                _raw(
                    artifact,
                    RichBlockKind.TABLE,
                    start + 1,
                    index + 1,
                    {"headers": [], "rows": rows, "caption": ""},
                )
            )
            index += 1
            continue
        figure = re.search(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", line
        )
        if figure:
            target = figure.group(1)
            asset = import_asset(target)
            output.append(
                _raw(
                    artifact,
                    RichBlockKind.FIGURE,
                    index + 1,
                    index + 1,
                    _figure_payload(
                        asset,
                        alt_text="",
                        caption=_tex_caption(line),
                        target=target,
                    ),
                )
            )
            index += 1
            continue
        start = index
        values: list[str] = []
        while index < len(lines) and lines[index].strip():
            if index > start and _tex_starts_block(lines[index]):
                break
            values.append(lines[index].strip())
            index += 1
        raw_text = " ".join(values)
        output.append(
            _raw(
                artifact,
                RichBlockKind.PARAGRAPH,
                start + 1,
                index,
                _tex_inline_payload(raw_text),
            )
        )
    return output


def _tex_starts_block(value: str) -> bool:
    return bool(
        re.search(
            r"\\(?:sub)*section|\\begin\{(?:equation|align|gather|multline|eqnarray|verbatim|lstlisting|itemize|enumerate|tabular)\}|\\includegraphics",
            value,
        )
        or value.strip().startswith((r"\[", "$$"))
    )


def _tex_heading_text(value: str) -> str:
    return _tex_plain_text(_unwrap_texorpdfstring(value))


def _tex_plain_text(value: str) -> str:
    value = value.replace(r"\{", "\0OPEN_BRACE\0").replace(
        r"\}", "\0CLOSE_BRACE\0"
    )
    value = re.sub(r"\\(?:label|tag)\{[^{}]*\}", "", value)
    value = re.sub(
        r"\\(?:textbf|textit|emph|mathrm|mathbf|mathcal)\{([^{}]*)\}", r"\1", value
    )
    value = re.sub(r"\\(?:href|url)\{([^{}]*)\}(?:\{([^{}]*)\})?", lambda match: match.group(2) or match.group(1), value)
    value = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", value)
    value = value.replace("{", "").replace("}", "")
    value = value.replace("\0OPEN_BRACE\0", "{").replace(
        "\0CLOSE_BRACE\0", "}"
    )
    return " ".join(value.split())


def _tex_inline_payload(value: str) -> dict[str, Any]:
    token = re.compile(
        r"(?P<href>\\href\{([^{}]+)\}\{([^{}]+)\})"
        r"|(?P<url>\\url\{([^{}]+)\})"
        r"|(?P<dollar>(?<!\\)(?<!\$)\$(?!\$)(.+?)(?<!\\)\$(?!\$))"
        r"|(?P<paren>\\\((.+?)\\\))"
    )
    parts: list[dict[str, str]] = []
    cursor = 0
    for match in token.finditer(value):
        _append_inline_part(parts, "text", _tex_text_segment(value[cursor : match.start()]))
        if match.lastgroup == "href":
            _append_inline_part(
                parts,
                "link",
                _tex_plain_text(match.group(3)),
                target=match.group(2),
            )
        elif match.lastgroup == "url":
            _append_inline_part(
                parts,
                "link",
                match.group(5),
                target=match.group(5),
            )
        else:
            source = match.group(0)
            tex = normalize_tex(source)
            if tex:
                _append_inline_part(
                    parts, "math", source, tex=tex, source=source
                )
            else:
                _append_inline_part(parts, "text", source)
        cursor = match.end()
    _append_inline_part(parts, "text", _tex_text_segment(value[cursor:]))
    return _inline_payload(parts)


def _tex_text_segment(value: str) -> str:
    leading = bool(re.match(r"\s", value))
    trailing = bool(re.search(r"\s$", value))
    text = _tex_plain_text(value)
    if not text:
        return " " if leading or trailing else ""
    return (" " if leading else "") + text + (" " if trailing else "")


def _tex_caption(value: str) -> str:
    match = re.search(r"\\caption\{([^{}]*)\}", value)
    return _tex_plain_text(match.group(1)) if match else ""


def _tex_label(value: str) -> str:
    match = re.search(r"\\(?:label|tag)\s*\{([^{}]+)\}", value)
    return match.group(1) if match else ""


def _append_inline_part(
    parts: list[dict[str, str]],
    kind: str,
    text: str,
    **metadata: str,
) -> None:
    if not text:
        return
    item = {"kind": kind, "text": text, **metadata}
    if kind == "text" and parts and parts[-1]["kind"] == "text":
        parts[-1]["text"] += text
    else:
        parts.append(item)


def _inline_payload(parts: list[dict[str, str]]) -> dict[str, Any]:
    normalized: list[dict[str, str]] = []
    for part in parts:
        item = dict(part)
        item["text"] = re.sub(r"\s+", " ", item["text"])
        if not item["text"]:
            continue
        if normalized and normalized[-1]["text"].endswith(" ") and item["text"].startswith(" "):
            item["text"] = item["text"][1:]
        if item["text"]:
            normalized.append(item)
    if normalized:
        normalized[0]["text"] = normalized[0]["text"].lstrip()
        normalized[-1]["text"] = normalized[-1]["text"].rstrip()
        normalized = [item for item in normalized if item["text"]]
    text = "".join(item["text"] for item in normalized)
    spans: list[dict[str, Any]] = []
    cursor = 0
    for part in normalized:
        end = cursor + len(part["text"])
        span: dict[str, Any] = {
            "kind": part["kind"],
            "start": cursor,
            "end": end,
            "text": part["text"],
        }
        if part["kind"] == "link":
            span["target"] = part["target"]
        elif part["kind"] == "math":
            span["tex"] = part["tex"]
            span["source"] = part["source"]
        spans.append(span)
        cursor = end
    return {
        "text": text,
        "inline_spans": spans,
    }


def _figure_payload(
    asset: RichAsset | None,
    *,
    alt_text: str,
    caption: str,
    target: str,
) -> dict[str, Any]:
    return {
        "asset_digest": asset.artifact_digest if asset else "",
        "alt_text": alt_text,
        "caption": caption,
        "target": target,
        "media_type": asset.media_type if asset else "",
        "logical_name": asset.logical_name if asset else target,
        "size": asset.size if asset else 0,
    }


def _raw(
    artifact: SourceArtifact,
    kind: RichBlockKind,
    line_start: int,
    line_end: int,
    payload: Mapping[str, Any],
) -> _RawBlock:
    return _RawBlock(
        kind=kind,
        locator=SourceLocator(
            source_format=artifact.source_format,
            line_start=line_start,
            column_start=None,
            line_end=line_end,
            column_end=None,
        ),
        payload=payload,
    )


def _finalize_document(
    artifact: SourceArtifact,
    raw_blocks: list[_RawBlock],
    *,
    assets: tuple[RichAsset, ...],
    metadata: Mapping[str, Any],
) -> RichDocument:
    section_specs: list[dict[str, Any]] = []
    paths: list[tuple[str, ...]] = []
    stack: list[tuple[int, str]] = []
    synthetic_id = "sec-" + hashlib.sha256(
        json_bytes(
            {
                "source": artifact.content_identity,
                "role": "synthetic-document-section",
            }
        )
    ).hexdigest()[:20]
    for ordinal, raw in enumerate(raw_blocks):
        if raw.section_reset_to_level is not None:
            while stack and stack[-1][0] > raw.section_reset_to_level:
                _level, closed_section_id = stack.pop()
                for spec in reversed(section_specs):
                    if spec["section_id"] == closed_section_id:
                        spec["block_end"] = ordinal
                        break
        if raw.kind is RichBlockKind.HEADING:
            level = int(raw.payload["level"])
            title = str(raw.payload["text"])
            material = {
                "source": artifact.content_identity,
                "ordinal": ordinal,
                "level": level,
                "title": title,
            }
            section_id = (
                f"sec-{hashlib.sha256(json_bytes(material)).hexdigest()[:20]}"
            )
            while stack and stack[-1][0] >= level:
                stack.pop()
            path = tuple(item[1] for item in stack) + (section_id,)
            stack.append((level, section_id))
            section_specs.append(
                {
                    "section_id": section_id,
                    "title": title,
                    "level": level,
                    "path": path,
                    "block_start": ordinal,
                }
            )
        elif not stack:
            if not section_specs or section_specs[-1]["section_id"] != synthetic_id:
                section_specs.append(
                    {
                        "section_id": synthetic_id,
                        "title": "Document",
                        "level": 1,
                        "path": (synthetic_id,),
                        "block_start": ordinal,
                    }
                )
            stack = [(1, synthetic_id)]
        paths.append(tuple(item[1] for item in stack))
    blocks: list[RichBlock] = []
    for ordinal, (raw, section_path) in enumerate(zip(raw_blocks, paths, strict=True)):
        material = {
            "source": artifact.content_identity,
            "ordinal": ordinal,
            "kind": raw.kind.value,
            "locator": {
                "line_start": raw.locator.line_start,
                "line_end": raw.locator.line_end,
                "selector": raw.locator.selector,
                "source_id": raw.locator.source_id,
            },
            "payload": raw.payload,
        }
        block_id = "block-" + hashlib.sha256(
            json_bytes(material)
        ).hexdigest()[:24]
        blocks.append(
            RichBlock(
                block_id=block_id,
                ordinal=ordinal,
                kind=raw.kind,
                section_path=section_path,
                locator=raw.locator,
                payload=raw.payload,
            )
        )
    sections: list[RichSection] = []
    for ordinal, spec in enumerate(section_specs):
        following = [
            int(other["block_start"])
            for other in section_specs[ordinal + 1 :]
            if len(other["path"]) <= len(spec["path"])
        ]
        block_end = int(
            spec.get(
                "block_end",
                min(following) if following else len(blocks),
            )
        )
        sections.append(
            RichSection(
                section_id=str(spec["section_id"]),
                title=str(spec["title"]),
                level=int(spec["level"]),
                ordinal=ordinal,
                path=tuple(spec["path"]),
                block_start=int(spec["block_start"]),
                block_end=block_end,
            )
        )
    if not sections and blocks:
        sections.append(
            RichSection(
                section_id=synthetic_id,
                title="Document",
                level=1,
                ordinal=0,
                path=(synthetic_id,),
                block_start=0,
                block_end=len(blocks),
            )
        )
    return RichDocument(
        source=artifact,
        blocks=tuple(blocks),
        sections=tuple(sections),
        assets=assets,
        metadata=metadata,
    )


def json_bytes(value: Any) -> bytes:
    import json

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _is_local_asset_target(value: str) -> bool:
    parsed = urlparse(value)
    return bool(value) and not parsed.scheme and not parsed.netloc and not value.startswith(("#", "/"))


def resolve_local_asset_path(source_path: str | Path, target: str) -> Path | None:
    """Resolve a source-relative asset target without interpreting remote URLs."""

    if not _is_local_asset_target(target):
        return None
    clean_target = target.split("#", 1)[0].split("?", 1)[0]
    return Path(source_path).parent / clean_target


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in values if item))


__all__ = [
    "AssetImporter",
    "RichSourceParseResult",
    "SOURCE_PAGE_BOUNDARIES_METADATA_KEY",
    "SOURCE_PAGE_BOUNDARIES_SCHEMA",
    "parse_rich_artifact_bytes",
    "resolve_local_asset_path",
]
