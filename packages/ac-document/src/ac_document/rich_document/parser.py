from __future__ import annotations

import hashlib
import mimetypes
import re
from math import gcd
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from .._parsing import ParseError, normalize_tex
from .._parsing.html_source import (
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
    RichListPathEntry,
    RichPageMapEntry,
    RichSection,
    SourceLocator,
)
from .source_targets import (
    SOURCE_TARGET_MANIFEST_METADATA_KEY,
    SOURCE_TARGET_MANIFEST_SCHEMA,
)
from .source_fidelity import (
    SOURCE_FRONT_MATTER_METADATA_KEY,
    SOURCE_FRONT_MATTER_SCHEMA,
    SOURCE_NOTES_METADATA_KEY,
    SOURCE_NOTES_SCHEMA,
)
from .source_presentation import (
    SOURCE_PRESENTATION_METADATA_KEY,
    SOURCE_PRESENTATION_SCHEMA,
)


AssetImporter = Callable[[str], RichAsset | None]
WarningEmitter = Callable[[str], None]
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
SOURCE_PAGE_BOUNDARIES_SCHEMA = "ac.document.source_page_boundaries.v1"
SOURCE_PAGE_BOUNDARIES_METADATA_KEY = "source_page_boundaries"
DOCUMENT_NOTES_SCHEMA = "ac.document.document_notes.v1"
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
    target_panels: tuple[Mapping[str, Any], ...] = ()
    list_path: tuple[RichListPathEntry, ...] = ()
    notes: tuple[_HTMLNoteSpec, ...] = ()
    presentation_roles: tuple[str, ...] = ()
    presentation_fields: tuple[Mapping[str, Any], ...] = ()
    figure_presentation: Mapping[str, Any] | None = None
    table_presentation: Mapping[str, Any] | None = None
    caption_presentation: Mapping[str, Any] | None = None
    outline_heading: bool = True


@dataclass(frozen=True)
class _HTMLNoteSpec:
    note_id: str
    locator: SourceLocator
    marker: str
    body_payload: Mapping[str, Any]
    owner_locator: SourceLocator
    anchor: Mapping[str, Any]


@dataclass
class _HTMLListOwner:
    container_id: str
    container_source_id: str
    container_selector: str
    item_id: str
    item_source_id: str
    item_selector: str
    item_index: int
    item_count: int
    depth: int
    ordered: bool
    next_segment_index: int = 0


@dataclass(frozen=True)
class _HTMLSectionTarget:
    alias: str
    selector: str
    block_start: int
    block_end: int


@dataclass(frozen=True)
class _HTMLBlockTarget:
    alias: str
    selector: str
    block_index: int


@dataclass(frozen=True)
class _HTMLClassificationFlow:
    classification_id: str
    locator: SourceLocator
    heading_block_index: int
    value_block_indexes: tuple[int, ...]


@dataclass(frozen=True)
class _HTMLParseResult:
    blocks: tuple[_RawBlock, ...]
    sections: tuple[_HTMLSectionTarget, ...]
    block_targets: tuple[_HTMLBlockTarget, ...]
    front_matter: tuple[Mapping[str, Any], ...]
    classifications: tuple[_HTMLClassificationFlow, ...]


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
_HTML_TABLE_SPAN_LIMIT = 4096
_HTML_TABLE_COVERAGE_LIMIT = 65_536


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

    html_sections: tuple[_HTMLSectionTarget, ...] = ()
    html_block_targets: tuple[_HTMLBlockTarget, ...] = ()
    html_front_matter: tuple[Mapping[str, Any], ...] = ()
    html_classifications: tuple[_HTMLClassificationFlow, ...] = ()
    if artifact.source_format is SourceFormat.MARKDOWN:
        raw = _parse_markdown(text, artifact, import_asset)
    elif artifact.source_format is SourceFormat.HTML:
        html_result = _parse_html(
            text,
            artifact,
            import_asset,
            warnings.append,
        )
        raw = list(html_result.blocks)
        html_sections = html_result.sections
        html_block_targets = html_result.block_targets
        html_front_matter = html_result.front_matter
        html_classifications = html_result.classifications
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
        html_sections=html_sections,
        html_block_targets=html_block_targets,
        html_front_matter=html_front_matter,
        html_classifications=html_classifications,
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


def _html_locator(
    node: Tag,
    source_format: SourceFormat,
    *,
    ordinal: int | None = None,
) -> SourceLocator:
    line_start, column_start, line_end, column_end = html_source_position(node)
    source_id = str(node.get("id") or "")
    if ordinal is not None:
        selector = rich_html_selector(node, ordinal)
    elif source_id:
        selector = f"#{source_id}"
    else:
        selector = ""
    return SourceLocator(
        source_format=source_format,
        line_start=line_start,
        column_start=column_start,
        line_end=line_end,
        column_end=column_end,
        selector=selector,
        source_id=source_id,
    )


def _html_authors_node(
    values: tuple[Tag | NavigableString, ...],
) -> Tag | None:
    authors = [
        value
        for value in values
        if isinstance(value, Tag)
        and "ltx_authors" in {
            str(class_name).casefold()
            for class_name in value.get("class") or ()
        }
    ]
    if len(authors) != 1:
        return None
    if any(
        (isinstance(value, NavigableString) and str(value).strip())
        or (
            isinstance(value, Tag)
            and value is not authors[0]
            and not _html_is_pubnotes(value)
        )
        for value in values
    ):
        return None
    return authors[0]


def _html_is_pubnotes(node: Tag) -> bool:
    return "ltx_pubnotes" in {
        str(class_name).casefold() for class_name in node.get("class") or ()
    }


def _html_front_matter_entry(
    node: Tag,
    *,
    artifact: SourceArtifact,
    block_index: int,
) -> Mapping[str, Any] | None:
    creators = [
        child
        for child in node.children
        if isinstance(child, Tag)
        and "ltx_creator"
        in {str(value).casefold() for value in child.get("class") or ()}
    ]
    if not creators or any(
        "ltx_role_author"
        not in {str(value).casefold() for value in creator.get("class") or ()}
        for creator in creators
    ):
        return None
    authors = []
    affiliations = []
    affiliation_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    creator_flow = []
    for creator_ordinal, creator in enumerate(creators):
        people = [
            person
            for person in creator.find_all(
                class_=lambda value: value
                and "ltx_personname" in str(value).casefold().split()
            )
            if isinstance(person, Tag)
            and person.find_parent(
                class_=lambda value: value
                and "ltx_creator" in str(value).casefold().split()
            )
            is creator
        ]
        if len(people) != 1:
            return None
        person = people[0]
        markers = [
            _html_visible_text(marker)
            for marker in person.find_all("sup", recursive=False)
            if _html_visible_text(marker)
        ]
        name = _html_text_excluding(
            person,
            lambda value: value.name == "sup"
            or "ltx_orcid" in {
                str(class_name).casefold()
                for class_name in value.get("class") or ()
            },
        )
        if not name:
            return None
        orcid_links = [
            link
            for link in creator.find_all(
                "a",
                class_=lambda value: value
                and "ltx_orcid" in str(value).casefold().split(),
            )
            if link.find_parent(
                class_=lambda value: value
                and "ltx_creator" in str(value).casefold().split()
            )
            is creator
        ]
        if len(orcid_links) > 1:
            return None
        orcid_link = orcid_links[0] if orcid_links else None
        orcid_url = (
            str(orcid_link.get("href") or "")
            if isinstance(orcid_link, Tag)
            else ""
        )
        orcid = (
            orcid_url.removeprefix("https://orcid.org/")
            if orcid_url.startswith("https://orcid.org/")
            else ""
        )
        contact_nodes = [
            contact
            for contact in creator.find_all(
                class_=lambda value: value
                and "ltx_contact" in str(value).casefold().split()
            )
            if isinstance(contact, Tag)
            and contact.find_parent(
                class_=lambda value: value
                and "ltx_creator" in str(value).casefold().split()
            )
            is creator
            and "ltx_role_orcid"
            not in {
                str(class_name).casefold()
                for class_name in contact.get("class") or ()
            }
        ]
        contacts = []
        contact_indexes: dict[int, int] = {}
        affiliation_references: dict[int, str] = {}
        for contact in contact_nodes:
            classes = {
                str(class_name).casefold()
                for class_name in contact.get("class") or ()
            }
            label_node = contact.find(
                class_=lambda value: value
                and "ltx_contact_name" in str(value).casefold().split()
            )
            label = (
                _html_visible_text(label_node)
                if isinstance(label_node, Tag)
                else ""
            )
            if "ltx_role_affiliation" in classes:
                marker_node = contact.find("sup")
                marker = (
                    _html_visible_text(marker_node)
                    if isinstance(marker_node, Tag)
                    else ""
                )
                text = _html_text_excluding(
                    contact,
                    lambda value: value is label_node
                    or value is marker_node
                    or value.name == "br",
                )
                key = (marker, text)
                if not text:
                    return None
                affiliation = affiliation_by_key.get(key)
                if affiliation is None:
                    source_id = str(contact.get("id") or "")
                    affiliation = {
                        "affiliation_id": _html_structural_identity(
                            artifact,
                            contact,
                            role="front-affiliation",
                        ),
                        "source_id": source_id,
                        "selector": f"#{source_id}" if source_id else "",
                        "marker": marker,
                        "text": text,
                    }
                    affiliation_by_key[key] = affiliation
                    affiliations.append(affiliation)
                affiliation_references[id(contact)] = str(
                    affiliation["affiliation_id"]
                )
                continue
            role = next(
                (
                    class_name.removeprefix("ltx_role_")
                    for class_name in sorted(classes)
                    if class_name.startswith("ltx_role_")
                ),
                "contact",
            )
            link = contact.find("a", href=True)
            contact_value = _html_text_excluding(
                contact,
                lambda value: value is label_node,
            )
            if not contact_value:
                continue
            contact_indexes[id(contact)] = len(contacts)
            contacts.append(
                {
                    "kind": role,
                    "label": label,
                    "value": contact_value,
                    "target": (
                        str(link.get("href") or "")
                        if isinstance(link, Tag)
                        else ""
                    ),
                }
            )
        source_id = str(creator.get("id") or "")
        author_id = _html_structural_identity(
            artifact,
            creator,
            role="front-author",
        )
        authors.append(
            {
                "author_id": author_id,
                "source_id": source_id,
                "selector": f"#{source_id}" if source_id else "",
                "name": name,
                "markers": markers,
                "orcid": orcid,
                "orcid_url": orcid_url,
                "contacts": contacts,
            }
        )
        slot_nodes = sorted(
            [
                person,
                *[
                    contact
                    for contact in contact_nodes
                    if id(contact) in contact_indexes
                    or id(contact) in affiliation_references
                ],
            ],
            key=_html_tag_path,
        )
        slots = []
        for slot_ordinal, slot_node in enumerate(slot_nodes):
            if slot_node is person:
                kind = "author"
                contact_index = None
                affiliation_id = ""
            elif id(slot_node) in contact_indexes:
                kind = "contact"
                contact_index = contact_indexes[id(slot_node)]
                affiliation_id = ""
            elif id(slot_node) in affiliation_references:
                kind = "affiliation"
                contact_index = None
                affiliation_id = affiliation_references[id(slot_node)]
            else:
                return None
            slots.append(
                {
                    "slot_id": _html_structural_identity(
                        artifact,
                        slot_node,
                        role="front-creator-slot",
                    ),
                    "ordinal": slot_ordinal,
                    "kind": kind,
                    "locator": _locator_to_document(
                        _html_locator(slot_node, artifact.source_format)
                    ),
                    "contact_index": contact_index,
                    "affiliation_id": affiliation_id,
                }
            )
        if not slots or slots[0]["kind"] != "author":
            return None
        creator_flow.append(
            {
                "creator_id": _html_structural_identity(
                    artifact,
                    creator,
                    role="front-creator",
                ),
                "ordinal": creator_ordinal,
                "locator": _locator_to_document(
                    _html_locator(creator, artifact.source_format)
                ),
                "author_id": author_id,
                "slots": slots,
            }
        )
    return {
        "front_matter_id": _html_structural_identity(
            artifact,
            node,
            role="front-matter",
        ),
        "kind": "authors",
        "block_index": block_index,
        "locator": _locator_to_document(
            _html_locator(node, artifact.source_format)
        ),
        "authors": authors,
        "affiliations": affiliations,
        "creator_flow": {
            "creator_count": len(creator_flow),
            "slot_count": sum(
                len(creator["slots"]) for creator in creator_flow
            ),
            "creators": creator_flow,
        },
    }


def _html_text_excluding(
    node: Tag,
    exclude: Callable[[Tag], bool],
) -> str:
    values: list[str] = []

    def visit(value: Tag | NavigableString) -> None:
        if isinstance(value, Comment):
            return
        if isinstance(value, NavigableString):
            values.append(str(value))
            return
        if exclude(value):
            return
        if value.name == "math":
            values.append(_html_visible_math_text(value))
            return
        if value.name in {"annotation", "annotation-xml"}:
            return
        for child in value.children:
            if isinstance(child, (Tag, NavigableString)):
                visit(child)

    visit(node)
    return re.sub(r"\s+", " ", "".join(values)).strip()


def _html_event_is_documented_exclusion(node: Tag) -> bool:
    if not _html_values_have_visible_content([node]):
        return True
    if node.name in {"table", "figure", "math"}:
        return True
    if node.name in {"ul", "ol"} and not node.find("li", recursive=False):
        return True
    return False


def _parse_html(
    text: str,
    artifact: SourceArtifact,
    import_asset: AssetImporter,
    warn: WarningEmitter,
) -> _HTMLParseResult:
    soup = BeautifulSoup(text, "html.parser")
    roots = html_roots(soup)
    events: list[Tag | _HTMLFallback] = []
    for root in roots:
        events.extend(_html_visible_body_events(root))
    output: list[_RawBlock] = []
    section_ranges: dict[int, list[int]] = {}
    classification_ranges: dict[int, list[int]] = {}
    front_matter: list[Mapping[str, Any]] = []
    for ordinal, event in enumerate(events):
        node = event.locator_node if isinstance(event, _HTMLFallback) else event
        locator = _html_locator(
            node,
            artifact.source_format,
            ordinal=ordinal,
        )
        event_output_start = len(output)

        def finish_event() -> None:
            if len(output) <= event_output_start:
                return
            classification = _html_classification_ancestor(node)
            if isinstance(classification, Tag):
                key = id(classification)
                if key not in classification_ranges:
                    classification_ranges[key] = [
                        event_output_start,
                        len(output),
                    ]
                else:
                    classification_ranges[key][0] = min(
                        classification_ranges[key][0],
                        event_output_start,
                    )
                    classification_ranges[key][1] = max(
                        classification_ranges[key][1],
                        len(output),
                    )
            section = node.find_parent("section")
            while isinstance(section, Tag):
                key = id(section)
                if key not in section_ranges:
                    section_ranges[key] = [event_output_start, len(output)]
                else:
                    section_ranges[key][0] = min(
                        section_ranges[key][0], event_output_start
                    )
                    section_ranges[key][1] = max(
                        section_ranges[key][1], len(output)
                    )
                section = section.find_parent("section")

        def require_covered(*, structured: bool = False) -> None:
            if (
                structured
                or len(output) > event_output_start
                or _html_event_is_documented_exclusion(node)
            ):
                return
            raise ParseError(
                "uncovered_html_flow",
                "visible HTML flow was neither emitted nor explicitly excluded",
                artifact=artifact,
            )

        if isinstance(event, _HTMLFallback):
            authors = _html_authors_node(event.values)
            if authors is not None:
                entry = _html_front_matter_entry(
                    authors,
                    artifact=artifact,
                    block_index=len(output),
                )
                if entry is None:
                    raise ParseError(
                        "uncovered_html_flow",
                        "uncovered_html_flow: structured HTML authors could not be represented",
                        artifact=artifact,
                    )
                front_matter.append(entry)
                finish_event()
                require_covered(structured=True)
                continue
            _append_html_fallback_blocks(
                output,
                locator=locator,
                values=event.values,
                import_asset=import_asset,
            )
            finish_event()
            require_covered()
            continue
        if re.fullmatch(r"h[1-6]", node.name or ""):
            segment = _html_single_inline_segment(node)
            if segment.get("_notes"):
                raise ValueError(
                    "source notes in HTML headings cannot be represented"
                )
            level, roles = _html_heading_semantics(node)
            output.append(
                _RawBlock(
                    RichBlockKind.HEADING,
                    locator,
                    {"text": segment["text"], "level": level},
                    presentation_roles=roles,
                    presentation_fields=(
                        _html_presentation_field(segment, field="text"),
                    ),
                    outline_heading="classification" not in roles,
                    section_reset_to_level=(
                        1 if "classification" in roles else None
                    ),
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
                node=node,
                artifact=artifact,
                import_asset=import_asset,
                warn=warn,
            )
        elif node.name == "pre":
            output.append(_html_code_block(locator, node))
        elif node.name == "table":
            if _html_is_equation_table(node):
                _append_html_equation_table_blocks(
                    output,
                    locator=locator,
                    node=node,
                    import_asset=import_asset,
                )
                finish_event()
                require_covered()
                continue
            _append_html_table_node_blocks(
                output,
                locator=locator,
                node=node,
                caption_node=node.find("caption", recursive=False),
                import_asset=import_asset,
            )
        elif _html_is_latexml_table_figure(node):
            tables = _html_descendant_tables(node)
            if len(tables) == 1:
                _append_html_table_node_blocks(
                    output,
                    locator=locator,
                    node=tables[0],
                    caption_node=_html_owned_caption(node, "figcaption"),
                    import_asset=import_asset,
                )
        elif node.name in {"figure", "img"}:
            figure = _html_figure_block(
                locator=locator,
                node=node,
                import_asset=import_asset,
                warn=warn,
            )
            if figure is not None:
                output.append(figure)
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
        finish_event()
        require_covered()
    sections = []
    for section in soup.find_all("section"):
        if not isinstance(section, Tag) or id(section) not in section_ranges:
            continue
        alias = str(section.get("id") or "")
        if not alias:
            continue
        block_start, block_end = section_ranges[id(section)]
        sections.append(
            _HTMLSectionTarget(
                alias=alias,
                selector=f"#{alias}",
                block_start=block_start,
                block_end=block_end,
            )
        )
    classifications = []
    for container in soup.find_all(
        class_=lambda value: value
        and "ltx_classification" in str(value).casefold().split()
    ):
        if not isinstance(container, Tag) or id(container) not in classification_ranges:
            continue
        if _html_classification_ancestor(container) is not container:
            continue
        if any(
            isinstance(descendant, Tag)
            and descendant is not container
            and "ltx_classification"
            in {
                str(class_name).casefold()
                for class_name in descendant.get("class") or ()
            }
            for descendant in container.find_all(True)
        ):
            continue
        semantic_titles = [
            child
            for child in container.find_all(re.compile(r"h[1-6]"), recursive=False)
            if isinstance(child, Tag)
            and {"ltx_title", "ltx_title_classification"}
            <= {
                str(class_name).casefold()
                for class_name in child.get("class") or ()
            }
        ]
        block_start, block_end = classification_ranges[id(container)]
        heading_indexes = [
            index
            for index in range(block_start, block_end)
            if output[index].kind is RichBlockKind.HEADING
        ]
        value_indexes = tuple(
            index
            for index in range(block_start, block_end)
            if output[index].kind is not RichBlockKind.HEADING
        )
        if (
            len(semantic_titles) != 1
            or heading_indexes != [block_start]
            or not value_indexes
        ):
            continue
        classifications.append(
            _HTMLClassificationFlow(
                classification_id=_html_structural_identity(
                    artifact,
                    container,
                    role="classification",
                ),
                locator=_html_locator(container, artifact.source_format),
                heading_block_index=block_start,
                value_block_indexes=value_indexes,
            )
        )
    raw_indexes_by_source_id: dict[str, list[int]] = {}
    for index, block in enumerate(output):
        if block.locator.source_id:
            raw_indexes_by_source_id.setdefault(
                block.locator.source_id, []
            ).append(index)
    referenced_aliases = {
        unquote(str(link.get("href") or "")[1:])
        for link in soup.find_all("a", href=True)
        if str(link.get("href") or "").startswith("#")
        and str(link.get("href") or "")[1:]
    }
    block_targets = []
    for alias in sorted(referenced_aliases):
        target = soup.find(id=alias)
        if not isinstance(target, Tag) or _html_is_source_note(target):
            continue
        owner_indexes: set[int] = set()
        owner: Tag | None = target
        while isinstance(owner, Tag):
            source_id = str(owner.get("id") or "")
            indexes = raw_indexes_by_source_id.get(source_id, ())
            if len(indexes) == 1:
                owner_indexes.add(indexes[0])
            owner = owner.parent if isinstance(owner.parent, Tag) else None
        if len(owner_indexes) != 1:
            continue
        block_index = next(iter(owner_indexes))
        if output[block_index].locator.source_id == alias:
            continue
        block_targets.append(
            _HTMLBlockTarget(alias, f"#{alias}", block_index)
        )
    return _HTMLParseResult(
        tuple(output),
        tuple(sections),
        tuple(block_targets),
        tuple(front_matter),
        tuple(classifications),
    )


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
            payload, notes = _html_segment_payload(segment)
            notes = _html_anchor_notes(notes, field="text")
            output.append(
                _RawBlock(
                    RichBlockKind.PARAGRAPH,
                    locator,
                    payload,
                    notes=notes,
                    presentation_fields=(
                        _html_presentation_field(segment, field="text"),
                    ),
                )
            )


def _append_html_list_blocks(
    output: list[_RawBlock],
    *,
    node: Tag,
    artifact: SourceArtifact,
    import_asset: AssetImporter,
    warn: WarningEmitter,
    parent_path: tuple[_HTMLListOwner, ...] = (),
) -> None:
    ordered = node.name == "ol"
    items = [
        item
        for item in node.find_all("li", recursive=False)
        if isinstance(item, Tag)
    ]
    container_source_id = str(node.get("id") or "")
    container_id = _html_list_identity(artifact, node, role="container")
    for item_index, item in enumerate(items):
        item_source_id = str(item.get("id") or "")
        owner = _HTMLListOwner(
            container_id=container_id,
            container_source_id=container_source_id,
            container_selector=(
                f"#{container_source_id}" if container_source_id else ""
            ),
            item_id=_html_list_identity(artifact, item, role="item"),
            item_source_id=item_source_id,
            item_selector=f"#{item_source_id}" if item_source_id else "",
            item_index=item_index,
            item_count=len(items),
            depth=len(parent_path),
            ordered=ordered,
        )
        _append_html_list_item_content(
            output,
            item,
            artifact=artifact,
            import_asset=import_asset,
            warn=warn,
            path=parent_path + (owner,),
        )
        if owner.next_segment_index == 0:
            _append_html_owned_block(
                output,
                _RawBlock(
                    RichBlockKind.LIST,
                    _html_locator(item, artifact.source_format),
                    {
                        "ordered": ordered,
                        "items": [{"text": "", "inline_spans": []}],
                    },
                    presentation_fields=(
                        _html_presentation_field(
                            _inline_payload([]),
                            field="list_item",
                            item_index=0,
                        ),
                    ),
                ),
                parent_path + (owner,),
            )


def _append_html_list_item_content(
    output: list[_RawBlock],
    node: Tag,
    *,
    artifact: SourceArtifact,
    import_asset: AssetImporter,
    warn: WarningEmitter,
    path: tuple[_HTMLListOwner, ...],
) -> None:
    pending: list[Tag | NavigableString] = []

    def flush() -> None:
        if not pending:
            return
        _append_html_list_inline_values(
            output,
            tuple(pending),
            locator=_html_locator(node, artifact.source_format),
            import_asset=import_asset,
            path=path,
        )
        pending.clear()

    for child in node.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            pending.append(child)
            continue
        if not isinstance(child, Tag):
            continue
        if _html_is_latexml_list_marker(child):
            continue
        if _html_list_is_block_node(child):
            flush()
            _append_html_list_block_node(
                output,
                child,
                artifact=artifact,
                import_asset=import_asset,
                warn=warn,
                path=path,
            )
            continue
        if _html_list_contains_block_node(child):
            flush()
            _append_html_list_item_content(
                output,
                child,
                artifact=artifact,
                import_asset=import_asset,
                warn=warn,
                path=path,
            )
            continue
        pending.append(child)
    flush()


def _append_html_list_block_node(
    output: list[_RawBlock],
    node: Tag,
    *,
    artifact: SourceArtifact,
    import_asset: AssetImporter,
    warn: WarningEmitter,
    path: tuple[_HTMLListOwner, ...],
) -> None:
    locator = _html_locator(node, artifact.source_format)
    if node.name == "p":
        _append_html_list_inline_values(
            output,
            (node,),
            locator=locator,
            import_asset=import_asset,
            path=path,
        )
        return
    if node.name in {"ul", "ol"}:
        _append_html_list_blocks(
            output,
            node=node,
            artifact=artifact,
            import_asset=import_asset,
            warn=warn,
            parent_path=path,
        )
        return
    start = len(output)
    if node.name == "pre":
        output.append(_html_code_block(locator, node))
    elif node.name == "table":
        if _html_is_equation_table(node):
            _append_html_equation_table_blocks(
                output,
                locator=locator,
                node=node,
                import_asset=import_asset,
            )
        else:
            _append_html_table_node_blocks(
                output,
                locator=locator,
                node=node,
                caption_node=node.find("caption", recursive=False),
                import_asset=import_asset,
            )
    elif _html_is_latexml_table_figure(node):
        tables = _html_descendant_tables(node)
        if len(tables) == 1:
            _append_html_table_node_blocks(
                output,
                locator=locator,
                node=tables[0],
                caption_node=_html_owned_caption(node, "figcaption"),
                import_asset=import_asset,
            )
    elif node.name in {"figure", "img"}:
        figure = _html_figure_block(
            locator=locator,
            node=node,
            import_asset=import_asset,
            warn=warn,
        )
        if figure is not None:
            output.append(figure)
    elif node.name == "math":
        embedded = _html_embedded_block(locator, node, import_asset)
        if embedded is not None:
            output.append(embedded)
    _apply_list_path_to_new_blocks(output, start, path)


def _append_html_list_inline_values(
    output: list[_RawBlock],
    values: tuple[Tag | NavigableString, ...],
    *,
    locator: SourceLocator,
    import_asset: AssetImporter,
    path: tuple[_HTMLListOwner, ...],
) -> None:
    for segment in _html_inline_segments_from_values(values):
        if isinstance(segment, Tag):
            embedded = _html_embedded_block(
                _html_locator(segment, locator.source_format),
                segment,
                import_asset,
            )
            if embedded is not None:
                _append_html_owned_block(output, embedded, path)
        elif segment["text"]:
            payload, notes = _html_segment_payload(segment)
            notes = _html_anchor_notes(
                notes,
                field="list_item",
                item_index=0,
            )
            _append_html_owned_block(
                output,
                _RawBlock(
                    RichBlockKind.LIST,
                    locator,
                    {
                        "ordered": path[-1].ordered,
                        "items": [payload],
                    },
                    notes=notes,
                    presentation_fields=(
                        _html_presentation_field(
                            segment,
                            field="list_item",
                            item_index=0,
                        ),
                    ),
                ),
                path,
            )


def _apply_list_path_to_new_blocks(
    output: list[_RawBlock],
    start: int,
    path: tuple[_HTMLListOwner, ...],
) -> None:
    for index in range(start, len(output)):
        block = output[index]
        output[index] = _owned_raw_block(block, path)


def _append_html_owned_block(
    output: list[_RawBlock],
    block: _RawBlock,
    path: tuple[_HTMLListOwner, ...],
) -> None:
    output.append(_owned_raw_block(block, path))


def _owned_raw_block(
    block: _RawBlock,
    path: tuple[_HTMLListOwner, ...],
) -> _RawBlock:
    entries = []
    for owner in path:
        segment_index = owner.next_segment_index
        entries.append(
            RichListPathEntry(
                container_id=owner.container_id,
                container_source_id=owner.container_source_id,
                container_selector=owner.container_selector,
                item_id=owner.item_id,
                item_source_id=owner.item_source_id,
                item_selector=owner.item_selector,
                item_index=owner.item_index,
                item_count=owner.item_count,
                depth=owner.depth,
                ordered=owner.ordered,
                segment_index=segment_index,
                continuation=segment_index > 0,
            )
        )
        owner.next_segment_index += 1
    return _RawBlock(
        kind=block.kind,
        locator=block.locator,
        payload=block.payload,
        section_reset_to_level=block.section_reset_to_level,
        target_panels=block.target_panels,
        list_path=tuple(entries),
        notes=block.notes,
        presentation_roles=block.presentation_roles,
        presentation_fields=block.presentation_fields,
        figure_presentation=block.figure_presentation,
        table_presentation=block.table_presentation,
        caption_presentation=block.caption_presentation,
        outline_heading=block.outline_heading,
    )


def _html_list_identity(
    artifact: SourceArtifact,
    node: Tag,
    *,
    role: str,
) -> str:
    prefix = "list" if role == "container" else "list-item"
    material = {
        "source": artifact.content_identity,
        "role": f"html-list-{role}",
        "path": _html_tag_path(node),
    }
    return f"{prefix}-{hashlib.sha256(json_bytes(material)).hexdigest()[:24]}"


def _html_structural_identity(
    artifact: SourceArtifact,
    node: Tag,
    *,
    role: str,
) -> str:
    material = {
        "source": artifact.content_identity,
        "role": role,
        "path": _html_tag_path(node),
    }
    return f"{role}-{hashlib.sha256(json_bytes(material)).hexdigest()[:24]}"


def _locator_to_document(locator: SourceLocator) -> dict[str, Any]:
    return {
        "source_format": locator.source_format.value,
        "line_start": locator.line_start,
        "column_start": locator.column_start,
        "line_end": locator.line_end,
        "column_end": locator.column_end,
        "selector": locator.selector,
        "source_id": locator.source_id,
    }


def _html_tag_path(node: Tag) -> tuple[tuple[str, int], ...]:
    output = []
    current: Tag | None = node
    while isinstance(current, Tag) and current.name != "[document]":
        parent = current.parent
        siblings = (
            [child for child in parent.children if isinstance(child, Tag)]
            if isinstance(parent, Tag)
            else [current]
        )
        output.append((str(current.name), siblings.index(current)))
        current = parent if isinstance(parent, Tag) else None
    return tuple(reversed(output))


def _html_is_latexml_list_marker(node: Tag) -> bool:
    classes = {str(value).casefold() for value in node.get("class") or ()}
    return "ltx_tag_item" in classes


def _html_list_is_block_node(node: Tag) -> bool:
    if node.name == "math":
        return str(node.get("display") or "").casefold() == "block"
    return node.name in {"p", "ul", "ol", "pre", "table", "figure", "img"}


def _html_list_contains_block_node(node: Tag) -> bool:
    return any(
        isinstance(descendant, Tag) and _html_list_is_block_node(descendant)
        for descendant in node.descendants
    )


def _html_code_block(locator: SourceLocator, node: Tag) -> _RawBlock:
    code = node.find("code")
    language = ""
    if isinstance(code, Tag):
        language = next(
            (
                str(class_name)[9:]
                for class_name in code.get("class") or ()
                if str(class_name).startswith("language-")
            ),
            "",
        )
    return _RawBlock(
        RichBlockKind.CODE,
        locator,
        {
            "text": (code or node).get_text("", strip=False),
            "language": language,
        },
    )


def _html_is_equation_table(node: Tag) -> bool:
    return any(
        "equation" in str(class_name).casefold()
        for class_name in node.get("class") or ()
    )


def _html_is_latexml_table_figure(node: Tag) -> bool:
    return node.name == "figure" and any(
        str(class_name).casefold() == "ltx_table"
        for class_name in node.get("class") or ()
    )


def _html_descendant_tables(node: Tag) -> list[Tag]:
    return [
        table
        for table in node.find_all("table")
        if isinstance(table, Tag)
    ]


def _html_figure_block(
    *,
    locator: SourceLocator,
    node: Tag,
    import_asset: AssetImporter,
    warn: WarningEmitter,
) -> _RawBlock | None:
    media_nodes = _html_figure_media_nodes(node)
    if not media_nodes:
        _html_figure_presentation(node, media_nodes=(), panels=())
        return None
    panel_values = tuple(
        _html_figure_panel(
            media,
            panel_index=index,
            import_asset=import_asset,
            warn=warn,
        )
        for index, media in enumerate(media_nodes)
    )
    panels = tuple(value[0] for value in panel_values)
    figure_presentation = _html_figure_presentation(
        node,
        media_nodes=media_nodes,
        panels=panels,
    )
    caption_node = (
        _html_owned_caption(node, "figcaption")
        if node.name == "figure"
        else None
    )
    caption_segment = (
        _html_single_inline_segment(caption_node)
        if isinstance(caption_node, Tag)
        else _inline_payload([])
    )
    if caption_segment.get("_notes"):
        raise ValueError(
            "source notes in HTML Figure captions cannot be represented"
        )
    caption = str(caption_segment["text"])
    panel = panels[0]
    single_supported = len(panels) == 1 and panel["status"] in {
        "available",
        "missing",
    }
    asset = panel_values[0][1] if single_supported else None
    return _RawBlock(
        RichBlockKind.FIGURE,
        locator,
        _figure_payload(
            asset,
            alt_text=str(panel["alt_text"]) if single_supported else "",
            caption=caption,
            target=str(panel["target"]) if single_supported else "",
        ),
        target_panels=panels,
        presentation_fields=(
            _html_presentation_field(caption_segment, field="caption"),
        ),
        figure_presentation=figure_presentation,
        caption_presentation=(
            _html_caption_presentation(
                kind="figure",
                container=node,
                caption=caption_node,
                content_nodes=media_nodes,
            )
            if isinstance(caption_node, Tag) and caption
            else None
        ),
    )


def _html_caption_presentation(
    *,
    kind: str,
    container: Tag,
    caption: Tag,
    content_nodes: tuple[Tag, ...],
) -> dict[str, Any]:
    alignment, alignment_sources = _html_caption_alignment(caption)
    return {
        "kind": kind,
        "placement": _html_caption_placement(
            container,
            caption,
            content_nodes=content_nodes,
        ),
        "alignment": alignment,
        "alignment_sources": list(alignment_sources),
    }


def _html_owned_caption(container: Tag, name: str) -> Tag | None:
    captions = [
        caption
        for caption in container.find_all(name)
        if isinstance(caption, Tag)
        and (
            caption.find_parent("figure") is container
            if container.name == "figure"
            else caption.find_parent(container.name) is container
        )
    ]
    if len(captions) > 1:
        raise ValueError("multiple authored captions conflict")
    return captions[0] if captions else None


def _html_caption_alignment(
    caption: Tag,
) -> tuple[str | None, tuple[str, ...]]:
    evidence: list[tuple[str, str]] = []
    unsupported_inline_alignment = False
    for declaration in str(caption.get("style") or "").split(";"):
        name, separator, value = declaration.partition(":")
        if not separator or name.strip().casefold() != "text-align":
            continue
        normalized = re.sub(
            r"\s*!important\s*$",
            "",
            value.strip(),
            flags=re.IGNORECASE,
        ).casefold()
        if normalized in {"start", "center", "end"}:
            evidence.append(
                (
                    normalized,
                    f"style:text-align:{normalized}",
                )
            )
        else:
            unsupported_inline_alignment = True
    class_alignment = {
        "ltx_centering": "center",
        "ltx_align_center": "center",
    }
    for class_name in caption.get("class") or ():
        normalized = str(class_name).casefold()
        alignment = class_alignment.get(normalized)
        if alignment is not None:
            evidence.append((alignment, f"class:{normalized}"))
    alignments = {alignment for alignment, _source in evidence}
    if (unsupported_inline_alignment and evidence) or len(alignments) > 1:
        raise ValueError("conflicting caption alignment evidence")
    if unsupported_inline_alignment or not evidence:
        return None, ()
    sources = tuple(dict.fromkeys(source for _alignment, source in evidence))
    return evidence[0][0], sources


def _html_caption_placement(
    container: Tag,
    caption: Tag,
    *,
    content_nodes: tuple[Tag, ...],
) -> str:
    if caption.parent is container and caption.name == "caption":
        return "embedded"
    direct_children = [
        child for child in container.children if isinstance(child, Tag)
    ]
    direct_caption = caption
    while (
        isinstance(direct_caption.parent, Tag)
        and direct_caption.parent is not container
    ):
        direct_caption = direct_caption.parent
    if direct_caption.parent is not container or direct_caption not in direct_children:
        raise ValueError("caption is not exactly owned by its container")
    caption_index = direct_children.index(direct_caption)
    content_indexes = []
    for node in content_nodes:
        direct = node
        while isinstance(direct.parent, Tag) and direct.parent is not container:
            direct = direct.parent
        if direct.parent is not container or direct not in direct_children:
            raise ValueError("caption content is not owned by its container")
        content_indexes.append(direct_children.index(direct))
    if not content_indexes:
        raise ValueError("caption has no exact container content")
    if caption_index < min(content_indexes):
        return "before_content"
    if caption_index > max(content_indexes):
        return "after_content"
    raise ValueError("caption placement conflicts with container content")


def _html_figure_media_nodes(node: Tag) -> tuple[Tag, ...]:
    if node.name == "img":
        return (node,)
    output = []
    for media in node.find_all(["object", "img"]):
        if not isinstance(media, Tag) or media.find_parent("figure") is not node:
            continue
        if media.name == "img" and isinstance(media.find_parent("object"), Tag):
            continue
        output.append(media)
    return tuple(output)


_HTML_FIGURE_DIMENSION_LIMIT = 1_000_000


def _html_figure_presentation(
    node: Tag,
    *,
    media_nodes: tuple[Tag, ...],
    panels: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    neutral = _html_neutral_figure_presentation(panels)
    if node.name != "figure" or "ltx_figure" not in _html_classes(node):
        return neutral

    flex_roots = [
        descendant
        for descendant in node.find_all(class_=True)
        if isinstance(descendant, Tag)
        and "ltx_flex_figure" in _html_classes(descendant)
    ]
    flex_signals = [
        descendant
        for descendant in node.find_all(class_=True)
        if isinstance(descendant, Tag)
        and _html_classes(descendant)
        & {
            "ltx_flex_cell",
            "ltx_flex_break",
            "ltx_figure_panel",
        }
    ]
    if flex_roots:
        if len(flex_roots) != 1 or flex_roots[0].parent is not node:
            raise ValueError("Figure panel layout has ambiguous flex roots")
        return _html_flex_figure_presentation(
            flex_roots[0],
            media_nodes=media_nodes,
            panels=panels,
        )
    if flex_signals:
        raise ValueError("Figure panel layout has flex children without a root")

    exact_graphics = [
        media for media in media_nodes if "ltx_graphics" in _html_classes(media)
    ]
    if not exact_graphics:
        return neutral
    if (
        len(media_nodes) != 1
        or len(exact_graphics) != 1
        or exact_graphics[0].parent is not node
    ):
        raise ValueError("Figure panel layout has ambiguous direct graphics")
    return {
        "layout": {
            "kind": "single",
            "column_count": 1,
            "row_count": 1,
            "rows": [[0]],
            "column_source": "latexml_ar5iv_direct_graphic",
            "row_sources": ["latexml_ar5iv_direct_graphic"],
            "break_after_panel_indexes": [],
            "break_source": None,
        },
        "panels": [
            _html_figure_layout_panel(
                exact_graphics[0],
                panels[0],
                row_index=0,
                column_index=0,
            )
        ],
    }


def _html_neutral_figure_presentation(
    panels: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    return {
        "layout": {
            "kind": "neutral",
            "column_count": None,
            "row_count": None,
            "rows": [],
            "column_source": None,
            "row_sources": [],
            "break_after_panel_indexes": [],
            "break_source": None,
        },
        "panels": [
            {
                "panel_index": int(panel["panel_index"]),
                "source_id": str(panel["source_id"]),
                "row_index": None,
                "column_index": None,
                "display_width": None,
                "display_height": None,
                "dimension_source": None,
                "aspect_ratio": None,
                "aspect_ratio_source": None,
            }
            for panel in panels
        ],
    }


def _html_flex_figure_presentation(
    root: Tag,
    *,
    media_nodes: tuple[Tag, ...],
    panels: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    if any(
        value.startswith("ltx_flex_") and value != "ltx_flex_figure"
        for value in _html_classes(root)
    ):
        raise ValueError("Figure panel layout flex root has conflicting classes")
    children = [child for child in root.children if isinstance(child, Tag)]
    graphics: list[Tag] = []
    break_after: list[int] = []
    rows: list[list[int]] = []
    row_size_sources: list[list[str]] = []
    current_row: list[int] = []
    current_size_sources: list[str] = []
    last_was_break = False

    for child in children:
        classes = _html_classes(child)
        unknown_flex = {
            value
            for value in classes
            if value.startswith("ltx_flex_")
            and value
            not in {
                "ltx_flex_cell",
                "ltx_flex_break",
                "ltx_flex_size_1",
                "ltx_flex_size_2",
                "ltx_flex_size_3",
            }
        }
        if unknown_flex:
            raise ValueError("Figure panel layout has an unknown flex class")
        if "ltx_flex_break" in classes:
            if (
                "ltx_flex_cell" in classes
                or bool(
                    classes
                    & {
                        "ltx_flex_size_1",
                        "ltx_flex_size_2",
                        "ltx_flex_size_3",
                    }
                )
                or list(child.find_all(True))
                or not current_row
                or last_was_break
            ):
                raise ValueError("Figure panel layout has an empty or mixed row break")
            break_after.append(current_row[-1])
            rows.append(current_row)
            row_size_sources.append(current_size_sources)
            current_row = []
            current_size_sources = []
            last_was_break = True
            continue
        if "ltx_flex_cell" not in classes:
            raise ValueError("Figure panel layout has an unknown flex child")
        size_classes = classes & {
            "ltx_flex_size_1",
            "ltx_flex_size_2",
            "ltx_flex_size_3",
        }
        if len(size_classes) != 1:
            raise ValueError("Figure panel layout has an ambiguous flex size")
        current_size_sources.append(f"class:{next(iter(size_classes))}")
        owned = [child for child in child.children if isinstance(child, Tag)]
        if len(owned) != 1 or owned[0].name not in {"img", "object"}:
            raise ValueError("Figure panel layout cell does not own one graphic")
        graphic = owned[0]
        if not {
            "ltx_graphics",
            "ltx_figure_panel",
        }.issubset(_html_classes(graphic)):
            raise ValueError("Figure panel layout cell graphic is not exact")
        graphics.append(graphic)
        current_row.append(len(graphics) - 1)
        last_was_break = False

    if last_was_break or not graphics:
        raise ValueError("Figure panel layout has an empty final row")
    normalized_rows: list[list[int]] = []
    normalized_row_sources: list[str] = []
    for authored_row, authored_sources in zip(
        [*rows, current_row],
        [*row_size_sources, current_size_sources],
        strict=True,
    ):
        sources = set(authored_sources)
        if len(sources) != 1:
            raise ValueError("Figure panel layout mixes sizes within a row")
        row_source = authored_sources[0]
        row_column_count = int(row_source.rsplit("_", 1)[1])
        while len(authored_row) > row_column_count:
            normalized_rows.append(authored_row[:row_column_count])
            normalized_row_sources.append(row_source)
            authored_row = authored_row[row_column_count:]
        if authored_row:
            normalized_rows.append(authored_row)
            normalized_row_sources.append(row_source)
    column_count = max(
        int(source.rsplit("_", 1)[1]) for source in normalized_row_sources
    )
    column_sources = set(normalized_row_sources)
    column_source = (
        normalized_row_sources[0] if len(column_sources) == 1 else None
    )
    if [id(graphic) for graphic in graphics] != [id(media) for media in media_nodes]:
        raise ValueError("Figure panel layout differs from source panel order")
    if len(graphics) != len(panels):
        raise ValueError("Figure panel layout differs from source panel count")
    positions = {
        panel_index: (row_index, column_index)
        for row_index, row in enumerate(normalized_rows)
        for column_index, panel_index in enumerate(row)
    }
    return {
        "layout": {
            "kind": "flex",
            "column_count": column_count,
            "row_count": len(normalized_rows),
            "rows": normalized_rows,
            "column_source": column_source,
            "row_sources": normalized_row_sources,
            "break_after_panel_indexes": break_after,
            "break_source": (
                "class:ltx_flex_break" if break_after else None
            ),
        },
        "panels": [
            _html_figure_layout_panel(
                graphic,
                panel,
                row_index=positions[index][0],
                column_index=positions[index][1],
            )
            for index, (graphic, panel) in enumerate(
                zip(graphics, panels, strict=True)
            )
        ],
    }


def _html_figure_layout_panel(
    node: Tag,
    panel: Mapping[str, Any],
    *,
    row_index: int,
    column_index: int,
) -> dict[str, Any]:
    width, height, dimension_source = _html_figure_dimensions(node)
    aspect_ratio, aspect_source = _html_figure_aspect_ratio(node)
    if (
        width is not None
        and aspect_ratio is not None
        and _normalized_ratio(width, height) != aspect_ratio
    ):
        raise ValueError("Figure panel layout aspect ratio differs from dimensions")
    return {
        "panel_index": int(panel["panel_index"]),
        "source_id": str(panel["source_id"]),
        "row_index": row_index,
        "column_index": column_index,
        "display_width": width,
        "display_height": height,
        "dimension_source": dimension_source,
        "aspect_ratio": list(aspect_ratio) if aspect_ratio is not None else None,
        "aspect_ratio_source": aspect_source,
    }


def _html_figure_dimensions(
    node: Tag,
) -> tuple[int | None, int | None, str | None]:
    raw_width = node.get("width")
    raw_height = node.get("height")
    if raw_width is None and raw_height is None:
        return None, None, None
    if raw_width is None or raw_height is None:
        raise ValueError("Figure panel layout dimensions are incomplete")
    values = []
    for raw in (raw_width, raw_height):
        value = str(raw).strip()
        if re.fullmatch(r"[1-9][0-9]*", value) is None:
            raise ValueError("Figure panel layout dimension is malformed")
        parsed = int(value)
        if parsed > _HTML_FIGURE_DIMENSION_LIMIT:
            raise ValueError("Figure panel layout dimension exceeds its bound")
        values.append(parsed)
    return values[0], values[1], "attributes:width,height"


def _html_figure_aspect_ratio(
    node: Tag,
) -> tuple[tuple[int, int] | None, str | None]:
    values = []
    for declaration in str(node.get("style") or "").split(";"):
        name, separator, value = declaration.partition(":")
        if not separator or name.strip().casefold() != "aspect-ratio":
            continue
        match = re.fullmatch(
            r"\s*([1-9][0-9]*)\s*/\s*([1-9][0-9]*)\s*",
            value,
        )
        if match is None:
            raise ValueError("Figure panel layout aspect ratio is malformed")
        numerator, denominator = (int(match.group(1)), int(match.group(2)))
        if max(numerator, denominator) > _HTML_FIGURE_DIMENSION_LIMIT:
            raise ValueError("Figure panel layout aspect ratio exceeds its bound")
        values.append(_normalized_ratio(numerator, denominator))
    if len(values) > 1:
        raise ValueError("Figure panel layout has conflicting aspect ratios")
    if not values:
        return None, None
    return values[0], "style:aspect-ratio"


def _normalized_ratio(numerator: int, denominator: int) -> tuple[int, int]:
    divisor = gcd(numerator, denominator)
    return numerator // divisor, denominator // divisor


def _html_classes(node: Tag) -> set[str]:
    return {str(value).casefold() for value in node.get("class") or ()}


def _html_figure_panel(
    node: Tag,
    *,
    panel_index: int,
    import_asset: AssetImporter,
    warn: WarningEmitter,
) -> tuple[dict[str, Any], RichAsset | None]:
    is_object = node.name == "object"
    target = str(node.get("data" if is_object else "src") or "")
    declared_media_type = str(node.get("type") or "").casefold()
    guessed_media_type = mimetypes.guess_type(target)[0] or ""
    media_type = declared_media_type or guessed_media_type
    supported = bool(target) and (
        not is_object or declared_media_type == "image/svg+xml"
    )
    asset = None
    if supported and _is_local_asset_target(target):
        asset = import_asset(target)
        status = "available" if asset is not None else "missing"
    else:
        status = "unsupported"
        warn(
            "unsupported figure panel was not imported: "
            f"{target or '<missing target>'}"
        )
    source_id = str(node.get("id") or "")
    alt_text = str(
        node.get("alt")
        or node.get("aria-label")
        or node.get("title")
        or ""
    )
    return (
        {
            "panel_index": panel_index,
            "source_id": source_id,
            "selector": f"#{source_id}" if source_id else "",
            "target": target,
            "media_type": asset.media_type if asset else media_type,
            "alt_text": alt_text,
            "status": status,
            "asset_digest": asset.artifact_digest if asset else "",
            "logical_name": target,
            "size": asset.size if asset else 0,
        },
        asset,
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


def _append_html_table_node_blocks(
    output: list[_RawBlock],
    *,
    locator: SourceLocator,
    node: Tag,
    caption_node: Tag | None,
    import_asset: AssetImporter,
) -> None:
    rows = [
        row
        for row in node.find_all("tr")
        if isinstance(row, Tag) and row.find_parent("table") is node
    ]
    expanded_result = _html_expand_table_rows(rows)
    if expanded_result is None:
        return
    expanded, cells = expanded_result
    data_start = (
        1
        if rows and isinstance(rows[0].find("th", recursive=False), Tag)
        else 0
    )
    caption_segment = (
        _html_single_inline_segment(caption_node)
        if isinstance(caption_node, Tag)
        else _inline_payload([])
    )
    if caption_segment.get("_notes"):
        raise ValueError(
            "source notes in HTML Table captions cannot be represented"
        )
    _append_html_table_blocks(
        output,
        locator=locator,
        headers=expanded[0] if data_start else [],
        rows=expanded[data_start:],
        caption_segment=caption_segment,
        caption_presentation=(
            _html_caption_presentation(
                kind="table",
                container=caption_node.parent,
                caption=caption_node,
                content_nodes=(node,),
            )
            if isinstance(caption_node, Tag)
            and isinstance(caption_node.parent, Tag)
            and caption_segment["text"]
            else None
        ),
        cells=cells,
        import_asset=import_asset,
    )


def _html_expand_table_rows(
    rows: list[Tag],
) -> tuple[list[list[Tag | None]], list[dict[str, Any]]] | None:
    output: list[list[Tag | None]] = []
    cells: list[dict[str, Any]] = []
    rowspans: dict[int, int] = {}
    coverage_area = 0
    for row_index, row in enumerate(rows):
        expanded: list[Tag | None] = []
        column = 0

        def fill_spans() -> None:
            nonlocal column
            while rowspans.get(column, 0) > 0:
                expanded.append(None)
                column += 1

        for cell in row.find_all(["th", "td"], recursive=False):
            if not isinstance(cell, Tag):
                continue
            fill_spans()
            colspan = _html_table_span(cell.get("colspan"))
            rowspan = _html_table_span(cell.get("rowspan"))
            if colspan is None or rowspan is None:
                return None
            cell_area = rowspan * colspan
            if cell_area > _HTML_TABLE_COVERAGE_LIMIT - coverage_area:
                return None
            coverage_area += cell_area
            if column + colspan > _HTML_TABLE_SPAN_LIMIT:
                return None
            if any(
                rowspans.get(occupied, 0) > 0
                for occupied in range(column, column + colspan)
            ):
                return None
            expanded.append(cell)
            expanded.extend(None for _ in range(colspan - 1))
            cells.append(
                {
                    "row_index": row_index,
                    "column_index": column,
                    "row_span": rowspan,
                    "column_span": colspan,
                    "kind": "header" if cell.name == "th" else "data",
                    "locator": _locator_to_document(
                        _html_locator(cell, SourceFormat.HTML)
                    ),
                    **_html_table_cell_presentation(cell),
                }
            )
            if rowspan > 1:
                for occupied in range(column, column + colspan):
                    rowspans[occupied] = max(
                        rowspans.get(occupied, 0),
                        rowspan,
                    )
            column += colspan
        if rowspans:
            final_column = max(column, max(rowspans) + 1)
            while column < final_column:
                expanded.append(None)
                column += 1
        output.append(expanded)
        for occupied in tuple(rowspans):
            rowspans[occupied] -= 1
            if rowspans[occupied] <= 0:
                del rowspans[occupied]
    if rowspans:
        return None
    width = max((len(row) for row in output), default=0)
    for row in output:
        row.extend(None for _ in range(width - len(row)))
    return output, cells


def _html_table_cell_presentation(cell: Tag) -> dict[str, Any]:
    alignment_evidence: list[tuple[str, str]] = []
    unsupported_inline_alignment = False
    for declaration in str(cell.get("style") or "").split(";"):
        name, separator, value = declaration.partition(":")
        if not separator:
            continue
        property_name = name.strip().casefold()
        if property_name.startswith("border"):
            raise ValueError(
                "unsupported Table cell presentation border style"
            )
        if property_name != "text-align":
            continue
        normalized = re.sub(
            r"\s*!important\s*$",
            "",
            value.strip(),
            flags=re.IGNORECASE,
        ).casefold()
        if normalized in {"left", "center", "right", "start", "end"}:
            alignment_evidence.append(
                (
                    normalized,
                    f"style:text-align:{normalized}",
                )
            )
        else:
            unsupported_inline_alignment = True
    class_alignment = {
        "ltx_align_left": "left",
        "ltx_align_center": "center",
        "ltx_align_right": "right",
    }
    for class_name in cell.get("class") or ():
        normalized = str(class_name).casefold()
        alignment = class_alignment.get(normalized)
        if alignment is not None:
            alignment_evidence.append(
                (alignment, f"class:{normalized}")
            )
    alignments = {
        alignment for alignment, _source in alignment_evidence
    }
    if (
        unsupported_inline_alignment and alignment_evidence
    ) or len(alignments) > 1:
        raise ValueError("conflicting Table cell presentation alignment")
    if unsupported_inline_alignment or not alignment_evidence:
        alignment = None
        alignment_sources: tuple[str, ...] = ()
    else:
        alignment = alignment_evidence[0][0]
        alignment_sources = tuple(
            dict.fromkeys(
                source for _alignment, source in alignment_evidence
            )
        )

    rule_classes = {
        "ltx_border_t": "top",
        "ltx_border_tt": "top",
        "ltx_border_T": "top",
        "ltx_border_r": "right",
        "ltx_border_rr": "right",
        "ltx_border_R": "right",
        "ltx_border_r_dashed": "right",
        "ltx_border_b": "bottom",
        "ltx_border_bb": "bottom",
        "ltx_border_B": "bottom",
        "ltx_border_b_dashed": "bottom",
        "ltx_border_l": "left",
        "ltx_border_ll": "left",
        "ltx_border_L": "left",
    }
    edges: dict[str, str] = {}
    for raw_class_name in cell.get("class") or ():
        class_name = str(raw_class_name)
        edge = rule_classes.get(class_name)
        if edge is None:
            if class_name.startswith("ltx_border_"):
                raise ValueError(
                    "unsupported Table cell presentation border class"
                )
            continue
        if edge in edges:
            raise ValueError("conflicting Table cell presentation rule edge")
        edges[edge] = f"class:{class_name}"
    edge_order = {"top": 0, "right": 1, "bottom": 2, "left": 3}
    return {
        "horizontal_alignment": alignment,
        "horizontal_alignment_sources": list(alignment_sources),
        "rule_edges": [
            {"edge": edge, "source": source}
            for edge, source in sorted(
                edges.items(),
                key=lambda item: edge_order[item[0]],
            )
        ],
    }


def _html_table_span(value: Any) -> int | None:
    if value is None:
        return 1
    try:
        span = int(str(value))
    except ValueError:
        return None
    if not 1 <= span <= _HTML_TABLE_SPAN_LIMIT:
        return None
    return span


def _append_html_table_blocks(
    output: list[_RawBlock],
    *,
    locator: SourceLocator,
    headers: list[Tag | None],
    rows: list[list[Tag | None]],
    caption_segment: Mapping[str, Any],
    caption_presentation: Mapping[str, Any] | None,
    cells: list[Mapping[str, Any]],
    import_asset: AssetImporter,
) -> None:
    shape = [headers, *rows]
    pending = [["" for _ in row] for row in shape]
    pending_views = [
        [_inline_payload([]) for _ in row]
        for row in shape
    ]
    pending_notes: list[_HTMLNoteSpec] = []
    emitted_table = False
    empty_caption_segment = _inline_payload([])

    def flush(*, force: bool = False) -> None:
        nonlocal emitted_table
        if not force and not any(cell for row in pending for cell in row):
            return
        owns_caption = not emitted_table
        active_caption = (
            caption_segment if owns_caption else empty_caption_segment
        )
        output.append(
            _RawBlock(
                RichBlockKind.TABLE,
                locator,
                {
                    "headers": list(pending[0]),
                    "rows": [list(row) for row in pending[1:]],
                    "caption": active_caption["text"],
                },
                notes=tuple(pending_notes),
                presentation_fields=(
                    _html_presentation_field(
                        active_caption,
                        field="caption",
                    ),
                    *(
                        _html_presentation_field(
                            pending_views[0][column],
                            field="table_header",
                            column_index=column,
                        )
                        for column in range(len(pending_views[0]))
                    ),
                    *(
                        _html_presentation_field(
                            pending_views[row_index][column],
                            field="table_cell",
                            row_index=row_index - 1,
                            column_index=column,
                        )
                        for row_index in range(1, len(pending_views))
                        for column in range(len(pending_views[row_index]))
                    ),
                ),
                table_presentation={
                    "cells": [dict(cell) for cell in cells],
                },
                caption_presentation=(
                    caption_presentation if owns_caption else None
                ),
            )
        )
        emitted_table = True
        pending_notes.clear()
        for row in pending:
            for column in range(len(row)):
                row[column] = ""
        for row in pending_views:
            for column in range(len(row)):
                row[column] = _inline_payload([])

    for row_index, row in enumerate(shape):
        for column, cell in enumerate(row):
            if cell is None:
                continue
            for segment in _html_inline_segments(
                cell,
                visible_text_only=True,
            ):
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
                    _payload, notes = _html_segment_payload(segment)
                    existing = pending[row_index][column]
                    offset = len(existing) + (1 if existing else 0)
                    notes = _html_anchor_notes(
                        notes,
                        field=(
                            "table_header"
                            if row_index == 0 and headers
                            else "table_cell"
                        ),
                        row_index=(
                            None
                            if row_index == 0 and headers
                            else row_index - 1
                        ),
                        column_index=column,
                        offset=offset,
                    )
                    pending_notes.extend(notes)
                    pending_views[row_index][column] = (
                        _merge_html_inline_payloads(
                            pending_views[row_index][column],
                            segment,
                        )
                    )
                    pending[row_index][column] = " ".join(
                        value
                        for value in (
                            pending[row_index][column],
                            segment["text"],
                        )
                        if value
                    )
    flush(force=not emitted_table)


def _merge_html_inline_payloads(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    if not left["text"]:
        return {
            "text": right["text"],
            "inline_spans": [dict(span) for span in right["inline_spans"]],
            "_marks": [dict(mark) for mark in right.get("_marks", ())],
        }
    offset = len(left["text"]) + 1
    spans = [dict(span) for span in left["inline_spans"]]
    spans.append(
        {
            "kind": "text",
            "start": len(left["text"]),
            "end": offset,
            "text": " ",
        }
    )
    for raw in right["inline_spans"]:
        span = dict(raw)
        span["start"] = int(span["start"]) + offset
        span["end"] = int(span["end"]) + offset
        spans.append(span)
    marks = [dict(mark) for mark in left.get("_marks", ())]
    for raw in right.get("_marks", ()):
        mark = dict(raw)
        mark["start"] = int(mark["start"]) + offset
        mark["end"] = int(mark["end"]) + offset
        marks.append(mark)
    return {
        "text": f"{left['text']} {right['text']}",
        "inline_spans": spans,
        "_marks": marks,
    }


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
        presentation_fields=(
            _html_presentation_field(
                _inline_payload([]),
                field="caption",
            ),
        ),
    )


def _html_inline_segments(
    node: Tag,
    *,
    include_equation_tables: bool = False,
    visible_text_only: bool = False,
) -> list[dict[str, Any] | Tag]:
    return _html_inline_segments_from_values(
        (node,),
        include_equation_tables=include_equation_tables,
        visible_text_only=visible_text_only,
    )


def _html_inline_segments_from_values(
    values: tuple[Tag | NavigableString, ...],
    *,
    include_equation_tables: bool = False,
    visible_text_only: bool = False,
    capture_notes: bool = True,
) -> list[dict[str, Any] | Tag]:
    segments: list[dict[str, Any] | Tag] = []
    parts: list[dict[str, Any]] = []
    pending_notes: list[_HTMLNoteSpec] = []
    next_mark_id = 0

    def flush() -> None:
        payload = _inline_payload(parts)
        note_offsets = payload.pop("_note_offsets", {})
        if payload["text"]:
            if pending_notes:
                if set(note_offsets) != set(range(len(pending_notes))):
                    raise ValueError("source note markers lost normalized anchors")
                payload["_notes"] = tuple(
                    replace(
                        note,
                        anchor={
                            **note.anchor,
                            "start": note_offsets[index][0],
                            "end": note_offsets[index][1],
                        },
                    )
                    for index, note in enumerate(pending_notes)
                )
            segments.append(payload)
        parts.clear()
        pending_notes.clear()

    def visit(
        value: Tag | NavigableString,
        *,
        link_target: str | None = None,
        active_marks: tuple[tuple[int, str], ...] = (),
    ) -> None:
        nonlocal next_mark_id
        if isinstance(value, NavigableString):
            text = re.sub(r"\s+", " ", str(value))
            if link_target is None:
                _append_inline_part(parts, "text", text, marks=active_marks)
            else:
                _append_inline_part(
                    parts,
                    "link",
                    text,
                    target=link_target,
                    marks=active_marks,
                )
            return
        if _html_is_pubnotes(value):
            return
        if capture_notes and _html_is_source_note(value):
            note = _html_note_spec(value, anchor_start=0)
            if note is None:
                raise ValueError("source notes cannot represent authored note")
            note_index = len(pending_notes)
            pending_notes.append(note)
            _append_inline_part(
                parts,
                "text",
                note.marker,
                marks=active_marks,
                _note_index=note_index,
                _note_marker_length=len(note.marker),
            )
            return
        if visible_text_only and any(
            str(class_name).casefold() == "ltx_note_outer"
            for class_name in value.get("class") or ()
        ):
            return
        if (
            include_equation_tables
            and value.name == "table"
            and _html_is_equation_table(value)
        ):
            flush()
            segments.append(value)
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
                source = _html_visible_text(value) or tex
                _append_inline_part(
                    parts,
                    "math",
                    source,
                    tex=tex,
                    source=source,
                    marks=active_marks,
                )
            return
        mark_kinds = _html_authored_mark_kinds(value)
        nested_marks = active_marks
        for mark_kind in mark_kinds:
            nested_marks = (*nested_marks, (next_mark_id, mark_kind))
            next_mark_id += 1
        nested_link = (
            str(value.get("href") or "")
            if value.name == "a"
            else link_target
        )
        for child in value.children:
            if isinstance(child, (Tag, NavigableString)):
                visit(
                    child,
                    link_target=nested_link,
                    active_marks=nested_marks,
                )

    for value in values:
        visit(value)
    flush()
    return segments


def _html_authored_mark_kinds(node: Tag) -> tuple[str, ...]:
    classes = {str(value).casefold() for value in node.get("class") or ()}
    output = []
    if node.name in {"strong", "b"} or "ltx_font_bold" in classes:
        output.append("strong")
    if node.name in {"em", "i"} or "ltx_font_italic" in classes:
        output.append("emphasis")
    return tuple(output)


def _html_segment_payload(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[_HTMLNoteSpec, ...]]:
    return (
        {
            "text": value["text"],
            "inline_spans": value["inline_spans"],
        },
        tuple(value.get("_notes", ())),
    )


def _html_single_inline_segment(node: Tag) -> Mapping[str, Any]:
    segments = _html_inline_segments(node)
    if len(segments) != 1 or not isinstance(segments[0], Mapping):
        raise ValueError("HTML inline field cannot contain a block element")
    return segments[0]


def _html_heading_semantics(node: Tag) -> tuple[int, tuple[str, ...]]:
    raw_level = int((node.name or "h6")[1:])
    classes = {str(value).casefold() for value in node.get("class") or ()}
    roles = []
    if "ltx_title_classification" in classes or _html_has_ancestor_class(
        node, "ltx_classification"
    ):
        roles.append("classification")

    if _html_is_unique_semantic_title(
        node,
        wrapper_class="ltx_abstract",
        title_class="ltx_title_abstract",
    ):
        roles.append("abstract")
    if _html_is_unique_semantic_title(
        node,
        wrapper_class="ltx_acknowledgements",
        title_class="ltx_title_acknowledgements",
    ):
        roles.append("acknowledgements")
    if len(roles) > 1:
        raise ValueError("HTML semantic heading conventions conflict")
    if not roles or roles == ["classification"]:
        return raw_level, tuple(roles)

    parent_level = _html_authored_parent_heading_level(node)
    if parent_level >= 6:
        raise ValueError("HTML semantic heading exceeds the supported hierarchy")
    level = parent_level + 1
    if roles == ["abstract"] and level != 2:
        raise ValueError(
            "HTML semantic heading places an abstract outside front matter"
        )
    return level, tuple(roles)


def _html_is_unique_semantic_title(
    node: Tag,
    *,
    wrapper_class: str,
    title_class: str,
) -> bool:
    classes = _html_classes(node)
    ancestors = _html_ancestors_with_class(node, wrapper_class)
    if len(ancestors) > 1:
        raise ValueError("HTML semantic heading convention is nested ambiguously")
    if not ancestors:
        return title_class in classes
    wrapper = ancestors[0]
    candidates = []
    for candidate in wrapper.find_all(re.compile(r"h[1-6]")):
        if not isinstance(candidate, Tag):
            continue
        candidate_ancestors = _html_ancestors_with_class(
            candidate,
            wrapper_class,
        )
        if not candidate_ancestors or candidate_ancestors[0] is not wrapper:
            continue
        if candidate.parent is wrapper or title_class in _html_classes(candidate):
            candidates.append(candidate)
    if len(candidates) > 1:
        raise ValueError("HTML semantic heading convention has multiple titles")
    return candidates == [node]


def _html_ancestors_with_class(node: Tag, class_name: str) -> tuple[Tag, ...]:
    matches = []
    current = node.parent
    while isinstance(current, Tag):
        classes = {
            str(value).casefold() for value in current.get("class") or ()
        }
        if class_name in classes:
            matches.append(current)
        current = current.parent
    return tuple(matches)


def _html_authored_parent_heading_level(node: Tag) -> int:
    sections = [
        ancestor
        for ancestor in node.parents
        if isinstance(ancestor, Tag) and ancestor.name == "section"
    ]
    if not sections:
        return 1
    for section in sections:
        owner = node
        while isinstance(owner.parent, Tag) and owner.parent is not section:
            owner = owner.parent
        headings = [
            child
            for child in section.children
            if isinstance(child, Tag)
            and re.fullmatch(r"h[1-6]", child.name or "")
            and child is not node
            and any(sibling is owner for sibling in child.next_siblings)
        ]
        if len(headings) > 1:
            raise ValueError("HTML semantic heading parent is ambiguous")
        if headings:
            heading = headings[0]
            for descendant in section.descendants:
                if descendant is node:
                    raise ValueError(
                        "HTML semantic heading precedes its authored parent"
                    )
                if descendant is heading:
                    break
            return int((heading.name or "h6")[1:])
        if node.parent is not section:
            raise ValueError("HTML semantic heading parent is missing")
    return 1


def _html_classification_ancestor(node: Tag) -> Tag | None:
    matches = []
    current: Tag | None = node
    while isinstance(current, Tag):
        classes = {
            str(value).casefold() for value in current.get("class") or ()
        }
        if "ltx_classification" in classes:
            matches.append(current)
        current = current.parent if isinstance(current.parent, Tag) else None
    return matches[0] if len(matches) == 1 else None


def _html_has_ancestor_class(node: Tag, class_name: str) -> bool:
    return isinstance(
        node.find_parent(
            class_=lambda value: value
            and class_name in str(value).casefold().split()
        ),
        Tag,
    )


def _html_presentation_field(
    value: Mapping[str, Any],
    *,
    field: str,
    item_index: int | None = None,
    row_index: int | None = None,
    column_index: int | None = None,
) -> dict[str, Any]:
    return {
        "field": field,
        "item_index": item_index,
        "row_index": row_index,
        "column_index": column_index,
        "text": str(value["text"]),
        "inline_spans": [dict(span) for span in value["inline_spans"]],
        "marks": [dict(mark) for mark in value.get("_marks", ())],
    }


def _html_anchor_notes(
    notes: tuple[_HTMLNoteSpec, ...],
    *,
    field: str,
    item_index: int | None = None,
    row_index: int | None = None,
    column_index: int | None = None,
    offset: int = 0,
) -> tuple[_HTMLNoteSpec, ...]:
    return tuple(
        replace(
            note,
            anchor={
                "field": field,
                "item_index": item_index,
                "row_index": row_index,
                "column_index": column_index,
                "start": int(note.anchor["start"]) + offset,
                "end": int(note.anchor["end"]) + offset,
            },
        )
        for note in notes
    )


def _html_is_source_note(node: Tag) -> bool:
    classes = {str(value).casefold() for value in node.get("class") or ()}
    return "ltx_note" in classes and "ltx_role_footnotemark" not in classes


def _html_note_spec(
    node: Tag,
    *,
    anchor_start: int,
) -> _HTMLNoteSpec | None:
    marker_node = next(
        (
            child
            for child in node.find_all("sup", recursive=False)
            if "ltx_note_mark" in {
                str(value).casefold()
                for value in child.get("class") or ()
            }
        ),
        None,
    )
    content = node.find(
        class_=lambda value: value
        and "ltx_note_content" in str(value).casefold().split()
    )
    owner = node.parent
    if (
        not isinstance(marker_node, Tag)
        or not isinstance(content, Tag)
        or not isinstance(owner, Tag)
    ):
        return None
    marker = _html_visible_text(marker_node)
    body_values = tuple(
        child
        for child in content.children
        if isinstance(child, (Tag, NavigableString))
        and not (
            isinstance(child, Tag)
            and (
                (
                    child.name == "sup"
                    and "ltx_note_mark" in {
                        str(value).casefold()
                        for value in child.get("class") or ()
                    }
                )
                or "ltx_tag_note" in {
                    str(value).casefold()
                    for value in child.get("class") or ()
                }
            )
        )
    )
    body_segments = _html_inline_segments_from_values(
        body_values,
        visible_text_only=True,
        capture_notes=False,
    )
    if len(body_segments) != 1 or not isinstance(body_segments[0], Mapping):
        return None
    body_payload, nested_notes = _html_segment_payload(body_segments[0])
    if nested_notes or not marker or not body_payload["text"]:
        return None
    source_id = str(node.get("id") or "")
    note_id = source_id or (
        "note-"
        + hashlib.sha256(
            json_bytes({"role": "html-note", "path": _html_tag_path(node)})
        ).hexdigest()[:24]
    )
    return _HTMLNoteSpec(
        note_id=note_id,
        locator=_html_locator(node, SourceFormat.HTML),
        marker=marker,
        body_payload=body_payload,
        owner_locator=_html_locator(owner, SourceFormat.HTML),
        anchor={
            "field": "inline",
            "item_index": None,
            "row_index": None,
            "column_index": None,
            "start": anchor_start,
            "end": anchor_start + len(marker),
        },
    )


def _html_visible_text(node: Tag) -> str:
    values: list[str] = []

    def visit(value: Tag | NavigableString) -> None:
        if isinstance(value, Comment):
            return
        if isinstance(value, NavigableString):
            values.append(str(value))
            return
        if value.name == "math":
            values.append(_html_visible_math_text(value))
            return
        if value.name in {"annotation", "annotation-xml"} or any(
            str(class_name).casefold() == "ltx_note_outer"
            for class_name in value.get("class") or ()
        ):
            return
        for child in value.children:
            if isinstance(child, (Tag, NavigableString)):
                visit(child)

    visit(node)
    return re.sub(r"\s+", " ", "".join(values)).strip()


def _html_visible_math_text(node: Tag) -> str:
    values = [
        str(value)
        for value in node.find_all(string=True)
        if not isinstance(value, Comment)
        and str(value).strip()
        and not isinstance(value.find_parent(["annotation", "annotation-xml"]), Tag)
    ]
    return re.sub(r"\s+", " ", "".join(values)).strip()


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
    parts: list[dict[str, Any]],
    kind: str,
    text: str,
    *,
    marks: tuple[tuple[int, str], ...] = (),
    **metadata: Any,
) -> None:
    if not text:
        return
    item = {"kind": kind, "text": text, "_marks": marks, **metadata}
    if (
        kind == "text"
        and parts
        and parts[-1]["kind"] == "text"
        and parts[-1].get("_marks", ()) == marks
        and "_note_index" not in item
        and "_note_index" not in parts[-1]
    ):
        parts[-1]["text"] += text
    else:
        parts.append(item)


def _inline_payload(parts: list[dict[str, Any]]) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
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
    mark_ranges: dict[int, dict[str, Any]] = {}
    note_offsets: dict[int, tuple[int, int]] = {}
    cursor = 0
    previous_span_marks: tuple[tuple[int, str], ...] | None = None
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
        part_marks = tuple(part.get("_marks", ()))
        if (
            span["kind"] == "text"
            and spans
            and spans[-1]["kind"] == "text"
            and previous_span_marks == part_marks
        ):
            spans[-1]["end"] = end
            spans[-1]["text"] += part["text"]
        else:
            spans.append(span)
        previous_span_marks = part_marks
        if "_note_index" in part:
            note_index = int(part["_note_index"])
            marker_end = cursor + int(part["_note_marker_length"])
            if note_index in note_offsets or marker_end > end:
                raise ValueError("source note marker normalization is invalid")
            note_offsets[note_index] = (cursor, marker_end)
        for mark_id, mark_kind in part.get("_marks", ()):
            mark = mark_ranges.setdefault(
                mark_id,
                {"kind": mark_kind, "start": cursor, "end": end},
            )
            mark["end"] = end
        cursor = end
    payload = {
        "text": text,
        "inline_spans": spans,
    }
    if mark_ranges:
        payload["_marks"] = sorted(
            mark_ranges.values(),
            key=lambda mark: (
                int(mark["start"]),
                -int(mark["end"]),
                str(mark["kind"]),
            ),
        )
    if note_offsets:
        payload["_note_offsets"] = note_offsets
    return payload


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
    html_sections: tuple[_HTMLSectionTarget, ...] = (),
    html_block_targets: tuple[_HTMLBlockTarget, ...] = (),
    html_front_matter: tuple[Mapping[str, Any], ...] = (),
    html_classifications: tuple[_HTMLClassificationFlow, ...] = (),
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
        if raw.kind is RichBlockKind.HEADING and raw.outline_heading:
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
        if raw.list_path:
            material["list_path"] = [
                {
                    "container_id": item.container_id,
                    "container_source_id": item.container_source_id,
                    "container_selector": item.container_selector,
                    "item_id": item.item_id,
                    "item_source_id": item.item_source_id,
                    "item_selector": item.item_selector,
                    "item_index": item.item_index,
                    "item_count": item.item_count,
                    "depth": item.depth,
                    "ordered": item.ordered,
                    "segment_index": item.segment_index,
                    "continuation": item.continuation,
                }
                for item in raw.list_path
            ]
        if raw.notes:
            material["notes"] = [
                {
                    "note_id": note.note_id,
                    "locator": _locator_to_document(note.locator),
                    "marker": note.marker,
                    "body_payload": note.body_payload,
                    "owner_locator": _locator_to_document(
                        note.owner_locator
                    ),
                    "anchor": dict(note.anchor),
                }
                for note in raw.notes
            ]
        if (
            raw.presentation_roles
            or raw.presentation_fields
            or raw.figure_presentation
            or raw.table_presentation
            or raw.caption_presentation
        ):
            presentation_material = {
                "roles": list(raw.presentation_roles),
                "fields": [dict(field) for field in raw.presentation_fields],
                "table": (
                    dict(raw.table_presentation)
                    if raw.table_presentation is not None
                    else None
                ),
                "caption": (
                    dict(raw.caption_presentation)
                    if raw.caption_presentation is not None
                    else None
                ),
                "outline_heading": raw.outline_heading,
            }
            if raw.figure_presentation is not None:
                presentation_material["figure"] = dict(
                    raw.figure_presentation
                )
            material["source_presentation"] = presentation_material
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
                list_path=raw.list_path,
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
    target_candidates = []
    for raw, block in zip(raw_blocks, blocks, strict=True):
        alias = block.locator.source_id
        if not alias:
            continue
        target_candidates.append(
            {
                "alias": alias,
                "selector": block.locator.selector,
                "kind": block.kind.value,
                "block_id": block.block_id,
                "block_start": block.ordinal,
                "block_end": block.ordinal + 1,
                "section_id": "",
                "panels": [dict(panel) for panel in raw.target_panels],
            }
        )
    sections_by_range: dict[tuple[int, int], list[RichSection]] = {}
    for section in sections:
        sections_by_range.setdefault(
            (section.block_start, section.block_end), []
        ).append(section)
    for source_section in html_sections:
        matches = sections_by_range.get(
            (source_section.block_start, source_section.block_end),
            [],
        )
        if len(matches) != 1 or source_section.block_start >= len(blocks):
            continue
        section = matches[0]
        block = blocks[source_section.block_start]
        if block.kind is not RichBlockKind.HEADING:
            continue
        target_candidates.append(
            {
                "alias": source_section.alias,
                "selector": source_section.selector,
                "kind": "section",
                "block_id": block.block_id,
                "block_start": source_section.block_start,
                "block_end": source_section.block_end,
                "section_id": section.section_id,
                "panels": [],
            }
        )
    for source_target in html_block_targets:
        if not 0 <= source_target.block_index < len(blocks):
            continue
        block = blocks[source_target.block_index]
        target_candidates.append(
            {
                "alias": source_target.alias,
                "selector": source_target.selector,
                "kind": block.kind.value,
                "block_id": block.block_id,
                "block_start": block.ordinal,
                "block_end": block.ordinal + 1,
                "section_id": "",
                "panels": [],
            }
        )
    alias_counts: dict[str, int] = {}
    for target in target_candidates:
        alias = str(target["alias"])
        alias_counts[alias] = alias_counts.get(alias, 0) + 1
    targets = [
        target
        for target in target_candidates
        if alias_counts[str(target["alias"])] == 1
    ]
    targets.sort(
        key=lambda target: (
            int(target["block_start"]),
            0 if target["kind"] == "section" else 1,
            str(target["alias"]),
        )
    )
    metadata_value = dict(metadata)
    if html_front_matter:
        metadata_value[SOURCE_FRONT_MATTER_METADATA_KEY] = {
            "schema_version": SOURCE_FRONT_MATTER_SCHEMA,
            "entries": [dict(entry) for entry in html_front_matter],
        }
    source_notes = []
    for raw, block in zip(raw_blocks, blocks, strict=True):
        for note in raw.notes:
            source_notes.append(
                {
                    "note_id": note.note_id,
                    "ordinal": len(source_notes),
                    "marker": note.marker,
                    "body": note.body_payload["text"],
                    "inline_spans": note.body_payload["inline_spans"],
                    "locator": _locator_to_document(note.locator),
                    "owner_block_id": block.block_id,
                    "owner_locator": _locator_to_document(
                        note.owner_locator
                    ),
                    "anchor": dict(note.anchor),
                }
            )
    if source_notes:
        metadata_value[SOURCE_NOTES_METADATA_KEY] = {
            "schema_version": SOURCE_NOTES_SCHEMA,
            "notes": source_notes,
        }
    if artifact.source_format is SourceFormat.HTML:
        figure_target_block_ids = {
            str(target["block_id"])
            for target in targets
            if target["kind"] == "figure" and target["panels"]
        }
        metadata_value[SOURCE_PRESENTATION_METADATA_KEY] = {
            "schema_version": SOURCE_PRESENTATION_SCHEMA,
            "blocks": [
                {
                    "block_id": block.block_id,
                    "roles": list(raw.presentation_roles),
                    "fields": [dict(field) for field in raw.presentation_fields],
                }
                for raw, block in zip(raw_blocks, blocks, strict=True)
                if raw.kind
                in {
                    RichBlockKind.HEADING,
                    RichBlockKind.PARAGRAPH,
                    RichBlockKind.LIST,
                    RichBlockKind.TABLE,
                    RichBlockKind.FIGURE,
                }
            ],
            "classifications": [
                {
                    "classification_id": relation.classification_id,
                    "locator": _locator_to_document(relation.locator),
                    "heading_block_id": blocks[
                        relation.heading_block_index
                    ].block_id,
                    "value_block_ids": [
                        blocks[index].block_id
                        for index in relation.value_block_indexes
                    ],
                    "composition": "inline",
                    "separator": ": ",
                    "separator_source": (
                        "latexml_ar5iv_classification_after"
                    ),
                }
                for relation in html_classifications
            ],
            "captions": [
                {
                    "block_id": block.block_id,
                    "kind": raw.caption_presentation["kind"],
                    "placement": raw.caption_presentation["placement"],
                    "alignment": raw.caption_presentation["alignment"],
                    "alignment_sources": list(
                        raw.caption_presentation["alignment_sources"]
                    ),
                }
                for raw, block in zip(raw_blocks, blocks, strict=True)
                if raw.caption_presentation is not None
            ],
            "figures": [
                {
                    "block_id": block.block_id,
                    "layout": dict(raw.figure_presentation["layout"]),
                    "panels": [
                        dict(panel)
                        for panel in raw.figure_presentation["panels"]
                    ],
                }
                for raw, block in zip(raw_blocks, blocks, strict=True)
                if raw.kind is RichBlockKind.FIGURE
                and raw.figure_presentation is not None
                and block.block_id in figure_target_block_ids
            ],
            "tables": [
                {
                    "block_id": block.block_id,
                    "cells": [
                        dict(cell)
                        for cell in raw.table_presentation["cells"]
                    ],
                }
                for raw, block in zip(raw_blocks, blocks, strict=True)
                if raw.kind is RichBlockKind.TABLE
                and raw.table_presentation is not None
            ],
        }
        metadata_value[SOURCE_TARGET_MANIFEST_METADATA_KEY] = {
            "schema_version": SOURCE_TARGET_MANIFEST_SCHEMA,
            "targets": targets,
        }
    return RichDocument(
        source=artifact,
        blocks=tuple(blocks),
        sections=tuple(sections),
        assets=assets,
        metadata=metadata_value,
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
