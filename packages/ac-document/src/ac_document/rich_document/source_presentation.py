"""Validated source-authored flow, inline, caption, Figure, and Table metadata."""

from __future__ import annotations

import re
from math import gcd
from collections.abc import Mapping, Sequence
from typing import Any

from .source_targets import SOURCE_TARGET_MANIFEST_METADATA_KEY


SOURCE_PRESENTATION_METADATA_KEY = "source_presentation"
SOURCE_PRESENTATION_SCHEMA = "ac.document.source_presentation.v1"

_ENVELOPE_FIELDS = {
    "schema_version",
    "blocks",
    "classifications",
    "captions",
    "figures",
    "tables",
}
_BLOCK_FIELDS = {"block_id", "roles", "fields"}
_VIEW_FIELDS = {
    "field",
    "item_index",
    "row_index",
    "column_index",
    "text",
    "inline_spans",
    "marks",
}
_MARK_FIELDS = {"kind", "start", "end"}
_CLASSIFICATION_FIELDS = {
    "classification_id",
    "locator",
    "heading_block_id",
    "value_block_ids",
    "composition",
    "separator",
    "separator_source",
}
_CAPTION_FIELDS = {
    "block_id",
    "kind",
    "placement",
    "alignment",
    "alignment_sources",
}
_FIGURE_FIELDS = {"block_id", "layout", "panels"}
_FIGURE_LAYOUT_FIELDS = {
    "kind",
    "column_count",
    "row_count",
    "rows",
    "column_source",
    "row_sources",
    "break_after_panel_indexes",
    "break_source",
}
_FIGURE_PANEL_FIELDS = {
    "panel_index",
    "source_id",
    "row_index",
    "column_index",
    "display_width",
    "display_height",
    "dimension_source",
    "aspect_ratio",
    "aspect_ratio_source",
}
_TABLE_FIELDS = {"block_id", "cells"}
_CELL_FIELDS = {
    "row_index",
    "column_index",
    "row_span",
    "column_span",
    "kind",
    "locator",
    "horizontal_alignment",
    "horizontal_alignment_sources",
    "rule_edges",
}
_RULE_EDGE_FIELDS = {"edge", "source"}
_LOCATOR_FIELDS = {
    "source_format",
    "line_start",
    "column_start",
    "line_end",
    "column_end",
    "selector",
    "source_id",
}
_INLINE_FIELDS = {
    "text": {"kind", "start", "end", "text"},
    "link": {"kind", "start", "end", "text", "target"},
    "math": {"kind", "start", "end", "text", "tex", "source"},
}
_ROLES = {"abstract", "classification", "acknowledgements"}
_MARK_KINDS = {"strong", "emphasis"}
_CAPTION_PLACEMENTS = {"before_content", "after_content", "embedded"}
_CAPTION_ALIGNMENTS = {"start", "center", "end"}
_CAPTION_ALIGNMENT_SOURCES = {
    "class:ltx_centering": "center",
    "class:ltx_align_center": "center",
    "style:text-align:start": "start",
    "style:text-align:center": "center",
    "style:text-align:end": "end",
}
_TABLE_ALIGNMENTS = {"left", "center", "right", "start", "end"}
_TABLE_ALIGNMENT_SOURCES = {
    "class:ltx_align_left": "left",
    "class:ltx_align_center": "center",
    "class:ltx_align_right": "right",
    "style:text-align:left": "left",
    "style:text-align:center": "center",
    "style:text-align:right": "right",
    "style:text-align:start": "start",
    "style:text-align:end": "end",
}
_TABLE_RULE_SOURCES = {
    "class:ltx_border_t": "top",
    "class:ltx_border_tt": "top",
    "class:ltx_border_T": "top",
    "class:ltx_border_r": "right",
    "class:ltx_border_rr": "right",
    "class:ltx_border_R": "right",
    "class:ltx_border_r_dashed": "right",
    "class:ltx_border_b": "bottom",
    "class:ltx_border_bb": "bottom",
    "class:ltx_border_B": "bottom",
    "class:ltx_border_b_dashed": "bottom",
    "class:ltx_border_l": "left",
    "class:ltx_border_ll": "left",
    "class:ltx_border_L": "left",
}
_TABLE_RULE_EDGE_ORDER = {"top": 0, "right": 1, "bottom": 2, "left": 3}
_TABLE_COVERAGE_LIMIT = 65_536
_FIGURE_COLUMN_SOURCES = {
    "latexml_ar5iv_direct_graphic": ("single", 1),
    "class:ltx_flex_size_1": ("flex", 1),
    "class:ltx_flex_size_2": ("flex", 2),
    "class:ltx_flex_size_3": ("flex", 3),
}
_FIGURE_BREAK_SOURCE = "class:ltx_flex_break"
_FIGURE_DIMENSION_SOURCE = "attributes:width,height"
_FIGURE_ASPECT_RATIO_SOURCE = "style:aspect-ratio"
_FIGURE_DIMENSION_LIMIT = 1_000_000
_CLASSIFICATION_ID_RE = re.compile(r"classification-[0-9a-f]{24}")


def source_presentation(document: Any) -> Mapping[str, Any] | None:
    """Return authoritative authored presentation metadata when present."""

    if SOURCE_PRESENTATION_METADATA_KEY not in document.metadata:
        return None
    value = document.metadata[SOURCE_PRESENTATION_METADATA_KEY]
    validate_source_presentation_metadata(
        document.metadata,
        blocks=document.blocks,
        source=document.source,
    )
    return value


def validate_source_presentation_metadata(
    metadata: Mapping[str, Any],
    *,
    blocks: Sequence[Any],
    source: Any,
) -> None:
    """Validate present presentation metadata against plain RichBlock fields."""

    if SOURCE_PRESENTATION_METADATA_KEY not in metadata:
        return
    raw = metadata[SOURCE_PRESENTATION_METADATA_KEY]
    envelope = _mapping(raw)
    _require_fields(envelope, _ENVELOPE_FIELDS)
    if envelope.get("schema_version") != SOURCE_PRESENTATION_SCHEMA:
        _invalid("has an unsupported schema")

    block_by_id = {block.block_id: block for block in blocks}
    expected_blocks = [
        block for block in blocks if _expected_view_keys(block)
    ]
    block_entries = _sequence(envelope.get("blocks"))
    if len(block_entries) != len(expected_blocks):
        _invalid("does not cover every presentable block")
    seen_block_ids: set[str] = set()
    for expected, raw_entry in zip(expected_blocks, block_entries, strict=True):
        entry = _mapping(raw_entry)
        _require_fields(entry, _BLOCK_FIELDS)
        block_id = _nonempty_string(entry, "block_id")
        if block_id in seen_block_ids or block_id != expected.block_id:
            _invalid("block identities are duplicate or out of source order")
        seen_block_ids.add(block_id)
        roles = _sequence(entry.get("roles"))
        if any(not isinstance(role, str) or role not in _ROLES for role in roles):
            _invalid("contains an unknown block role")
        if len(set(roles)) != len(roles):
            _invalid("contains duplicate block roles")
        if len(roles) > 1:
            _invalid("contains conflicting block roles")
        if roles and _kind(expected) != "heading":
            _invalid("assigns a heading role to a non-heading block")
        role_values = tuple(roles)
        if role_values == ("abstract",) and expected.payload.get("level") != 2:
            _invalid("binds the abstract role to a non-semantic heading level")
        if role_values == (
            "acknowledgements",
        ) and not _is_semantic_child_level(
            expected.payload.get("level")
        ):
            _invalid(
                "binds the acknowledgements role to a non-semantic heading level"
            )
        fields = _sequence(entry.get("fields"))
        expected_keys = _expected_view_keys(expected)
        actual_keys: list[tuple[str, int | None, int | None, int | None]] = []
        for raw_view in fields:
            view = _mapping(raw_view)
            _require_fields(view, _VIEW_FIELDS)
            key = _view_key(view)
            actual_keys.append(key)
            plain = _plain_field(expected, key)
            text = _string(view, "text")
            if text != plain:
                _invalid("rich field does not match its plain block field")
            _validate_inline_spans(view.get("inline_spans"), text=text)
            _validate_marks(view.get("marks"), text=text)
        if actual_keys != expected_keys or len(set(actual_keys)) != len(actual_keys):
            _invalid("rich fields are missing, duplicate, or out of order")

    _validate_classifications(
        envelope.get("classifications"),
        block_by_id=block_by_id,
        block_entries=block_entries,
        source_format=source.source_format.value,
    )
    _validate_figures(
        envelope.get("figures"),
        metadata=metadata,
        block_by_id=block_by_id,
    )

    table_blocks = [block for block in blocks if _kind(block) == "table"]
    table_entries = _sequence(envelope.get("tables"))
    if len(table_entries) != len(table_blocks):
        _invalid("does not cover every Table block")
    authored_ids = {
        block.locator.source_id for block in blocks if block.locator.source_id
    }
    for block, raw_entry in zip(table_blocks, table_entries, strict=True):
        entry = _mapping(raw_entry)
        _require_fields(entry, _TABLE_FIELDS)
        if _nonempty_string(entry, "block_id") != block.block_id:
            _invalid("Table identities are duplicate or out of source order")
        _validate_table_cells(
            entry.get("cells"),
            block=block,
            source_format=source.source_format.value,
            authored_ids=authored_ids,
        )
    _validate_captions(
        envelope.get("captions"),
        blocks=blocks,
        block_by_id=block_by_id,
    )
    if seen_block_ids != set(block_by_id) - {
        block.block_id for block in blocks if not _expected_view_keys(block)
    }:
        _invalid("references an unknown or missing block")


def _validate_classifications(
    value: Any,
    *,
    block_by_id: Mapping[str, Any],
    block_entries: Sequence[Any],
    source_format: str,
) -> None:
    relations = _sequence(value)
    roles_by_block_id = {
        _string(_mapping(entry), "block_id"): tuple(
            _sequence(_mapping(entry).get("roles"))
        )
        for entry in block_entries
    }
    identities: set[str] = set()
    claimed_blocks: set[str] = set()
    previous_heading_ordinal = -1
    for raw in relations:
        relation = _mapping(raw)
        _require_fields(relation, _CLASSIFICATION_FIELDS)
        identity = _nonempty_string(relation, "classification_id")
        if (
            _CLASSIFICATION_ID_RE.fullmatch(identity) is None
            or identity in identities
        ):
            _invalid("classification identity is invalid or duplicate")
        identities.add(identity)
        locator = _mapping(relation.get("locator"))
        _require_fields(locator, _LOCATOR_FIELDS)
        if locator.get("source_format") != source_format:
            _invalid("classification locator format differs from its source")
        _validate_locator(locator, description="classification")
        heading_id = _nonempty_string(relation, "heading_block_id")
        heading = block_by_id.get(heading_id)
        if (
            heading is None
            or _kind(heading) != "heading"
            or roles_by_block_id.get(heading_id) != ("classification",)
        ):
            _invalid("classification heading binding is invalid")
        value_ids = tuple(
            _nonempty_sequence_string(item, "classification value block ID")
            for item in _sequence(relation.get("value_block_ids"))
        )
        if not value_ids or len(set(value_ids)) != len(value_ids):
            _invalid("classification values are empty or duplicate")
        values = [block_by_id.get(block_id) for block_id in value_ids]
        if any(block is None or _kind(block) == "heading" for block in values):
            _invalid("classification value binding is invalid")
        ordinals = [heading.ordinal, *(block.ordinal for block in values)]
        if (
            heading.ordinal <= previous_heading_ordinal
            or ordinals != list(range(heading.ordinal, heading.ordinal + len(ordinals)))
            or any(block.section_path != heading.section_path for block in values)
        ):
            _invalid("classification flow is out of source order")
        previous_heading_ordinal = heading.ordinal
        relation_blocks = {heading_id, *value_ids}
        if relation_blocks & claimed_blocks:
            _invalid("classification flows reuse a block")
        claimed_blocks.update(relation_blocks)
        if _string(relation, "composition") != "inline":
            _invalid("classification composition is unsupported")
        if (
            _string(relation, "separator") != ": "
            or _string(relation, "separator_source")
            != "latexml_ar5iv_classification_after"
        ):
            _invalid("classification separator provenance is invalid")


def _validate_captions(
    value: Any,
    *,
    blocks: Sequence[Any],
    block_by_id: Mapping[str, Any],
) -> None:
    captions = _sequence(value)
    seen: set[str] = set()
    previous_ordinal = -1
    for raw in captions:
        caption = _mapping(raw)
        _require_fields(caption, _CAPTION_FIELDS)
        block_id = _nonempty_string(caption, "block_id")
        block = block_by_id.get(block_id)
        kind = _string(caption, "kind")
        placement = _string(caption, "placement")
        if (
            block is None
            or block_id in seen
            or _kind(block) != kind
            or kind not in {"figure", "table"}
            or not block.payload["caption"]
            or block.ordinal <= previous_ordinal
            or placement not in _CAPTION_PLACEMENTS
            or (kind == "figure" and placement == "embedded")
        ):
            _invalid("caption identity, kind, placement, or order is invalid")
        seen.add(block_id)
        previous_ordinal = block.ordinal
        alignment = caption.get("alignment")
        sources = tuple(
            _nonempty_sequence_string(item, "caption alignment source")
            for item in _sequence(caption.get("alignment_sources"))
        )
        if alignment is None:
            if sources:
                _invalid("neutral caption alignment has source evidence")
        elif not isinstance(alignment, str) or alignment not in _CAPTION_ALIGNMENTS:
            _invalid("caption alignment is unknown")
        elif (
            not sources
            or len(set(sources)) != len(sources)
            or any(
                _CAPTION_ALIGNMENT_SOURCES.get(source) != alignment
                for source in sources
            )
        ):
            _invalid("caption alignment evidence conflicts")
    required = {
        block.block_id
        for block in blocks
        if _kind(block) in {"figure", "table"} and block.payload["caption"]
    }
    if required != seen:
        _invalid("caption registry differs from visible captions")


def _validate_figures(
    value: Any,
    *,
    metadata: Mapping[str, Any],
    block_by_id: Mapping[str, Any],
) -> None:
    targets = _figure_panel_targets(metadata)
    expected = sorted(
        targets.items(),
        key=lambda item: block_by_id[item[0]].ordinal
        if item[0] in block_by_id
        else -1,
    )
    figures = _sequence(value)
    if len(figures) != len(expected):
        _invalid("Figure layout registry does not cover exact panel targets")
    seen: set[str] = set()
    for (expected_block_id, target_panels), raw in zip(
        expected,
        figures,
        strict=True,
    ):
        figure = _mapping(raw)
        _require_fields(figure, _FIGURE_FIELDS)
        block_id = _nonempty_string(figure, "block_id")
        block = block_by_id.get(block_id)
        if (
            block is None
            or _kind(block) != "figure"
            or block_id != expected_block_id
            or block_id in seen
        ):
            _invalid("Figure layout identity is duplicate or out of source order")
        seen.add(block_id)
        panels = _sequence(figure.get("panels"))
        if not panels or len(panels) != len(target_panels):
            _invalid("Figure layout panel coverage differs from its target")
        _validate_figure_layout(
            figure.get("layout"),
            panels=panels,
            target_panels=target_panels,
        )


def _figure_panel_targets(
    metadata: Mapping[str, Any],
) -> dict[str, Sequence[Any]]:
    raw_manifest = metadata.get(SOURCE_TARGET_MANIFEST_METADATA_KEY)
    if raw_manifest is None:
        return {}
    manifest = _mapping(raw_manifest)
    targets = _sequence(manifest.get("targets"))
    output: dict[str, Sequence[Any]] = {}
    for raw_target in targets:
        target = _mapping(raw_target)
        if target.get("kind") != "figure":
            continue
        panels = _sequence(target.get("panels"))
        if not panels:
            continue
        block_id = _nonempty_string(target, "block_id")
        if block_id in output:
            _invalid("Figure panel targets reuse a canonical block")
        output[block_id] = panels
    return output


def _validate_figure_layout(
    value: Any,
    *,
    panels: Sequence[Any],
    target_panels: Sequence[Any],
) -> None:
    layout = _mapping(value)
    _require_fields(layout, _FIGURE_LAYOUT_FIELDS)
    kind = _string(layout, "kind")
    column_count = layout.get("column_count")
    row_count = layout.get("row_count")
    rows = _sequence(layout.get("rows"))
    column_source = layout.get("column_source")
    row_sources = _sequence(layout.get("row_sources"))
    breaks = _sequence(layout.get("break_after_panel_indexes"))
    break_source = layout.get("break_source")

    if kind == "neutral":
        if (
            column_count is not None
            or row_count is not None
            or rows
            or column_source is not None
            or row_sources
            or breaks
            or break_source is not None
        ):
            _invalid("neutral Figure layout claims authored geometry")
        expected_positions: dict[int, tuple[int, int]] = {}
    else:
        if (
            not isinstance(column_count, int)
            or isinstance(column_count, bool)
            or column_count < 1
            or not isinstance(row_count, int)
            or isinstance(row_count, bool)
            or row_count < 1
            or row_count != len(rows)
            or row_count != len(row_sources)
        ):
            _invalid("Figure layout row or column count is invalid")
        normalized_rows: list[tuple[int, ...]] = []
        flattened: list[int] = []
        normalized_sources: list[str] = []
        row_capacities: list[int] = []
        for raw_row, raw_source in zip(rows, row_sources, strict=True):
            row = tuple(
                _nonnegative_integer(item, "Figure layout panel row index")
                for item in _sequence(raw_row)
            )
            if not isinstance(raw_source, str):
                _invalid("Figure layout row provenance is missing")
            source_contract = _FIGURE_COLUMN_SOURCES.get(raw_source)
            if source_contract is None or source_contract[0] != kind:
                _invalid("Figure layout row provenance conflicts")
            capacity = source_contract[1]
            if not row or len(row) > capacity:
                _invalid("Figure layout row is empty or exceeds its source columns")
            normalized_rows.append(row)
            normalized_sources.append(raw_source)
            row_capacities.append(capacity)
            flattened.extend(row)
        if column_count != max(row_capacities):
            _invalid("Figure layout column count differs from its rows")
        unique_sources = set(normalized_sources)
        expected_column_source = (
            normalized_sources[0] if len(unique_sources) == 1 else None
        )
        if column_source != expected_column_source:
            _invalid("Figure layout column provenance conflicts")
        if flattened != list(range(len(panels))):
            _invalid("Figure layout rows do not preserve complete panel order")
        expected_positions = {
            panel_index: (row_index, column_index)
            for row_index, row in enumerate(normalized_rows)
            for column_index, panel_index in enumerate(row)
        }
        normalized_breaks = [
            _nonnegative_integer(item, "Figure layout break index")
            for item in breaks
        ]
        row_ends = {row[-1] for row in normalized_rows[:-1]}
        if (
            normalized_breaks != sorted(set(normalized_breaks))
            or any(index not in row_ends for index in normalized_breaks)
            or any(index >= len(panels) - 1 for index in normalized_breaks)
        ):
            _invalid("Figure layout breaks are duplicate or out of row bounds")
        if normalized_breaks:
            if break_source != _FIGURE_BREAK_SOURCE:
                _invalid("Figure layout break provenance conflicts")
        elif break_source is not None:
            _invalid("Figure layout with no breaks claims break provenance")
        if kind == "single" and (
            len(panels) != 1 or normalized_rows != [(0,)] or breaks
        ):
            _invalid("single Figure layout does not contain exactly one panel")

    occupied: set[tuple[int, int]] = set()
    for index, (raw_panel, raw_target) in enumerate(
        zip(panels, target_panels, strict=True)
    ):
        panel = _mapping(raw_panel)
        target = _mapping(raw_target)
        _require_fields(panel, _FIGURE_PANEL_FIELDS)
        panel_index = _integer(panel, "panel_index")
        source_id = _string(panel, "source_id")
        if (
            panel_index != index
            or _integer(target, "panel_index") != index
            or source_id != _string(target, "source_id")
        ):
            _invalid("Figure layout panel binding differs from its target")
        row_index = panel.get("row_index")
        column_index = panel.get("column_index")
        if kind == "neutral":
            if row_index is not None or column_index is not None:
                _invalid("neutral Figure panel claims a grid position")
        else:
            row = _nonnegative_integer(row_index, "Figure panel row index")
            column = _nonnegative_integer(
                column_index,
                "Figure panel column index",
            )
            if (
                expected_positions.get(index) != (row, column)
                or row >= row_count
                or column >= column_count
                or (row, column) in occupied
            ):
                _invalid("Figure panel position overlaps or is out of bounds")
            occupied.add((row, column))
        _validate_figure_panel_dimensions(panel, neutral=kind == "neutral")


def _validate_figure_panel_dimensions(
    panel: Mapping[str, Any],
    *,
    neutral: bool,
) -> None:
    width = _optional_bounded_positive_integer(
        panel.get("display_width"),
        "Figure panel display width",
    )
    height = _optional_bounded_positive_integer(
        panel.get("display_height"),
        "Figure panel display height",
    )
    dimension_source = panel.get("dimension_source")
    if width is None and height is None:
        if dimension_source is not None:
            _invalid("Figure panel neutral dimensions claim provenance")
    elif (
        width is None
        or height is None
        or dimension_source != _FIGURE_DIMENSION_SOURCE
    ):
        _invalid("Figure panel dimensions or provenance are incomplete")

    raw_ratio = panel.get("aspect_ratio")
    ratio_source = panel.get("aspect_ratio_source")
    ratio: tuple[int, int] | None = None
    if raw_ratio is None:
        if ratio_source is not None:
            _invalid("Figure panel neutral aspect ratio claims provenance")
    else:
        values = _sequence(raw_ratio)
        if len(values) != 2:
            _invalid("Figure panel aspect ratio is invalid")
        numerator = _bounded_positive_integer(
            values[0],
            "Figure panel aspect numerator",
        )
        denominator = _bounded_positive_integer(
            values[1],
            "Figure panel aspect denominator",
        )
        ratio = (numerator, denominator)
        if gcd(numerator, denominator) != 1:
            _invalid("Figure panel aspect ratio is not normalized")
        if ratio_source != _FIGURE_ASPECT_RATIO_SOURCE:
            _invalid("Figure panel aspect-ratio provenance conflicts")
    if width is not None and ratio is not None:
        divisor = gcd(width, height)
        if (width // divisor, height // divisor) != ratio:
            _invalid("Figure panel aspect ratio differs from dimensions")
    if neutral and any(
        value is not None
        for value in (
            width,
            height,
            dimension_source,
            ratio,
            ratio_source,
        )
    ):
        _invalid("neutral Figure panel claims source display geometry")


def _expected_view_keys(
    block: Any,
) -> list[tuple[str, int | None, int | None, int | None]]:
    kind = _kind(block)
    if kind in {"heading", "paragraph"}:
        return [("text", None, None, None)]
    if kind == "list":
        return [
            ("list_item", index, None, None)
            for index, _item in enumerate(block.payload["items"])
        ]
    if kind == "figure":
        return [("caption", None, None, None)]
    if kind == "table":
        return [
            ("caption", None, None, None),
            *[
                ("table_header", None, None, column)
                for column, _value in enumerate(block.payload["headers"])
            ],
            *[
                ("table_cell", None, row, column)
                for row, values in enumerate(block.payload["rows"])
                for column, _value in enumerate(values)
            ],
        ]
    return []


def _is_semantic_child_level(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 2 <= value <= 6


def _plain_field(
    block: Any,
    key: tuple[str, int | None, int | None, int | None],
) -> str:
    field, item_index, row_index, column_index = key
    if field == "text":
        return str(block.payload["text"])
    if field == "list_item" and item_index is not None:
        return str(block.payload["items"][item_index]["text"])
    if field == "caption":
        return str(block.payload["caption"])
    if field == "table_header" and column_index is not None:
        return str(block.payload["headers"][column_index])
    if field == "table_cell" and row_index is not None and column_index is not None:
        return str(block.payload["rows"][row_index][column_index])
    _invalid("rich field does not match its block kind")


def _view_key(
    view: Mapping[str, Any],
) -> tuple[str, int | None, int | None, int | None]:
    field = _string(view, "field")
    if field not in {"text", "list_item", "caption", "table_header", "table_cell"}:
        _invalid("contains an unknown rich field")
    indices = tuple(
        _optional_index(view.get(key))
        for key in ("item_index", "row_index", "column_index")
    )
    item_index, row_index, column_index = indices
    valid = (
        field in {"text", "caption"} and indices == (None, None, None)
        or (
            field == "list_item"
            and item_index is not None
            and row_index is None
            and column_index is None
        )
        or (
            field == "table_header"
            and item_index is None
            and row_index is None
            and column_index is not None
        )
        or (
            field == "table_cell"
            and item_index is None
            and row_index is not None
            and column_index is not None
        )
    )
    if not valid:
        _invalid("rich field indices differ from its field")
    return field, item_index, row_index, column_index


def _validate_inline_spans(value: Any, *, text: str) -> None:
    spans = _sequence(value)
    cursor = 0
    reconstructed: list[str] = []
    for raw in spans:
        span = _mapping(raw)
        kind = span.get("kind")
        if kind not in _INLINE_FIELDS:
            _invalid("inline span kind is unknown")
        _require_fields(span, _INLINE_FIELDS[str(kind)])
        start = _integer(span, "start")
        end = _integer(span, "end")
        span_text = _string(span, "text")
        if start != cursor or end <= start or end - start != len(span_text):
            _invalid("inline spans are not a contiguous reconstruction")
        if kind == "link" and not _nonempty_string(span, "target"):
            _invalid("inline link target is empty")
        if kind == "math":
            _nonempty_string(span, "tex")
            _nonempty_string(span, "source")
        reconstructed.append(span_text)
        cursor = end
    if "".join(reconstructed) != text:
        _invalid("inline spans do not reconstruct the rich field")


def _validate_marks(value: Any, *, text: str) -> None:
    marks = _sequence(value)
    previous: tuple[int, int, str] | None = None
    seen: set[tuple[str, int, int]] = set()
    for raw in marks:
        mark = _mapping(raw)
        _require_fields(mark, _MARK_FIELDS)
        kind = _string(mark, "kind")
        start = _integer(mark, "start")
        end = _integer(mark, "end")
        if kind not in _MARK_KINDS or not 0 <= start < end <= len(text):
            _invalid("authored mark is unknown or out of bounds")
        identity = (kind, start, end)
        order = (start, -end, kind)
        if identity in seen or (previous is not None and order < previous):
            _invalid("authored marks are duplicate or out of source order")
        seen.add(identity)
        previous = order


def _validate_table_cells(
    value: Any,
    *,
    block: Any,
    source_format: str,
    authored_ids: set[str],
) -> None:
    cells = _sequence(value)
    headers = block.payload["headers"]
    rows = block.payload["rows"]
    height = (1 if headers else 0) + len(rows)
    width = max([len(headers), *(len(row) for row in rows)], default=0)
    occupied: set[tuple[int, int]] = set()
    origins: set[tuple[int, int]] = set()
    previous = (-1, -1)
    coverage_area = 0
    for raw in cells:
        cell = _mapping(raw)
        _require_fields(cell, _CELL_FIELDS)
        row = _integer(cell, "row_index")
        column = _integer(cell, "column_index")
        row_span = _integer(cell, "row_span")
        column_span = _integer(cell, "column_span")
        kind = _string(cell, "kind")
        if (
            row < 0
            or column < 0
            or row_span < 1
            or column_span < 1
            or row + row_span > height
            or column + column_span > width
            or kind not in {"header", "data"}
            or (row, column) <= previous
        ):
            _invalid("Table cell geometry is invalid or out of source order")
        previous = (row, column)
        _validate_table_cell_presentation(cell)
        cell_area = row_span * column_span
        if cell_area > _TABLE_COVERAGE_LIMIT - coverage_area:
            _invalid("Table cell coverage exceeds its aggregate limit")
        coverage_area += cell_area
        origin = (row, column)
        if origin in origins:
            _invalid("Table contains a duplicate cell origin")
        origins.add(origin)
        for grid_row in range(row, row + row_span):
            for grid_column in range(column, column + column_span):
                point = (grid_row, grid_column)
                if point in occupied:
                    _invalid("Table cell geometry overlaps")
                occupied.add(point)
        locator = _mapping(cell.get("locator"))
        _require_fields(locator, _LOCATOR_FIELDS)
        if locator.get("source_format") != source_format:
            _invalid("Table cell locator format differs from its source")
        _validate_locator(locator, description="Table cell")
        source_id = _string(locator, "source_id")
        if source_id:
            if source_id in authored_ids:
                _invalid("Table contains a duplicate authored cell ID")
            authored_ids.add(source_id)
    for row in range(height):
        values = headers if headers and row == 0 else rows[row - (1 if headers else 0)]
        for column, text in enumerate(values):
            if text and (row, column) not in origins:
                _invalid("Table rich geometry omits a nonempty cell origin")
            if (row, column) in occupied and (row, column) not in origins and text:
                _invalid("Table span-covered plain cell is not empty")


def _validate_table_cell_presentation(cell: Mapping[str, Any]) -> None:
    alignment = cell.get("horizontal_alignment")
    alignment_sources = tuple(
        _nonempty_sequence_string(item, "Table cell alignment source")
        for item in _sequence(cell.get("horizontal_alignment_sources"))
    )
    if alignment is None:
        if alignment_sources:
            _invalid("neutral Table cell alignment has source evidence")
    elif not isinstance(alignment, str) or alignment not in _TABLE_ALIGNMENTS:
        _invalid("Table cell alignment is unknown")
    elif (
        not alignment_sources
        or len(set(alignment_sources)) != len(alignment_sources)
        or any(
            _TABLE_ALIGNMENT_SOURCES.get(source) != alignment
            for source in alignment_sources
        )
    ):
        _invalid("Table cell alignment evidence conflicts")

    previous_edge_order = -1
    seen_edges: set[str] = set()
    for raw in _sequence(cell.get("rule_edges")):
        rule = _mapping(raw)
        _require_fields(rule, _RULE_EDGE_FIELDS)
        edge = _string(rule, "edge")
        source = _nonempty_string(rule, "source")
        edge_order = _TABLE_RULE_EDGE_ORDER.get(edge)
        if (
            edge_order is None
            or edge in seen_edges
            or edge_order <= previous_edge_order
            or _TABLE_RULE_SOURCES.get(source) != edge
        ):
            _invalid("Table cell rule edge is unknown, duplicate, or conflicting")
        seen_edges.add(edge)
        previous_edge_order = edge_order


def _validate_locator(
    locator: Mapping[str, Any],
    *,
    description: str,
) -> None:
    positions = [
        locator.get("line_start"),
        locator.get("column_start"),
        locator.get("line_end"),
        locator.get("column_end"),
    ]
    if any(
        item is not None
        and (not isinstance(item, int) or isinstance(item, bool) or item < 1)
        for item in positions
    ):
        _invalid(f"{description} locator positions are invalid")
    if (positions[0] is None) != (positions[2] is None) or (
        positions[1] is None
    ) != (positions[3] is None):
        _invalid(f"{description} locator range is incomplete")
    if positions[1] is not None and positions[0] is None:
        _invalid(f"{description} locator columns lack a line range")
    if positions[0] is not None and positions[2] < positions[0]:
        _invalid(f"{description} locator range is reversed")
    if (
        positions[0] == positions[2]
        and positions[1] is not None
        and positions[3] < positions[1]
    ):
        _invalid(f"{description} locator range is reversed")
    source_id = _string(locator, "source_id")
    selector = _string(locator, "selector")
    if source_id and selector != f"#{source_id}":
        _invalid(f"{description} selector differs from its source ID")
    if not source_id and selector:
        _invalid(f"{description} selector has no source ID")


def _kind(block: Any) -> str:
    return str(getattr(block.kind, "value", block.kind))


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _invalid("value must be an object")
    return value


def _sequence(value: Any) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        _invalid("value must be a list")
    return value


def _require_fields(value: Mapping[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        _invalid("has invalid fields")


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        _invalid(f"{key} must be a string")
    return item


def _nonempty_string(value: Mapping[str, Any], key: str) -> str:
    item = _string(value, key)
    if not item:
        _invalid(f"{key} is empty")
    return item


def _nonempty_sequence_string(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value:
        _invalid(f"{description} is empty or not a string")
    return value


def _integer(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        _invalid(f"{key} must be an integer")
    return item


def _nonnegative_integer(value: Any, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _invalid(f"{description} is invalid")
    return value


def _bounded_positive_integer(value: Any, description: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= _FIGURE_DIMENSION_LIMIT
    ):
        _invalid(f"{description} is invalid or out of bounds")
    return value


def _optional_bounded_positive_integer(
    value: Any,
    description: str,
) -> int | None:
    if value is None:
        return None
    return _bounded_positive_integer(value, description)


def _optional_index(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _invalid("rich field index is invalid")
    return value


def _invalid(message: str) -> None:
    raise ValueError(f"source presentation {message}")


__all__ = [
    "SOURCE_PRESENTATION_METADATA_KEY",
    "SOURCE_PRESENTATION_SCHEMA",
    "source_presentation",
    "validate_source_presentation_metadata",
]
