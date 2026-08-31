"""Versioned authoritative source-target metadata for RichDocument."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


SOURCE_TARGET_MANIFEST_METADATA_KEY = "source_target_manifest"
SOURCE_TARGET_MANIFEST_SCHEMA = "ac.document.source_target_manifest.v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MANIFEST_FIELDS = {"schema_version", "targets"}
_TARGET_FIELDS = {
    "alias",
    "selector",
    "kind",
    "block_id",
    "block_start",
    "block_end",
    "section_id",
    "panels",
}
_PANEL_FIELDS = {
    "panel_index",
    "source_id",
    "selector",
    "target",
    "media_type",
    "alt_text",
    "status",
    "asset_digest",
    "logical_name",
    "size",
}
_TARGET_KINDS = {
    "heading",
    "paragraph",
    "list",
    "code",
    "equation",
    "table",
    "figure",
    "section",
}
_PANEL_STATUSES = {"available", "missing", "unsupported"}


def source_target_manifest(document: Any) -> Mapping[str, Any] | None:
    """Return and revalidate authoritative source targets when present."""

    if SOURCE_TARGET_MANIFEST_METADATA_KEY not in document.metadata:
        return None
    value = document.metadata[SOURCE_TARGET_MANIFEST_METADATA_KEY]
    validate_source_target_manifest(
        value,
        blocks=document.blocks,
        sections=document.sections,
        assets=document.assets,
    )
    return value


def validate_source_target_manifest(
    value: Any,
    *,
    blocks: Sequence[Any],
    sections: Sequence[Any],
    assets: Sequence[Any],
) -> None:
    """Validate one manifest against the exact RichDocument it targets."""

    manifest = _mapping(value, "must be an object")
    _require_fields(manifest, _MANIFEST_FIELDS, "has invalid fields")
    if manifest.get("schema_version") != SOURCE_TARGET_MANIFEST_SCHEMA:
        _invalid("has an unsupported schema")
    targets = _sequence(manifest.get("targets"), "targets must be a list")
    block_by_id = {block.block_id: block for block in blocks}
    section_by_id = {section.section_id: section for section in sections}
    asset_by_id = {asset.artifact_digest: asset for asset in assets}
    aliases: set[str] = set()
    panel_source_ids: set[str] = set()
    for raw in targets:
        target = _mapping(raw, "target must be an object")
        _require_fields(target, _TARGET_FIELDS, "target has invalid fields")
        alias = _string(target, "alias")
        selector = _string(target, "selector")
        kind = _string(target, "kind")
        block_id = _string(target, "block_id")
        block_start = _integer(target, "block_start")
        block_end = _integer(target, "block_end")
        section_id = _string(target, "section_id")
        panels = _sequence(target.get("panels"), "panels must be a list")
        if not alias or alias in aliases:
            _invalid("contains an empty or duplicate alias")
        aliases.add(alias)
        if selector != f"#{alias}":
            _invalid("target selector differs from its exact alias")
        if kind not in _TARGET_KINDS:
            _invalid("target kind is unknown")
        if block_id not in block_by_id:
            _invalid("target block does not exist")
        if not 0 <= block_start < block_end <= len(blocks):
            _invalid("target range is out of bounds")
        block = block_by_id[block_id]
        if block.ordinal != block_start or blocks[block_start].block_id != block_id:
            _invalid("canonical block does not start the target range")
        if kind == "section":
            section = section_by_id.get(section_id)
            if section is None:
                _invalid("section target refers to an unknown section")
            if (
                section.block_start != block_start
                or section.block_end != block_end
                or getattr(block.kind, "value", block.kind) != "heading"
            ):
                _invalid("section target differs from the declared outline range")
        else:
            if section_id:
                _invalid("non-section target carries a section ID")
            if block_end != block_start + 1:
                _invalid("block target range must contain exactly one block")
            if kind != getattr(block.kind, "value", block.kind):
                _invalid("target kind differs from its canonical block")
        if kind != "figure" and panels:
            _invalid("non-figure target carries panels")
        _validate_panels(
            panels,
            asset_by_id=asset_by_id,
            panel_source_ids=panel_source_ids,
        )
    if aliases & panel_source_ids:
        _invalid("contains a panel source ID that conflicts with a target alias")


def _validate_panels(
    panels: Sequence[Any],
    *,
    asset_by_id: Mapping[str, Any],
    panel_source_ids: set[str],
) -> None:
    selectors: set[str] = set()
    targets: set[str] = set()
    for index, raw in enumerate(panels):
        panel = _mapping(raw, "panel must be an object")
        _require_fields(panel, _PANEL_FIELDS, "panel has invalid fields")
        panel_index = _integer(panel, "panel_index")
        source_id = _string(panel, "source_id")
        selector = _string(panel, "selector")
        target = _string(panel, "target")
        media_type = _string(panel, "media_type")
        _string(panel, "alt_text")
        status = _string(panel, "status")
        digest = _string(panel, "asset_digest")
        logical_name = _string(panel, "logical_name")
        size = _integer(panel, "size")
        if panel_index != index:
            _invalid("panel indexes are not contiguous and ordered")
        if source_id:
            if (
                source_id in panel_source_ids
                or selector in selectors
                or selector != f"#{source_id}"
            ):
                _invalid("contains duplicate panel source IDs or selectors")
            panel_source_ids.add(source_id)
            selectors.add(selector)
        elif selector:
            _invalid("panel selector has no source ID")
        if target:
            if target in targets:
                _invalid("contains duplicate panel targets")
            targets.add(target)
        if status not in _PANEL_STATUSES:
            _invalid("panel status is unknown")
        if logical_name != target:
            _invalid("panel logical name differs from its exact target")
        if size < 0:
            _invalid("panel size is negative")
        if status == "available":
            asset = asset_by_id.get(digest)
            if not target or asset is None or _SHA256_RE.fullmatch(digest) is None:
                _invalid("available panel does not refer to a document asset")
            if (
                media_type != asset.media_type
                or size != asset.size
            ):
                _invalid("available panel metadata differs from its asset")
        elif digest or size:
            _invalid("unavailable panel claims asset content")
        elif status == "missing" and not target:
            _invalid("missing panel has no target")


def _mapping(value: Any, message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _invalid(message)
    return value


def _sequence(value: Any, message: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        _invalid(message)
    return value


def _require_fields(
    value: Mapping[str, Any],
    fields: set[str],
    message: str,
) -> None:
    if set(value) != fields:
        _invalid(message)


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        _invalid(f"{key} must be a string")
    return item


def _integer(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        _invalid(f"{key} must be an integer")
    return item


def _invalid(message: str) -> None:
    raise ValueError(f"source target manifest {message}")


__all__ = [
    "SOURCE_TARGET_MANIFEST_METADATA_KEY",
    "SOURCE_TARGET_MANIFEST_SCHEMA",
    "source_target_manifest",
    "validate_source_target_manifest",
]
