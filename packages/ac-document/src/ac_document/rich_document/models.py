from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from ..sources import SourceArtifact, SourceFormat, SourceOrigin, SourceOriginKind


RICH_DOCUMENT_SCHEMA = "ac.document.rich_document.v2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class RichBlockKind(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    CODE = "code"
    EQUATION = "equation"
    TABLE = "table"
    FIGURE = "figure"


_PAYLOAD_FIELDS = {
    RichBlockKind.HEADING: {"text", "level"},
    RichBlockKind.PARAGRAPH: {"text", "inline_spans"},
    RichBlockKind.LIST: {"ordered", "items"},
    RichBlockKind.CODE: {"text", "language"},
    RichBlockKind.EQUATION: {"tex", "display", "label"},
    RichBlockKind.TABLE: {"headers", "rows", "caption"},
    RichBlockKind.FIGURE: {
        "asset_digest",
        "alt_text",
        "caption",
        "target",
        "media_type",
        "logical_name",
        "size",
    },
}


@dataclass(frozen=True)
class SourceLocator:
    """Format-neutral locator into the immutable primary source."""

    source_format: SourceFormat
    line_start: int | None = None
    column_start: int | None = None
    line_end: int | None = None
    column_end: int | None = None
    selector: str = ""
    source_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_format", SourceFormat(self.source_format))
        positions = (
            self.line_start,
            self.column_start,
            self.line_end,
            self.column_end,
        )
        if any(
            value is not None
            and (not isinstance(value, int) or isinstance(value, bool))
            for value in positions
        ):
            raise ValueError("source locator positions must be integers")
        if any(value is not None and value < 1 for value in positions):
            raise ValueError("source locator positions must be positive")
        if (self.line_start is None) != (self.line_end is None):
            raise ValueError("source locator line range must be complete")
        if (self.column_start is None) != (self.column_end is None):
            raise ValueError("source locator column range must be complete")
        if self.column_start is not None and self.line_start is None:
            raise ValueError("source locator columns require a line range")
        if self.line_start is not None and self.line_end < self.line_start:
            raise ValueError("source locator end cannot precede start")
        if (
            self.line_start == self.line_end
            and self.column_start is not None
            and self.column_end < self.column_start
        ):
            raise ValueError("source locator end cannot precede start")


@dataclass(frozen=True)
class RichBlock:
    block_id: str
    ordinal: int
    kind: RichBlockKind
    section_path: tuple[str, ...]
    locator: SourceLocator
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        kind = RichBlockKind(self.kind)
        if (
            not self.block_id
            or not isinstance(self.ordinal, int)
            or isinstance(self.ordinal, bool)
            or self.ordinal < 0
        ):
            raise ValueError("rich block identity is invalid")
        if any(not item for item in self.section_path):
            raise ValueError("rich block section path contains an empty ID")
        payload = dict(self.payload)
        if set(payload) != _PAYLOAD_FIELDS[kind]:
            raise ValueError(f"{kind.value} block payload has invalid fields")
        _validate_payload(kind, payload)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "section_path", tuple(self.section_path))
        object.__setattr__(self, "payload", _freeze_mapping(payload))


@dataclass(frozen=True)
class RichSection:
    section_id: str
    title: str
    level: int
    ordinal: int
    path: tuple[str, ...]
    block_start: int
    block_end: int

    def __post_init__(self) -> None:
        if (
            not self.section_id
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (
                    self.level,
                    self.ordinal,
                    self.block_start,
                    self.block_end,
                )
            )
            or self.level < 1
            or self.ordinal < 0
            or self.block_start < 0
            or self.block_end < self.block_start
        ):
            raise ValueError("rich section metadata is invalid")
        if not self.path or self.path[-1] != self.section_id:
            raise ValueError("rich section path must end with its section ID")
        object.__setattr__(self, "path", tuple(self.path))


@dataclass(frozen=True)
class RichAsset:
    artifact_digest: str
    media_type: str
    logical_name: str
    size: int

    def __post_init__(self) -> None:
        digest = self.artifact_digest.casefold()
        media_type = self.media_type.strip().casefold()
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("rich asset digest must be a SHA-256 digest")
        if not media_type or "/" not in media_type or ";" in media_type:
            raise ValueError("rich asset media type is invalid")
        if (
            not self.logical_name
            or not isinstance(self.size, int)
            or isinstance(self.size, bool)
            or self.size < 0
        ):
            raise ValueError("rich asset metadata is invalid")
        object.__setattr__(self, "artifact_digest", digest)
        object.__setattr__(self, "media_type", media_type)


@dataclass(frozen=True)
class RichPageMapEntry:
    block_id: str
    page_number: int

    def __post_init__(self) -> None:
        if (
            not self.block_id
            or not isinstance(self.page_number, int)
            or isinstance(self.page_number, bool)
            or self.page_number < 1
        ):
            raise ValueError("rich page map entry is invalid")


@dataclass(frozen=True)
class RichDocument:
    source: SourceArtifact
    blocks: tuple[RichBlock, ...]
    sections: tuple[RichSection, ...] = ()
    assets: tuple[RichAsset, ...] = ()
    page_map: tuple[RichPageMapEntry, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = RICH_DOCUMENT_SCHEMA
    document_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != RICH_DOCUMENT_SCHEMA:
            raise ValueError("unsupported rich document schema")
        if self.source.source_format not in {
            SourceFormat.MARKDOWN,
            SourceFormat.HTML,
            SourceFormat.TEX,
        }:
            raise ValueError("rich document requires a rich source format")
        blocks = tuple(self.blocks)
        sections = tuple(self.sections)
        assets = tuple(self.assets)
        page_map = tuple(self.page_map)
        if tuple(item.ordinal for item in blocks) != tuple(range(len(blocks))):
            raise ValueError("rich block ordinals must be contiguous")
        if len({item.block_id for item in blocks}) != len(blocks):
            raise ValueError("rich document contains duplicate block IDs")
        if tuple(item.ordinal for item in sections) != tuple(range(len(sections))):
            raise ValueError("rich section ordinals must be contiguous")
        section_ids = {item.section_id for item in sections}
        if len(section_ids) != len(sections):
            raise ValueError("rich document contains duplicate section IDs")
        section_paths = {item.path for item in sections}
        for section in sections:
            if section.block_end > len(blocks):
                raise ValueError("rich section block range exceeds the document")
            if any(section_id not in section_ids for section_id in section.path):
                raise ValueError("rich section path refers to an unknown section")
        for block in blocks:
            if any(section_id not in section_ids for section_id in block.section_path):
                raise ValueError("rich block refers to an unknown section")
            if block.section_path and block.section_path not in section_paths:
                raise ValueError("rich block section path is not a declared path")
            if block.locator.source_format is not self.source.source_format:
                raise ValueError("rich block locator format differs from its source")
        if len({item.artifact_digest for item in assets}) != len(assets):
            raise ValueError("rich document contains duplicate assets")
        asset_ids = {item.artifact_digest for item in assets}
        if any(
            block.kind is RichBlockKind.FIGURE
            and block.payload["asset_digest"]
            and block.payload["asset_digest"] not in asset_ids
            for block in blocks
        ):
            raise ValueError("rich figure refers to an unknown asset")
        block_ids = {item.block_id for item in blocks}
        if len({item.block_id for item in page_map}) != len(page_map):
            raise ValueError("rich page map contains duplicate block entries")
        if any(item.block_id not in block_ids for item in page_map):
            raise ValueError("rich page map refers to an unknown block")
        metadata = _freeze_mapping(dict(self.metadata))
        material = {
            "schema_version": self.schema_version,
            "source": _source_to_document(self.source),
            "blocks": [rich_block_to_document(item) for item in blocks],
            "sections": [_section_to_document(item) for item in sections],
            "assets": [_asset_to_document(item) for item in assets],
            "page_map": [_page_map_to_document(item) for item in page_map],
            "metadata": _thaw_json(metadata),
        }
        digest = hashlib.sha256(
            json.dumps(
                material,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "page_map", page_map)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "document_digest", digest)


_LOCATOR_FIELDS = {
    "source_format",
    "line_start",
    "column_start",
    "line_end",
    "column_end",
    "selector",
    "source_id",
}
_BLOCK_FIELDS = {
    "block_id",
    "ordinal",
    "kind",
    "section_path",
    "locator",
    "payload",
}
_SECTION_FIELDS = {
    "section_id",
    "title",
    "level",
    "ordinal",
    "path",
    "block_start",
    "block_end",
}
_ASSET_FIELDS = {"artifact_digest", "media_type", "logical_name", "size"}
_PAGE_MAP_FIELDS = {"block_id", "page_number"}
_SOURCE_FIELDS = {"source_format", "artifact_digest", "size", "media_type"}
_DOCUMENT_FIELDS = {
    "schema_version",
    "document_digest",
    "source",
    "blocks",
    "sections",
    "assets",
    "page_map",
    "metadata",
}


def rich_block_to_document(block: RichBlock) -> dict[str, Any]:
    return {
        "block_id": block.block_id,
        "ordinal": block.ordinal,
        "kind": block.kind.value,
        "section_path": list(block.section_path),
        "locator": {
            "source_format": block.locator.source_format.value,
            "line_start": block.locator.line_start,
            "column_start": block.locator.column_start,
            "line_end": block.locator.line_end,
            "column_end": block.locator.column_end,
            "selector": block.locator.selector,
            "source_id": block.locator.source_id,
        },
        "payload": _thaw_json(block.payload),
    }


def rich_block_from_document(value: Mapping[str, Any]) -> RichBlock:
    _require_json_document(value, "rich block")
    _require_fields(value, _BLOCK_FIELDS, "rich block")
    locator = _mapping(value.get("locator"), "rich block locator")
    _require_fields(locator, _LOCATOR_FIELDS, "rich block locator")
    section_path = _string_list(value.get("section_path"), "section_path")
    payload = _mapping(value.get("payload"), "rich block payload")
    _validate_codec_payload_lists(
        RichBlockKind(_string(value, "kind")), payload
    )
    return RichBlock(
        block_id=_string(value, "block_id"),
        ordinal=_integer(value, "ordinal"),
        kind=RichBlockKind(_string(value, "kind")),
        section_path=tuple(section_path),
        locator=SourceLocator(
            source_format=SourceFormat(_string(locator, "source_format")),
            line_start=_optional_integer(locator, "line_start"),
            column_start=_optional_integer(locator, "column_start"),
            line_end=_optional_integer(locator, "line_end"),
            column_end=_optional_integer(locator, "column_end"),
            selector=_string(locator, "selector"),
            source_id=_string(locator, "source_id"),
        ),
        payload=payload,
    )


def rich_document_to_document(document: RichDocument) -> dict[str, Any]:
    return {
        "schema_version": document.schema_version,
        "document_digest": document.document_digest,
        "source": _source_to_document(document.source),
        "blocks": [rich_block_to_document(item) for item in document.blocks],
        "sections": [_section_to_document(item) for item in document.sections],
        "assets": [_asset_to_document(item) for item in document.assets],
        "page_map": [_page_map_to_document(item) for item in document.page_map],
        "metadata": _thaw_json(document.metadata),
    }


def rich_document_from_document(value: Mapping[str, Any]) -> RichDocument:
    _require_json_document(value, "rich document")
    _require_fields(value, _DOCUMENT_FIELDS, "rich document")
    if value.get("schema_version") != RICH_DOCUMENT_SCHEMA:
        raise ValueError("unsupported rich document schema")
    source_value = _mapping(value.get("source"), "rich document source")
    _require_fields(source_value, _SOURCE_FIELDS, "rich document source")
    source = SourceArtifact(
        source_format=SourceFormat(_string(source_value, "source_format")),
        artifact_digest=_string(source_value, "artifact_digest"),
        size=_integer(source_value, "size"),
        media_type=_string(source_value, "media_type"),
        origin=SourceOrigin(
            SourceOriginKind.REPOSITORY,
            locator=(
                f"{source_value['source_format']}/sha256/"
                f"{source_value['artifact_digest']}"
            ),
        ),
    )
    blocks = tuple(
        rich_block_from_document(_mapping(item, "rich block"))
        for item in _list(value, "blocks")
    )
    sections = []
    for raw in _list(value, "sections"):
        item = _mapping(raw, "rich section")
        _require_fields(item, _SECTION_FIELDS, "rich section")
        sections.append(
            RichSection(
                section_id=_string(item, "section_id"),
                title=_string(item, "title"),
                level=_integer(item, "level"),
                ordinal=_integer(item, "ordinal"),
                path=tuple(_string_list(item.get("path"), "section path")),
                block_start=_integer(item, "block_start"),
                block_end=_integer(item, "block_end"),
            )
        )
    assets = []
    for raw in _list(value, "assets"):
        item = _mapping(raw, "rich asset")
        _require_fields(item, _ASSET_FIELDS, "rich asset")
        assets.append(
            RichAsset(
                artifact_digest=_string(item, "artifact_digest"),
                media_type=_string(item, "media_type"),
                logical_name=_string(item, "logical_name"),
                size=_integer(item, "size"),
            )
        )
    page_map = []
    for raw in _list(value, "page_map"):
        item = _mapping(raw, "rich page map entry")
        _require_fields(item, _PAGE_MAP_FIELDS, "rich page map entry")
        page_map.append(
            RichPageMapEntry(
                block_id=_string(item, "block_id"),
                page_number=_integer(item, "page_number"),
            )
        )
    document = RichDocument(
        source=source,
        blocks=blocks,
        sections=tuple(sections),
        assets=tuple(assets),
        page_map=tuple(page_map),
        metadata=_mapping(value.get("metadata"), "rich document metadata"),
    )
    if document.document_digest != _string(value, "document_digest"):
        raise ValueError("rich document digest does not match its content")
    return document


def _source_to_document(source: SourceArtifact) -> dict[str, Any]:
    return {
        "source_format": source.source_format.value,
        "artifact_digest": source.artifact_digest,
        "size": source.size,
        "media_type": source.media_type,
    }


def _section_to_document(section: RichSection) -> dict[str, Any]:
    return {
        "section_id": section.section_id,
        "title": section.title,
        "level": section.level,
        "ordinal": section.ordinal,
        "path": list(section.path),
        "block_start": section.block_start,
        "block_end": section.block_end,
    }


def _asset_to_document(asset: RichAsset) -> dict[str, Any]:
    return {
        "artifact_digest": asset.artifact_digest,
        "media_type": asset.media_type,
        "logical_name": asset.logical_name,
        "size": asset.size,
    }


def _page_map_to_document(item: RichPageMapEntry) -> dict[str, Any]:
    return {"block_id": item.block_id, "page_number": item.page_number}


def _validate_payload(kind: RichBlockKind, payload: dict[str, Any]) -> None:
    if kind is RichBlockKind.HEADING:
        _expect_string(payload["text"], "heading text")
        _expect_integer(payload["level"], "heading level", minimum=1)
    elif kind is RichBlockKind.PARAGRAPH:
        _expect_string(payload["text"], "paragraph text")
        _validate_inline_spans(
            payload["inline_spans"],
            text=payload["text"],
        )
    elif kind is RichBlockKind.LIST:
        if not isinstance(payload["ordered"], bool):
            raise ValueError("list ordered must be a boolean")
        items = _expect_list(payload["items"], "list items")
        for item in items:
            value = _mapping(item, "list item")
            _require_fields(value, {"text", "inline_spans"}, "list item")
            _expect_string(value["text"], "list item text")
            _validate_inline_spans(
                value["inline_spans"],
                text=value["text"],
            )
    elif kind is RichBlockKind.CODE:
        _expect_string(payload["text"], "code text")
        _expect_string(payload["language"], "code language")
    elif kind is RichBlockKind.EQUATION:
        _expect_string(payload["tex"], "equation TeX")
        if not payload["tex"]:
            raise ValueError("equation TeX cannot be empty")
        if not isinstance(payload["display"], bool):
            raise ValueError("equation display must be a boolean")
        _expect_string(payload["label"], "equation label")
    elif kind is RichBlockKind.TABLE:
        headers = _expect_list(payload["headers"], "table headers")
        rows = _expect_list(payload["rows"], "table rows")
        if any(not isinstance(item, str) for item in headers):
            raise ValueError("table headers must be strings")
        if any(
            not isinstance(row, (list, tuple))
            or any(not isinstance(cell, str) for cell in row)
            for row in rows
        ):
            raise ValueError("table rows must contain strings")
        _expect_string(payload["caption"], "table caption")
    else:
        digest = payload["asset_digest"]
        if digest and (
            not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None
        ):
            raise ValueError("figure asset digest must be empty or a SHA-256 digest")
        for key in ("alt_text", "caption", "target", "media_type", "logical_name"):
            _expect_string(payload[key], f"figure {key}")
        _expect_integer(payload["size"], "figure size", minimum=0)
        if digest:
            if not payload["media_type"] or "/" not in payload["media_type"]:
                raise ValueError("imported figure media type is invalid")
            if not payload["logical_name"]:
                raise ValueError("imported figure logical name is required")
        elif payload["media_type"] or payload["size"]:
            raise ValueError("unimported figure cannot claim asset metadata")


def _validate_inline_spans(
    value: Any,
    *,
    text: str,
) -> None:
    spans = _expect_list(value, "inline spans")
    cursor = 0
    reconstructed: list[str] = []
    for raw in spans:
        item = _mapping(raw, "inline span")
        kind = item.get("kind")
        fields = {
            "text": {"kind", "start", "end", "text"},
            "link": {"kind", "start", "end", "text", "target"},
            "math": {"kind", "start", "end", "text", "tex", "source"},
        }
        if kind not in fields:
            raise ValueError("inline span kind is invalid")
        _require_fields(item, fields[kind], "inline span")
        start = item["start"]
        end = item["end"]
        _expect_integer(start, "inline span start", minimum=0)
        _expect_integer(end, "inline span end", minimum=0)
        _expect_string(item["text"], "inline span text")
        if start != cursor or end <= start or end - start != len(item["text"]):
            raise ValueError("inline spans must contiguously cover their text")
        reconstructed.append(item["text"])
        cursor = end
        if kind == "link":
            _expect_string(item["target"], "inline span target")
        elif kind == "math":
            _expect_string(item["tex"], "inline span TeX")
            _expect_string(item["source"], "inline span math source")
    if "".join(reconstructed) != text:
        raise ValueError("inline spans do not reconstruct text")


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if any(not isinstance(key, str) for key in value):
        raise ValueError("rich document object keys must be strings")
    return MappingProxyType(
        {key: _freeze_json(item) for key, item in value.items()}
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("rich document numbers must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("rich document values must be JSON-compatible")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _require_fields(
    value: Mapping[str, Any], expected: set[str], description: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{description} has invalid fields")


def _mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _list(value: Mapping[str, Any], key: str) -> list[Any]:
    item = value.get(key)
    if not isinstance(item, list):
        raise ValueError(f"rich document {key} must be a list")
    return item


def _expect_list(value: Any, description: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{description} must be a list")
    return value


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    _expect_string(item, key)
    return item


def _expect_string(value: Any, description: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{description} must be a string")


def _integer(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    _expect_integer(item, key)
    return item


def _expect_integer(value: Any, description: str, *, minimum: int | None = None) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{description} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{description} must be at least {minimum}")


def _optional_integer(value: Mapping[str, Any], key: str) -> int | None:
    item = value.get(key)
    if item is None:
        return None
    return _integer(value, key)


def _string_list(value: Any, description: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be a list")
    items = value
    if any(not isinstance(item, str) for item in items):
        raise ValueError(f"{description} must contain strings")
    return list(items)


def _validate_codec_payload_lists(
    kind: RichBlockKind, payload: Mapping[str, Any]
) -> None:
    list_fields: tuple[str, ...] = ()
    if kind is RichBlockKind.PARAGRAPH:
        list_fields = ("inline_spans",)
    elif kind is RichBlockKind.LIST:
        list_fields = ("items",)
    elif kind is RichBlockKind.TABLE:
        list_fields = ("headers", "rows")
    for key in list_fields:
        if not isinstance(payload.get(key), list):
            raise ValueError(f"{kind.value} block {key} must be a list")
    if kind is RichBlockKind.LIST:
        for item in payload["items"]:
            if isinstance(item, Mapping):
                if not isinstance(item.get("inline_spans"), list):
                    raise ValueError("list item inline_spans must be a list")
    if kind is RichBlockKind.TABLE and any(
        not isinstance(row, list) for row in payload["rows"]
    ):
        raise ValueError("table rows must be lists")


def _require_json_document(value: Any, description: str) -> None:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{description} object keys must be strings")
        for item in value.values():
            _require_json_document(item, description)
        return
    if isinstance(value, list):
        for item in value:
            _require_json_document(item, description)
        return
    if isinstance(value, tuple):
        raise ValueError(f"{description} arrays must be lists")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{description} numbers must be finite")
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{description} must contain JSON-compatible values")


__all__ = [
    "RICH_DOCUMENT_SCHEMA",
    "RichAsset",
    "RichBlock",
    "RichBlockKind",
    "RichDocument",
    "RichPageMapEntry",
    "RichSection",
    "SourceLocator",
    "rich_block_from_document",
    "rich_block_to_document",
    "rich_document_from_document",
    "rich_document_to_document",
]
