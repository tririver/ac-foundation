"""Cross-block list ancestry validation for RichDocument."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def validate_list_paths(
    blocks: Sequence[Any],
    *,
    sections: Sequence[Any] = (),
    source_target_manifest: Mapping[str, Any] | None = None,
) -> None:
    """Validate flat block ownership by authored list items."""

    block_by_id = {block.block_id: block for block in blocks}
    container_definitions: dict[str, tuple[Any, ...]] = {}
    item_definitions: dict[str, tuple[Any, ...]] = {}
    container_ordinals: dict[str, list[int]] = {}
    item_entries: dict[str, list[tuple[int, Any]]] = {}
    container_items: dict[str, dict[int, str]] = {}
    source_owners: dict[str, tuple[str, str]] = {}
    section_by_path = {tuple(section.path): section for section in sections}

    for block in blocks:
        path = tuple(block.list_path)
        if path and sections:
            section = section_by_path.get(tuple(block.section_path))
            if section is None or not (
                section.block_start <= block.ordinal < section.block_end
            ):
                _invalid("block lies outside its declared section owner")
        if path and getattr(block.kind, "value", block.kind) == "heading":
            _invalid("cannot attach ancestry to a heading block")
        if path and getattr(block.kind, "value", block.kind) == "list":
            items = tuple(block.payload["items"])
            if len(items) != 1:
                _invalid("LIST blocks with ancestry must carry exactly one item")
            if block.payload["ordered"] is not path[-1].ordered:
                _invalid("LIST payload ordering differs from its owner")
        seen_containers: set[str] = set()
        seen_items: set[str] = set()
        prefix: list[tuple[str, str]] = []
        for depth, entry in enumerate(path):
            if entry.depth != depth:
                _invalid("depth differs from the path position")
            if entry.container_id in seen_containers or entry.item_id in seen_items:
                _invalid("contains a repeated owner in one path")
            seen_containers.add(entry.container_id)
            seen_items.add(entry.item_id)
            parent_prefix = tuple(prefix)
            container_definition = (
                entry.container_source_id,
                entry.container_selector,
                entry.item_count,
                entry.depth,
                entry.ordered,
                parent_prefix,
                tuple(block.section_path),
            )
            if not _bind_definition(
                container_definitions,
                entry.container_id,
                container_definition,
            ):
                _invalid("container identity has conflicting ownership")
            item_definition = (
                entry.container_id,
                entry.item_source_id,
                entry.item_selector,
                entry.item_index,
                entry.item_count,
                entry.depth,
                entry.ordered,
                parent_prefix,
                tuple(block.section_path),
            )
            if not _bind_definition(
                item_definitions,
                entry.item_id,
                item_definition,
            ):
                _invalid("item identity has conflicting ownership")
            _bind_source_owner(
                source_owners,
                entry.container_source_id,
                ("container", entry.container_id),
            )
            _bind_source_owner(
                source_owners,
                entry.item_source_id,
                ("item", entry.item_id),
            )
            indexes = container_items.setdefault(entry.container_id, {})
            existing_item = indexes.setdefault(entry.item_index, entry.item_id)
            if existing_item != entry.item_id:
                _invalid("container item index has conflicting identities")
            container_ordinals.setdefault(entry.container_id, []).append(
                block.ordinal
            )
            item_entries.setdefault(entry.item_id, []).append(
                (block.ordinal, entry)
            )
            prefix.append((entry.container_id, entry.item_id))

    for container_id, ordinals in container_ordinals.items():
        _require_contiguous_ordinals(ordinals, "container")
        indexed_items = container_items[container_id]
        item_count = int(container_definitions[container_id][2])
        indexes = sorted(indexed_items)
        if item_count != len(indexes) or any(
            index != expected for expected, index in enumerate(indexes)
        ):
            _invalid("container item count or indexes do not cover authored items")
        first_ordinals = {
            item_id: item_entries[item_id][0][0]
            for item_id in indexed_items.values()
        }
        ordered_items = sorted(
            indexed_items.items(), key=lambda item: first_ordinals[item[1]]
        )
        if [index for index, _item_id in ordered_items] != sorted(indexed_items):
            _invalid("container items are out of source order")

    for entries in item_entries.values():
        entries.sort(key=lambda item: item[0])
        _require_contiguous_ordinals(
            [ordinal for ordinal, _entry in entries],
            "item",
        )
        if [entry.segment_index for _ordinal, entry in entries] != list(
            range(len(entries))
        ):
            _invalid("item segments are not contiguous in block order")

    if source_target_manifest is None:
        return
    for target in source_target_manifest.get("targets", ()):
        alias = target["alias"]
        owner = source_owners.get(alias)
        if owner is None:
            continue
        block = block_by_id.get(target["block_id"])
        if block is None:
            _invalid("source target refers to an unknown owner block")
        owner_kind, owner_id = owner
        if not any(
            (
                entry.container_id
                if owner_kind == "container"
                else entry.item_id
            )
            == owner_id
            for entry in block.list_path
        ):
            _invalid("source target alias conflicts with a list owner")


def _bind_definition(
    values: dict[str, tuple[Any, ...]],
    identity: str,
    definition: tuple[Any, ...],
) -> bool:
    existing = values.setdefault(identity, definition)
    return existing == definition


def _bind_source_owner(
    values: dict[str, tuple[str, str]],
    source_id: str,
    owner: tuple[str, str],
) -> None:
    if not source_id:
        return
    existing = values.setdefault(source_id, owner)
    if existing != owner:
        _invalid("authored source ID has conflicting list owners")


def _require_contiguous_ordinals(ordinals: list[int], owner: str) -> None:
    unique = sorted(set(ordinals))
    if unique != list(range(unique[0], unique[-1] + 1)):
        _invalid(f"{owner} blocks are not contiguous")


def _invalid(message: str) -> None:
    raise ValueError(f"rich list path {message}")


__all__ = ["validate_list_paths"]
