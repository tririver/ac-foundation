"""Administration over document-owned cache projections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ._cache_root import resolve_cache_root
from ._full_text_catalog import FullTextCatalog
from .terms import TermInventoryStore


@dataclass(frozen=True)
class DocumentCacheComponent:
    name: str
    cached_at: str
    storage_entry_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentCacheEntry:
    entry_id: str
    kind: str
    document_ids: tuple[str, ...]
    local_source_identity: Mapping[str, Any] | None
    components: tuple[DocumentCacheComponent, ...]
    cached_at: str


@dataclass(frozen=True)
class DocumentCacheListResult:
    as_of: str
    since_seconds: int | None
    threshold_at: str | None
    entries: tuple[DocumentCacheEntry, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentCacheRemoveResult:
    dry_run: bool
    selected: tuple[DocumentCacheEntry, ...]
    removed_entry_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()


class DocumentCacheAdministrator:
    """List and remove neutral full-text and term-inventory cache entries."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = resolve_cache_root(root)
        self.catalog = FullTextCatalog(self.root)
        self.term_inventory = TermInventoryStore(self.root)

    def list(
        self,
        *,
        document_ids: Sequence[str] = (),
        entry_ids: Sequence[str] = (),
        since_seconds: int | None = None,
        now: datetime | None = None,
    ) -> DocumentCacheListResult:
        if since_seconds is not None and (
            isinstance(since_seconds, bool)
            or not isinstance(since_seconds, int)
            or since_seconds <= 0
        ):
            raise ValueError("since_seconds must be a positive integer")
        as_of = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        threshold = (
            as_of - timedelta(seconds=since_seconds)
            if since_seconds is not None
            else None
        )
        entries: dict[str, DocumentCacheEntry] = {}
        for item in self.catalog.admin_entries():
            entries[item.entry_id] = _merge_component(
                entries.get(item.entry_id),
                entry_id=item.entry_id,
                kind=item.entry.kind,
                document_ids=item.entry.document_ids,
                local_source_identity=item.entry.local_source_identity,
                component=DocumentCacheComponent(
                    "full-text", item.cached_at, (item.entry_id,)
                ),
            )
        for item in self.term_inventory.admin_entries():
            entries[item.entry_id] = _merge_component(
                entries.get(item.entry_id),
                entry_id=item.entry_id,
                kind="local",
                document_ids=(),
                local_source_identity=item.source_identity,
                component=DocumentCacheComponent(
                    "term-inventory", item.cached_at, (item.entry_id,)
                ),
            )

        requested_documents = {str(item) for item in document_ids if str(item)}
        requested_entries = {str(item) for item in entry_ids if str(item)}
        values = list(entries.values())
        if requested_documents or requested_entries:
            values = [
                item
                for item in values
                if requested_documents.intersection(item.document_ids)
                or item.entry_id in requested_entries
            ]
        if threshold is not None:
            values = [
                item for item in values if _parse_utc(item.cached_at) >= threshold
            ]
        values.sort(
            key=lambda item: (
                -_parse_utc(item.cached_at).timestamp(),
                item.kind == "local",
                item.document_ids,
                item.entry_id,
            )
        )
        return DocumentCacheListResult(
            as_of=_format_utc(as_of),
            since_seconds=since_seconds,
            threshold_at=_format_utc(threshold) if threshold is not None else None,
            entries=tuple(values),
        )

    def remove(
        self,
        *,
        document_ids: Sequence[str] = (),
        entry_ids: Sequence[str] = (),
        dry_run: bool = True,
    ) -> DocumentCacheRemoveResult:
        if not document_ids and not entry_ids:
            raise ValueError(
                "cache remove requires at least one exact document or entry id"
            )
        selected = self.list(
            document_ids=document_ids, entry_ids=entry_ids
        ).entries
        if dry_run:
            return DocumentCacheRemoveResult(True, selected, ())
        removed: list[str] = []
        warnings: list[str] = []
        for entry in selected:
            changed = self.term_inventory.remove_admin_entry(entry.entry_id)
            changed = self.catalog.remove_admin_entry(entry.entry_id) or changed
            if changed:
                removed.append(entry.entry_id)
            else:
                warnings.append(f"cache_entry_already_absent:{entry.entry_id}")
        return DocumentCacheRemoveResult(
            False, selected, tuple(removed), tuple(warnings)
        )


def _merge_component(
    previous: DocumentCacheEntry | None,
    *,
    entry_id: str,
    kind: str,
    document_ids: Sequence[str],
    local_source_identity: Mapping[str, Any] | None,
    component: DocumentCacheComponent,
) -> DocumentCacheEntry:
    by_name = (
        {item.name: item for item in previous.components}
        if previous is not None
        else {}
    )
    existing = by_name.get(component.name)
    if existing is not None:
        component = DocumentCacheComponent(
            component.name,
            max(existing.cached_at, component.cached_at, key=_parse_utc),
            tuple(
                sorted(
                    set(existing.storage_entry_ids + component.storage_entry_ids)
                )
            ),
        )
    by_name[component.name] = component
    components = tuple(sorted(by_name.values(), key=lambda item: item.name))
    cached_at = max(components, key=lambda item: _parse_utc(item.cached_at)).cached_at
    return DocumentCacheEntry(
        entry_id=entry_id,
        kind=previous.kind if previous is not None else kind,
        document_ids=(
            previous.document_ids
            if previous is not None
            else tuple(sorted(set(document_ids), key=str.casefold))
        ),
        local_source_identity=(
            previous.local_source_identity
            if previous is not None
            else (
                dict(local_source_identity)
                if local_source_identity is not None
                else None
            )
        ),
        components=components,
        cached_at=cached_at,
    )


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("cache timestamp must include UTC offset")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "DocumentCacheAdministrator",
    "DocumentCacheComponent",
    "DocumentCacheEntry",
    "DocumentCacheListResult",
    "DocumentCacheRemoveResult",
]
