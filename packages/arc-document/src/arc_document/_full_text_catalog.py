"""Logical locators for the current parsed full-text cache.

The catalog contains no source bytes, parsed text, or physical cache paths.
Each representation points to immutable content identities that must still be
verified by :class:`ParsedDocumentCache` before use.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterator

from arc_jobs import canonical_json_bytes as _canonical_json_bytes

from ._cache_root import resolve_cache_root
from ._durable_io import atomic_write_bytes
from ._file_lock import exclusive_file_lock
from .sources import SourceArtifact

if TYPE_CHECKING:
    from .parse.models import ParsedDocument


FULL_TEXT_CATALOG_SCHEMA = "arc.document.full_text_catalog.v2"
FULL_TEXT_CATALOG_ADMIN_SCHEMA = "arc.document.full_text_catalog_admin.v2"


def _origin_document_id(source: SourceArtifact) -> str:
    """Return an optional provider-neutral identity supplied by acquisition."""

    value = str(source.origin.metadata.get("document_id") or "").strip()
    return value


def _valid_document_id(value: str) -> bool:
    return bool(value.strip())


@dataclass(frozen=True)
class FullTextCatalogDialect:
    """Persistence vocabulary for one identified-document catalog."""

    schema_version: str
    admin_schema_version: str
    directory: str
    identified_kind: str
    identifier_field: str
    admin_prefix: str
    identify_source: Callable[[SourceArtifact], str]
    validate_identifier: Callable[[str], bool]


DOCUMENT_FULL_TEXT_CATALOG_DIALECT = FullTextCatalogDialect(
    schema_version=FULL_TEXT_CATALOG_SCHEMA,
    admin_schema_version=FULL_TEXT_CATALOG_ADMIN_SCHEMA,
    directory="document-full-text-catalog",
    identified_kind="identified",
    identifier_field="document_ids",
    admin_prefix="document",
    identify_source=_origin_document_id,
    validate_identifier=_valid_document_id,
)
_REPRESENTATION_FIELDS = {
    "source_identity",
    "parser_contract",
    "parsed_cache_key",
    "document_digest",
}
_SOURCE_IDENTITY_FIELDS = {
    "source_format",
    "media_type",
    "artifact_digest",
    "size",
}
_ADMIN_FIELDS = {
    "schema_version",
    "cached_at",
}


@dataclass(frozen=True)
class FullTextRepresentation:
    """One current format-specific projection selected by a locator."""

    source_identity: Mapping[str, Any]
    parser_contract: str
    parsed_cache_key: str
    document_digest: str

    def __post_init__(self) -> None:
        identity = _validated_source_identity(self.source_identity)
        parser_contract = str(self.parser_contract).strip()
        if not parser_contract:
            raise ValueError("parser_contract is required")
        if not _is_sha256(self.parsed_cache_key):
            raise ValueError("parsed_cache_key must be a SHA-256 digest")
        if not _is_sha256(self.document_digest):
            raise ValueError("document_digest must be a SHA-256 digest")
        object.__setattr__(self, "source_identity", MappingProxyType(identity))
        object.__setattr__(self, "parser_contract", parser_contract)

    @property
    def source_format(self) -> str:
        return str(self.source_identity["source_format"])


@dataclass(frozen=True)
class FullTextCatalogEntry:
    """One current identified or local full-text locator."""

    kind: str
    document_ids: tuple[str, ...]
    local_source_identity: Mapping[str, Any] | None
    representations: tuple[FullTextRepresentation, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"identified", "local"}:
            raise ValueError("catalog entry kind must be identified or local")
        document_ids = tuple(sorted(set(self.document_ids), key=str.casefold))
        representations = tuple(
            sorted(self.representations, key=lambda item: item.source_format)
        )
        formats = [item.source_format for item in representations]
        if len(formats) != len(set(formats)):
            raise ValueError("catalog entry contains duplicate source formats")
        if not representations:
            raise ValueError("catalog entry requires a representation")
        if self.kind == "identified":
            if not document_ids or any(not item.strip() for item in document_ids):
                raise ValueError("remote catalog entry requires document IDs")
            if self.local_source_identity is not None:
                raise ValueError("identified catalog entry cannot have a local identity")
            local_identity = None
        else:
            if document_ids:
                raise ValueError("local catalog entry cannot have document IDs")
            if self.local_source_identity is None:
                raise ValueError("local catalog entry requires a source identity")
            local_identity = MappingProxyType(
                _validated_source_identity(self.local_source_identity)
            )
            if any(
                dict(item.source_identity) != dict(local_identity)
                for item in representations
            ):
                raise ValueError("local representation must match locator identity")
        object.__setattr__(self, "document_ids", document_ids)
        object.__setattr__(self, "local_source_identity", local_identity)
        object.__setattr__(self, "representations", representations)


@dataclass(frozen=True)
class FullTextCatalogAdminEntry:
    entry_id: str
    entry: FullTextCatalogEntry
    cached_at: str


class FullTextCatalog:
    """Atomic per-entry locator index for materialized full text."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        _dialect: FullTextCatalogDialect = DOCUMENT_FULL_TEXT_CATALOG_DIALECT,
    ) -> None:
        self.root = resolve_cache_root(root)
        self._dialect = _dialect

    def record(
        self,
        source: SourceArtifact,
        document: ParsedDocument,
        *,
        parser_contract: str,
        parsed_cache_key: str,
    ) -> FullTextCatalogEntry:
        if document.source.content_identity != source.content_identity:
            raise ValueError("catalog document source does not match source")
        representation = FullTextRepresentation(
            source_identity=_source_identity(source),
            parser_contract=parser_contract,
            parsed_cache_key=parsed_cache_key,
            document_digest=document.document_digest,
        )
        locator_kind, locator_identity, document_id = _locator_identity(
            source, self._dialect
        )
        locator_key = _locator_key(locator_kind, locator_identity, self._dialect)
        path = self._entry_path(locator_key)
        with self._entry_lock(locator_key):
            previous = self._read_path(path)
            if previous is not None and not _entry_matches_locator(
                previous, locator_kind, locator_identity
            ):
                previous = None
            by_format = (
                {item.source_format: item for item in previous.representations}
                if previous is not None
                else {}
            )
            by_format[representation.source_format] = representation
            if locator_kind == "identified":
                document_ids = set(previous.document_ids if previous is not None else ())
                document_ids.add(document_id)
                entry = FullTextCatalogEntry(
                    kind="identified",
                    document_ids=tuple(document_ids),
                    local_source_identity=None,
                    representations=tuple(by_format.values()),
                )
            else:
                entry = FullTextCatalogEntry(
                    kind="local",
                    document_ids=(),
                    local_source_identity=locator_identity,
                    representations=tuple(by_format.values()),
                )
            payload = _canonical_json_bytes(_entry_document(entry, self._dialect))
            try:
                unchanged = path.read_bytes() == payload
            except OSError:
                unchanged = False
            if not unchanged or self._read_admin(path.parent) is None:
                atomic_write_bytes(
                    path.parent / "admin.json",
                    _canonical_json_bytes(
                        {
                            "schema_version": self._dialect.admin_schema_version,
                            "cached_at": _utc_now(),
                        }
                    ),
                )
                atomic_write_bytes(path, payload)
            return entry

    def current_entries(self) -> tuple[FullTextCatalogEntry, ...]:
        """Return only strict current locators; damaged entries are ignored."""

        entries_root = self.root / self._dialect.directory / "v2" / "entries"
        if not entries_root.is_dir():
            return ()
        entries = tuple(
            entry
            for path in entries_root.glob("*/*/locator.json")
            if (entry := self._read_path(path)) is not None
        )
        return tuple(sorted(entries, key=_entry_sort_key))

    def admin_entries(self) -> tuple[FullTextCatalogAdminEntry, ...]:
        """Return current locators with their last successful publication time."""

        entries_root = self.root / self._dialect.directory / "v2" / "entries"
        if not entries_root.is_dir():
            return ()
        entries: list[FullTextCatalogAdminEntry] = []
        for path in entries_root.glob("*/*/locator.json"):
            entry = self._read_path(path)
            if entry is None:
                continue
            cached_at = self._read_admin(path.parent)
            if cached_at is None:
                continue
            entries.append(
                FullTextCatalogAdminEntry(
                    entry_id=_admin_entry_id(entry, self._dialect),
                    entry=entry,
                    cached_at=cached_at,
                )
            )
        return tuple(sorted(entries, key=lambda item: item.entry_id))

    def remove_admin_entry(self, entry_id: str) -> bool:
        """Physically remove one locator and its selected source/parsed objects."""

        selected = next(
            (item for item in self.admin_entries() if item.entry_id == entry_id),
            None,
        )
        if selected is None:
            return False
        removed = False
        for locator_key in sorted(_locator_keys(selected.entry, self._dialect)):
            path = self._entry_path(locator_key)
            with self._entry_lock(locator_key):
                if path.parent.exists():
                    shutil.rmtree(path.parent)
                    removed = True
        from ._parsed_document_cache import ParsedDocumentCache
        from .source_repository import SourceRepository

        repository = SourceRepository(self.root)
        for representation in selected.entry.representations:
            cache = ParsedDocumentCache(
                self.root,
                repository=repository,
                parser_contract=representation.parser_contract,
            )
            removed = (
                cache.remove_by_key(representation.parsed_cache_key) or removed
            )
            removed = (
                repository.remove(
                    representation.source_identity["source_format"],
                    representation.source_identity["artifact_digest"],
                )
                or removed
            )
        return removed

    def _read_path(self, path: Path) -> FullTextCatalogEntry | None:
        try:
            if self._read_admin(path.parent) is None:
                return None
            value = json.loads(path.read_text(encoding="utf-8"))
            entry = _entry_from_document(value, self._dialect)
            if path.parent.name not in _locator_keys(entry, self._dialect):
                return None
            return entry
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def _read_admin(self, entry_dir: Path) -> str | None:
        try:
            value = json.loads(
                (entry_dir / "admin.json").read_text(encoding="utf-8")
            )
            if (
                not isinstance(value, dict)
                or set(value) != _ADMIN_FIELDS
                or value.get("schema_version")
                != self._dialect.admin_schema_version
                or not isinstance(value.get("cached_at"), str)
                or not value["cached_at"]
                or not _is_utc_timestamp(value["cached_at"])
            ):
                return None
            return value["cached_at"]
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            return None

    def _entry_path(self, locator_key: str) -> Path:
        return (
            self.root
            / self._dialect.directory
            / "v2"
            / "entries"
            / locator_key[:2]
            / locator_key
            / "locator.json"
        )

    @contextmanager
    def _entry_lock(self, locator_key: str) -> Iterator[None]:
        path = (
            self.root
            / self._dialect.directory
            / "v2"
            / "locks"
            / f"{locator_key}.lock"
        )
        with exclusive_file_lock(path):
            yield


def _locator_identity(
    source: SourceArtifact,
    dialect: FullTextCatalogDialect,
) -> tuple[str, str | dict[str, Any], str]:
    document_id = dialect.identify_source(source)
    if document_id:
        canonical = document_id
        if not dialect.validate_identifier(canonical):
            raise ValueError("catalog source identifier is invalid")
        return "identified", canonical, canonical
    identity = _source_identity(source)
    return "local", identity, ""


def _locator_key(
    kind: str,
    identity: str | Mapping[str, Any],
    dialect: FullTextCatalogDialect,
) -> str:
    stored_kind = dialect.identified_kind if kind == "identified" else kind
    return hashlib.sha256(
        _canonical_json_bytes({"kind": stored_kind, "identity": identity})
    ).hexdigest()


def _entry_matches_locator(
    entry: FullTextCatalogEntry,
    kind: str,
    identity: str | Mapping[str, Any],
) -> bool:
    if entry.kind != kind:
        return False
    if kind == "identified":
        return identity in entry.document_ids
    return dict(entry.local_source_identity or {}) == dict(identity)


def _locator_keys(
    entry: FullTextCatalogEntry,
    dialect: FullTextCatalogDialect,
) -> set[str]:
    if entry.kind == "identified":
        return {
            _locator_key("identified", document_id, dialect)
            for document_id in entry.document_ids
        }
    return {
        _locator_key("local", dict(entry.local_source_identity or {}), dialect)
    }


def _entry_document(
    entry: FullTextCatalogEntry,
    dialect: FullTextCatalogDialect,
) -> dict[str, Any]:
    return {
        "schema_version": dialect.schema_version,
        "kind": (
            dialect.identified_kind if entry.kind == "identified" else entry.kind
        ),
        dialect.identifier_field: list(entry.document_ids),
        "local_source_identity": (
            dict(entry.local_source_identity)
            if entry.local_source_identity is not None
            else None
        ),
        "representations": [
            {
                "source_identity": dict(item.source_identity),
                "parser_contract": item.parser_contract,
                "parsed_cache_key": item.parsed_cache_key,
                "document_digest": item.document_digest,
            }
            for item in entry.representations
        ],
    }


def _entry_from_document(
    value: object,
    dialect: FullTextCatalogDialect,
) -> FullTextCatalogEntry:
    entry_fields = {
        "schema_version",
        "kind",
        dialect.identifier_field,
        "local_source_identity",
        "representations",
    }
    if not isinstance(value, Mapping) or set(value) != entry_fields:
        raise ValueError("catalog locator has invalid fields")
    if value.get("schema_version") != dialect.schema_version:
        raise ValueError("catalog locator has unsupported schema")
    stored_kind = value.get("kind")
    if stored_kind not in {dialect.identified_kind, "local"}:
        raise ValueError("catalog locator has invalid kind")
    document_ids = value.get(dialect.identifier_field)
    representations = value.get("representations")
    if not isinstance(document_ids, list) or not all(
        isinstance(item, str) for item in document_ids
    ):
        raise ValueError(f"catalog {dialect.identifier_field} must be strings")
    if any(not dialect.validate_identifier(item) for item in document_ids):
        raise ValueError("catalog identifier is invalid")
    if not isinstance(representations, list):
        raise ValueError("catalog representations must be a list")
    decoded: list[FullTextRepresentation] = []
    for item in representations:
        if not isinstance(item, Mapping) or set(item) != _REPRESENTATION_FIELDS:
            raise ValueError("catalog representation has invalid fields")
        decoded.append(
            FullTextRepresentation(
                source_identity=item["source_identity"],
                parser_contract=item["parser_contract"],
                parsed_cache_key=item["parsed_cache_key"],
                document_digest=item["document_digest"],
            )
        )
    local_identity = value.get("local_source_identity")
    if local_identity is not None and not isinstance(local_identity, Mapping):
        raise ValueError("catalog local identity must be an object or null")
    return FullTextCatalogEntry(
        kind="identified" if stored_kind == dialect.identified_kind else "local",
        document_ids=tuple(document_ids),
        local_source_identity=local_identity,
        representations=tuple(decoded),
    )


def _source_identity(source: SourceArtifact) -> dict[str, Any]:
    return {
        "source_format": source.source_format.value,
        "media_type": source.media_type,
        "artifact_digest": source.artifact_digest,
        "size": source.size,
    }


def _validated_source_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SOURCE_IDENTITY_FIELDS:
        raise ValueError("source identity has invalid fields")
    source_format = value.get("source_format")
    media_type = value.get("media_type")
    artifact_digest = value.get("artifact_digest")
    size = value.get("size")
    if (
        source_format not in {"html", "markdown", "tex", "pdf"}
        or not isinstance(media_type, str)
        or not media_type
        or "/" not in media_type
        or ";" in media_type
        or not _is_sha256(artifact_digest)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
    ):
        raise ValueError("source identity is invalid")
    return {
        "source_format": source_format,
        "media_type": media_type,
        "artifact_digest": artifact_digest,
        "size": size,
    }


def _entry_sort_key(entry: FullTextCatalogEntry) -> tuple[str, str]:
    if entry.kind == "identified":
        return entry.kind, entry.document_ids[0].casefold()
    return entry.kind, str(
        (entry.local_source_identity or {}).get("artifact_digest", "")
    )


def _admin_entry_id(
    entry: FullTextCatalogEntry,
    dialect: FullTextCatalogDialect,
) -> str:
    if entry.kind == "identified":
        return f"{dialect.admin_prefix}:{entry.document_ids[0]}"
    identity = entry.local_source_identity or {}
    return (
        f"local:{identity.get('source_format', '')}:"
        f"{identity.get('artifact_digest', '')}"
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_utc_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "FULL_TEXT_CATALOG_ADMIN_SCHEMA",
    "FULL_TEXT_CATALOG_SCHEMA",
    "DOCUMENT_FULL_TEXT_CATALOG_DIALECT",
    "FullTextCatalog",
    "FullTextCatalogAdminEntry",
    "FullTextCatalogDialect",
    "FullTextCatalogEntry",
    "FullTextRepresentation",
]
