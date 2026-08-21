"""Typed keyword inventory models and the independent derived cache."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ac_jobs import canonical_json_bytes as _canonical_json_bytes

from ._cache_root import resolve_cache_root
from ._durable_io import atomic_write_bytes
from ._file_lock import exclusive_file_lock
from .document_structure import (
    DocumentStructureNodeKind,
    DocumentStructureOverlay,
)
from .parse import ParsedDocument
from .rich_document import RichBlock, RichBlockKind, RichDocument


TERM_INVENTORY_SCHEMA = "ac.document.term_inventory.v1"
KEYWORD_RESULT_SCHEMA = "ac.document.keyword_result.v1"
TERM_INVENTORY_STORE_SCHEMA = "ac.document.term_inventory_store.v1"
TERM_INVENTORY_CONTRACT = "ac.document.term_inventory_builder.v1"
TERM_INVENTORY_BATCH_SCHEMA = "ac.document.term_inventory_batch.v1"
TERM_INVENTORY_CURRENT_SCHEMA = "ac.document.term_inventory_current.v1"
TERM_INVENTORY_CORRUPT_WARNING = (
    "term-inventory derived cache was corrupt and was rebuilt"
)

_PAYLOAD_FIELDS = {
    "schema_version",
    "lineage_key",
    "document_digest",
    "source_digest",
    "generation",
    "high_water",
    "explicit_disposition",
    "terms",
    "first_occurrence",
    "inventory_digest",
    "cached_at",
}


@dataclass(frozen=True)
class MatchedSentence:
    text: str
    section_id: str
    page_number: int | None
    matched_surface: str
    clipped: bool

    def to_document(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "section_id": self.section_id,
            "page_number": self.page_number,
            "matched_surface": self.matched_surface,
            "clipped": self.clipped,
        }


@dataclass(frozen=True)
class KeywordTerm:
    term_id: str
    term: str
    aliases: tuple[str, ...]
    occurrence_count: int
    source_refs: tuple[str, ...]
    matched_sentences: tuple[MatchedSentence, ...]

    def to_document(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id,
            "term": self.term,
            "aliases": list(self.aliases),
            "occurrence_count": self.occurrence_count,
            "source_refs": list(self.source_refs),
            "matched_sentences": [
                item.to_document() for item in self.matched_sentences
            ],
        }


@dataclass(frozen=True)
class KeywordResult:
    schema_version: str
    document_digest: str
    source_digest: str
    approx_count: int
    planned_count: int
    returned_count: int
    terms: tuple[KeywordTerm, ...]
    inventory_digest: str
    warnings: tuple[str, ...] = ()

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_digest": self.document_digest,
            "source_digest": self.source_digest,
            "approx_count": self.approx_count,
            "planned_count": self.planned_count,
            "returned_count": self.returned_count,
            "terms": [item.to_document() for item in self.terms],
            "inventory_digest": self.inventory_digest,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class TermCandidate:
    term: str
    aliases: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.term, str) or not 1 <= len(self.term.strip()) <= 300:
            raise ValueError("term candidate must contain 1..300 characters")
        if any(
            not isinstance(item, str) or not 1 <= len(item.strip()) <= 300
            for item in self.aliases
        ):
            raise ValueError("term candidate aliases must contain 1..300 characters")
        object.__setattr__(self, "term", self.term.strip())
        object.__setattr__(
            self, "aliases", tuple(item.strip() for item in self.aliases)
        )
        object.__setattr__(self, "source_refs", tuple(self.source_refs))


@dataclass(frozen=True)
class StoredTermInventory:
    lineage_key: str
    document_digest: str
    source_digest: str
    generation: int
    high_water: int
    explicit_disposition: str
    terms: tuple[KeywordTerm, ...]
    first_occurrence: Mapping[str, int | None]
    inventory_digest: str
    cached_at: str


@dataclass(frozen=True)
class TermInventoryAdminEntry:
    entry_id: str
    source_identity: Mapping[str, Any]
    cached_at: str
    time_basis: str = "recorded_utc"


class TermInventoryStoreError(RuntimeError):
    code = "term_inventory_batch_corrupt"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class _CurrentProjectionCorrupt(RuntimeError):
    pass


@dataclass(frozen=True)
class TermInventoryLineage:
    document_digest: str
    source_digest: str
    source_format: str
    source_media_type: str
    source_size: int
    parser_contract: str
    discovery_contract: str
    review_contract: str
    normalization_contract: str
    count_contract: str
    model_requirement: Mapping[str, Any]
    source_structure: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _sha256_string(self.document_digest, "document digest")
        _sha256_string(self.source_digest, "source digest")
        if not self.source_format or not self.source_media_type:
            raise ValueError("source format and media type are required")
        if (
            not isinstance(self.source_size, int)
            or isinstance(self.source_size, bool)
            or self.source_size < 0
        ):
            raise ValueError("source_size must be non-negative")
        for name in (
            "parser_contract",
            "discovery_contract",
            "review_contract",
            "normalization_contract",
            "count_contract",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        object.__setattr__(
            self, "model_requirement", dict(self.model_requirement)
        )
        object.__setattr__(self, "source_structure", dict(self.source_structure))

    def to_document(self) -> dict[str, Any]:
        value = {
            "document_digest": self.document_digest,
            "source_digest": self.source_digest,
            "source_format": self.source_format,
            "source_media_type": self.source_media_type,
            "source_size": self.source_size,
            "parser_contract": self.parser_contract,
            "discovery_contract": self.discovery_contract,
            "review_contract": self.review_contract,
            "normalization_contract": self.normalization_contract,
            "count_contract": self.count_contract,
            "model_requirement": dict(self.model_requirement),
        }
        if self.source_structure:
            value["source_structure"] = dict(self.source_structure)
        return value

    @property
    def key(self) -> str:
        return hashlib.sha256(
            _canonical_json_bytes(self.to_document())
        ).hexdigest()


class TermInventoryStore:
    """Lazy, content-addressed term inventory cache.

    Lineages include document and every behavior/model contract. Successful
    growth writes an immutable batch before atomically advancing ``current``.
    Paths and provider IDs remain outside this component's semantic contract.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = resolve_cache_root(root)

    def load(
        self,
        document: KeywordDocument,
        lineage: TermInventoryLineage,
    ) -> tuple[StoredTermInventory | None, tuple[str, ...]]:
        self._validate_lineage(document, lineage)
        try:
            return self._read_current(lineage), ()
        except _CurrentProjectionCorrupt:
            with exclusive_file_lock(self._lock_path(lineage.key)):
                rebuilt = self._rebuild_current(lineage)
            if rebuilt is None:
                raise TermInventoryStoreError(
                    "term inventory current projection has no verified batch"
                )
            return rebuilt, (TERM_INVENTORY_CORRUPT_WARNING,)

    def merge(
        self,
        document: KeywordDocument,
        lineage: TermInventoryLineage,
        *,
        high_water: int,
        explicit_disposition: str,
        candidates: Sequence[TermCandidate],
    ) -> StoredTermInventory:
        if high_water < 1:
            raise ValueError("term inventory high_water must be positive")
        if explicit_disposition not in {"absent", "reviewed", "discarded"}:
            raise ValueError("invalid explicit term disposition")
        self._validate_lineage(document, lineage)
        key = lineage.key
        with exclusive_file_lock(self._lock_path(key)):
            try:
                previous = self._read_current(lineage)
            except _CurrentProjectionCorrupt:
                previous = self._rebuild_current(lineage)
                if previous is None:
                    raise TermInventoryStoreError(
                        "term inventory current projection has no verified batch"
                    )
            merged_candidates = list(candidates)
            if previous is not None:
                merged_candidates.extend(
                    TermCandidate(item.term, item.aliases, item.source_refs)
                    for item in previous.terms
                )
                high_water = max(high_water, previous.high_water)
                explicit_disposition = _merged_disposition(
                    previous.explicit_disposition, explicit_disposition
                )
            terms = build_keyword_terms(document, merged_candidates)
            first_occurrence = _first_occurrence_positions(document, terms)
            inventory_digest = _inventory_digest(
                lineage.key,
                document.document_digest,
                document.source.artifact_digest,
                terms,
                first_occurrence,
            )
            cached_at = _utc_now()
            value = StoredTermInventory(
                lineage.key,
                document.document_digest,
                document.source.artifact_digest,
                1 if previous is None else previous.generation + 1,
                high_water,
                explicit_disposition,
                terms,
                first_occurrence,
                inventory_digest,
                cached_at,
            )
            self._write_batch_and_current(lineage, value)
            return value

    def admin_entries(self) -> tuple[TermInventoryAdminEntry, ...]:
        base = self.root / "term-inventory" / "v1" / "lineages"
        if not base.is_dir():
            return ()
        values: list[TermInventoryAdminEntry] = []
        for current_path in base.glob("*/*/current.json"):
            try:
                current = self._read_current_document(current_path)
                lineage_key = _sha256_string(
                    current.get("lineage_key"), "lineage key"
                )
                if self._entry_dir(lineage_key) != current_path.parent:
                    continue
                value = self._read_batch(
                    lineage_key,
                    _sha256_string(
                        current.get("current_batch_digest"), "batch digest"
                    ),
                    expected_lineage=None,
                )
                values.append(
                    TermInventoryAdminEntry(
                        entry_id=f"term-inventory:{lineage_key}",
                        source_identity={
                            "source_format": str(
                                current["lineage"]["source_format"]
                            ),
                            "media_type": str(
                                current["lineage"]["source_media_type"]
                            ),
                            "artifact_digest": value.source_digest,
                            "size": int(current["lineage"]["source_size"]),
                        },
                        cached_at=value.cached_at,
                    )
                )
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                ValueError,
                _CurrentProjectionCorrupt,
                TermInventoryStoreError,
                TypeError,
                KeyError,
            ):
                continue
        return tuple(sorted(values, key=lambda item: item.entry_id))

    def remove_admin_entry(self, entry_id: str) -> bool:
        prefix = "term-inventory:"
        if not entry_id.startswith(prefix):
            return False
        key = entry_id[len(prefix) :]
        if not _is_sha256(key):
            return False
        entry_dir = self._entry_dir(key)
        with exclusive_file_lock(self._lock_path(key)):
            if not entry_dir.exists():
                return False
            shutil.rmtree(entry_dir)
            return True

    def _read_current(
        self, lineage: TermInventoryLineage
    ) -> StoredTermInventory | None:
        current_path = self._entry_dir(lineage.key) / "current.json"
        if not current_path.exists():
            if any(self._batch_root(lineage.key).glob("*/*.json")):
                raise _CurrentProjectionCorrupt(
                    "term inventory current projection is missing"
                )
            return None
        try:
            current = self._read_current_document(current_path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise _CurrentProjectionCorrupt(
                "term inventory current projection is unreadable"
            ) from exc
        try:
            if (
                current.get("schema_version") != TERM_INVENTORY_CURRENT_SCHEMA
                or current.get("lineage_key") != lineage.key
                or current.get("lineage") != lineage.to_document()
            ):
                raise _CurrentProjectionCorrupt(
                    "term inventory current lineage mismatch"
                )
            return self._read_batch(
                lineage.key,
                _sha256_string(
                    current.get("current_batch_digest"), "batch digest"
                ),
                expected_lineage=lineage,
            )
        except ValueError as exc:
            raise _CurrentProjectionCorrupt(
                "term inventory current projection is invalid"
            ) from exc

    def _read_current_document(self, path: Path) -> Mapping[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise _CurrentProjectionCorrupt(
                "term inventory current projection is invalid"
            ) from exc
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "lineage_key",
            "lineage",
            "current_batch_digest",
        }:
            raise _CurrentProjectionCorrupt(
                "term inventory current projection has invalid fields"
            )
        return value

    def _read_batch(
        self,
        lineage_key: str,
        batch_digest: str,
        *,
        expected_lineage: TermInventoryLineage | None,
    ) -> StoredTermInventory:
        path = self._batch_path(lineage_key, batch_digest)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise TermInventoryStoreError(
                "term inventory current batch is missing"
            ) from exc
        if hashlib.sha256(payload).hexdigest() != batch_digest:
            raise TermInventoryStoreError(
                "term inventory batch digest mismatch"
            )
        try:
            document = json.loads(payload)
            value = _stored_inventory_from_document(document)
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise TermInventoryStoreError(
                "term inventory batch is semantically invalid"
            ) from exc
        if value.lineage_key != lineage_key:
            raise TermInventoryStoreError(
                "term inventory batch lineage mismatch"
            )
        if expected_lineage is not None and (
            value.document_digest != expected_lineage.document_digest
            or value.source_digest != expected_lineage.source_digest
        ):
            raise TermInventoryStoreError(
                "term inventory batch identity mismatch"
            )
        if (
            value.inventory_digest
            != _inventory_digest(
                lineage_key,
                value.document_digest,
                value.source_digest,
                value.terms,
                value.first_occurrence,
            )
        ):
            raise TermInventoryStoreError(
                "term inventory batch semantic digest mismatch"
            )
        return value

    def _rebuild_current(
        self, lineage: TermInventoryLineage
    ) -> StoredTermInventory | None:
        valid: list[tuple[str, StoredTermInventory]] = []
        for path in self._batch_root(lineage.key).glob("*/*.json"):
            digest = path.stem
            if not _is_sha256(digest):
                continue
            try:
                valid.append(
                    (
                        digest,
                        self._read_batch(
                            lineage.key,
                            digest,
                            expected_lineage=lineage,
                        ),
                    )
                )
            except TermInventoryStoreError:
                raise
        if not valid:
            return None
        digest, value = max(
            valid,
            key=lambda item: (
                item[1].generation,
                item[1].cached_at,
                item[0],
            ),
        )
        self._write_current(lineage, digest)
        return value

    def _write_batch_and_current(
        self,
        lineage: TermInventoryLineage,
        value: StoredTermInventory,
    ) -> None:
        payload = _canonical_json_bytes(_stored_inventory_document(value))
        digest = hashlib.sha256(payload).hexdigest()
        path = self._batch_path(lineage.key, digest)
        if path.exists():
            if path.read_bytes() != payload:
                raise TermInventoryStoreError(
                    "immutable term inventory batch conflicts"
                )
        else:
            atomic_write_bytes(path, payload)
        self._write_current(lineage, digest)

    def _write_current(
        self, lineage: TermInventoryLineage, batch_digest: str
    ) -> None:
        current = {
            "schema_version": TERM_INVENTORY_CURRENT_SCHEMA,
            "lineage_key": lineage.key,
            "lineage": lineage.to_document(),
            "current_batch_digest": batch_digest,
        }
        atomic_write_bytes(
            self._entry_dir(lineage.key) / "current.json",
            _canonical_json_bytes(current),
        )

    def _validate_lineage(
        self, document: KeywordDocument, lineage: TermInventoryLineage
    ) -> None:
        if (
            lineage.document_digest != document.document_digest
            or lineage.source_digest != document.source.artifact_digest
        ):
            raise ValueError("term inventory lineage differs from document")

    def _entry_dir(self, lineage_key: str) -> Path:
        return (
            self.root
            / "term-inventory"
            / "v1"
            / "lineages"
            / lineage_key[:2]
            / lineage_key
        )

    def _batch_root(self, lineage_key: str) -> Path:
        return self._entry_dir(lineage_key) / "batches"

    def _batch_path(self, lineage_key: str, batch_digest: str) -> Path:
        return (
            self._batch_root(lineage_key)
            / batch_digest[:2]
            / f"{batch_digest}.json"
        )

    def _lock_path(self, lineage_key: str) -> Path:
        return (
            self.root
            / "term-inventory"
            / "v1"
            / "locks"
            / f"{lineage_key}.lock"
        )


def validate_approx_count(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= 200
    ):
        raise ValueError("approx_count must be an integer between 1 and 200")
    return value


def normalize_term(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"\s+", " ", normalized).strip(" \t\r\n,;:.")
    return normalized.casefold()


def literal_term_occurs(text: str, surfaces: Sequence[str]) -> bool:
    """Return whether a term surface occurs without crossing a word boundary."""

    return _literal_term_pattern(surfaces).search(text) is not None


def _literal_term_pattern(surfaces: Sequence[str]) -> re.Pattern[str]:
    alternatives = []
    for surface in sorted(
        {item for item in surfaces if item},
        key=lambda item: (-len(item), item.casefold(), item),
    ):
        prefix = r"(?<!\w)" if re.match(r"\w", surface[0]) else ""
        suffix = r"(?!\w)" if re.match(r"\w", surface[-1]) else ""
        alternatives.append(f"{prefix}{re.escape(surface)}{suffix}")
    return re.compile("|".join(alternatives) or r"(?!x)x", re.IGNORECASE)


def build_keyword_terms(
    document: KeywordDocument, candidates: Iterable[TermCandidate]
) -> tuple[KeywordTerm, ...]:
    grouped: dict[str, dict[str, set[str]]] = {}
    for candidate in candidates:
        surfaces = (candidate.term, *candidate.aliases)
        normalized = normalize_term(candidate.term)
        if not normalized:
            continue
        value = grouped.setdefault(
            normalized, {"surfaces": set(), "source_refs": set()}
        )
        value["surfaces"].update(
            surface.strip()
            for surface in surfaces
            if isinstance(surface, str) and normalize_term(surface)
        )
        value["source_refs"].update(
            item.strip() for item in candidate.source_refs if item.strip()
        )
    output: list[KeywordTerm] = []
    for normalized, value in grouped.items():
        surfaces = sorted(
            value["surfaces"],
            key=lambda item: (item.casefold(), item),
        )
        canonical = next(
            (item for item in surfaces if normalize_term(item) == normalized),
            surfaces[0],
        )
        aliases = tuple(item for item in surfaces if item != canonical)
        occurrence_count, sentences = _literal_occurrences(
            document, (canonical, *aliases)
        )
        output.append(
            KeywordTerm(
                term_id=f"term-{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:20]}",
                term=canonical,
                aliases=aliases,
                occurrence_count=occurrence_count,
                source_refs=tuple(sorted(value["source_refs"])),
                matched_sentences=sentences,
            )
        )
    return tuple(sorted(output, key=lambda item: item.term_id))


def result_from_inventory(
    value: StoredTermInventory,
    *,
    approx_count: int,
    planned_count: int,
    warnings: Sequence[str] = (),
) -> KeywordResult:
    approx_count = validate_approx_count(approx_count)
    if planned_count != (3 * approx_count + 1) // 2:
        raise ValueError("planned_count must equal ceil(1.5 * approx_count)")
    ordered = tuple(
        sorted(
            value.terms,
            key=lambda item: (
                -item.occurrence_count,
                (
                    value.first_occurrence.get(item.term_id)
                    if value.first_occurrence.get(item.term_id) is not None
                    else 2**63 - 1
                ),
                normalize_term(item.term),
                item.term,
            ),
        )[:planned_count]
    )
    return KeywordResult(
        schema_version=KEYWORD_RESULT_SCHEMA,
        document_digest=value.document_digest,
        source_digest=value.source_digest,
        approx_count=approx_count,
        planned_count=planned_count,
        returned_count=len(ordered),
        terms=ordered,
        inventory_digest=value.inventory_digest,
        warnings=tuple(dict.fromkeys(item for item in warnings if item)),
    )


def keyword_result_from_document(value: Mapping[str, Any]) -> KeywordResult:
    if value.get("schema_version") != KEYWORD_RESULT_SCHEMA:
        raise ValueError("unsupported keyword result schema")
    terms = tuple(_keyword_term_from_document(item) for item in _list(value, "terms"))
    returned_count = _integer(value, "returned_count")
    if returned_count != len(terms):
        raise ValueError("keyword returned_count does not match terms")
    if len({item.term_id for item in terms}) != len(terms):
        raise ValueError("keyword result contains duplicate term IDs")
    if any(
        len(item.matched_sentences) > 10
        or len({sentence.text for sentence in item.matched_sentences})
        != len(item.matched_sentences)
        for item in terms
    ):
        raise ValueError("keyword matched_sentences must be distinct and limited to 10")
    planned_count = _positive_integer(value.get("planned_count"), "planned_count")
    approx_count = validate_approx_count(value.get("approx_count"))
    if planned_count != (3 * approx_count + 1) // 2:
        raise ValueError("keyword planned_count does not match approx_count")
    if returned_count > planned_count:
        raise ValueError("keyword result exceeds planned_count")
    warnings = _string_tuple(value.get("warnings"), "warnings")
    return KeywordResult(
        KEYWORD_RESULT_SCHEMA,
        _sha256_string(value.get("document_digest"), "document digest"),
        _sha256_string(value.get("source_digest"), "source digest"),
        approx_count,
        planned_count,
        returned_count,
        terms,
        _sha256_string(value.get("inventory_digest"), "inventory digest"),
        warnings,
    )


def _literal_occurrences(
    document: KeywordDocument, surfaces: Sequence[str]
) -> tuple[int, tuple[MatchedSentence, ...]]:
    ordered_surfaces = tuple(
        sorted(
            {item for item in surfaces if item},
            key=lambda item: (-len(item), item.casefold(), item),
        )
    )
    if not ordered_surfaces:
        return 0, ()
    pattern = _literal_term_pattern(ordered_surfaces)
    count = 0
    matched_sentences: list[MatchedSentence] = []
    seen_sentences: set[str] = set()
    for unit in keyword_text_units(document):
        count += sum(1 for _ in pattern.finditer(unit.text))
        for sentence in _sentences(unit.text):
            match = pattern.search(sentence)
            if match is None or len(matched_sentences) >= 10:
                continue
            clipped_text, clipped = _clip_sentence(sentence, match.start(), match.end())
            clipped_match = pattern.search(clipped_text)
            if clipped_text in seen_sentences:
                continue
            seen_sentences.add(clipped_text)
            matched_sentences.append(
                MatchedSentence(
                    text=clipped_text,
                    section_id=unit.section_id,
                    page_number=unit.page_number,
                    matched_surface=(
                        clipped_match.group(0)
                        if clipped_match is not None
                        else match.group(0)
                    ),
                    clipped=clipped,
                )
            )
    return count, tuple(matched_sentences)


def _first_occurrence_positions(
    document: KeywordDocument,
    terms: Sequence[KeywordTerm],
) -> dict[str, int | None]:
    units = keyword_text_units(document)
    positions: dict[str, int | None] = {}
    for term in terms:
        surfaces = tuple(
            sorted(
                {term.term, *term.aliases},
                key=lambda item: (-len(item), item.casefold(), item),
            )
        )
        pattern = _literal_term_pattern(surfaces)
        offset = 0
        found: int | None = None
        for unit in units:
            match = pattern.search(unit.text)
            if match is not None:
                found = offset + match.start()
                break
            offset += len(unit.text) + 1
        positions[term.term_id] = found
    return positions


KeywordDocument = ParsedDocument | RichDocument


@dataclass(frozen=True)
class KeywordTextUnit:
    section_id: str
    title: str
    text: str
    page_number: int | None = None


def keyword_chapters(
    document: KeywordDocument,
    *,
    structure: DocumentStructureOverlay | None = None,
    section_ids: Sequence[str] | None = None,
) -> tuple[KeywordTextUnit, ...]:
    if structure is not None:
        return _structured_keyword_chapters(
            document,
            structure=structure,
            section_ids=section_ids,
        )
    if section_ids is not None:
        raise ValueError("section_ids requires a document structure overlay")
    if isinstance(document, ParsedDocument):
        return tuple(
            KeywordTextUnit(
                section.section_id,
                section.title,
                section.text,
                section.page_start,
            )
            for section in document.sections
            if not _is_explicit_term_title(section.title)
        )
    if not document.sections:
        text = "\n".join(
            value for block in document.blocks if (value := _rich_block_text(block))
        )
        return (
            (KeywordTextUnit("document", "Document", text),) if text else ()
        )
    minimum_level = min(section.level for section in document.sections)
    chapters = tuple(
        section
        for section in document.sections
        if section.level == minimum_level
        and not _is_explicit_term_title(section.title)
    )
    values: list[KeywordTextUnit] = []
    for section in chapters:
        text = "\n".join(
            value
            for block in document.blocks
            if block.section_path
            and block.section_path[0] == section.section_id
            and (value := _rich_block_text(block))
        )
        if text:
            values.append(
                KeywordTextUnit(section.section_id, section.title, text)
            )
    return tuple(values)


def _structured_keyword_chapters(
    document: KeywordDocument,
    *,
    structure: DocumentStructureOverlay,
    section_ids: Sequence[str] | None,
) -> tuple[KeywordTextUnit, ...]:
    if not isinstance(document, RichDocument):
        raise ValueError(
            "document structure keyword grouping requires a RichDocument"
        )
    if not isinstance(structure, DocumentStructureOverlay):
        raise TypeError("structure must be a DocumentStructureOverlay")
    cached = structure.document
    source = document.source
    if (
        cached.source_format is not source.source_format
        or cached.source_sha256 != source.artifact_digest
        or cached.source_size != source.size
        or cached.media_type != source.media_type
    ):
        raise ValueError("document structure overlay differs from keyword source")

    requested = None
    if section_ids is not None:
        requested = tuple(str(item) for item in section_ids)
        if not requested or any(not item for item in requested):
            raise ValueError("section_ids must contain non-empty IDs")
        if len(set(requested)) != len(requested):
            raise ValueError("section_ids must be unique")
    allowed_kinds = (
        {DocumentStructureNodeKind.CONTENT}
        if requested is None
        else {
            DocumentStructureNodeKind.CONTENT,
            DocumentStructureNodeKind.INTERNAL,
        }
    )
    content = tuple(
        item
        for item in structure.entries
        if item.kind in allowed_kinds
        and not _is_explicit_term_title(item.title)
        and (requested is None or item.section_id in requested)
    )
    if requested is not None and {item.section_id for item in content} != set(
        requested
    ):
        raise ValueError(
            "section_ids must identify content or internal entries in the "
            "structure overlay"
        )

    values: list[KeywordTextUnit] = []
    seen_blocks: set[str] = set()
    for entry in content:
        selected: list[str] = []
        for block in document.blocks:
            line_start = block.locator.line_start
            line_end = block.locator.line_end
            if line_start is None or line_end is None:
                continue
            if (
                line_end < entry.source_line_start
                or line_start > entry.source_line_end
            ):
                continue
            if block.block_id in seen_blocks:
                raise ValueError(
                    "document structure content ranges overlap keyword blocks"
                )
            text = _rich_block_text(block)
            if text:
                selected.append(text)
                seen_blocks.add(block.block_id)
        text = "\n".join(selected)
        if text:
            values.append(
                KeywordTextUnit(
                    entry.section_id,
                    entry.title,
                    text,
                    entry.pdf_page_start,
                )
            )
    if not values:
        raise ValueError(
            "document structure contains no mapped keyword content"
        )
    return tuple(values)


def keyword_text_units(
    document: KeywordDocument,
) -> tuple[KeywordTextUnit, ...]:
    if isinstance(document, ParsedDocument):
        return tuple(
            KeywordTextUnit(
                unit.section_id,
                unit.title,
                _without_explicit_regions(document, unit.text),
                unit.page_number,
            )
            for unit in keyword_chapters(document)
        )
    pages = {item.block_id: item.page_number for item in document.page_map}
    values: list[KeywordTextUnit] = []
    titles = {item.section_id: item.title for item in document.sections}
    for block in document.blocks:
        text = _rich_block_text(block)
        if not text:
            continue
        section_id = block.section_path[-1] if block.section_path else "document"
        if _is_explicit_term_title(titles.get(section_id, "")):
            continue
        values.append(
            KeywordTextUnit(
                section_id,
                titles.get(section_id, "Document"),
                _without_explicit_regions(document, text),
                pages.get(block.block_id),
            )
        )
    return tuple(values)


def _is_explicit_term_title(value: str) -> bool:
    title = re.sub(r"[^a-z]", "", value.casefold())
    return title in {
        "keyword",
        "keywords",
        "keyterms",
        "index",
        "subjectindex",
        "indexterms",
    }


def _without_explicit_regions(
    document: KeywordDocument, text: str
) -> str:
    fields = document.metadata.get("explicit_term_fields")
    if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
        return text
    output = text
    for raw in fields:
        if not isinstance(raw, Mapping):
            continue
        entries = raw.get("entries")
        if not isinstance(entries, Sequence) or isinstance(
            entries, (str, bytes)
        ):
            continue
        values = [str(item).strip() for item in entries if str(item).strip()]
        if not values:
            continue
        label = str(raw.get("label") or raw.get("kind") or "").strip()
        normalized_values = {normalize_term(item) for item in values}
        visible_lines = [
            normalize_term(re.sub(r"^\s*[-*+]\s*", "", line))
            for line in output.splitlines()
            if line.strip()
        ]
        if visible_lines and all(
            line in normalized_values for line in visible_lines
        ):
            return ""
        if label:
            output = re.sub(
                rf"(?ms)^\s*{re.escape(label)}\s*:\s*"
                rf"(?:\[[^\n]*\]|[^\n]*(?:\n\s*[-*+]\s*[^\n]+)*)",
                " ",
                output,
                flags=re.IGNORECASE,
            )
        joined = (
            r"\s*(?:[,;]|\s)\s*"
        ).join(re.escape(item) for item in values)
        output = re.sub(joined, " ", output, flags=re.IGNORECASE)
        output = re.sub(
            r"\\(?:keywords?|keyterms?)\*?\s*\{"
            + r"\s*[,;]\s*".join(re.escape(item) for item in values)
            + r"\}",
            " ",
            output,
            flags=re.IGNORECASE | re.DOTALL,
        )
    return output


def _rich_block_text(block: RichBlock) -> str:
    payload = block.payload
    if block.kind in {RichBlockKind.HEADING, RichBlockKind.PARAGRAPH, RichBlockKind.CODE}:
        return str(payload.get("text") or "").strip()
    if block.kind is RichBlockKind.LIST:
        return "\n".join(
            str(item.get("text") or "")
            for item in payload.get("items", ())
            if isinstance(item, Mapping) and str(item.get("text") or "")
        )
    if block.kind is RichBlockKind.EQUATION:
        return str(payload.get("tex") or "").strip()
    if block.kind is RichBlockKind.TABLE:
        parts = [
            str(payload.get("caption") or ""),
            *(str(item) for item in payload.get("headers", ())),
            *(
                " | ".join(str(cell) for cell in row)
                for row in payload.get("rows", ())
            ),
        ]
        return "\n".join(item for item in parts if item)
    if block.kind is RichBlockKind.FIGURE:
        return str(payload.get("caption") or "").strip()
    return ""


def _sentences(text: str) -> Iterable[str]:
    for value in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = re.sub(r"\s+", " ", value).strip()
        if sentence:
            yield sentence


def _clip_sentence(value: str, start: int, end: int) -> tuple[str, bool]:
    if len(value) <= 400:
        return value, False
    match_width = end - start
    context_before = max(0, (400 - match_width) // 2)
    window_start = max(0, min(start - context_before, len(value) - 400))
    window_end = min(len(value), window_start + 400)
    clipped = value[window_start:window_end]
    if window_start:
        clipped = "…" + clipped[1:]
    if window_end < len(value):
        clipped = clipped[:-1] + "…"
    return clipped, True


def _stored_inventory_document(value: StoredTermInventory) -> dict[str, Any]:
    return {
        "schema_version": TERM_INVENTORY_BATCH_SCHEMA,
        "lineage_key": value.lineage_key,
        "document_digest": value.document_digest,
        "source_digest": value.source_digest,
        "generation": value.generation,
        "high_water": value.high_water,
        "explicit_disposition": value.explicit_disposition,
        "terms": [item.to_document() for item in value.terms],
        "first_occurrence": {
            key: item
            for key, item in sorted(value.first_occurrence.items())
        },
        "inventory_digest": value.inventory_digest,
        "cached_at": value.cached_at,
    }


def _stored_inventory_from_document(value: Any) -> StoredTermInventory:
    if not isinstance(value, Mapping) or set(value) != _PAYLOAD_FIELDS:
        raise ValueError("invalid term inventory fields")
    if value.get("schema_version") != TERM_INVENTORY_BATCH_SCHEMA:
        raise ValueError("unsupported term inventory schema")
    disposition = value.get("explicit_disposition")
    if disposition not in {"absent", "reviewed", "discarded"}:
        raise ValueError("invalid explicit term disposition")
    first_occurrence_raw = value.get("first_occurrence")
    if not isinstance(first_occurrence_raw, Mapping):
        raise ValueError("first_occurrence must be an object")
    first_occurrence: dict[str, int | None] = {}
    for key, item in first_occurrence_raw.items():
        if not isinstance(key, str) or (
            item is not None
            and (
                not isinstance(item, int)
                or isinstance(item, bool)
                or item < 0
            )
        ):
            raise ValueError("first_occurrence entry is invalid")
        first_occurrence[key] = item
    terms = tuple(
        _keyword_term_from_document(item) for item in _list(value, "terms")
    )
    if tuple(item.term_id for item in terms) != tuple(
        sorted(item.term_id for item in terms)
    ):
        raise ValueError("stored terms must be ordered by term_id")
    if set(first_occurrence) != {item.term_id for item in terms}:
        raise ValueError("first_occurrence keys differ from terms")
    return StoredTermInventory(
        lineage_key=_sha256_string(value.get("lineage_key"), "lineage key"),
        document_digest=_sha256_string(
            value.get("document_digest"), "document digest"
        ),
        source_digest=_sha256_string(value.get("source_digest"), "source digest"),
        generation=_positive_integer(value.get("generation"), "generation"),
        high_water=_positive_integer(value.get("high_water"), "high_water"),
        explicit_disposition=disposition,
        terms=terms,
        first_occurrence=first_occurrence,
        inventory_digest=_sha256_string(
            value.get("inventory_digest"), "inventory digest"
        ),
        cached_at=_timestamp(value.get("cached_at")),
    )


def _keyword_term_from_document(value: Any) -> KeywordTerm:
    if not isinstance(value, Mapping):
        raise ValueError("keyword term must be an object")
    sentences = tuple(
        _matched_sentence_from_document(item)
        for item in _list(value, "matched_sentences")
    )
    count = _integer(value, "occurrence_count")
    if count < 0:
        raise ValueError("keyword occurrence_count must be non-negative")
    term = _required_string(value, "term")
    expected_id = (
        f"term-{hashlib.sha256(normalize_term(term).encode('utf-8')).hexdigest()[:20]}"
    )
    if value.get("term_id") != expected_id:
        raise ValueError("keyword term ID does not match normalized term")
    return KeywordTerm(
        expected_id,
        term,
        _string_tuple(value.get("aliases"), "aliases"),
        count,
        _string_tuple(value.get("source_refs"), "source_refs"),
        sentences,
    )


def _matched_sentence_from_document(value: Any) -> MatchedSentence:
    if not isinstance(value, Mapping):
        raise ValueError("matched sentence must be an object")
    page = value.get("page_number")
    if page is not None and (
        not isinstance(page, int) or isinstance(page, bool) or page < 1
    ):
        raise ValueError("matched sentence page_number is invalid")
    clipped = value.get("clipped")
    if not isinstance(clipped, bool):
        raise ValueError("matched sentence clipped must be boolean")
    text = _required_string(value, "text")
    if len(text) > 400:
        raise ValueError("matched sentence exceeds 400 characters")
    return MatchedSentence(
        text,
        _required_string(value, "section_id"),
        page,
        _required_string(value, "matched_surface"),
        clipped,
    )


def _inventory_digest(
    lineage_key: str,
    document_digest: str,
    source_digest: str,
    terms: Sequence[KeywordTerm],
    first_occurrence: Mapping[str, int | None],
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "contract": TERM_INVENTORY_CONTRACT,
                "lineage_key": lineage_key,
                "document_digest": document_digest,
                "source_digest": source_digest,
                "terms": [item.to_document() for item in terms],
                "first_occurrence": {
                    key: item
                    for key, item in sorted(first_occurrence.items())
                },
            }
        )
    ).hexdigest()


def _merged_disposition(previous: str, incoming: str) -> str:
    order = {"absent": 0, "discarded": 1, "reviewed": 2}
    return max((previous, incoming), key=order.__getitem__)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _timestamp(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("cache timestamp must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("cache timestamp must have a UTC offset")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _sha256_string(value: Any, description: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{description} must be a SHA-256 digest")
    return value


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return item


def _list(value: Mapping[str, Any], key: str) -> list[Any]:
    item = value.get(key)
    if not isinstance(item, list):
        raise ValueError(f"{key} must be an array")
    return item


def _integer(value: Mapping[str, Any], key: str) -> int:
    return _positive_or_zero_integer(value.get(key), key)


def _positive_integer(value: Any, key: str) -> int:
    item = _positive_or_zero_integer(value, key)
    if item < 1:
        raise ValueError(f"{key} must be positive")
    return item


def _positive_or_zero_integer(value: Any, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _string_tuple(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"{key} must be an array of strings")
    return tuple(value)


__all__ = [
    "KEYWORD_RESULT_SCHEMA",
    "KeywordDocument",
    "KeywordResult",
    "KeywordTerm",
    "KeywordTextUnit",
    "MatchedSentence",
    "StoredTermInventory",
    "TERM_INVENTORY_CORRUPT_WARNING",
    "TERM_INVENTORY_BATCH_SCHEMA",
    "TERM_INVENTORY_CURRENT_SCHEMA",
    "TERM_INVENTORY_SCHEMA",
    "TermCandidate",
    "TermInventoryAdminEntry",
    "TermInventoryLineage",
    "TermInventoryStore",
    "TermInventoryStoreError",
    "build_keyword_terms",
    "keyword_result_from_document",
    "keyword_chapters",
    "keyword_text_units",
    "literal_term_occurs",
    "normalize_term",
    "result_from_inventory",
    "validate_approx_count",
]
