from __future__ import annotations

import json
from pathlib import Path

import pytest

from arc_document import (
    CachedFullTextSearchError,
    CachedFullTextSearchMode,
    CachedFullTextSearcher,
    DocumentParserService,
    SourceFormat,
    SourceOrigin,
    SourceOriginKind,
    SourceRepository,
)
from arc_document._full_text_catalog import FullTextCatalog


class AllCandidates:
    def ensure_available(self) -> None:
        pass

    def files_with_matches(
        self,
        patterns,
        paths,
        *,
        case_sensitive: bool,
    ) -> tuple[Path, ...]:
        return tuple(Path(item) for item in paths)


def _materialize(
    repository: SourceRepository,
    payload: bytes,
    *,
    document_id: str = "",
):
    source = repository.store_bytes(
        payload,
        source_format=SourceFormat.MARKDOWN,
        origin=SourceOrigin(
            (
                SourceOriginKind.REMOTE_PROVIDER
                if document_id
                else SourceOriginKind.LOCAL_IMPORT
            ),
            provider="fixture" if document_id else "",
            metadata={"document_id": document_id} if document_id else {},
        ),
    )
    return source, DocumentParserService(repository).parse_source(source)


def test_document_catalog_preserves_document_dialect(tmp_path: Path) -> None:
    repository = SourceRepository(tmp_path / "cache")
    _materialize(
        repository,
        b"# Document title\n\nneedle once\nneedle twice\n",
        document_id="urn:fixture:one",
    )

    entry = FullTextCatalog(repository.root).current_entries()[0]
    locator = next(
        repository.root.glob(
            "document-full-text-catalog/v2/entries/*/*/locator.json"
        )
    )
    payload = json.loads(locator.read_text(encoding="utf-8"))

    assert entry.kind == "identified"
    assert entry.document_ids == ("urn:fixture:one",)
    assert payload["schema_version"] == "arc.document.full_text_catalog.v2"
    assert payload["kind"] == "identified"
    assert payload["document_ids"] == ["urn:fixture:one"]
    assert "paper_ids" not in payload


def test_document_search_maps_neutral_engine_to_document_fields(
    tmp_path: Path,
) -> None:
    repository = SourceRepository(tmp_path / "cache")
    _materialize(
        repository,
        b"# Document title\n\nneedle once\nneedle twice\n",
        document_id="urn:fixture:one",
    )
    searcher = CachedFullTextSearcher(
        repository.root,
        candidate_selector=AllCandidates(),
    )

    result = searcher.search(("needle",), limit=10, context_lines=1)
    broad = searcher.search(("needle",), limit=1)

    assert result.mode is CachedFullTextSearchMode.OCCURRENCES
    assert result.total_occurrences == 2
    assert result.occurrences[0].source_kind == "identified"
    assert result.occurrences[0].document_ids == ("urn:fixture:one",)
    assert result.documents[0].document_ids == ("urn:fixture:one",)
    assert result.top_document_titles == ()
    assert broad.mode is CachedFullTextSearchMode.REFINEMENT_REQUIRED
    assert broad.top_document_titles == ("Document title",)


def test_document_catalog_keeps_local_identity_and_removes_entry(
    tmp_path: Path,
) -> None:
    repository = SourceRepository(tmp_path / "cache")
    source, _ = _materialize(repository, b"# Local\n\nneedle\n")
    catalog = FullTextCatalog(repository.root)
    entry = catalog.admin_entries()[0]

    assert entry.entry.kind == "local"
    assert entry.entry.document_ids == ()
    assert entry.entry.local_source_identity["artifact_digest"] == (
        source.artifact_digest
    )
    assert entry.entry_id == f"local:markdown:{source.artifact_digest}"
    assert catalog.remove_admin_entry(entry.entry_id)
    assert catalog.current_entries() == ()


def test_document_search_retains_typed_invalid_request(tmp_path: Path) -> None:
    searcher = CachedFullTextSearcher(
        tmp_path / "cache",
        candidate_selector=AllCandidates(),
    )

    with pytest.raises(CachedFullTextSearchError) as error:
        searcher.search(())

    assert error.value.code == "invalid_search_request"
