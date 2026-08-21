from __future__ import annotations

from pathlib import Path

import pytest

from ac_document import (
    AcDocumentService,
    DocumentInputError,
    SourceFormat,
    SourceOrigin,
    SourceOriginKind,
)


def test_document_cache_lists_and_removes_one_local_projection(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "note.md"
    source_path.write_text("# Note\nbody\n", encoding="utf-8")
    service = AcDocumentService(cache_root=tmp_path / "cache")

    source = service.import_source(source_path)
    listed = service.list_cache()

    assert len(listed.entries) == 1
    entry = listed.entries[0]
    assert entry.entry_id == f"local:markdown:{source.artifact_digest}"
    assert entry.kind == "local"
    assert entry.document_ids == ()
    assert [item.name for item in entry.components] == ["full-text"]
    assert service.remove_cache(entry_ids=(entry.entry_id,)).dry_run
    removed = service.remove_cache(
        entry_ids=(entry.entry_id,), dry_run=False
    )
    assert removed.removed_entry_ids == (entry.entry_id,)
    assert service.list_cache().entries == ()


def test_document_cache_filters_identified_projection(tmp_path: Path) -> None:
    service = AcDocumentService(cache_root=tmp_path / "cache")
    source = service.repository.store_bytes(
        b"# Identified\nbody\n",
        source_format=SourceFormat.MARKDOWN,
        origin=SourceOrigin(
            SourceOriginKind.REMOTE_PROVIDER,
            provider="fixture",
            metadata={"document_id": "arXiv:0911.3380"},
        ),
    )
    service.parser.parse_source(source)

    result = service.list_cache(document_ids=("arXiv:0911.3380",))

    assert len(result.entries) == 1
    assert result.entries[0].kind == "identified"
    assert result.entries[0].document_ids == ("arXiv:0911.3380",)


def test_document_cache_remove_requires_exact_selector(tmp_path: Path) -> None:
    service = AcDocumentService(cache_root=tmp_path / "cache")

    with pytest.raises(DocumentInputError, match="at least one exact"):
        service.remove_cache()
