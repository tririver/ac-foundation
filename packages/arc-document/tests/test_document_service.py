from __future__ import annotations

import json
import types
from pathlib import Path

from arc_document import (
    CACHED_DOCUMENT_REF_SCHEMA,
    DOCUMENT_STRUCTURE_OVERLAY_SCHEMA,
    PARSED_DOCUMENT_SCHEMA,
    RICH_DOCUMENT_SCHEMA,
    ArcDocumentService,
    cached_document_ref_to_document,
)
from arc_document.cli import main


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "source.md"
    path.write_text("# One\n\nAlpha.\n\n## Two\n\nBeta.\n", encoding="utf-8")
    return path


def test_generic_schemas_are_owned_by_arc_document() -> None:
    import arc_document

    assert isinstance(arc_document.parse, types.ModuleType)
    assert arc_document.parse.__name__ == "arc_document.parse"
    assert PARSED_DOCUMENT_SCHEMA == "arc.document.parsed_document.v2"
    assert RICH_DOCUMENT_SCHEMA == "arc.document.rich_document.v2"
    assert CACHED_DOCUMENT_REF_SCHEMA == "arc.document.cached_document_ref.v1"
    assert DOCUMENT_STRUCTURE_OVERLAY_SCHEMA.startswith("arc.document.")


def test_service_caches_and_reads_exact_local_document(tmp_path: Path) -> None:
    service = ArcDocumentService(cache_root=tmp_path / "cache")
    source = service.import_source(_source(tmp_path))
    reference = service.cache_document(source)

    selected = service.read_cached_source_range(reference, 1, 3)
    toc = service.get_cached_table_of_contents(reference)

    assert selected.text == "# One\n\nAlpha."
    assert [entry.title for entry in toc.entries] == ["One", "Two"]


def test_cli_reads_content_addressed_source_range(
    tmp_path: Path, capsys
) -> None:
    service = ArcDocumentService(cache_root=tmp_path / "cache")
    reference = service.cache_document(service.import_source(_source(tmp_path)))

    assert main(
        [
            "read-cached-source-range",
            "--document-ref",
            json.dumps(cached_document_ref_to_document(reference)),
            "--cache-root",
            str(service.cache_root),
            "2",
            "3",
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["text"] == "\nAlpha."
