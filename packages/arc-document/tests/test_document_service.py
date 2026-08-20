from __future__ import annotations

from dataclasses import replace
import json
import types
from pathlib import Path

import pytest

from arc_document import (
    CACHED_DOCUMENT_REF_SCHEMA,
    DOCUMENT_STRUCTURE_OVERLAY_SCHEMA,
    PARSED_DOCUMENT_SCHEMA,
    RICH_DOCUMENT_SCHEMA,
    ArcDocumentService,
    CachedDocumentError,
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


@pytest.mark.parametrize(
    ("replacement", "code"),
    [
        ({"source_size": 1}, "cached_document_source_mismatch"),
        ({"media_type": "text/plain"}, "cached_document_source_mismatch"),
        (
            {"parser_contract": "arc.document.parser.future"},
            "cached_document_parser_contract_mismatch",
        ),
        (
            {"parsed_document_sha256": "0" * 64},
            "cached_document_digest_mismatch",
        ),
    ],
)
def test_service_revalidates_cached_document_identity(
    tmp_path: Path,
    replacement: dict[str, object],
    code: str,
) -> None:
    service = ArcDocumentService(cache_root=tmp_path / "cache")
    reference = service.cache_document(service.import_source(_source(tmp_path)))

    with pytest.raises(CachedDocumentError) as error:
        service.get_cached_table_of_contents(replace(reference, **replacement))

    assert error.value.code == code


def test_service_rejects_untyped_cached_document_reference(tmp_path: Path) -> None:
    service = ArcDocumentService(cache_root=tmp_path / "cache")

    with pytest.raises(CachedDocumentError) as error:
        service.get_cached_table_of_contents({})  # type: ignore[arg-type]

    assert error.value.code == "invalid_cached_document_ref"


@pytest.mark.parametrize(
    ("start_line", "end_line"),
    [
        (True, 1),
        (1, True),
        (1.0, 1),
        (1, "2"),
        (0, 1),
        (2, 1),
    ],
)
def test_service_validates_source_range_integer_contract(
    tmp_path: Path,
    start_line: object,
    end_line: object,
) -> None:
    service = ArcDocumentService(cache_root=tmp_path / "cache")
    reference = service.cache_document(service.import_source(_source(tmp_path)))

    with pytest.raises(CachedDocumentError) as error:
        service.read_cached_source_range(
            reference,
            start_line,  # type: ignore[arg-type]
            end_line,  # type: ignore[arg-type]
        )

    assert error.value.code == "invalid_source_range"


def test_service_validates_source_range_options_and_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ArcDocumentService(cache_root=tmp_path / "cache")
    reference = service.cache_document(service.import_source(_source(tmp_path)))

    with pytest.raises(CachedDocumentError) as error:
        service.read_cached_source_range(
            reference,
            1,
            1,
            text_only="yes",  # type: ignore[arg-type]
        )
    assert error.value.code == "invalid_text_only"

    monkeypatch.setattr(service.repository, "read_bytes", lambda _source: b"\xff")
    with pytest.raises(CachedDocumentError) as error:
        service.read_cached_source_range(reference, 1, 1)
    assert error.value.code == "cached_source_not_utf8"


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
