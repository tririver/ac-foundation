from __future__ import annotations

import json
from pathlib import Path

from arc_document import ArcDocumentService, cached_document_ref_to_document
from arc_document.cli import main
from arc_document.operation_registry import OperationSpec
from arc_document.registry import (
    OPERATION_REGISTRY,
    OperationRequestError,
    dispatch_operation,
    registry_document,
)


def test_document_registry_has_stable_ids_and_safe_projection() -> None:
    assert OPERATION_REGISTRY["import-source"].operation_id == (
        "arc-document.import-source.v1"
    )
    document = registry_document()
    assert document["schema_version"] == "arc.document.operation_registry.v1"
    assert "cache-list" not in {
        item["name"] for item in document["operations"]
    }


def test_document_registry_rejects_extra_parameters() -> None:
    try:
        dispatch_operation(
            "import-source", {"path": "missing.md", "unexpected": True}
        )
    except OperationRequestError as exc:
        assert exc.code == "invalid_parameters"
    else:  # pragma: no cover
        raise AssertionError("strict registry accepted an extra parameter")


def test_document_cli_uses_command_result_protocol(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "short.md"
    source.write_text("# Title\n\nBody.\n", encoding="utf-8")
    cache_root = tmp_path / "cache"
    service = ArcDocumentService(cache_root=cache_root)
    reference = service.cache_document(service.import_source(source))

    assert main(
        [
            "get-table-of-contents",
            "--document-ref",
            json.dumps(cached_document_ref_to_document(reference)),
            "--cache-root",
            str(cache_root),
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert result["data"]["entries"][0]["title"] == "Title"
