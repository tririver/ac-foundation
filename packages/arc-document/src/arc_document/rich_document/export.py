from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from .._durable_io import atomic_write_bytes
from ..source_repository import SourceRepository
from ..sources import SourceBundle, SourceFormat
from .models import rich_document_to_document
from .service import RichDocumentParserService


class RichDocumentExportError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def export_rich_document_workspace(
    repository: SourceRepository,
    source: str | Path,
    *,
    output_dir: str | Path,
    validator: str | Path | None = None,
    source_format: SourceFormat | str | None = None,
) -> dict[str, object]:
    """Parse one local rich source into a portable ``arc-render`` handoff."""

    output = Path(output_dir).expanduser().resolve(strict=False)
    _require_available_output(output)

    primary = repository.import_path(source, source_format=source_format)
    validators = (
        (
            repository.import_path(
                validator,
                source_format=SourceFormat.PDF,
            ),
        )
        if validator is not None
        else ()
    )
    outcome = RichDocumentParserService(repository).parse(
        SourceBundle(primary=primary, validators=validators)
    )

    resources: list[tuple[dict[str, object], bytes]] = []
    for asset in outcome.document.assets:
        stored = repository.get_asset(asset.artifact_digest)
        payload = repository.read_asset_bytes(stored)
        if (
            stored.media_type != asset.media_type
            or stored.size != asset.size
            or len(payload) != asset.size
            or hashlib.sha256(payload).hexdigest() != asset.artifact_digest
        ):
            raise RichDocumentExportError(
                "rich_document_asset_mismatch",
                f"rich document asset differs from verified bytes: {asset.artifact_digest}",
            )
        relative = f"resources/{asset.artifact_digest}"
        resources.append(
            (
                {
                    "artifact_digest": asset.artifact_digest,
                    "media_type": asset.media_type,
                    "logical_name": asset.logical_name,
                    "size": asset.size,
                    "path": relative,
                },
                payload,
            )
        )

    source_payload = _json_bytes(rich_document_to_document(outcome.document))
    metadata_payload = _json_bytes(
        {
            "glossary": [],
            "bibliography": [],
            "labels": {},
            "resources": [item for item, _payload in resources],
            "reader_profile": {},
        }
    )
    _publish_workspace(
        output,
        source_payload=source_payload,
        metadata_payload=metadata_payload,
        resources=resources,
    )

    return {
        "source": str(output / "rich-source.json"),
        "metadata": str(output / "metadata.json"),
        "resources": [
            {
                **item,
                "path": str(output / str(item["path"])),
            }
            for item, _payload in resources
        ],
        "document_digest": outcome.document.document_digest,
        "warnings": list(outcome.warnings),
    }


def _require_available_output(output: Path) -> None:
    if not output.exists():
        return
    if not output.is_dir():
        raise RichDocumentExportError(
            "rich_document_output_exists",
            f"output path exists and is not a directory: {output}",
        )
    try:
        nonempty = next(output.iterdir(), None) is not None
    except OSError as exc:
        raise RichDocumentExportError(
            "rich_document_output_unreadable",
            f"output directory cannot be inspected: {output}",
        ) from exc
    if nonempty:
        raise RichDocumentExportError(
            "rich_document_output_not_empty",
            f"output directory must be absent or empty: {output}",
        )


def _publish_workspace(
    output: Path,
    *,
    source_payload: bytes,
    metadata_payload: bytes,
    resources: list[tuple[dict[str, object], bytes]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.export-",
            dir=output.parent,
        )
    )
    replaced_empty = False
    try:
        resource_dir = staging / "resources"
        resource_dir.mkdir()
        atomic_write_bytes(staging / "rich-source.json", source_payload)
        atomic_write_bytes(staging / "metadata.json", metadata_payload)
        for item, payload in resources:
            atomic_write_bytes(staging / str(item["path"]), payload)

        _require_available_output(output)
        if output.exists():
            output.rmdir()
            replaced_empty = True
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if replaced_empty and not output.exists():
            output.mkdir()
        raise


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


__all__ = [
    "RichDocumentExportError",
    "export_rich_document_workspace",
]
