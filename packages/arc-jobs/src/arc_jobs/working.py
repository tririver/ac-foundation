"""Agent-editable working state for durable runs.

The immutable run spec and artifact object store remain the historical record.
This module owns the small, public working tree that an agent may repair before
explicitly resuming a failed run.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Mapping

from .artifacts import decode_artifact_ref
from .identity import canonical_json_bytes, validate_artifact_id
from .models import (
    ArtifactDigest,
    ArtifactRef,
    ArtifactSourceRef,
    JsonValue,
    RunError,
    RunSpec,
    VerifiedArtifact,
)
from .storage import (
    ImmutableArtifactStore,
    atomic_write_bytes,
    atomic_write_json,
    read_json_object,
    utc_now,
)


class WorkingState:
    """Manage one run's editable state and recovery snapshots."""

    def __init__(self, run_directory: Path):
        self.run_directory = run_directory
        self.root = run_directory / "working"
        self.semantic_input_path = self.root / "semantic-input.json"
        self.index_path = self.root / "index.json"
        self.artifacts_directory = self.root / "artifacts"
        self.candidates_directory = self.root / "candidates"
        self.last_error_path = self.root / "last-error.json"

    def materialize(self, spec: RunSpec, *, include_legacy_artifacts: bool = False) -> None:
        """Create missing public working files without overwriting agent edits."""

        was_missing = not self.root.exists()
        self.artifacts_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.candidates_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.semantic_input_path.exists():
            atomic_write_json(self.semantic_input_path, dict(spec.semantic_input))
        if include_legacy_artifacts and was_missing:
            self._materialize_legacy_artifacts()
        if not self.index_path.exists():
            atomic_write_json(
                self.index_path,
                {
                    "schema_version": "arc.jobs.working_index.v1",
                    "recovery_epoch": 0,
                    "updated_at": utc_now(),
                    "snapshot_path": None,
                    "files": self._file_index(),
                    "warnings": [],
                },
            )

    def read_semantic_input(self) -> dict[str, JsonValue]:
        """Read the current semantic input, retaining the precise path on error."""

        return read_json_object(self.semantic_input_path)

    def prepare_recovery(
        self,
        spec: RunSpec,
        *,
        recovery_epoch: int,
    ) -> tuple[dict[str, JsonValue], tuple[dict[str, JsonValue], ...]]:
        """Snapshot and index the bytes an agent selected for a recovery epoch."""

        self.materialize(spec, include_legacy_artifacts=True)
        previous = read_json_object(self.index_path)
        previous_files = previous.get("files")
        if not isinstance(previous_files, dict):
            previous_files = {}
        current_files = self._file_index()
        changed = sorted(
            path
            for path in set(previous_files) | set(current_files)
            if previous_files.get(path) != current_files.get(path)
        )
        warnings: list[dict[str, JsonValue]] = []
        if changed:
            warnings.append(
                {
                    "code": "working_state_modified",
                    "message": "Editable working state differs from the previous recovery baseline.",
                    "paths": changed,
                }
            )
        semantic_changed = "semantic-input.json" in changed
        downstream = [
            path
            for path in current_files
            if path.startswith("artifacts/") or path.startswith("candidates/")
        ]
        if semantic_changed and downstream:
            warnings.append(
                {
                    "code": "working_state_may_be_stale",
                    "message": (
                        "Semantic input changed while downstream working files remain; "
                        "delete files that should be regenerated."
                    ),
                    "paths": sorted(downstream),
                }
            )
        snapshot_root = (
            self.run_directory
            / "recovery"
            / f"epoch-{recovery_epoch:04d}"
            / "working"
        )
        if snapshot_root.exists():
            shutil.rmtree(snapshot_root)
        shutil.copytree(self.root, snapshot_root)
        semantic_input = self.read_semantic_input()
        atomic_write_json(
            self.index_path,
            {
                "schema_version": "arc.jobs.working_index.v1",
                "recovery_epoch": recovery_epoch,
                "updated_at": utc_now(),
                "snapshot_path": snapshot_root.relative_to(
                    self.run_directory
                ).as_posix(),
                "files": current_files,
                "warnings": warnings,
            },
        )
        return semantic_input, tuple(warnings)

    def record_error(self, error: RunError, *, attempt: int, recovery_epoch: int) -> None:
        atomic_write_json(
            self.last_error_path,
            {
                "schema_version": "arc.jobs.working_error.v1",
                "attempt": attempt,
                "recovery_epoch": recovery_epoch,
                "recorded_at": utc_now(),
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "details": dict(error.details),
                },
            },
        )

    def clear_error(self) -> None:
        self.last_error_path.unlink(missing_ok=True)

    def candidate_path(self, candidate_id: str) -> Path:
        return self.candidates_directory / validate_artifact_id(candidate_id)

    def find_candidate(self, candidate_id: str) -> Path | None:
        path = self.candidate_path(candidate_id)
        return path if path.is_file() else None

    def write_candidate_json(
        self, candidate_id: str, value: Mapping[str, JsonValue]
    ) -> Path:
        path = self.candidate_path(candidate_id)
        atomic_write_json(path, value)
        return path

    def read_candidate_json(self, candidate_id: str) -> dict[str, JsonValue]:
        return read_json_object(self.candidate_path(candidate_id))

    def _file_index(self) -> dict[str, JsonValue]:
        files: dict[str, JsonValue] = {}
        if not self.root.exists():
            return files
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path == self.index_path:
                continue
            relative = path.relative_to(self.root).as_posix()
            content = path.read_bytes()
            files[relative] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        return files

    def _materialize_legacy_artifacts(self) -> None:
        manifests = self.run_directory / "artifacts" / "manifests"
        if not manifests.exists():
            return
        for manifest_path in sorted(manifests.glob("*.json")):
            document = read_json_object(manifest_path)
            try:
                ref = decode_artifact_ref(
                    {
                        "artifact_id": document["artifact_id"],
                        "digest": document["digest"],
                        "media_type": document["media_type"],
                        "relative_path": document["relative_path"],
                    }
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"cannot materialize artifact manifest: {manifest_path}"
                ) from exc
            source = self.run_directory / ref.relative_path
            target = self.artifacts_directory / ref.artifact_id
            if not target.exists():
                atomic_write_bytes(target, source.read_bytes())


class EditableArtifactStore:
    """Immutable publication backed by an agent-editable logical working copy."""

    def __init__(
        self,
        immutable: ImmutableArtifactStore,
        working: WorkingState,
        *,
        recovery_epoch: int,
        prefix: str | None = None,
    ):
        self.immutable = immutable
        self.working = working
        self.recovery_epoch = recovery_epoch
        self.prefix = prefix

    def scoped(
        self, prefix: str
    ) -> "EditableArtifactStore | ImmutableArtifactStore":
        validate_artifact_id(prefix)
        combined = prefix if self.prefix is None else f"{self.prefix}/{prefix}"
        if combined.startswith("llm/tasks/"):
            # Provider task protocol records are ARC-managed execution state,
            # not agent-editable scientific working artifacts.
            return self.immutable.scoped(combined)
        return EditableArtifactStore(
            self.immutable,
            self.working,
            recovery_epoch=self.recovery_epoch,
            prefix=combined,
        )

    def _logical_id(self, artifact_id: str) -> str:
        validate_artifact_id(artifact_id)
        return validate_artifact_id(
            artifact_id
            if self.prefix is None
            else f"{self.prefix}/{artifact_id}"
        )

    def _working_path(self, logical_id: str) -> Path:
        return self.working.artifacts_directory / logical_id

    def _immutable_id(self, logical_id: str) -> str:
        if self.recovery_epoch == 0:
            return logical_id
        return f"recovery-{self.recovery_epoch}/{logical_id}"

    def _known_media_type(self, logical_id: str) -> str:
        for epoch in range(self.recovery_epoch - 1, -1, -1):
            artifact_id = (
                logical_id if epoch == 0 else f"recovery-{epoch}/{logical_id}"
            )
            ref = self.immutable.find(artifact_id)
            if ref is not None:
                return ref.media_type
        return "application/octet-stream"

    def find(self, artifact_id: str) -> ArtifactRef | None:
        logical_id = self._logical_id(artifact_id)
        path = self._working_path(logical_id)
        if not path.is_file():
            return None
        if self.recovery_epoch == 0:
            return self.immutable.find(logical_id)
        return self.immutable.publish_bytes(
            self._immutable_id(logical_id),
            path.read_bytes(),
            media_type=self._known_media_type(logical_id),
        )

    def publish_bytes(
        self, artifact_id: str, content: bytes, *, media_type: str
    ) -> ArtifactRef:
        logical_id = self._logical_id(artifact_id)
        ref = self.immutable.publish_bytes(
            self._immutable_id(logical_id),
            content,
            media_type=media_type,
        )
        atomic_write_bytes(self._working_path(logical_id), bytes(content))
        return ref

    def publish_json(self, artifact_id: str, value: JsonValue) -> ArtifactRef:
        return self.publish_bytes(
            artifact_id,
            canonical_json_bytes(value) + b"\n",
            media_type="application/json",
        )

    def read_bytes(self, ref: ArtifactRef) -> bytes:
        return self.immutable.read_bytes(ref)

    def verify(self, ref: ArtifactRef) -> None:
        self.immutable.verify(ref)

    def read_source(self, source: ArtifactSourceRef) -> VerifiedArtifact:
        return self.immutable.read_source(source)

    def adopt(
        self,
        source: ArtifactSourceRef,
        *,
        artifact_id: str,
        expected_verified_digest: ArtifactDigest,
    ) -> ArtifactRef:
        verified = self.read_source(source)
        if verified.digest != expected_verified_digest:
            from .errors import ArtifactConflictError

            raise ArtifactConflictError(
                "verified source digest changed before adoption"
            )
        return self.publish_bytes(
            artifact_id,
            verified.content,
            media_type=verified.media_type,
        )


__all__ = ["EditableArtifactStore", "WorkingState"]
