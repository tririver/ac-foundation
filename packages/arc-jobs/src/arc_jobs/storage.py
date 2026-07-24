from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Generic, Mapping, TypeVar

from .contracts import StateContract
from .errors import (
    ArtifactConflictError,
    CorruptStateError,
    RevisionConflictError,
    StateConflictError,
    UnsupportedSchemaError,
)
from .artifacts import decode_artifact_digest, encode_artifact_digest
from .identity import canonical_json_bytes, validate_artifact_id, validate_simple_id
from .lease import FileLease
from .models import (
    ArtifactDigest,
    ArtifactRef,
    ArtifactSourceRef,
    JsonValue,
    VerifiedArtifact,
)

T = TypeVar("T")


def utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":  # pragma: no cover - Windows
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, content: bytes, *, exclusive: bool = False) -> None:
    _ensure_directory(path.parent)
    if exclusive and path.exists():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and path.exists():
            raise FileExistsError(path)
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Mapping[str, JsonValue], *, exclusive: bool = False) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n", exclusive=exclusive)


def read_json_object(path: Path) -> dict[str, JsonValue]:
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorruptStateError(f"cannot read JSON document: {path}") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CorruptStateError(f"expected JSON object: {path}")
    return value


def require_fields(
    document: Mapping[str, JsonValue],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    keys = set(document)
    missing = required - keys
    unknown = keys - required - optional
    if missing or unknown:
        raise CorruptStateError(
            f"invalid fields; missing={sorted(missing)!r}, unknown={sorted(unknown)!r}"
        )


class AtomicStateStore(Generic[T]):
    def __init__(self, path: Path, contract: StateContract[T]):
        self.path = path
        self.contract = contract

    def _encode(self, value: T) -> dict[str, JsonValue]:
        revision = getattr(value, "revision", None)
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise CorruptStateError("revisioned state must expose a non-negative integer revision")
        encoded = dict(self.contract.encode(value))
        return {
            "schema_version": "arc.jobs.state.v1",
            "contract_schema_version": self.contract.schema_version,
            "revision": revision,
            "value": encoded,
        }

    def _decode(self, document: Mapping[str, JsonValue]) -> T:
        require_fields(
            document,
            required={"schema_version", "contract_schema_version", "revision", "value"},
        )
        if document["schema_version"] != "arc.jobs.state.v1":
            raise UnsupportedSchemaError(str(document["schema_version"]))
        if document["contract_schema_version"] != self.contract.schema_version:
            raise UnsupportedSchemaError(str(document["contract_schema_version"]))
        revision = document["revision"]
        value = document["value"]
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise CorruptStateError("invalid state revision")
        if not isinstance(value, dict):
            raise CorruptStateError("state value must be an object")
        decoded = self.contract.decode(value)
        if getattr(decoded, "revision", None) != revision:
            raise CorruptStateError("outer and inner revisions differ")
        return decoded

    def read(self) -> T | None:
        if not self.path.exists():
            return None
        return self._decode(read_json_object(self.path))

    def create(self, value: T) -> T:
        self.contract.validate_transition(None, value)
        encoded = self._encode(value)
        lease = FileLease(self.path.with_suffix(f"{self.path.suffix}.lock")).acquire(
            blocking=True
        )
        try:
            if self.path.exists():
                current = self.read()
                if current == value:
                    return value
                raise StateConflictError(f"state already exists: {self.path}")
            atomic_write_json(self.path, encoded)
            return value
        finally:
            lease.release()

    def compare_and_swap(self, expected_revision: int, value: T) -> T:
        lease = FileLease(self.path.with_suffix(f"{self.path.suffix}.lock")).acquire(
            blocking=True
        )
        try:
            current = self.read()
            if current is None:
                raise RevisionConflictError("state does not exist")
            current_revision = getattr(current, "revision", None)
            next_revision = getattr(value, "revision", None)
            if current_revision != expected_revision or next_revision != expected_revision + 1:
                raise RevisionConflictError(
                    f"expected revision {expected_revision}, found {current_revision}, "
                    f"next is {next_revision}"
                )
            self.contract.validate_transition(current, value)
            atomic_write_json(self.path, self._encode(value))
            return value
        finally:
            lease.release()

    def validate(self) -> T | None:
        return self.read()


class ImmutableArtifactStore:
    def __init__(
        self,
        run_directory: Path,
        *,
        repository_root: Path | None = None,
        prefix: str | None = None,
    ):
        self.run_directory = run_directory
        self.repository_root = repository_root
        self.prefix = prefix
        self.artifact_root = run_directory / "artifacts"

    def scoped(self, prefix: str) -> "ImmutableArtifactStore":
        validate_artifact_id(prefix)
        combined = prefix if self.prefix is None else f"{self.prefix}/{prefix}"
        validate_artifact_id(combined)
        return ImmutableArtifactStore(
            self.run_directory,
            repository_root=self.repository_root,
            prefix=combined,
        )

    def _logical_id(self, artifact_id: str) -> str:
        validate_artifact_id(artifact_id)
        value = artifact_id if self.prefix is None else f"{self.prefix}/{artifact_id}"
        return validate_artifact_id(value)

    def _manifest_path(self, artifact_id: str) -> Path:
        identifier_digest = hashlib.sha256(artifact_id.encode("utf-8")).hexdigest()
        return self.artifact_root / "manifests" / f"{identifier_digest}.json"

    def _object_path(self, content_sha256: str) -> Path:
        return self.artifact_root / "objects" / "sha256" / content_sha256[:2] / content_sha256

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.run_directory).as_posix()

    def _publish(
        self,
        artifact_id: str,
        content: bytes,
        *,
        media_type: str,
        reused_from: Mapping[str, JsonValue] | None = None,
    ) -> ArtifactRef:
        logical_id = self._logical_id(artifact_id)
        if not isinstance(media_type, str) or not media_type:
            raise ValueError("media_type must be non-empty")
        digest_value = hashlib.sha256(content).hexdigest()
        digest = ArtifactDigest("sha256", digest_value, len(content))
        object_path = self._object_path(digest_value)
        object_lease = FileLease(
            object_path.with_suffix(f"{object_path.suffix}.lock")
        ).acquire(blocking=True)
        try:
            if object_path.exists():
                if object_path.read_bytes() != content:
                    raise ArtifactConflictError(
                        "content-addressed object does not match its digest"
                    )
            else:
                atomic_write_bytes(object_path, content)
        finally:
            object_lease.release()
        relative_path = self._relative(object_path)
        manifest: dict[str, JsonValue] = {
            "schema_version": "arc.jobs.artifact.v1",
            "artifact_id": logical_id,
            "digest": encode_artifact_digest(digest),
            "media_type": media_type,
            "relative_path": relative_path,
            "reused_from": dict(reused_from) if reused_from is not None else None,
        }
        manifest_path = self._manifest_path(logical_id)
        manifest_lease = FileLease(
            manifest_path.with_suffix(f"{manifest_path.suffix}.lock")
        ).acquire(blocking=True)
        try:
            if manifest_path.exists():
                if read_json_object(manifest_path) != manifest:
                    raise ArtifactConflictError(f"artifact id already published: {logical_id}")
            else:
                atomic_write_json(manifest_path, manifest)
        finally:
            manifest_lease.release()
        return ArtifactRef(logical_id, digest, media_type, relative_path)

    def publish_bytes(self, artifact_id: str, content: bytes, *, media_type: str) -> ArtifactRef:
        return self._publish(artifact_id, bytes(content), media_type=media_type)

    def publish_json(self, artifact_id: str, value: JsonValue) -> ArtifactRef:
        return self._publish(
            artifact_id,
            canonical_json_bytes(value) + b"\n",
            media_type="application/json",
        )

    def _read_manifest(self, artifact_id: str) -> dict[str, JsonValue]:
        validate_artifact_id(artifact_id)
        manifest = read_json_object(self._manifest_path(artifact_id))
        require_fields(
            manifest,
            required={
                "schema_version",
                "artifact_id",
                "digest",
                "media_type",
                "relative_path",
                "reused_from",
            },
        )
        if manifest["schema_version"] != "arc.jobs.artifact.v1":
            raise UnsupportedSchemaError(str(manifest["schema_version"]))
        if manifest["artifact_id"] != artifact_id:
            raise CorruptStateError("artifact manifest id mismatch")
        return manifest

    def find(self, artifact_id: str) -> ArtifactRef | None:
        """Return a verified immutable artifact reference when it is published.

        This is a read-only replay primitive. Missing artifacts return ``None``;
        malformed manifests or content still raise the normal corruption errors.
        """

        logical_id = self._logical_id(artifact_id)
        manifest_path = self._manifest_path(logical_id)
        if not manifest_path.exists():
            return None
        ref = self._ref_from_manifest(self._read_manifest(logical_id))
        self.verify(ref)
        return ref

    @staticmethod
    def _digest_from_json(value: JsonValue) -> ArtifactDigest:
        try:
            return decode_artifact_digest(value)
        except ValueError as exc:
            raise CorruptStateError("invalid artifact digest") from exc

    def _ref_from_manifest(self, manifest: Mapping[str, JsonValue]) -> ArtifactRef:
        artifact_id = manifest["artifact_id"]
        media_type = manifest["media_type"]
        relative_path = manifest["relative_path"]
        if not all(isinstance(item, str) for item in (artifact_id, media_type, relative_path)):
            raise CorruptStateError("invalid artifact manifest strings")
        digest = self._digest_from_json(manifest["digest"])
        expected = self._relative(self._object_path(digest.value))
        if relative_path != expected:
            raise CorruptStateError("non-canonical artifact object path")
        return ArtifactRef(artifact_id, digest, media_type, relative_path)

    def read_bytes(self, ref: ArtifactRef) -> bytes:
        manifest = self._read_manifest(ref.artifact_id)
        canonical_ref = self._ref_from_manifest(manifest)
        if canonical_ref != ref:
            raise CorruptStateError("artifact reference does not match manifest")
        path = self.run_directory / canonical_ref.relative_path
        content = path.read_bytes()
        actual = ArtifactDigest("sha256", hashlib.sha256(content).hexdigest(), len(content))
        if actual != ref.digest:
            raise CorruptStateError("artifact content digest mismatch")
        return content

    def verify(self, ref: ArtifactRef) -> None:
        self.read_bytes(ref)

    def read_source(self, source: ArtifactSourceRef) -> VerifiedArtifact:
        if self.repository_root is None:
            raise ValueError("repository_root is required for artifact adoption")
        validate_simple_id(source.source_run_id, label="source run id")
        validate_artifact_id(source.source_artifact_id)
        source_store = ImmutableArtifactStore(
            self.repository_root / "runs" / source.source_run_id,
            repository_root=self.repository_root,
        )
        manifest = source_store._read_manifest(source.source_artifact_id)
        ref = source_store._ref_from_manifest(manifest)
        if ref.digest != source.expected_digest:
            raise ArtifactConflictError("source artifact digest differs from expected digest")
        content = source_store.read_bytes(ref)
        return VerifiedArtifact(source, ref.digest, ref.media_type, content)

    def adopt(
        self,
        source: ArtifactSourceRef,
        *,
        artifact_id: str,
        expected_verified_digest: ArtifactDigest,
    ) -> ArtifactRef:
        verified = self.read_source(source)
        if verified.digest != expected_verified_digest:
            raise ArtifactConflictError("verified source digest changed before adoption")
        provenance: dict[str, JsonValue] = {
            "source_run_id": source.source_run_id,
            "source_artifact_id": source.source_artifact_id,
            "digest": encode_artifact_digest(verified.digest),
        }
        return self._publish(
            artifact_id,
            verified.content,
            media_type=verified.media_type,
            reused_from=provenance,
        )
