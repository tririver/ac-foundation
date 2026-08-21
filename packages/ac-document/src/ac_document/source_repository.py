from __future__ import annotations

import hashlib
import json
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ._durable_io import atomic_write_bytes, payload_matches
from ._file_lock import exclusive_file_lock
from .sources import SourceArtifact, SourceFormat, SourceOrigin, SourceOriginKind


SOURCE_REPOSITORY_SCHEMA = "ac.document.source_repository.v1"
SOURCE_ASSET_SCHEMA = "ac.document.source_asset.v1"
DEFAULT_MEDIA_TYPES = {
    SourceFormat.HTML: "text/html",
    SourceFormat.MARKDOWN: "text/markdown",
    SourceFormat.TEX: "text/x-tex",
    SourceFormat.PDF: "application/pdf",
}
SOURCE_SUFFIXES = {
    ".html": SourceFormat.HTML,
    ".htm": SourceFormat.HTML,
    ".md": SourceFormat.MARKDOWN,
    ".markdown": SourceFormat.MARKDOWN,
    ".tex": SourceFormat.TEX,
    ".pdf": SourceFormat.PDF,
}
_MANIFEST_FIELDS = {
    "schema_version",
    "source_format",
    "media_type",
    "artifact_digest",
    "size",
}
_ASSET_MANIFEST_FIELDS = {
    "schema_version",
    "media_type",
    "artifact_digest",
    "size",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class RepositoryAsset:
    """One immutable non-document resource stored by ``SourceRepository``."""

    artifact_digest: str
    size: int
    media_type: str

    def __post_init__(self) -> None:
        digest = self.artifact_digest.casefold()
        media_type = self.media_type.strip().casefold()
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("artifact_digest must be a SHA-256 digest")
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 0:
            raise ValueError("asset size cannot be negative")
        if not media_type or "/" not in media_type or ";" in media_type:
            raise ValueError("asset media_type must be a normalized MIME type")
        object.__setattr__(self, "artifact_digest", digest)
        object.__setattr__(self, "media_type", media_type)

    @property
    def content_identity(self) -> tuple[str, str, int]:
        return (self.media_type, self.artifact_digest, self.size)


class SourceRepositoryError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class SourceRepository:
    """Content-addressed storage for immutable document source bytes.

    The repository owns source bytes and integrity metadata only. It deliberately
    does not own workflow state, retries, queues, or run recovery.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def import_path(
        self,
        path: str | Path,
        *,
        source_format: SourceFormat | str | None = None,
    ) -> SourceArtifact:
        source_path = Path(path)
        resolved_format = (
            SourceFormat(source_format)
            if source_format is not None
            else self._format_for_path(source_path)
        )
        try:
            payload = source_path.read_bytes()
        except OSError as exc:
            raise SourceRepositoryError(
                "source_read_failed", f"unable to read source: {source_path}"
            ) from exc
        return self.store_bytes(
            payload,
            source_format=resolved_format,
            origin=SourceOrigin(
                kind=SourceOriginKind.LOCAL_IMPORT,
                locator=str(source_path),
            ),
        )

    def store_bytes(
        self,
        payload: bytes,
        *,
        source_format: SourceFormat | str,
        origin: SourceOrigin,
        media_type: str | None = None,
    ) -> SourceArtifact:
        if not isinstance(payload, bytes):
            raise TypeError("source payload must be bytes")
        resolved_format = SourceFormat(source_format)
        resolved_media_type = self._normalize_media_type(
            media_type or DEFAULT_MEDIA_TYPES[resolved_format]
        )
        digest = hashlib.sha256(payload).hexdigest()
        object_dir = self._object_dir(resolved_format, digest)
        payload_path = object_dir / "source"
        manifest_path = object_dir / "manifest.json"

        with self._content_lock(resolved_format, digest):
            if manifest_path.exists():
                artifact, _ = self._read_verified_unlocked(
                    resolved_format, digest, origin=origin
                )
                if (
                    artifact.size != len(payload)
                    or artifact.media_type != resolved_media_type
                ):
                    raise SourceRepositoryError(
                        "source_metadata_conflict",
                        "stored source metadata conflicts with the requested source",
                    )
                return artifact

            object_dir.mkdir(parents=True, exist_ok=True)
            if not self._payload_matches(payload_path, digest, len(payload)):
                self._atomic_write(payload_path, payload)
            manifest = {
                "schema_version": SOURCE_REPOSITORY_SCHEMA,
                "source_format": resolved_format.value,
                "media_type": resolved_media_type,
                "artifact_digest": digest,
                "size": len(payload),
            }
            self._atomic_write(
                manifest_path,
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
            artifact, _ = self._read_verified_unlocked(
                resolved_format, digest, origin=origin
            )
            return artifact

    def import_asset_path(
        self,
        path: str | Path,
        *,
        media_type: str,
    ) -> RepositoryAsset:
        """Import a local relative resource referenced by a rich document."""

        asset_path = Path(path)
        try:
            payload = asset_path.read_bytes()
        except OSError as exc:
            raise SourceRepositoryError(
                "asset_read_failed", f"unable to read asset: {asset_path}"
            ) from exc
        return self.store_asset_bytes(payload, media_type=media_type)

    def store_asset_bytes(
        self,
        payload: bytes,
        *,
        media_type: str,
    ) -> RepositoryAsset:
        """Store arbitrary immutable resource bytes beside document sources."""

        if not isinstance(payload, bytes):
            raise TypeError("asset payload must be bytes")
        normalized_media_type = self._normalize_media_type(media_type)
        digest = hashlib.sha256(payload).hexdigest()
        object_dir = self._asset_object_dir(digest)
        payload_path = object_dir / "asset"
        manifest_path = object_dir / "manifest.json"
        with self._asset_content_lock(digest):
            if manifest_path.exists():
                asset = self._read_asset_verified(digest)
                if (
                    asset.size != len(payload)
                    or asset.media_type != normalized_media_type
                ):
                    raise SourceRepositoryError(
                        "asset_metadata_conflict",
                        "stored asset metadata conflicts with the requested asset",
                    )
                return asset
            object_dir.mkdir(parents=True, exist_ok=True)
            if not self._payload_matches(payload_path, digest, len(payload)):
                self._atomic_write(payload_path, payload)
            manifest = {
                "schema_version": SOURCE_ASSET_SCHEMA,
                "media_type": normalized_media_type,
                "artifact_digest": digest,
                "size": len(payload),
            }
            self._atomic_write(
                manifest_path,
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
        return self._read_asset_verified(digest)

    def get_asset(self, artifact_digest: str) -> RepositoryAsset:
        return self._read_asset_verified(artifact_digest.casefold())

    def read_asset_bytes(self, asset: RepositoryAsset) -> bytes:
        verified = self._read_asset_verified(asset.artifact_digest)
        if verified.content_identity != asset.content_identity:
            raise SourceRepositoryError(
                "asset_artifact_mismatch",
                "asset metadata does not match repository content",
            )
        return (self._asset_object_dir(asset.artifact_digest) / "asset").read_bytes()

    def get(
        self,
        source_format: SourceFormat | str,
        artifact_digest: str,
    ) -> SourceArtifact:
        resolved_format = SourceFormat(source_format)
        digest = artifact_digest.casefold()
        return self._read_verified(
            resolved_format,
            digest,
            origin=SourceOrigin(
                kind=SourceOriginKind.REPOSITORY,
                locator=f"{resolved_format.value}/sha256/{digest}",
            ),
        )

    def read_bytes(self, artifact: SourceArtifact) -> bytes:
        with self._content_lock(
            artifact.source_format, artifact.artifact_digest
        ):
            verified, payload = self._read_verified_unlocked(
                artifact.source_format,
                artifact.artifact_digest,
                origin=artifact.origin,
            )
            if verified.content_identity != artifact.content_identity:
                raise SourceRepositoryError(
                    "source_artifact_mismatch",
                    "source artifact metadata does not match repository content",
                )
            return payload

    def remove(
        self,
        source_format: SourceFormat | str,
        artifact_digest: str,
    ) -> bool:
        """Physically delete one exact source object owned by the repository."""

        resolved_format = SourceFormat(source_format)
        digest = str(artifact_digest).casefold()
        if _SHA256_RE.fullmatch(digest) is None:
            raise SourceRepositoryError(
                "invalid_artifact_digest",
                "artifact digest must be a SHA-256 digest",
            )
        object_dir = self._object_dir(resolved_format, digest)
        with self._content_lock(resolved_format, digest):
            if not object_dir.exists():
                return False
            shutil.rmtree(object_dir)
            return True

    def _read_verified(
        self,
        source_format: SourceFormat,
        digest: str,
        *,
        origin: SourceOrigin,
    ) -> SourceArtifact:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise SourceRepositoryError(
                "invalid_artifact_digest", "artifact digest must be a SHA-256 digest"
            )
        with self._content_lock(source_format, digest):
            artifact, _ = self._read_verified_unlocked(
                source_format, digest, origin=origin
            )
            return artifact

    def _read_verified_unlocked(
        self,
        source_format: SourceFormat,
        digest: str,
        *,
        origin: SourceOrigin,
    ) -> tuple[SourceArtifact, bytes]:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise SourceRepositoryError(
                "invalid_artifact_digest", "artifact digest must be a SHA-256 digest"
            )
        object_dir = self._object_dir(source_format, digest)
        manifest_path = object_dir / "manifest.json"
        payload_path = object_dir / "source"
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SourceRepositoryError(
                "source_not_found", f"source is not present: {source_format.value}/{digest}"
            ) from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SourceRepositoryError(
                "source_manifest_invalid", "source manifest is unreadable or malformed"
            ) from exc
        if not isinstance(value, dict) or set(value) != _MANIFEST_FIELDS:
            raise SourceRepositoryError(
                "source_manifest_invalid", "source manifest has an invalid schema"
            )
        expected = {
            "schema_version": SOURCE_REPOSITORY_SCHEMA,
            "source_format": source_format.value,
            "artifact_digest": digest,
        }
        if any(value.get(key) != item for key, item in expected.items()):
            raise SourceRepositoryError(
                "source_manifest_invalid", "source manifest identity does not match its key"
            )
        media_type = value.get("media_type")
        size = value.get("size")
        if (
            not isinstance(media_type, str)
            or not media_type
            or ";" in media_type
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise SourceRepositoryError(
                "source_manifest_invalid", "source manifest metadata is invalid"
            )
        try:
            payload = payload_path.read_bytes()
        except OSError as exc:
            raise SourceRepositoryError(
                "source_corrupt", "source bytes do not match the manifest"
            ) from exc
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            raise SourceRepositoryError(
                "source_corrupt", "source bytes do not match the manifest"
            )
        return (
            SourceArtifact(
                source_format=source_format,
                artifact_digest=digest,
                size=size,
                media_type=media_type,
                origin=origin,
            ),
            payload,
        )

    def _object_dir(self, source_format: SourceFormat, digest: str) -> Path:
        return (
            self.root
            / "source-repository"
            / "v1"
            / source_format.value
            / "sha256"
            / digest[:2]
            / digest
        )

    def _asset_object_dir(self, digest: str) -> Path:
        return (
            self.root
            / "source-repository"
            / "v1"
            / "asset"
            / "sha256"
            / digest[:2]
            / digest
        )

    def _read_asset_verified(self, digest: str) -> RepositoryAsset:
        if _SHA256_RE.fullmatch(digest) is None:
            raise SourceRepositoryError(
                "invalid_artifact_digest", "artifact digest must be a SHA-256 digest"
            )
        object_dir = self._asset_object_dir(digest)
        manifest_path = object_dir / "manifest.json"
        payload_path = object_dir / "asset"
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SourceRepositoryError(
                "asset_not_found", f"asset is not present: {digest}"
            ) from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SourceRepositoryError(
                "asset_manifest_invalid", "asset manifest is unreadable or malformed"
            ) from exc
        if not isinstance(value, dict) or set(value) != _ASSET_MANIFEST_FIELDS:
            raise SourceRepositoryError(
                "asset_manifest_invalid", "asset manifest has an invalid schema"
            )
        if (
            value.get("schema_version") != SOURCE_ASSET_SCHEMA
            or value.get("artifact_digest") != digest
        ):
            raise SourceRepositoryError(
                "asset_manifest_invalid", "asset manifest identity does not match its key"
            )
        media_type = value.get("media_type")
        size = value.get("size")
        if (
            not isinstance(media_type, str)
            or not media_type
            or ";" in media_type
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise SourceRepositoryError(
                "asset_manifest_invalid", "asset manifest metadata is invalid"
            )
        if not self._payload_matches(payload_path, digest, size):
            raise SourceRepositoryError(
                "asset_corrupt", "asset bytes do not match the manifest"
            )
        return RepositoryAsset(
            artifact_digest=digest,
            size=size,
            media_type=media_type,
        )

    @contextmanager
    def _content_lock(
        self, source_format: SourceFormat, digest: str
    ) -> Iterator[None]:
        lock_path = (
            self.root
            / "source-repository"
            / "v1"
            / "locks"
            / source_format.value
            / f"{digest}.lock"
        )
        with exclusive_file_lock(lock_path):
            yield

    @contextmanager
    def _asset_content_lock(self, digest: str) -> Iterator[None]:
        lock_path = (
            self.root
            / "source-repository"
            / "v1"
            / "locks"
            / "asset"
            / f"{digest}.lock"
        )
        with exclusive_file_lock(lock_path):
            yield

    @staticmethod
    def _payload_matches(path: Path, digest: str, size: int) -> bool:
        return payload_matches(path, digest, size)

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        atomic_write_bytes(path, payload)

    @staticmethod
    def _normalize_media_type(media_type: str) -> str:
        normalized = media_type.strip().casefold()
        if not normalized or ";" in normalized or "/" not in normalized:
            raise ValueError("media_type must be a normalized MIME type")
        return normalized

    @staticmethod
    def _format_for_path(path: Path) -> SourceFormat:
        try:
            return SOURCE_SUFFIXES[path.suffix.casefold()]
        except KeyError as exc:
            raise SourceRepositoryError(
                "unsupported_source",
                f"unsupported local source suffix: {path.suffix or '<none>'}",
            ) from exc


__all__ = [
    "DEFAULT_MEDIA_TYPES",
    "RepositoryAsset",
    "SOURCE_ASSET_SCHEMA",
    "SOURCE_REPOSITORY_SCHEMA",
    "SourceRepository",
    "SourceRepositoryError",
]
