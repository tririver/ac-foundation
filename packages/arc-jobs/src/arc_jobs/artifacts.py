"""Shared codecs for immutable artifact identities.

The artifact store remains responsible for validating the canonical object
location recorded in a manifest.  This module deliberately owns only the
closed wire representation shared by durable state documents.
"""

from __future__ import annotations

from collections.abc import Mapping

from .models import ArtifactDigest, ArtifactRef, JsonValue

_DIGEST_FIELDS = {"algorithm", "value", "size_bytes"}
_REF_FIELDS = {"artifact_id", "digest", "media_type", "relative_path"}
_HEX = frozenset("0123456789abcdef")


def encode_artifact_digest(value: ArtifactDigest) -> dict[str, JsonValue]:
    """Encode one validated SHA-256 artifact digest."""

    _validate_digest(value.algorithm, value.value, value.size_bytes)
    return {
        "algorithm": value.algorithm,
        "value": value.value,
        "size_bytes": value.size_bytes,
    }


def decode_artifact_digest(value: JsonValue) -> ArtifactDigest:
    """Decode one closed SHA-256 artifact digest document."""

    if not isinstance(value, Mapping) or set(value) != _DIGEST_FIELDS:
        raise ValueError("artifact digest must use the closed digest shape")
    algorithm = value["algorithm"]
    digest = value["value"]
    size_bytes = value["size_bytes"]
    _validate_digest(algorithm, digest, size_bytes)
    return ArtifactDigest("sha256", digest, size_bytes)


def encode_artifact_ref(value: ArtifactRef) -> dict[str, JsonValue]:
    """Encode one validated closed artifact reference document."""

    _validate_ref_strings(value.artifact_id, value.media_type, value.relative_path)
    return {
        "artifact_id": value.artifact_id,
        "digest": encode_artifact_digest(value.digest),
        "media_type": value.media_type,
        "relative_path": value.relative_path,
    }


def decode_artifact_ref(value: JsonValue) -> ArtifactRef:
    """Decode one closed artifact reference document.

    Callers that permit ``None`` retain that outer nullable-schema handling.
    """

    if not isinstance(value, Mapping) or set(value) != _REF_FIELDS:
        raise ValueError("artifact ref must use the closed reference shape")
    artifact_id = value["artifact_id"]
    media_type = value["media_type"]
    relative_path = value["relative_path"]
    _validate_ref_strings(artifact_id, media_type, relative_path)
    return ArtifactRef(
        artifact_id,
        decode_artifact_digest(value["digest"]),
        media_type,
        relative_path,
    )


def _validate_digest(algorithm: object, digest: object, size_bytes: object) -> None:
    if (
        algorithm != "sha256"
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in _HEX for character in digest)
        or not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 0
    ):
        raise ValueError("invalid artifact digest")


def _validate_ref_strings(
    artifact_id: object, media_type: object, relative_path: object
) -> None:
    if not all(isinstance(item, str) for item in (artifact_id, media_type, relative_path)):
        raise ValueError("artifact reference fields must be strings")
