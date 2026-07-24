from __future__ import annotations

import json
from collections.abc import Mapping

from arc_jobs import ArtifactDigest, ArtifactRef, JsonValue


def request_artifact_id() -> str:
    return "request"


def proposal_artifact_id(loop_id: str, round_number: int, worker_id: str) -> str:
    return f"loops/{loop_id}/rounds/{round_number:03d}/proposals/{worker_id}"


def review_artifact_id(loop_id: str, round_number: int, worker_id: str) -> str:
    return f"loops/{loop_id}/rounds/{round_number:03d}/reviews/{worker_id}"


def transcript_artifact_id(loop_id: str, round_number: int, turn_id: str) -> str:
    return f"loops/{loop_id}/rounds/{round_number:03d}/transcript/{turn_id}"


def loop_result_artifact_id(loop_id: str) -> str:
    return f"loops/{loop_id}/result"


def batch_result_artifact_id() -> str:
    return "batch/result"


def artifact_ref_to_document(ref: ArtifactRef) -> dict[str, JsonValue]:
    return {
        "artifact_id": ref.artifact_id,
        "digest": {
            "algorithm": ref.digest.algorithm,
            "value": ref.digest.value,
            "size_bytes": ref.digest.size_bytes,
        },
        "media_type": ref.media_type,
        "relative_path": ref.relative_path,
    }


def artifact_ref_from_document(value: JsonValue) -> ArtifactRef:
    if not isinstance(value, Mapping) or set(value) != {
        "artifact_id",
        "digest",
        "media_type",
        "relative_path",
    }:
        raise ValueError("invalid artifact reference")
    digest = value["digest"]
    if not isinstance(digest, Mapping) or set(digest) != {
        "algorithm",
        "value",
        "size_bytes",
    }:
        raise ValueError("invalid artifact digest")
    if (
        digest["algorithm"] != "sha256"
        or not isinstance(digest["value"], str)
        or type(digest["size_bytes"]) is not int
        or not isinstance(value["artifact_id"], str)
        or not isinstance(value["media_type"], str)
        or not isinstance(value["relative_path"], str)
    ):
        raise ValueError("invalid artifact reference fields")
    return ArtifactRef(
        artifact_id=value["artifact_id"],
        digest=ArtifactDigest(
            algorithm="sha256",
            value=digest["value"],
            size_bytes=digest["size_bytes"],
        ),
        media_type=value["media_type"],
        relative_path=value["relative_path"],
    )


def read_json_artifact(store: object, ref: ArtifactRef) -> JsonValue:
    content = store.read_bytes(ref)  # type: ignore[attr-defined]
    return json.loads(content.decode("utf-8"))
