from __future__ import annotations

import json
from ac_jobs import ArtifactRef, JsonValue, decode_artifact_ref, encode_artifact_ref


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
    return encode_artifact_ref(ref)


def artifact_ref_from_document(value: JsonValue) -> ArtifactRef:
    try:
        return decode_artifact_ref(value)
    except ValueError as exc:
        raise ValueError("invalid artifact reference") from exc


def read_json_artifact(store: object, ref: ArtifactRef) -> JsonValue:
    content = store.read_bytes(ref)  # type: ignore[attr-defined]
    return json.loads(content.decode("utf-8"))
