from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, cast

from arc_jobs import ArtifactRef, JsonValue, encode_artifact_ref

from .artifacts import artifact_ref_from_document


_TRANSCRIPT_TURN_SCHEMA = "arc.proposer_reviewer.transcript_turn.v1"


@dataclass(frozen=True)
class TranscriptTurn:
    role: Literal["proposer", "reviewer"]
    worker_id: str
    round_number: int
    content_ref: ArtifactRef
    addressed_worker_ids: tuple[str, ...] = ()
    interaction_provenance_refs: tuple[ArtifactRef, ...] = ()


def encode_transcript_turn(turn: TranscriptTurn) -> dict[str, JsonValue]:
    return {
        "schema_version": _TRANSCRIPT_TURN_SCHEMA,
        "role": turn.role,
        "worker_id": turn.worker_id,
        "round": turn.round_number,
        "content_ref": _artifact_ref(turn.content_ref),
        "addressed_worker_ids": list(turn.addressed_worker_ids),
        "interaction_provenance_refs": [
            _artifact_ref(ref) for ref in turn.interaction_provenance_refs
        ],
    }


def decode_transcript_turn(value: JsonValue) -> TranscriptTurn:
    """Decode one closed, persisted transcript-turn document."""

    if not isinstance(value, Mapping):
        raise ValueError("transcript turn must be an object")
    expected = {
        "schema_version",
        "role",
        "worker_id",
        "round",
        "content_ref",
        "addressed_worker_ids",
        "interaction_provenance_refs",
    }
    if set(value) != expected:
        raise ValueError("transcript turn uses an invalid closed shape")
    if value["schema_version"] != _TRANSCRIPT_TURN_SCHEMA:
        raise ValueError("unsupported transcript turn schema")
    role = value["role"]
    worker_id = value["worker_id"]
    round_number = value["round"]
    addressed_worker_ids = value["addressed_worker_ids"]
    provenance_refs = value["interaction_provenance_refs"]
    if role not in {"proposer", "reviewer"}:
        raise ValueError("transcript turn role is invalid")
    if not isinstance(worker_id, str) or not worker_id:
        raise ValueError("transcript turn worker_id is invalid")
    if (
        type(round_number) is not int
        or round_number < 1
        or not isinstance(addressed_worker_ids, list)
        or not all(isinstance(item, str) and item for item in addressed_worker_ids)
        or not isinstance(provenance_refs, list)
    ):
        raise ValueError("transcript turn fields are invalid")
    return TranscriptTurn(
        role=cast(Literal["proposer", "reviewer"], role),
        worker_id=worker_id,
        round_number=round_number,
        content_ref=artifact_ref_from_document(value["content_ref"]),
        addressed_worker_ids=tuple(addressed_worker_ids),
        interaction_provenance_refs=tuple(
            artifact_ref_from_document(item) for item in provenance_refs
        ),
    )


def _artifact_ref(ref: ArtifactRef) -> dict[str, JsonValue]:
    return encode_artifact_ref(ref)
