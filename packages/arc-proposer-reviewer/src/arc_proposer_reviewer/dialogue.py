from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from arc_jobs import ArtifactRef, JsonValue, encode_artifact_ref


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
        "schema_version": "arc.proposer_reviewer.transcript_turn.v1",
        "role": turn.role,
        "worker_id": turn.worker_id,
        "round": turn.round_number,
        "content_ref": _artifact_ref(turn.content_ref),
        "addressed_worker_ids": list(turn.addressed_worker_ids),
        "interaction_provenance_refs": [
            _artifact_ref(ref) for ref in turn.interaction_provenance_refs
        ],
    }


def _artifact_ref(ref: ArtifactRef) -> dict[str, JsonValue]:
    return encode_artifact_ref(ref)
