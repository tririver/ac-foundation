from __future__ import annotations

from copy import deepcopy

import pytest

from arc_jobs import ArtifactDigest, ArtifactRef, Awaiting, JsonValue, ResumeReason
from arc_proposer_reviewer.state import (
    _PauseRecord,
    _pause_document,
    _pause_from_document,
)


def _record(*, role: str = "proposer", input_required: bool = False) -> _PauseRecord:
    request_ref = (
        ArtifactRef(
            artifact_id="pause-request",
            digest=ArtifactDigest("sha256", "a" * 64, 2),
            media_type="application/json",
            relative_path="artifacts/pause-request.json",
        )
        if input_required
        else None
    )
    return _PauseRecord(
        role=role,
        worker_id="worker-a",
        round_number=1,
        task_id="pr-proposer-abc",
        awaiting=Awaiting(
            reason=ResumeReason.INTERACTION_REQUIRED,
            resume_key="resume-abc-1",
            input_required=input_required,
            request_ref=request_ref,
            response_contract=(
                "arc.proposer_reviewer.pause_response.v1"
                if input_required
                else None
            ),
            details={"worker_id": "worker-a"},
        ),
    )


@pytest.mark.parametrize("role", ["proposer", "reviewer"])
@pytest.mark.parametrize("input_required", [False, True])
def test_pause_codec_round_trips_valid_roles_and_nullable_strings(
    role: str,
    input_required: bool,
) -> None:
    record = _record(role=role, input_required=input_required)

    assert _pause_from_document(_pause_document(record)) == record


@pytest.mark.parametrize(
    ("scope", "field", "invalid"),
    [
        ("pause", "role", "observer"),
        ("pause", "role", True),
        ("pause", "worker_id", ""),
        ("pause", "worker_id", "worker id"),
        ("pause", "round", True),
        ("pause", "round", 1.0),
        ("pause", "round", "1"),
        ("pause", "round", 0),
        ("pause", "task_id", None),
        ("pause", "task_id", "task/id"),
        ("awaiting", "reason", "unknown"),
        ("awaiting", "resume_key", ""),
        ("awaiting", "resume_key", "resume key"),
        ("awaiting", "input_required", 1),
        ("awaiting", "input_required", "true"),
        ("awaiting", "response_contract", False),
        ("awaiting", "response_contract", 1),
    ],
)
def test_pause_codec_rejects_corrupt_scalar_fields(
    scope: str,
    field: str,
    invalid: JsonValue,
) -> None:
    document = deepcopy(_pause_document(_record()))
    target = document if scope == "pause" else document["awaiting"]
    assert isinstance(target, dict)
    target[field] = invalid

    with pytest.raises(ValueError):
        _pause_from_document(document)
