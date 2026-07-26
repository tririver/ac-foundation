from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from arc_jobs import (
    AtomicStateStore,
    Awaiting,
    EventWriter,
    FailureMode,
    ImmutableArtifactStore,
    ResumeReason,
    RunContext,
    RunError,
    RunRepository,
    RunSpec,
    RunStatus,
    UnitResult,
    WorkUnit,
)
from arc_proposer_reviewer.artifacts import (
    proposal_artifact_id,
    review_artifact_id,
    transcript_artifact_id,
)
from arc_proposer_reviewer.dialogue import TranscriptTurn, encode_transcript_turn
from arc_proposer_reviewer.handler import ProposerReviewerHandler
from arc_proposer_reviewer.models import (
    BATCH_SCHEMA_VERSION,
    BatchFailurePolicy,
    BatchRequest,
    LoopSpec,
    WorkerSpec,
)
from arc_proposer_reviewer.projection import (
    BatchProjectionIntegrityError,
    inspect_batch,
    read_batch_round,
    read_batch_trace,
)
from arc_proposer_reviewer.protocol import encode_batch_request
from arc_proposer_reviewer.state import (
    _LoopState,
    _LoopStateContract,
    _PauseRecord,
    batch_group_id,
    state_namespace,
)


SCHEMA = {"type": "object", "additionalProperties": True}


def _request() -> BatchRequest:
    return BatchRequest(
        BATCH_SCHEMA_VERSION,
        "projection-batch",
        (
            LoopSpec(
                "loop-a",
                {"question": "Q"},
                (WorkerSpec("proposer-a", "Propose.", SCHEMA),),
                WorkerSpec("reviewer-a", "Review.", SCHEMA),
                2,
            ),
        ),
    )


def _running_repository(tmp_path: Path) -> tuple[RunRepository, object, LoopSpec]:
    request = _request()
    repository = RunRepository(tmp_path)
    snapshot = repository.create(
        RunSpec(
            "run-a",
            ProposerReviewerHandler.name,
            encode_batch_request(request),
        )
    )
    running = repository._snapshot_store("run-a").compare_and_swap(  # type: ignore[attr-defined]
        snapshot.revision,
        replace(
            snapshot,
            revision=snapshot.revision + 1,
            status=RunStatus.RUNNING,
            attempt=1,
        ),
    )
    return repository, running, request.loops[0]


def _state_store(repository: RunRepository, loop: LoopSpec):
    return AtomicStateStore(
        repository.run_directory("run-a")
        / "state"
        / f"{state_namespace(loop.loop_id)}.json",
        _LoopStateContract(),
    )


def _empty_state(loop: LoopSpec) -> _LoopState:
    return _LoopState(
        revision=0,
        loop_id=loop.loop_id,
        rounds_completed=0,
        proposal_refs={},
        current_proposer_ids=(),
        review_ref=None,
        proposer_sessions={},
        reviewer_session=None,
        transcript_refs=(),
        pauses={},
        termination=None,
    )


def _commit_one_round(
    repository: RunRepository,
    loop: LoopSpec,
    *,
    corrupt_transcript: bool = False,
) -> None:
    store = _state_store(repository, loop)
    initial = store.create(_empty_state(loop))
    artifacts = ImmutableArtifactStore(
        repository.run_directory("run-a"), repository_root=repository.root
    ).scoped("proposer-reviewer")
    proposal = artifacts.publish_json(
        proposal_artifact_id(loop.loop_id, 1, "proposer-a"),
        {"proposal": "committed"},
    )
    review = artifacts.publish_json(
        review_artifact_id(loop.loop_id, 1, "reviewer-a"),
        {"review": "committed"},
    )
    if corrupt_transcript:
        proposal_turn = artifacts.publish_json(
            transcript_artifact_id(loop.loop_id, 1, "001"),
            {"schema_version": "not-a-transcript"},
        )
    else:
        proposal_turn = artifacts.publish_json(
            transcript_artifact_id(loop.loop_id, 1, "001"),
            encode_transcript_turn(
                TranscriptTurn(
                    "proposer",
                    "proposer-a",
                    1,
                    proposal,
                    ("reviewer-a",),
                )
            ),
        )
    review_turn = artifacts.publish_json(
        transcript_artifact_id(loop.loop_id, 1, "002"),
        encode_transcript_turn(
            TranscriptTurn(
                "reviewer",
                "reviewer-a",
                1,
                review,
                ("proposer-a",),
            )
        ),
    )
    store.compare_and_swap(
        initial.revision,
        replace(
            initial,
            revision=initial.revision + 1,
            rounds_completed=1,
            proposal_refs={"proposer-a": proposal},
            current_proposer_ids=("proposer-a",),
            review_ref=review,
            transcript_refs=(proposal_turn, review_turn),
        ),
    )


def test_running_batch_with_no_loop_state_has_empty_trace(tmp_path: Path) -> None:
    repository, _snapshot, loop = _running_repository(tmp_path)

    inspection = inspect_batch(repository, "run-a")

    assert inspection.durable_lifecycle == "running"
    assert inspection.loop_revisions == {loop.loop_id: None}
    assert inspection.loops[0].lifecycle == "pending"
    assert inspection.loops[0].phase == "not_started"
    assert inspection.loops[0].activity.best_effort is True
    assert read_batch_trace(repository, "run-a").loops[0].rounds == ()


def test_trace_uses_only_committed_transcript_frontier_and_show_round_is_safe(
    tmp_path: Path,
) -> None:
    repository, _snapshot, loop = _running_repository(tmp_path)
    _commit_one_round(repository, loop)
    artifacts = ImmutableArtifactStore(
        repository.run_directory("run-a"), repository_root=repository.root
    ).scoped("proposer-reviewer")
    orphan = artifacts.publish_json(
        proposal_artifact_id(loop.loop_id, 2, "proposer-a"),
        {"proposal": "orphaned before CAS"},
    )

    trace = read_batch_trace(repository, "run-a")
    round_ref = trace.loops[0].rounds[0]
    expanded = read_batch_round(repository, "run-a", loop.loop_id, 1)

    assert trace.run_revision == 1
    assert trace.loop_revisions == {loop.loop_id: 1}
    assert len(trace.loops[0].rounds) == 1
    assert all(
        ref.artifact_id != orphan.artifact_id
        for ref in (*round_ref.proposal_refs.values(), round_ref.review_ref, *round_ref.transcript_refs)
    )
    assert expanded.proposals == {"proposer-a": {"proposal": "committed"}}
    assert expanded.review == {"review": "committed"}
    assert "relative_path" not in asdict(round_ref.review_ref)
    assert "relative_path" not in asdict(expanded)


def test_paused_loop_uses_safe_pause_summary(tmp_path: Path) -> None:
    repository, snapshot, loop = _running_repository(tmp_path)
    store = _state_store(repository, loop)
    initial = store.create(_empty_state(loop))
    artifacts = ImmutableArtifactStore(
        repository.run_directory("run-a"), repository_root=repository.root
    ).scoped("proposer-reviewer")
    pause_request = artifacts.publish_json("pause-request", {"request": "input"})
    paused = _PauseRecord(
        "proposer",
        "proposer-a",
        1,
        "private-task-id",
        Awaiting(
            ResumeReason.INTERACTION_REQUIRED,
            "private-resume-key",
            True,
            request_ref=pause_request,
            response_contract="projection-test-input.v1",
            details={"private": "detail"},
        ),
    )
    store.compare_and_swap(
        initial.revision,
        replace(
            initial,
            revision=initial.revision + 1,
            pauses={"proposer.proposer-a": paused},
        ),
    )
    repository._snapshot_store("run-a").compare_and_swap(  # type: ignore[attr-defined]
        snapshot.revision,
        replace(
            snapshot,
            revision=snapshot.revision + 1,
            status=RunStatus.PAUSED,
            awaiting=paused.awaiting,
        ),
    )

    loop_inspection = inspect_batch(repository, "run-a").loops[0]

    assert loop_inspection.lifecycle == "paused"
    assert loop_inspection.phase == "paused"
    assert loop_inspection.pause is not None
    assert len(loop_inspection.pause.entries) == 1
    pause_entry = loop_inspection.pause.entries[0]
    assert pause_entry.worker_id == "proposer-a"
    assert pause_entry.role == "proposer"
    assert pause_entry.round_number == 1
    assert pause_entry.reason == "interaction_required"
    assert pause_entry.code is None
    assert pause_entry.resume_key == "private-resume-key"
    assert pause_entry.response_contract == "projection-test-input.v1"
    assert pause_entry.request_ref is not None
    assert not hasattr(pause_entry.request_ref, "relative_path")
    assert pause_entry.resume_action == "provide_input"
    assert "private-task-id" not in repr(loop_inspection)
    assert "detail" not in repr(loop_inspection)


def test_failed_group_unit_controls_loop_lifecycle(tmp_path: Path) -> None:
    repository, snapshot, loop = _running_repository(tmp_path)
    context = RunContext(
        repository,
        snapshot,
        resume_input=None,
        execution_slice=None,
    )
    context.run_group(
        batch_group_id(),
        (WorkUnit(loop.loop_id, {"loop": loop.loop_id}),),
        lambda unit: UnitResult(
            unit.unit_id,
            "failed",
            error=RunError(
                "proposer_failed",
                "one or more proposers failed",
                {
                    "causes": [
                        {
                            "worker_id": "proposer-a",
                            "code": "invalid_request",
                            "message": "/private/provider/path",
                        }
                    ]
                },
            ),
        ),
        max_workers=1,
        failure_mode=FailureMode.COLLECT,
    )

    loop_inspection = inspect_batch(repository, "run-a").loops[0]

    assert loop_inspection.lifecycle == "failed"
    assert loop_inspection.activity.loop_group_status == "failed"
    assert loop_inspection.failure is not None
    assert loop_inspection.failure.code == "proposer_failed"
    assert loop_inspection.failure.worker_causes[0].code == "invalid_request"
    assert "/private/provider/path" not in repr(loop_inspection)


def test_fail_fast_skipped_loop_projects_as_failed_not_integrity_error(
    tmp_path: Path,
) -> None:
    first = _request().loops[0]
    second = replace(
        first,
        loop_id="loop-b",
        proposers=(WorkerSpec("proposer-b", "Propose.", SCHEMA),),
        reviewer=WorkerSpec("reviewer-b", "Review.", SCHEMA),
    )
    request = BatchRequest(
        BATCH_SCHEMA_VERSION,
        "fail-fast-projection",
        (first, second),
        BatchFailurePolicy.FAIL_FAST,
    )
    repository = RunRepository(tmp_path)
    pending = repository.create(
        RunSpec(
            "run-a",
            ProposerReviewerHandler.name,
            encode_batch_request(request),
        )
    )
    running = repository._snapshot_store("run-a").compare_and_swap(  # type: ignore[attr-defined]
        pending.revision,
        replace(
            pending,
            revision=pending.revision + 1,
            status=RunStatus.RUNNING,
            attempt=1,
        ),
    )
    context = RunContext(
        repository,
        running,
        resume_input=None,
        execution_slice=None,
    )
    context.run_group(
        batch_group_id(),
        (
            WorkUnit(first.loop_id, {"loop": first.loop_id}),
            WorkUnit(second.loop_id, {"loop": second.loop_id}),
        ),
        lambda unit: UnitResult(
            unit.unit_id,
            "failed",
            error=RunError("proposer_failed", "proposer failed"),
        ),
        max_workers=1,
        failure_mode=FailureMode.FAIL_FAST,
    )
    repository._snapshot_store("run-a").compare_and_swap(  # type: ignore[attr-defined]
        running.revision,
        replace(
            running,
            revision=running.revision + 1,
            status=RunStatus.SUCCEEDED,
        ),
    )

    inspection = inspect_batch(repository, "run-a")

    assert [loop.lifecycle for loop in inspection.loops] == ["failed", "failed"]
    assert inspection.loops[1].integrity_error is None
    assert inspection.loops[1].failure is not None
    assert inspection.loops[1].failure.code == "fail_fast_skipped"


def test_inspection_projects_active_worker_without_private_ids(
    tmp_path: Path,
) -> None:
    repository, _snapshot, _loop = _running_repository(tmp_path)
    events = EventWriter(
        repository.run_directory("run-a") / "events.jsonl",
        run_id="run-a",
    )
    events.emit(
        "proposer_reviewer_loop_started",
        {"loop_id": "loop-a"},
    )
    events.emit(
        "proposer_reviewer_worker_started",
        {
            "loop_id": "loop-a",
            "round": 1,
            "role": "proposer",
            "worker_id": "proposer-a",
        },
    )
    inspection = inspect_batch(repository, "run-a")
    active = inspection.loops[0].active_workers

    assert inspection.durable_lifecycle == "running"
    assert inspection.lifecycle_counts == {
        "pending": 1,
        "running": 0,
        "paused": 0,
        "succeeded": 0,
        "failed": 0,
        "integrity_error": 0,
    }
    assert len(active) == 1
    assert active[0].worker_id == "proposer-a"
    assert inspection.loops[0].last_activity_at is not None

    events.emit(
        "proposer_reviewer_worker_finished",
        {
            "loop_id": "loop-a",
            "round": 1,
            "role": "proposer",
            "worker_id": "proposer-a",
            "status": "succeeded",
        },
    )
    assert inspect_batch(repository, "run-a").loops[0].active_workers == ()


def test_inspection_marks_truncated_runtime_activity_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _snapshot, _loop = _running_repository(tmp_path)

    class TruncatedEvents:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def validate(self) -> None:
            pass

        def tail(self):
            return (
                {
                    "sequence": 9,
                    "event": "proposer_reviewer_loop_started",
                    "emitted_at": "2026-07-25T00:00:00Z",
                    "data": {"loop_id": "loop-a"},
                },
            )

    monkeypatch.setattr(
        "arc_proposer_reviewer.projection.EventWriter",
        TruncatedEvents,
    )

    inspection = inspect_batch(repository, "run-a")

    assert (
        inspection.activity_integrity_error
        == "runtime_activity_history_truncated"
    )


def test_inspection_distinguishes_malformed_runtime_activity_log(
    tmp_path: Path,
) -> None:
    repository, _snapshot, _loop = _running_repository(tmp_path)
    events_path = repository.run_directory("run-a") / "events.jsonl"
    with events_path.open("ab") as handle:
        handle.write(b'{"malformed":"closed-record"}\n')

    inspection = inspect_batch(repository, "run-a")

    assert (
        inspection.activity_integrity_error
        == "runtime_activity_integrity_error"
    )
    assert inspection.loops[0].active_workers == ()


def test_succeeded_batch_without_loop_state_is_an_integrity_error(tmp_path: Path) -> None:
    repository, snapshot, loop = _running_repository(tmp_path)
    repository._snapshot_store("run-a").compare_and_swap(  # type: ignore[attr-defined]
        snapshot.revision,
        replace(
            snapshot,
            revision=snapshot.revision + 1,
            status=RunStatus.SUCCEEDED,
        ),
    )

    inspection = inspect_batch(repository, "run-a")

    assert inspection.loops[0].loop_id == loop.loop_id
    assert inspection.loops[0].lifecycle == "integrity_error"
    assert inspection.loops[0].integrity_error == "missing_loop_state_after_success"
    with pytest.raises(BatchProjectionIntegrityError):
        read_batch_trace(repository, "run-a")


def test_corrupt_committed_transcript_keeps_inspection_but_closes_trace(
    tmp_path: Path,
) -> None:
    repository, _snapshot, loop = _running_repository(tmp_path)
    _commit_one_round(repository, loop, corrupt_transcript=True)

    inspection = inspect_batch(repository, "run-a")

    assert inspection.loops[0].lifecycle == "integrity_error"
    assert inspection.loops[0].integrity_error is not None
    with pytest.raises(BatchProjectionIntegrityError):
        read_batch_trace(repository, "run-a")
    with pytest.raises(BatchProjectionIntegrityError):
        read_batch_round(repository, "run-a", loop.loop_id, 1)
