"""Read-only projections for durable proposer-reviewer batches.

The projection intentionally reads the executor's existing CAS state and work
groups.  It neither creates state nor discovers artifacts by directory scan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Mapping, cast

from arc_jobs import (
    ArcJobsError,
    AtomicStateStore,
    EventWriter,
    GroupUnitView,
    GroupView,
    ImmutableArtifactStore,
    JsonValue,
    RunRepository,
    RunStatus,
)

from .artifacts import (
    proposal_artifact_id,
    review_artifact_id,
    transcript_artifact_id,
)
from .dialogue import TranscriptTurn, decode_transcript_turn
from .handler import ProposerReviewerHandler
from .models import (
    BatchFailurePolicy,
    BatchRequest,
    LoopSpec,
    LoopTermination,
)
from .protocol import decode_batch_request
from .state import (
    _LoopState,
    _LoopStateContract,
    _PauseRecord,
    batch_group_id,
    proposer_group_id,
    state_namespace,
)


_LoopLifecycle = Literal[
    "pending", "running", "paused", "succeeded", "failed", "integrity_error"
]
_LoopPhase = Literal["not_started", "proposers", "reviewer", "paused", "completed"]


class BatchProjectionIntegrityError(ValueError):
    """Raised when a strict trace cannot be verified from durable data."""


class CommittedRoundNotFoundError(ValueError):
    """Raised when a requested round is not part of the committed frontier."""


@dataclass(frozen=True)
class SafeArtifactRef:
    """A verified public artifact reference with no physical path."""

    artifact_id: str
    sha256: str
    size_bytes: int
    media_type: str


@dataclass(frozen=True)
class PauseEntry:
    worker_id: str
    role: Literal["proposer", "reviewer"]
    round_number: int
    reason: str
    code: str | None
    input_required: bool
    resume_key: str
    response_contract: str | None
    request_ref: SafeArtifactRef | None
    resume_action: Literal["resume", "provide_input"]


@dataclass(frozen=True)
class PauseSummary:
    entries: tuple[PauseEntry, ...]
    input_required: bool


@dataclass(frozen=True)
class ActiveWorkerSummary:
    worker_id: str
    role: Literal["proposer", "reviewer"]
    round_number: int
    last_activity_at: str


@dataclass(frozen=True)
class WorkerFailureCause:
    worker_id: str
    code: str
    message: str


@dataclass(frozen=True)
class FailureSummary:
    code: str
    message: str
    worker_causes: tuple[WorkerFailureCause, ...]


@dataclass(frozen=True)
class BestEffortActivity:
    """Observed durable-group counts; not a recovery or ranking signal."""

    best_effort: Literal[True]
    loop_group_status: str | None
    proposer_pending: int
    proposer_succeeded: int
    proposer_failed: int


@dataclass(frozen=True)
class LoopInspection:
    loop_id: str
    lifecycle: _LoopLifecycle
    phase: _LoopPhase
    current_round: int | None
    rounds_completed: int
    revision: int | None
    pause: PauseSummary | None
    activity: BestEffortActivity
    active_workers: tuple[ActiveWorkerSummary, ...] = ()
    last_activity_at: str | None = None
    failure: FailureSummary | None = None
    integrity_error: str | None = None


@dataclass(frozen=True)
class BatchInspection:
    run_id: str
    durable_lifecycle: str
    run_revision: int
    lifecycle_counts: Mapping[str, int]
    loop_revisions: Mapping[str, int | None]
    loops: tuple[LoopInspection, ...]
    activity_integrity_error: str | None = None


@dataclass(frozen=True)
class CommittedRoundRef:
    loop_id: str
    round_number: int
    proposal_refs: Mapping[str, SafeArtifactRef]
    review_ref: SafeArtifactRef
    transcript_refs: tuple[SafeArtifactRef, ...]


@dataclass(frozen=True)
class LoopTrace:
    loop_id: str
    revision: int | None
    rounds: tuple[CommittedRoundRef, ...]


@dataclass(frozen=True)
class BatchTrace:
    run_id: str
    run_revision: int
    loop_revisions: Mapping[str, int | None]
    loops: tuple[LoopTrace, ...]


@dataclass(frozen=True)
class CommittedRound:
    loop_id: str
    round_number: int
    proposals: Mapping[str, JsonValue]
    review: JsonValue
    proposal_refs: Mapping[str, SafeArtifactRef]
    review_ref: SafeArtifactRef
    transcript_refs: tuple[SafeArtifactRef, ...]


@dataclass(frozen=True)
class _CommittedRoundData:
    reference: CommittedRoundRef
    proposal_refs: Mapping[str, object]
    review_ref: object


@dataclass(frozen=True)
class _LoopData:
    state: _LoopState | None
    rounds: tuple[_CommittedRoundData, ...]
    integrity_error: str | None
    proposer_group: GroupView | None = None


@dataclass(frozen=True)
class _RuntimeActivity:
    active_workers: tuple[ActiveWorkerSummary, ...] = ()
    last_activity_at: str | None = None


class BatchProjection:
    """One non-persistent read projection over a batch's durable frontier."""

    def __init__(self, repository: RunRepository, run_id: str) -> None:
        self.repository = repository
        self.run_id = run_id
        self.run_view = repository.inspect(run_id)
        spec = repository.read_spec(run_id)
        if spec.handler != ProposerReviewerHandler.name:
            raise ValueError("run is not a proposer-reviewer batch")
        self.request: BatchRequest = decode_batch_request(spec.semantic_input)
        self.artifacts = ImmutableArtifactStore(
            repository.run_directory(run_id), repository_root=repository.root
        ).scoped("proposer-reviewer")
        self.batch_group, self.batch_group_error = self._read_batch_group()
        (
            self.runtime_activity,
            self.runtime_activity_error,
        ) = self._read_runtime_activity()
        batch_units = _group_units(self.batch_group)
        self._loop_data: dict[str, _LoopData] = {}
        for loop in self.request.loops:
            data = self._read_loop(loop)
            proposer_group, activity_error = self._read_proposer_group(loop, data.state)
            missing_after_success = (
                self.run_view.snapshot.status is RunStatus.SUCCEEDED
                and data.state is None
                and (
                    batch_units.get(loop.loop_id) is None
                    or batch_units[loop.loop_id].status != "failed"
                )
                and not self._is_fail_fast_skipped(loop.loop_id, batch_units)
            )
            self._loop_data[loop.loop_id] = _LoopData(
                state=data.state,
                rounds=data.rounds,
                integrity_error=(
                    data.integrity_error
                    or activity_error
                    or (
                        "missing_loop_state_after_success"
                        if missing_after_success
                        else None
                    )
                ),
                proposer_group=proposer_group,
            )

    def inspect(self) -> BatchInspection:
        loop_units = _group_units(self.batch_group)
        loop_revisions = {
            loop.loop_id: self._loop_data[loop.loop_id].state.revision
            if self._loop_data[loop.loop_id].state is not None
            else None
            for loop in self.request.loops
        }
        inspections = tuple(
            self._inspect_loop(loop, loop_units.get(loop.loop_id))
            for loop in self.request.loops
        )
        lifecycle_counts = {
            lifecycle: 0
            for lifecycle in (
                "pending",
                "running",
                "paused",
                "succeeded",
                "failed",
                "integrity_error",
            )
        }
        for inspection in inspections:
            lifecycle_counts[inspection.lifecycle] += 1
        return BatchInspection(
            run_id=self.run_view.snapshot.run_id,
            durable_lifecycle=self.run_view.snapshot.status.value,
            run_revision=self.run_view.snapshot.revision,
            lifecycle_counts=lifecycle_counts,
            loop_revisions=loop_revisions,
            loops=inspections,
            activity_integrity_error=(
                self.batch_group_error or self.runtime_activity_error
            ),
        )

    def _read_runtime_activity(
        self,
    ) -> tuple[Mapping[str, _RuntimeActivity], str | None]:
        try:
            writer = EventWriter(
                self.repository.run_directory(self.run_id) / "events.jsonl",
                run_id=self.run_id,
            )
            writer.validate()
            events = writer.tail()
        except (ArcJobsError, ValueError):
            return {}, "runtime_activity_integrity_error"
        except OSError:
            return {}, "runtime_activity_unavailable"
        source_error = (
            "runtime_activity_history_truncated"
            if events
            and type(events[0].get("sequence")) is int
            and events[0]["sequence"] != 1
            else None
        )
        active: dict[str, dict[tuple[str, str], ActiveWorkerSummary]] = {}
        last_activity: dict[str, str] = {}
        for document in events:
            event = document.get("event")
            data = document.get("data")
            emitted_at = document.get("emitted_at")
            if (
                not isinstance(event, str)
                or not event.startswith("proposer_reviewer_")
                or not isinstance(data, Mapping)
                or not isinstance(emitted_at, str)
            ):
                continue
            loop_id = data.get("loop_id")
            if not isinstance(loop_id, str):
                continue
            last_activity[loop_id] = emitted_at
            workers = active.setdefault(loop_id, {})
            if event in {
                "proposer_reviewer_loop_started",
                "proposer_reviewer_loop_finished",
            }:
                workers.clear()
                continue
            worker_id = data.get("worker_id")
            role = data.get("role")
            round_number = data.get("round")
            if (
                not isinstance(worker_id, str)
                or role not in {"proposer", "reviewer"}
                or type(round_number) is not int
            ):
                continue
            key = (role, worker_id)
            if event == "proposer_reviewer_worker_finished":
                workers.pop(key, None)
                continue
            if event != "proposer_reviewer_worker_started":
                continue
            workers[key] = ActiveWorkerSummary(
                worker_id=worker_id,
                role=role,
                round_number=round_number,
                last_activity_at=emitted_at,
            )
        return (
            {
                loop_id: _RuntimeActivity(
                    active_workers=tuple(
                        sorted(
                            workers.values(),
                            key=lambda value: (
                                value.round_number,
                                value.role,
                                value.worker_id,
                            ),
                        )
                    ),
                    last_activity_at=last_activity.get(loop_id),
                )
                for loop_id, workers in active.items()
            },
            source_error,
        )

    def _is_fail_fast_skipped(
        self,
        loop_id: str,
        batch_units: Mapping[str, GroupUnitView],
    ) -> bool:
        if self.request.failure_policy is not BatchFailurePolicy.FAIL_FAST:
            return False
        for loop in self.request.loops:
            if loop.loop_id == loop_id:
                return False
            unit = batch_units.get(loop.loop_id)
            if unit is not None and unit.status == "failed":
                return True
        return False

    def trace(self) -> BatchTrace:
        if self.batch_group_error is not None:
            raise BatchProjectionIntegrityError(self.batch_group_error)
        loop_revisions: dict[str, int | None] = {}
        traces: list[LoopTrace] = []
        for loop in self.request.loops:
            data = self._require_loop(loop.loop_id)
            state = data.state
            loop_revisions[loop.loop_id] = None if state is None else state.revision
            traces.append(
                LoopTrace(
                    loop_id=loop.loop_id,
                    revision=None if state is None else state.revision,
                    rounds=tuple(item.reference for item in data.rounds),
                )
            )
        return BatchTrace(
            run_id=self.run_view.snapshot.run_id,
            run_revision=self.run_view.snapshot.revision,
            loop_revisions=loop_revisions,
            loops=tuple(traces),
        )

    def read_round(self, loop_id: str, round_number: int) -> CommittedRound:
        if type(round_number) is not int or round_number < 1:
            raise CommittedRoundNotFoundError("round number must be a positive integer")
        data = self._require_loop(loop_id)
        try:
            round_data = data.rounds[round_number - 1]
        except IndexError as exc:
            raise CommittedRoundNotFoundError(
                f"round {round_number} is not committed for loop {loop_id!r}"
            ) from exc
        if round_data.reference.round_number != round_number:
            raise BatchProjectionIntegrityError("committed round index is inconsistent")
        proposals = {
            worker_id: self._read_json(ref)
            for worker_id, ref in round_data.proposal_refs.items()
        }
        review = self._read_json(round_data.review_ref)
        return CommittedRound(
            loop_id=round_data.reference.loop_id,
            round_number=round_data.reference.round_number,
            proposals=proposals,
            review=review,
            proposal_refs=round_data.reference.proposal_refs,
            review_ref=round_data.reference.review_ref,
            transcript_refs=round_data.reference.transcript_refs,
        )

    def _read_batch_group(self):
        state_path = (
            self.repository.run_directory(self.run_id)
            / "groups"
            / batch_group_id()
            / "state.json"
        )
        if not state_path.exists():
            return None, None
        try:
            return self.repository.inspect_group(self.run_id, batch_group_id()), None
        except (ArcJobsError, OSError, ValueError) as exc:
            return None, _integrity_code("batch_group", exc)

    def _read_loop(self, loop: LoopSpec) -> _LoopData:
        store = AtomicStateStore(
            self.repository.run_directory(self.run_id)
            / "state"
            / f"{state_namespace(loop.loop_id)}.json",
            _LoopStateContract(),
        )
        state: _LoopState | None = None
        try:
            state = store.read()
            if state is None:
                return _LoopData(None, (), None)
            if state.loop_id != loop.loop_id:
                raise BatchProjectionIntegrityError("loop state identity does not match request")
            rounds = self._read_committed_rounds(loop, state)
            return _LoopData(state, rounds, None)
        except (ArcJobsError, OSError, ValueError, json.JSONDecodeError) as exc:
            return _LoopData(state, (), _integrity_code("loop", exc))

    def _read_committed_rounds(
        self, loop: LoopSpec, state: _LoopState
    ) -> tuple[_CommittedRoundData, ...]:
        if state.rounds_completed == 0:
            if (
                state.proposal_refs
                or state.current_proposer_ids
                or state.review_ref is not None
                or state.transcript_refs
            ):
                raise BatchProjectionIntegrityError("empty loop state has committed data")
            return ()
        turns_by_round: dict[int, list[tuple[object, TranscriptTurn]]] = {
            number: [] for number in range(1, state.rounds_completed + 1)
        }
        transcript_refs_by_round: dict[int, list[SafeArtifactRef]] = {
            number: [] for number in range(1, state.rounds_completed + 1)
        }
        turn_numbers: dict[int, int] = {
            number: 0 for number in range(1, state.rounds_completed + 1)
        }
        for transcript_ref in state.transcript_refs:
            turn = self._read_transcript(transcript_ref)
            if turn.round_number not in turns_by_round:
                raise BatchProjectionIntegrityError("transcript references an uncommitted round")
            turn_numbers[turn.round_number] += 1
            expected_transcript_id = _scoped_artifact_id(
                transcript_artifact_id(
                    loop.loop_id,
                    turn.round_number,
                    f"{turn_numbers[turn.round_number]:03d}",
                )
            )
            if transcript_ref.artifact_id != expected_transcript_id:
                raise BatchProjectionIntegrityError("transcript artifact locator is inconsistent")
            turns_by_round[turn.round_number].append((transcript_ref, turn))
            transcript_refs_by_round[turn.round_number].append(_safe_ref(transcript_ref))

        result: list[_CommittedRoundData] = []
        for round_number in range(1, state.rounds_completed + 1):
            result.append(
                self._validate_round(
                    loop,
                    state,
                    round_number,
                    turns_by_round[round_number],
                    tuple(transcript_refs_by_round[round_number]),
                )
            )
        latest = result[-1]
        latest_by_worker: dict[str, object] = {}
        for committed in result:
            latest_by_worker.update(committed.proposal_refs)
        if set(state.proposal_refs) != set(latest_by_worker):
            raise BatchProjectionIntegrityError("proposal state frontier is inconsistent")
        for worker_id, ref in state.proposal_refs.items():
            if latest_by_worker[worker_id] != ref:
                raise BatchProjectionIntegrityError("proposal state frontier is inconsistent")
        if tuple(latest.proposal_refs) != state.current_proposer_ids:
            raise BatchProjectionIntegrityError("current proposer frontier is inconsistent")
        for worker_id in state.current_proposer_ids:
            if state.proposal_refs.get(worker_id) != latest.proposal_refs[worker_id]:
                raise BatchProjectionIntegrityError("current proposal frontier is inconsistent")
        if state.review_ref != latest.review_ref:
            raise BatchProjectionIntegrityError("current review frontier is inconsistent")
        return tuple(result)

    def _validate_round(
        self,
        loop: LoopSpec,
        state: _LoopState,
        round_number: int,
        entries: list[tuple[object, TranscriptTurn]],
        transcript_refs: tuple[SafeArtifactRef, ...],
    ) -> _CommittedRoundData:
        if not entries:
            raise BatchProjectionIntegrityError("committed round has no transcript turns")
        proposer_ids = {worker.worker_id for worker in loop.proposers}
        proposal_refs: dict[str, object] = {}
        review_ref: object | None = None
        reviewer_seen = False
        for index, (_transcript_ref, turn) in enumerate(entries):
            if turn.role == "proposer":
                if reviewer_seen or turn.worker_id not in proposer_ids:
                    raise BatchProjectionIntegrityError("proposer transcript turn is invalid")
                if turn.worker_id in proposal_refs:
                    raise BatchProjectionIntegrityError("round contains a duplicate proposer")
                if turn.addressed_worker_ids != (loop.reviewer.worker_id,):
                    raise BatchProjectionIntegrityError("proposer transcript audience is invalid")
                expected = _scoped_artifact_id(
                    proposal_artifact_id(loop.loop_id, round_number, turn.worker_id)
                )
                if turn.content_ref.artifact_id != expected:
                    raise BatchProjectionIntegrityError("proposal artifact locator is inconsistent")
                self._verify_ref(turn.content_ref)
                proposal_refs[turn.worker_id] = turn.content_ref
                continue
            if reviewer_seen or turn.worker_id != loop.reviewer.worker_id:
                raise BatchProjectionIntegrityError("reviewer transcript turn is invalid")
            reviewer_seen = True
            if index != len(entries) - 1:
                raise BatchProjectionIntegrityError("reviewer must close a committed round")
            if turn.addressed_worker_ids != tuple(proposal_refs):
                raise BatchProjectionIntegrityError("reviewer transcript audience is invalid")
            expected = _scoped_artifact_id(
                review_artifact_id(loop.loop_id, round_number, turn.worker_id)
            )
            if turn.content_ref.artifact_id != expected:
                raise BatchProjectionIntegrityError("review artifact locator is inconsistent")
            self._verify_ref(turn.content_ref)
            review_ref = turn.content_ref
        if not proposal_refs or review_ref is None:
            raise BatchProjectionIntegrityError("committed round is incomplete")
        if round_number == state.rounds_completed and state.termination is LoopTermination.FAILED:
            raise BatchProjectionIntegrityError("failed loop state has a terminal frontier")
        reference = CommittedRoundRef(
            loop_id=loop.loop_id,
            round_number=round_number,
            proposal_refs={
                worker_id: _safe_ref(ref)
                for worker_id, ref in proposal_refs.items()
            },
            review_ref=_safe_ref(review_ref),
            transcript_refs=transcript_refs,
        )
        return _CommittedRoundData(reference, proposal_refs, review_ref)

    def _read_transcript(self, ref: object) -> TranscriptTurn:
        value = self._read_json(ref)
        return decode_transcript_turn(value)

    def _read_json(self, ref: object) -> JsonValue:
        content = self.artifacts.read_bytes(cast(object, ref))  # type: ignore[arg-type]
        try:
            value = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BatchProjectionIntegrityError("artifact is not valid JSON") from exc
        return cast(JsonValue, value)

    def _verify_ref(self, ref: object) -> None:
        self.artifacts.verify(cast(object, ref))  # type: ignore[arg-type]

    def _require_loop(self, loop_id: str) -> _LoopData:
        try:
            data = self._loop_data[loop_id]
        except KeyError as exc:
            raise CommittedRoundNotFoundError(f"unknown loop {loop_id!r}") from exc
        if data.integrity_error is not None:
            raise BatchProjectionIntegrityError(data.integrity_error)
        return data

    def _inspect_loop(
        self, loop: LoopSpec, batch_unit: GroupUnitView | None
    ) -> LoopInspection:
        data = self._loop_data[loop.loop_id]
        runtime = self.runtime_activity.get(loop.loop_id, _RuntimeActivity())
        if data.integrity_error is not None:
            state = data.state
            return LoopInspection(
                loop_id=loop.loop_id,
                lifecycle="integrity_error",
                phase=_phase(state, data.proposer_group),
                current_round=_current_round(state),
                rounds_completed=0 if state is None else state.rounds_completed,
                revision=None if state is None else state.revision,
                pause=_pause_summary(state),
                activity=_activity(batch_unit, data.proposer_group),
                active_workers=runtime.active_workers,
                last_activity_at=runtime.last_activity_at,
                failure=_failure_summary(batch_unit),
                integrity_error=data.integrity_error,
            )
        state = data.state
        activity = _activity(batch_unit, data.proposer_group)
        batch_units = _group_units(self.batch_group)
        fail_fast_skipped = self._is_fail_fast_skipped(
            loop.loop_id, batch_units
        )
        lifecycle = _lifecycle(
            self.run_view.snapshot.status,
            state,
            batch_unit,
        )
        if fail_fast_skipped:
            lifecycle = "failed"
        pause = _pause_summary(state)
        phase = _phase(state, data.proposer_group)
        return LoopInspection(
            loop_id=loop.loop_id,
            lifecycle=lifecycle,
            phase=phase,
            current_round=_current_round(state),
            rounds_completed=0 if state is None else state.rounds_completed,
            revision=None if state is None else state.revision,
            pause=pause,
            activity=activity,
            active_workers=runtime.active_workers,
            last_activity_at=runtime.last_activity_at,
            failure=(
                FailureSummary(
                    "fail_fast_skipped",
                    "loop was not started after an earlier loop failed",
                    (),
                )
                if fail_fast_skipped
                else _failure_summary(batch_unit)
            ),
        )

    def _read_proposer_group(
        self, loop: LoopSpec, state: _LoopState | None
    ) -> tuple[GroupView | None, str | None]:
        if state is None or state.termination is not None or state.pauses:
            return None, None
        group_id = proposer_group_id(loop.loop_id, state.rounds_completed + 1)
        state_path = (
            self.repository.run_directory(self.run_id)
            / "groups"
            / group_id
            / "state.json"
        )
        if not state_path.exists():
            return None, None
        try:
            return self.repository.inspect_group(self.run_id, group_id), None
        except (ArcJobsError, OSError, ValueError) as exc:
            return None, _integrity_code("proposer_group", exc)


def inspect_batch(repository: RunRepository, run_id: str) -> BatchInspection:
    return BatchProjection(repository, run_id).inspect()


def read_batch_trace(repository: RunRepository, run_id: str) -> BatchTrace:
    return BatchProjection(repository, run_id).trace()


def read_batch_round(
    repository: RunRepository, run_id: str, loop_id: str, round_number: int
) -> CommittedRound:
    return BatchProjection(repository, run_id).read_round(loop_id, round_number)


def _safe_ref(ref: object) -> SafeArtifactRef:
    artifact_ref = cast(object, ref)
    return SafeArtifactRef(
        artifact_id=artifact_ref.artifact_id,  # type: ignore[attr-defined]
        sha256=artifact_ref.digest.value,  # type: ignore[attr-defined]
        size_bytes=artifact_ref.digest.size_bytes,  # type: ignore[attr-defined]
        media_type=artifact_ref.media_type,  # type: ignore[attr-defined]
    )


def _scoped_artifact_id(artifact_id: str) -> str:
    return f"proposer-reviewer/{artifact_id}"


def _group_units(group: object) -> Mapping[str, GroupUnitView]:
    if group is None:
        return {}
    return {unit.unit_id: unit for unit in group.units}  # type: ignore[attr-defined]


def _activity(
    batch_unit: GroupUnitView | None, proposer_group: object
) -> BestEffortActivity:
    counts = {"pending": 0, "succeeded": 0, "failed": 0}
    if proposer_group is not None:
        for unit in proposer_group.units:  # type: ignore[attr-defined]
            counts[unit.status] += 1
    return BestEffortActivity(
        best_effort=True,
        loop_group_status=None if batch_unit is None else batch_unit.status,
        proposer_pending=counts["pending"],
        proposer_succeeded=counts["succeeded"],
        proposer_failed=counts["failed"],
    )


def _lifecycle(
    run_status: RunStatus,
    state: _LoopState | None,
    batch_unit: GroupUnitView | None,
) -> _LoopLifecycle:
    if batch_unit is not None:
        if batch_unit.status == "failed":
            return "failed"
        if batch_unit.status == "succeeded":
            return "succeeded"
    if state is not None:
        if state.termination is LoopTermination.FAILED:
            return "failed"
        if state.termination is not None:
            return "succeeded"
        if state.pauses:
            return "paused"
    if run_status is RunStatus.FAILED:
        return "failed"
    if run_status is RunStatus.PAUSED:
        return "paused" if state is not None else "pending"
    if run_status is RunStatus.RUNNING:
        return "running" if state is not None else "pending"
    return "pending"


def _phase(state: _LoopState | None, proposer_group: object) -> _LoopPhase:
    if state is None:
        return "not_started"
    if state.termination is not None:
        return "completed"
    if state.pauses:
        return "paused"
    if proposer_group is not None and all(
        unit.status != "pending" for unit in proposer_group.units  # type: ignore[attr-defined]
    ):
        return "reviewer"
    return "proposers"


def _current_round(state: _LoopState | None) -> int | None:
    if state is None or state.termination is not None:
        return None
    if state.pauses:
        return min(record.round_number for record in state.pauses.values())
    return state.rounds_completed + 1


def _pause_summary(state: _LoopState | None) -> PauseSummary | None:
    if state is None or not state.pauses:
        return None
    records = tuple(
        sorted(
            state.pauses.values(),
            key=lambda record: (
                record.round_number,
                record.role,
                record.worker_id,
            ),
        )
    )
    return PauseSummary(
        entries=tuple(
            PauseEntry(
                worker_id=record.worker_id,
                role=cast(Literal["proposer", "reviewer"], record.role),
                round_number=record.round_number,
                reason=record.awaiting.reason.value,
                code=_pause_code(record),
                input_required=record.awaiting.input_required,
                resume_key=record.awaiting.resume_key,
                response_contract=record.awaiting.response_contract,
                request_ref=(
                    None
                    if record.awaiting.request_ref is None
                    else _safe_ref(record.awaiting.request_ref)
                ),
                resume_action=(
                    "provide_input"
                    if record.awaiting.input_required
                    else "resume"
                ),
            )
            for record in records
        ),
        input_required=any(record.awaiting.input_required for record in records),
    )


def _pause_code(record: _PauseRecord) -> str | None:
    awaiting = record.awaiting
    for key in ("llm_code", "code"):
        code = awaiting.details.get(key)
        if isinstance(code, str) and code:
            return code
    return None


def _failure_summary(batch_unit: GroupUnitView | None) -> FailureSummary | None:
    if batch_unit is None or batch_unit.error is None:
        return None
    error = batch_unit.error
    causes: list[WorkerFailureCause] = []
    raw_causes = error.details.get("causes")
    if isinstance(raw_causes, list):
        for raw_cause in raw_causes:
            if not isinstance(raw_cause, Mapping):
                continue
            worker_id = raw_cause.get("worker_id")
            code = raw_cause.get("code")
            if not isinstance(worker_id, str) or not isinstance(code, str):
                continue
            causes.append(
                WorkerFailureCause(
                    worker_id=worker_id,
                    code=code,
                    message="worker execution failed",
                )
            )
    message = (
        "loop execution failed"
        if error.code == "worker_unhandled_exception"
        else error.message.splitlines()[0][:300]
    )
    return FailureSummary(error.code, message, tuple(causes))


def _integrity_code(scope: str, exc: Exception) -> str:
    if isinstance(exc, BatchProjectionIntegrityError):
        return f"{scope}_integrity_error"
    return f"{scope}_{type(exc).__name__}"
