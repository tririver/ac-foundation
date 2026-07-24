from __future__ import annotations

import hashlib
import os
import socket
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping, Protocol, TypeVar

from .cancellation import CancellationToken
from .contracts import StateContract
from .effects import EffectJournal
from .errors import (
    CancelledError,
    CorruptStateError,
    IdempotencyConflictError,
    InvalidTransitionError,
    ResumeInputConflictError,
    ResumeMismatchError,
    RevisionConflictError,
    RunNotFoundError,
    UnsafeEffectRecoveryError,
    UnsupportedSchemaError,
)
from .events import EventWriter
from .groups import WorkGroupRunner
from .identity import canonical_json_bytes, semantic_key, validate_simple_id
from .lease import FileLease
from .models import (
    ArtifactDigest,
    ArtifactRef,
    Awaiting,
    CancelRequest,
    ExecutionSlice,
    Failed,
    FailureMode,
    GroupExecutionResult,
    GroupResult,
    JsonValue,
    Paused,
    ResumeReason,
    RunError,
    RunOutcome,
    RunSnapshot,
    RunSpec,
    RunStatus,
    RunView,
    Succeeded,
    UnitResult,
    ValidationIssue,
    ValidationReport,
    WorkUnit,
)
from .storage import (
    AtomicStateStore,
    ImmutableArtifactStore,
    atomic_write_json,
    read_json_object,
    require_fields,
    utc_now,
)

T = TypeVar("T")
_TERMINAL = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
_PROCESS_START_IDENTITY = f"{os.getpid()}-{time.time_ns()}"


class RunHandler(Protocol):
    name: str

    def execute(self, context: "RunContext") -> RunOutcome: ...


class _SliceExpired(Exception):
    pass


def _digest_json(value: ArtifactDigest) -> dict[str, JsonValue]:
    return {
        "algorithm": value.algorithm,
        "value": value.value,
        "size_bytes": value.size_bytes,
    }


def _ref_json(value: ArtifactRef | None) -> JsonValue:
    if value is None:
        return None
    return {
        "artifact_id": value.artifact_id,
        "digest": _digest_json(value.digest),
        "media_type": value.media_type,
        "relative_path": value.relative_path,
    }


def _decode_digest(value: JsonValue) -> ArtifactDigest:
    if not isinstance(value, dict):
        raise CorruptStateError("digest must be an object")
    require_fields(value, required={"algorithm", "value", "size_bytes"})
    algorithm, digest, size = value["algorithm"], value["value"], value["size_bytes"]
    if (
        algorithm != "sha256"
        or not isinstance(digest, str)
        or len(digest) != 64
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
    ):
        raise CorruptStateError("invalid digest")
    return ArtifactDigest("sha256", digest, size)


def _decode_ref(value: JsonValue) -> ArtifactRef | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CorruptStateError("artifact ref must be an object")
    require_fields(
        value, required={"artifact_id", "digest", "media_type", "relative_path"}
    )
    artifact_id, media_type, relative_path = (
        value["artifact_id"],
        value["media_type"],
        value["relative_path"],
    )
    if not all(isinstance(part, str) for part in (artifact_id, media_type, relative_path)):
        raise CorruptStateError("invalid artifact ref strings")
    return ArtifactRef(
        artifact_id, _decode_digest(value["digest"]), media_type, relative_path
    )


def _awaiting_json(value: Awaiting | None) -> JsonValue:
    if value is None:
        return None
    return {
        "reason": value.reason.value,
        "resume_key": value.resume_key,
        "input_required": value.input_required,
        "request_ref": _ref_json(value.request_ref),
        "response_contract": value.response_contract,
        "details": dict(value.details),
    }


def _decode_awaiting(value: JsonValue) -> Awaiting | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CorruptStateError("awaiting must be an object")
    require_fields(
        value,
        required={
            "reason",
            "resume_key",
            "input_required",
            "request_ref",
            "response_contract",
            "details",
        },
    )
    try:
        reason = ResumeReason(str(value["reason"]))
    except ValueError as exc:
        raise CorruptStateError("unknown resume reason") from exc
    resume_key = value["resume_key"]
    input_required = value["input_required"]
    response_contract = value["response_contract"]
    details = value["details"]
    if (
        not isinstance(resume_key, str)
        or not isinstance(input_required, bool)
        or not (response_contract is None or isinstance(response_contract, str))
        or not isinstance(details, dict)
    ):
        raise CorruptStateError("invalid awaiting fields")
    awaiting = Awaiting(
        reason,
        resume_key,
        input_required,
        _decode_ref(value["request_ref"]),
        response_contract,
        details,
    )
    _validate_awaiting(awaiting)
    return awaiting


def _validate_awaiting(value: Awaiting) -> None:
    validate_simple_id(value.resume_key, label="resume key")
    if value.input_required and (
        value.request_ref is None or not value.response_contract
    ):
        raise InvalidTransitionError(
            "input-required pause needs request_ref and response_contract"
        )
    if not value.input_required and value.response_contract is not None:
        raise InvalidTransitionError(
            "no-input pause must not declare a response contract"
        )


class _SnapshotStore:
    def __init__(self, path: Path):
        self.path = path

    def _encode(self, value: RunSnapshot) -> dict[str, JsonValue]:
        return {
            "schema_version": "arc.jobs.run_snapshot.v1",
            "run_id": value.run_id,
            "revision": value.revision,
            "status": value.status.value,
            "attempt": value.attempt,
            "created_at": value.created_at,
            "updated_at": value.updated_at,
            "awaiting": _awaiting_json(value.awaiting),
            "result_ref": _ref_json(value.result_ref),
            "error": (
                {
                    "code": value.error.code,
                    "message": value.error.message,
                    "details": dict(value.error.details),
                }
                if value.error
                else None
            ),
            "interrupted": value.interrupted,
        }

    def _decode(self, document: Mapping[str, JsonValue]) -> RunSnapshot:
        require_fields(
            document,
            required={
                "schema_version",
                "run_id",
                "revision",
                "status",
                "attempt",
                "created_at",
                "updated_at",
                "awaiting",
                "result_ref",
                "error",
                "interrupted",
            },
        )
        if document["schema_version"] != "arc.jobs.run_snapshot.v1":
            raise UnsupportedSchemaError(str(document["schema_version"]))
        try:
            status = RunStatus(str(document["status"]))
        except ValueError as exc:
            raise CorruptStateError("unknown run status") from exc
        run_id, revision, attempt = (
            document["run_id"],
            document["revision"],
            document["attempt"],
        )
        created_at, updated_at, interrupted = (
            document["created_at"],
            document["updated_at"],
            document["interrupted"],
        )
        if (
            not isinstance(run_id, str)
            or not isinstance(revision, int)
            or isinstance(revision, bool)
            or revision < 0
            or not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or attempt < 0
            or not isinstance(created_at, str)
            or not isinstance(updated_at, str)
            or not isinstance(interrupted, bool)
        ):
            raise CorruptStateError("invalid snapshot scalar fields")
        error_json = document["error"]
        error = None
        if error_json is not None:
            if not isinstance(error_json, dict):
                raise CorruptStateError("run error must be an object")
            require_fields(error_json, required={"code", "message", "details"})
            if (
                not isinstance(error_json["code"], str)
                or not isinstance(error_json["message"], str)
                or not isinstance(error_json["details"], dict)
            ):
                raise CorruptStateError("invalid run error")
            error = RunError(
                error_json["code"], error_json["message"], error_json["details"]
            )
        snapshot = RunSnapshot(
            run_id,
            revision,
            status,
            attempt,
            created_at,
            updated_at,
            _decode_awaiting(document["awaiting"]),
            _decode_ref(document["result_ref"]),
            error,
            interrupted,
        )
        self._validate(None, snapshot, reading=True)
        return snapshot

    def _validate(
        self,
        previous: RunSnapshot | None,
        next_value: RunSnapshot,
        *,
        reading: bool = False,
    ) -> None:
        validate_simple_id(next_value.run_id, label="run id")
        if next_value.status is RunStatus.PAUSED:
            if next_value.awaiting is None:
                raise InvalidTransitionError("paused run requires awaiting descriptor")
            _validate_awaiting(next_value.awaiting)
        elif next_value.awaiting is not None:
            raise InvalidTransitionError("only paused runs may contain awaiting")
        if next_value.status is RunStatus.SUCCEEDED and next_value.error is not None:
            raise InvalidTransitionError("succeeded run cannot contain an error")
        if (
            next_value.status is not RunStatus.SUCCEEDED
            and next_value.result_ref is not None
        ):
            raise InvalidTransitionError("only succeeded runs may contain result_ref")
        if next_value.status is RunStatus.FAILED and next_value.error is None:
            raise InvalidTransitionError("failed run requires an error")
        if next_value.status is not RunStatus.FAILED and next_value.error is not None:
            raise InvalidTransitionError("only failed runs may contain an error")
        if reading or previous is None:
            return
        if next_value.run_id != previous.run_id or next_value.created_at != previous.created_at:
            raise InvalidTransitionError("run identity and created_at are immutable")
        allowed = {
            RunStatus.PENDING: {RunStatus.RUNNING, RunStatus.CANCELLED},
            RunStatus.RUNNING: {
                RunStatus.SUCCEEDED,
                RunStatus.PAUSED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            },
            RunStatus.PAUSED: {RunStatus.RUNNING, RunStatus.CANCELLED},
        }
        if next_value.status not in allowed.get(previous.status, set()):
            raise InvalidTransitionError(
                f"invalid run transition {previous.status} -> {next_value.status}"
            )
        expected_attempt = (
            previous.attempt + 1
            if next_value.status is RunStatus.RUNNING
            else previous.attempt
        )
        if next_value.attempt != expected_attempt:
            raise InvalidTransitionError("attempt changes only when entering RUNNING")

    def read(self) -> RunSnapshot:
        if not self.path.exists():
            raise RunNotFoundError(self.path.parent.name)
        return self._decode(read_json_object(self.path))

    def create(self, value: RunSnapshot) -> RunSnapshot:
        self._validate(None, value)
        try:
            atomic_write_json(self.path, self._encode(value), exclusive=True)
        except FileExistsError:
            current = self.read()
            if current != value:
                raise RevisionConflictError("snapshot already exists")
            return current
        return value

    def compare_and_swap(
        self, expected_revision: int, value: RunSnapshot
    ) -> RunSnapshot:
        current = self.read()
        if current.revision != expected_revision or value.revision != expected_revision + 1:
            raise RevisionConflictError(
                f"expected revision {expected_revision}, found {current.revision}"
            )
        self._validate(current, value)
        atomic_write_json(self.path, self._encode(value))
        return value


class RunRepository:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        (self.root / "runs").mkdir(parents=True, exist_ok=True, mode=0o700)

    def run_directory(self, run_id: str) -> Path:
        validate_simple_id(run_id, label="run id")
        return self.root / "runs" / run_id

    def _snapshot_store(self, run_id: str) -> _SnapshotStore:
        return _SnapshotStore(self.run_directory(run_id) / "snapshot.json")

    def _spec_document(self, spec: RunSpec) -> dict[str, JsonValue]:
        validate_simple_id(spec.run_id, label="run id")
        validate_simple_id(spec.handler, label="handler")
        projection = {
            "schema_version": "arc.jobs.run_semantic_key.v1",
            "handler": spec.handler,
            "semantic_input": dict(spec.semantic_input),
        }
        return {
            "schema_version": "arc.jobs.run_spec.v1",
            "run_id": spec.run_id,
            "handler": spec.handler,
            "semantic_input": dict(spec.semantic_input),
            "semantic_key_schema": "arc.jobs.run_semantic_key.v1",
            "semantic_key_sha256": semantic_key(projection).sha256,
        }

    def create(self, spec: RunSpec) -> RunSnapshot:
        run_directory = self.run_directory(spec.run_id)
        run_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        creation_lease = FileLease(run_directory / "create.lock").acquire(blocking=True)
        try:
            expected = self._spec_document(spec)
            spec_path = run_directory / "spec.json"
            if spec_path.exists():
                existing_spec = self.read_spec(spec.run_id)
                if self._spec_document(existing_spec) != expected:
                    raise IdempotencyConflictError(
                        f"run id {spec.run_id!r} already has different semantic input"
                    )
            else:
                if (run_directory / "snapshot.json").exists():
                    raise CorruptStateError(
                        "run snapshot exists without its immutable spec"
                    )
                atomic_write_json(spec_path, expected)
            store = self._snapshot_store(spec.run_id)
            if store.path.exists():
                return store.read()
            now = utc_now()
            return store.create(
                RunSnapshot(
                    spec.run_id,
                    0,
                    RunStatus.PENDING,
                    0,
                    now,
                    now,
                )
            )
        finally:
            creation_lease.release()

    def read_spec(self, run_id: str) -> RunSpec:
        document = read_json_object(self.run_directory(run_id) / "spec.json")
        require_fields(
            document,
            required={
                "schema_version",
                "run_id",
                "handler",
                "semantic_input",
                "semantic_key_schema",
                "semantic_key_sha256",
            },
        )
        if document["schema_version"] != "arc.jobs.run_spec.v1":
            raise UnsupportedSchemaError(str(document["schema_version"]))
        run_value, handler, semantic_input = (
            document["run_id"],
            document["handler"],
            document["semantic_input"],
        )
        if (
            run_value != run_id
            or not isinstance(handler, str)
            or not isinstance(semantic_input, dict)
        ):
            raise CorruptStateError("invalid run spec")
        spec = RunSpec(run_id, handler, semantic_input)
        if self._spec_document(spec) != document:
            raise CorruptStateError("run semantic key does not match spec")
        return spec

    def inspect(self, run_id: str) -> RunView:
        snapshot = self._snapshot_store(run_id).read()
        cancel = CancellationToken(self.run_directory(run_id) / "cancel.json").read()
        return RunView(snapshot, cancel)

    def request_cancel(self, run_id: str, *, reason: str | None = None) -> RunView:
        snapshot = self._snapshot_store(run_id).read()
        if snapshot.status in _TERMINAL:
            return self.inspect(run_id)
        token = CancellationToken(self.run_directory(run_id) / "cancel.json")
        persisted_request = token.request(reason=reason)
        events = EventWriter(self.run_directory(run_id) / "events.jsonl", run_id=run_id)
        if persisted_request.reason != reason:
            events.emit(
                "cancel_warning",
                {"code": "cancel_reason_not_replaced"},
            )
        lease = FileLease(self.run_directory(run_id) / "lease.lock")
        try:
            lease.acquire()
        except Exception as exc:
            from .errors import RunBusyError

            if isinstance(exc, RunBusyError):
                return self.inspect(run_id)
            raise
        try:
            snapshot = self._snapshot_store(run_id).read()
            if snapshot.status in _TERMINAL:
                return self.inspect(run_id)
            events.emit("run_cancelled", {"reason": reason})
            cancelled = replace(
                snapshot,
                revision=snapshot.revision + 1,
                status=RunStatus.CANCELLED,
                updated_at=utc_now(),
                awaiting=None,
                error=None,
                interrupted=snapshot.status is RunStatus.RUNNING or snapshot.interrupted,
            )
            self._snapshot_store(run_id).compare_and_swap(snapshot.revision, cancelled)
            return self.inspect(run_id)
        finally:
            lease.release()

    def validate(self, run_id: str) -> ValidationReport:
        issues: list[ValidationIssue] = []
        try:
            spec = self.read_spec(run_id)
            view = self.inspect(run_id)
            if spec.run_id != view.snapshot.run_id:
                issues.append(
                    ValidationIssue("run_id_mismatch", "spec and snapshot run IDs differ")
                )
            EventWriter(
                self.run_directory(run_id) / "events.jsonl", run_id=run_id
            ).validate()
            if view.snapshot.result_ref is not None:
                ImmutableArtifactStore(
                    self.run_directory(run_id), repository_root=self.root
                ).verify(view.snapshot.result_ref)
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    type(exc).__name__,
                    str(exc),
                )
            )
        return ValidationReport(tuple(issues))


class RunContext:
    def __init__(
        self,
        repository: RunRepository,
        snapshot: RunSnapshot,
        *,
        resume_input: Mapping[str, JsonValue] | None,
        execution_slice: ExecutionSlice | None,
    ):
        self.repository = repository
        self.run_id = snapshot.run_id
        self.attempt = snapshot.attempt
        self.spec = repository.read_spec(snapshot.run_id)
        self.semantic_input = self.spec.semantic_input
        self.resume_input = resume_input
        self.execution_slice = execution_slice
        self.run_directory = repository.run_directory(self.run_id)
        self.cancel = CancellationToken(self.run_directory / "cancel.json")
        self.events = EventWriter(
            self.run_directory / "events.jsonl", run_id=self.run_id
        )
        self.artifacts = ImmutableArtifactStore(
            self.run_directory, repository_root=repository.root
        )
        self.effects = EffectJournal(
            self.run_directory / "effects",
            artifacts=self.artifacts,
            cancel=self.cancel,
        )

    def state(
        self, namespace: str, contract: StateContract[T]
    ) -> AtomicStateStore[T]:
        validate_simple_id(namespace, label="state namespace")
        return AtomicStateStore(
            self.run_directory / "state" / f"{namespace}.json", contract
        )

    def checkpoint(self) -> None:
        self.cancel.raise_if_requested()
        if (
            self.execution_slice is not None
            and self.execution_slice.monotonic_deadline is not None
            and time.monotonic() >= self.execution_slice.monotonic_deadline
        ):
            raise _SliceExpired

    def run_group(
        self,
        group_id: str,
        units: tuple[WorkUnit, ...],
        worker: Callable[
            [WorkUnit], JsonValue | ArtifactRef | UnitResult | Paused
        ],
        *,
        max_workers: int,
        failure_mode: FailureMode,
    ) -> GroupExecutionResult:
        return WorkGroupRunner(
            self.run_directory / "groups",
            cancel=self.cancel,
            events=self.events,
            checkpoint=self.checkpoint,
        ).run(
            group_id,
            units,
            worker,
            max_workers=max_workers,
            failure_mode=failure_mode,
        )


class RunEngine:
    def __init__(self, repository: RunRepository):
        self.repository = repository

    def execute(
        self,
        spec: RunSpec,
        handler: RunHandler,
        *,
        execution_slice: ExecutionSlice | None = None,
    ) -> RunSnapshot:
        if handler.name != spec.handler:
            raise ResumeMismatchError("handler name does not match RunSpec.handler")
        snapshot = self.repository.create(spec)
        if snapshot.status in _TERMINAL:
            return snapshot
        return self._run(
            spec.run_id,
            handler,
            resume_input=None,
            execution_slice=execution_slice,
        )

    def resume(
        self,
        run_id: str,
        handler: RunHandler,
        *,
        input: Mapping[str, JsonValue] | None = None,
        execution_slice: ExecutionSlice | None = None,
    ) -> RunSnapshot:
        spec = self.repository.read_spec(run_id)
        if handler.name != spec.handler:
            raise ResumeMismatchError("handler does not match durable run spec")
        return self._run(
            run_id,
            handler,
            resume_input=input,
            execution_slice=execution_slice,
        )

    def _persist_resume_input(
        self,
        run_directory: Path,
        awaiting: Awaiting,
        resume_input: Mapping[str, JsonValue] | None,
    ) -> Mapping[str, JsonValue] | None:
        if awaiting.input_required:
            if resume_input is None:
                raise ResumeMismatchError("resume input is required")
            if resume_input.get("resume_key") != awaiting.resume_key:
                raise ResumeMismatchError("resume key mismatch")
        elif resume_input is None:
            return None
        elif resume_input.get("resume_key") != awaiting.resume_key:
            raise ResumeMismatchError("resume key mismatch")
        if resume_input is None:
            return None
        digest = hashlib.sha256(canonical_json_bytes(resume_input)).hexdigest()
        key_digest = hashlib.sha256(awaiting.resume_key.encode("utf-8")).hexdigest()
        path = run_directory / "resume-inputs" / f"{key_digest}.json"
        document: dict[str, JsonValue] = {
            "schema_version": "arc.jobs.resume_input_record.v1",
            "resume_key": awaiting.resume_key,
            "input_sha256": digest,
            "input": dict(resume_input),
        }
        if path.exists():
            existing = read_json_object(path)
            if existing != document:
                raise ResumeInputConflictError(
                    "the same resume key was submitted with different input"
                )
        else:
            try:
                atomic_write_json(path, document, exclusive=True)
            except FileExistsError:
                if read_json_object(path) != document:
                    raise ResumeInputConflictError(
                        "the same resume key was submitted with different input"
                    )
        return resume_input

    def _validate_replayed_resume_input(
        self, run_directory: Path, resume_input: Mapping[str, JsonValue]
    ) -> None:
        resume_key = resume_input.get("resume_key")
        if not isinstance(resume_key, str):
            raise ResumeMismatchError("resume input requires resume_key")
        key_digest = hashlib.sha256(resume_key.encode("utf-8")).hexdigest()
        path = run_directory / "resume-inputs" / f"{key_digest}.json"
        if not path.exists():
            raise ResumeMismatchError("resume key is not part of this run lineage")
        document = read_json_object(path)
        if document.get("input") != dict(resume_input):
            raise ResumeInputConflictError(
                "the same resume key was submitted with different input"
            )

    def _run(
        self,
        run_id: str,
        handler: RunHandler,
        *,
        resume_input: Mapping[str, JsonValue] | None,
        execution_slice: ExecutionSlice | None,
    ) -> RunSnapshot:
        run_directory = self.repository.run_directory(run_id)
        lease = FileLease(run_directory / "lease.lock").acquire()
        store = self.repository._snapshot_store(run_id)
        events = EventWriter(run_directory / "events.jsonl", run_id=run_id)
        try:
            snapshot = store.read()
            if snapshot.status in _TERMINAL:
                if resume_input is not None:
                    self._validate_replayed_resume_input(run_directory, resume_input)
                return snapshot
            if snapshot.status is RunStatus.RUNNING:
                awaiting = Awaiting(
                    ResumeReason.EXECUTION_INTERRUPTED,
                    f"interrupted-{snapshot.attempt}",
                    False,
                    details={"code": "orphaned_running_attempt"},
                )
                recovered = replace(
                    snapshot,
                    revision=snapshot.revision + 1,
                    status=RunStatus.PAUSED,
                    updated_at=utc_now(),
                    awaiting=awaiting,
                    interrupted=True,
                )
                events.emit(
                    "run_interrupted",
                    {"attempt": snapshot.attempt, "code": "orphaned_running_attempt"},
                )
                snapshot = store.compare_and_swap(snapshot.revision, recovered)
            cancel_token = CancellationToken(run_directory / "cancel.json")
            cancel_request = cancel_token.read()
            if cancel_request is not None:
                events.emit(
                    "run_cancelled",
                    {"reason": cancel_request.reason},
                )
                cancelled = replace(
                    snapshot,
                    revision=snapshot.revision + 1,
                    status=RunStatus.CANCELLED,
                    updated_at=utc_now(),
                    awaiting=None,
                    error=None,
                    interrupted=(
                        snapshot.interrupted
                        or snapshot.status is RunStatus.RUNNING
                    ),
                )
                return store.compare_and_swap(snapshot.revision, cancelled)
            if snapshot.status is RunStatus.PAUSED:
                assert snapshot.awaiting is not None
                resume_input = self._persist_resume_input(
                    run_directory, snapshot.awaiting, resume_input
                )
            elif resume_input is not None:
                raise ResumeMismatchError("pending run does not accept resume input")
            running = replace(
                snapshot,
                revision=snapshot.revision + 1,
                status=RunStatus.RUNNING,
                attempt=snapshot.attempt + 1,
                updated_at=utc_now(),
                awaiting=None,
                result_ref=None,
                error=None,
            )
            snapshot = store.compare_and_swap(snapshot.revision, running)
            atomic_write_json(
                run_directory / "lease.json",
                {
                    "schema_version": "arc.jobs.lease.v1",
                    "attempt": snapshot.attempt,
                    "acquired_at": utc_now(),
                    "pid": os.getpid(),
                    "process_start_identity": _PROCESS_START_IDENTITY,
                    "hostname": socket.gethostname(),
                },
            )
            context = RunContext(
                self.repository,
                snapshot,
                resume_input=resume_input,
                execution_slice=execution_slice,
            )
            try:
                context.checkpoint()
                outcome = handler.execute(context)
                context.checkpoint()
                if isinstance(outcome, Succeeded):
                    if outcome.result_ref is not None:
                        context.artifacts.verify(outcome.result_ref)
                    status = RunStatus.SUCCEEDED
                    next_snapshot = replace(
                        snapshot,
                        revision=snapshot.revision + 1,
                        status=status,
                        updated_at=utc_now(),
                        result_ref=outcome.result_ref,
                    )
                elif isinstance(outcome, Paused):
                    _validate_awaiting(outcome.awaiting)
                    next_snapshot = replace(
                        snapshot,
                        revision=snapshot.revision + 1,
                        status=RunStatus.PAUSED,
                        updated_at=utc_now(),
                        awaiting=outcome.awaiting,
                    )
                elif isinstance(outcome, Failed):
                    next_snapshot = replace(
                        snapshot,
                        revision=snapshot.revision + 1,
                        status=RunStatus.FAILED,
                        updated_at=utc_now(),
                        error=outcome.error,
                    )
                else:
                    raise TypeError("handler returned an invalid RunOutcome")
            except CancelledError:
                next_snapshot = replace(
                    snapshot,
                    revision=snapshot.revision + 1,
                    status=RunStatus.CANCELLED,
                    updated_at=utc_now(),
                )
            except _SliceExpired:
                next_snapshot = replace(
                    snapshot,
                    revision=snapshot.revision + 1,
                    status=RunStatus.PAUSED,
                    updated_at=utc_now(),
                    awaiting=Awaiting(
                        ResumeReason.EXECUTION_BUDGET_EXHAUSTED,
                        f"slice-{snapshot.attempt}",
                        False,
                        details={"code": "execution_slice_expired"},
                    ),
                )
            except UnsafeEffectRecoveryError as exc:
                next_snapshot = replace(
                    snapshot,
                    revision=snapshot.revision + 1,
                    status=RunStatus.PAUSED,
                    updated_at=utc_now(),
                    awaiting=Awaiting(
                        ResumeReason.SUPERVISION_REQUIRED,
                        f"effect-{snapshot.attempt}",
                        False,
                        details={
                            "code": "unsafe_effect_recovery",
                            "message": str(exc)[:300],
                        },
                    ),
                )
            except Exception as exc:
                next_snapshot = replace(
                    snapshot,
                    revision=snapshot.revision + 1,
                    status=RunStatus.FAILED,
                    updated_at=utc_now(),
                    error=RunError(
                        "handler_unhandled_exception",
                        f"{type(exc).__name__}: {str(exc)[:300]}",
                    ),
                )
            events.emit(
                "run_terminal" if next_snapshot.status in _TERMINAL else "run_paused",
                {
                    "status": next_snapshot.status.value,
                    "attempt": next_snapshot.attempt,
                },
            )
            return store.compare_and_swap(snapshot.revision, next_snapshot)
        finally:
            lease.release()
