from __future__ import annotations

import json
import threading
from dataclasses import replace

import pytest

import arc_jobs.engine as engine_module
import arc_jobs.groups as group_storage
import arc_jobs.lease as lease_module

from arc_jobs import (
    Awaiting,
    CorruptStateError,
    EventWriter,
    Failed,
    FileLease,
    FailureMode,
    InvalidTransitionError,
    Paused,
    ResumeReason,
    RunBusyError,
    RunEngine,
    RunError,
    RunRepository,
    RunSnapshot,
    RunSpec,
    RunStatus,
    RunContext,
    StoppedError,
    Succeeded,
    UnitResult,
    UnsupportedSchemaError,
    WorkUnit,
    validate_artifact_id,
    validate_simple_id,
)


def test_simple_and_hierarchical_identifier_contracts():
    assert validate_simple_id("run-1") == "run-1"
    assert validate_artifact_id("loops/a/rounds/001/result") == (
        "loops/a/rounds/001/result"
    )
    for value in ("a/b", "", "../x", "a\\b"):
        with pytest.raises(Exception):
            validate_simple_id(value)
    for value in ("/absolute", "a//b", "a/../b", "a\\b", "a/" + "x" * 129):
        with pytest.raises(Exception):
            validate_artifact_id(value)


def test_same_process_file_lease_is_exclusive(tmp_path):
    first = FileLease(tmp_path / "lease").acquire()
    try:
        with pytest.raises(RunBusyError):
            FileLease(tmp_path / "lease").acquire()
    finally:
        first.release()
    FileLease(tmp_path / "lease").acquire().release()


def test_file_lease_chmod_failure_releases_local_lease(tmp_path, monkeypatch):
    path = tmp_path / "lease"

    def fail_chmod(*_args):
        raise PermissionError("permission setup failed")

    with monkeypatch.context() as patched:
        patched.setattr(lease_module.os, "chmod", fail_chmod)
        with pytest.raises(PermissionError, match="permission setup failed"):
            FileLease(path).acquire()

    FileLease(path).acquire().release()


@pytest.mark.skipif(lease_module.fcntl is None, reason="POSIX flock required")
def test_file_lease_release_clears_local_lock_after_unlock_failure(tmp_path, monkeypatch):
    path = tmp_path / "lease"
    original_flock = lease_module.fcntl.flock
    held = FileLease(path).acquire()

    def fail_unlock(fd, flags):
        if flags == lease_module.fcntl.LOCK_UN:
            raise OSError("unlock failed")
        return original_flock(fd, flags)

    with monkeypatch.context() as patched:
        patched.setattr(lease_module.fcntl, "flock", fail_unlock)
        with pytest.raises(OSError, match="unlock failed"):
            held.release()

    FileLease(path).acquire().release()


def test_inspect_is_a_pure_read(tmp_path):
    repository = RunRepository(tmp_path)
    repository.create(RunSpec("run-1", "example.v1", {}))
    files = tuple(repository.run_directory("run-1").glob("**/*"))
    before = {
        path: path.stat().st_mtime_ns
        for path in files
        if path.is_file()
    }

    first = repository.inspect("run-1")
    second = repository.inspect("run-1")

    assert first == second
    assert {
        path: path.stat().st_mtime_ns
        for path in files
        if path.is_file()
    } == before


def test_event_limits_and_incomplete_tail(tmp_path):
    writer = EventWriter(tmp_path / "events.jsonl", run_id="run-1")
    writer.emit("step", {"summary": "safe", "content": "body"})
    with pytest.raises(ValueError):
        writer.emit("step", {"summary": "x" * (256 * 1024)})
    with (tmp_path / "events.jsonl").open("ab") as handle:
        handle.write(b'{"incomplete":')
    assert len(writer.tail()) == 1
    writer.validate()  # A trailing crash fragment is ignored.
    writer.emit("next_step", {"summary": "recovered"})
    writer.validate()
    assert [item["sequence"] for item in writer.tail()] == [1, 2]
    assert writer.tail()[-1]["data"]["summary"] == "recovered"


def test_event_writer_roundtrips_arbitrary_progress_bodies(tmp_path):
    writer = EventWriter(tmp_path / "events.jsonl", run_id="run-1")
    body_data = {
        "prompt": "task body",
        "nested": [{"content": "assistant body", "result": [1, 2, 3]}],
    }

    writer.emit("progress", body_data)

    assert writer.tail()[0]["data"] == body_data
    writer.validate()


def test_event_writer_reads_and_validates_complete_history(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path, run_id="run-1")
    writer.emit("first", {"value": 1})
    writer.emit("second", {"value": 2})

    assert [
        item["event"] for item in writer.read_all()
    ] == ["first", "second"]


def test_event_sink_observes_fsynced_event_after_lock_release(tmp_path):
    path = tmp_path / "events.jsonl"
    observed = []

    def sink(document):
        lease = FileLease(path.with_suffix(".lock")).acquire()
        lease.release()
        assert json.loads(path.read_text().splitlines()[-1]) == document
        observed.append(document)

    writer = EventWriter(path, run_id="run-1", event_sink=sink)
    writer.emit("progress", {"content": "visible body"})

    assert len(observed) == 1
    assert observed[0]["event"] == "progress"
    assert observed[0]["data"] == {"content": "visible body"}


def test_event_sink_failure_is_isolated_and_reported_once_per_writer(tmp_path):
    calls = []

    def fail(document):
        calls.append(document["event"])
        raise RuntimeError("presentation unavailable")

    writer = EventWriter(
        tmp_path / "events.jsonl",
        run_id="run-1",
        event_sink=fail,
    )
    writer.emit("first_step", {"index": 1})
    writer.emit("second_step", {"index": 2})

    assert calls == ["first_step", "second_step"]
    events = writer.tail()
    assert [event["event"] for event in events] == [
        "first_step",
        "progress_sink_failed",
        "second_step",
    ]
    diagnostic = events[1]["data"]
    assert diagnostic["source_event"] == "first_step"
    assert diagnostic["error_type"] == "RuntimeError"
    writer.validate()


def test_event_validation_rejects_tampered_identity(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path, run_id="run-1")
    writer.emit("step", {"summary": "safe"})
    document = json.loads(path.read_text())
    document["event_id"] = "0" * 64
    path.write_text(json.dumps(document) + "\n")
    with pytest.raises(Exception):
        writer.validate()


class MalformedFailureHandler:
    name = "malformed-failure.v1"

    def __init__(self, error):
        self.error = error

    def execute(self, context):
        return Failed(self.error)


@pytest.mark.parametrize(
    "error",
    (
        RunError(7, "message"),
        RunError("", "message"),
        RunError("error_code", 7),
        RunError("error_code", ""),
        RunError("error_code", "message", []),
        {"code": "error_code", "message": "message", "details": {}},
    ),
)
def test_malformed_failed_outcome_becomes_valid_terminal_error(tmp_path, error):
    repository = RunRepository(tmp_path)
    handler = MalformedFailureHandler(error)

    snapshot = RunEngine(repository).execute(
        RunSpec("run-1", handler.name, {}), handler
    )

    assert snapshot.status is RunStatus.FAILED
    assert snapshot.error is not None
    assert snapshot.error.code == "handler_unhandled_exception"
    assert isinstance(snapshot.error.message, str)
    assert repository.inspect("run-1").snapshot == snapshot
    assert repository.validate("run-1").ok


def test_snapshot_create_rejects_malformed_run_error_before_write(tmp_path):
    path = tmp_path / "snapshot.json"
    store = engine_module._SnapshotStore(path)
    snapshot = RunSnapshot(
        "run-1",
        0,
        RunStatus.FAILED,
        0,
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
        error=RunError("error_code", 7),
    )

    with pytest.raises(InvalidTransitionError, match="invalid run error fields"):
        store.create(snapshot)

    assert not path.exists()


def test_snapshot_compare_and_swap_rejects_malformed_run_error_before_write(
    tmp_path,
):
    repository = RunRepository(tmp_path)
    pending = repository.create(RunSpec("run-1", "example.v1", {}))
    store = repository._snapshot_store("run-1")
    running = replace(
        pending,
        revision=1,
        status=RunStatus.RUNNING,
        attempt=1,
        updated_at="2026-01-01T00:00:01Z",
    )
    store.compare_and_swap(pending.revision, running)
    failed = replace(
        running,
        revision=2,
        status=RunStatus.FAILED,
        updated_at="2026-01-01T00:00:02Z",
        error=RunError("error_code", "message", []),
    )

    with pytest.raises(InvalidTransitionError, match="invalid run error fields"):
        store.compare_and_swap(running.revision, failed)

    assert store.read() == running


class BlockingHandler:
    name = "blocking.v1"

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def execute(self, context):
        self.started.set()
        self.release.wait(timeout=5)
        context.checkpoint()
        return Succeeded()


def test_stop_running_handler_is_visible_and_cooperative(tmp_path):
    repository = RunRepository(tmp_path)
    engine = RunEngine(repository)
    handler = BlockingHandler()
    result = []
    thread = threading.Thread(
        target=lambda: result.append(
            engine.execute(RunSpec("run-1", handler.name, {}), handler)
        )
    )
    thread.start()
    assert handler.started.wait(timeout=5)

    view = repository.request_stop("run-1", reason="stop")
    assert view.snapshot.status is RunStatus.RUNNING
    assert view.stop_request is not None
    assert view.stop_request.reason == "stop"

    handler.release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result[0].status is RunStatus.PAUSED
    assert result[0].awaiting is not None
    assert result[0].awaiting.reason is ResumeReason.EXECUTION_STOPPED
    assert result[0].awaiting.details["reason"] == "stop"


def test_repeated_stop_keeps_first_request_metadata(tmp_path):
    repository = RunRepository(tmp_path)
    engine = RunEngine(repository)
    handler = BlockingHandler()
    result = []
    thread = threading.Thread(
        target=lambda: result.append(
            engine.execute(RunSpec("run-1", handler.name, {}), handler)
        )
    )
    thread.start()
    assert handler.started.wait(timeout=5)

    first = repository.request_stop("run-1", reason="first")
    second = repository.request_stop("run-1", reason="later")

    assert first.stop_request is not None
    assert second.stop_request == first.stop_request
    assert second.stop_request.reason == "first"
    handler.release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result[0].awaiting is not None
    assert result[0].awaiting.details["reason"] == "first"
    assert "requested_at" in result[0].awaiting.details

    events = EventWriter(
        repository.run_directory("run-1") / "events.jsonl", run_id="run-1"
    ).tail()
    requests = [event for event in events if event["event"] == "attempt_stop_requested"]
    assert len(requests) == 1
    assert requests[0]["data"]["reason"] == "first"


def test_terminal_winning_stop_race_hides_stale_request_and_event(
    tmp_path, monkeypatch
):
    repository = RunRepository(tmp_path)
    pending = repository.create(RunSpec("run-1", "example.v1", {}))
    running = replace(
        pending,
        revision=pending.revision + 1,
        status=RunStatus.RUNNING,
        attempt=1,
    )
    store = repository._snapshot_store("run-1")
    store.compare_and_swap(pending.revision, running)
    original_request = engine_module.StopToken.request

    def request_then_finish(token, *, reason=None):
        request = original_request(token, reason=reason)
        current = store.read()
        succeeded = replace(
            current,
            revision=current.revision + 1,
            status=RunStatus.SUCCEEDED,
        )
        store.compare_and_swap(current.revision, succeeded)
        return request

    monkeypatch.setattr(engine_module.StopToken, "request", request_then_finish)

    view = repository.request_stop("run-1", reason="too late")

    assert view.snapshot.status is RunStatus.SUCCEEDED
    assert view.stop_request is None
    assert repository.inspect("run-1").stop_request is None
    assert (
        repository.run_directory("run-1") / "stop-requests" / "1.json"
    ).exists()
    events = EventWriter(
        repository.run_directory("run-1") / "events.jsonl", run_id="run-1"
    ).tail()
    assert all(event["event"] != "attempt_stop_requested" for event in events)


def test_stop_is_idempotent_for_an_already_paused_run(tmp_path):
    repository = RunRepository(tmp_path)
    handler = ReplayGroupHandler()
    paused = RunEngine(repository).execute(
        RunSpec("run-1", handler.name, {}), handler
    )

    view = repository.request_stop("run-1", reason="too late")

    assert paused.status is RunStatus.PAUSED
    assert view.snapshot == paused
    assert view.stop_request is None


def test_resume_ignores_stop_request_for_previous_attempt(tmp_path):
    repository = RunRepository(tmp_path)
    handler = ReplayGroupHandler()
    engine = RunEngine(repository)
    paused = engine.execute(RunSpec("run-1", handler.name, {}), handler)
    assert paused.status is RunStatus.PAUSED
    from arc_jobs import StopToken

    StopToken(
        repository.run_directory("run-1") / "stop-requests" / "1.json",
        target_attempt=1,
    ).request(
        reason="stale stop"
    )
    resumed = engine.resume("run-1", handler)
    assert resumed.status is RunStatus.SUCCEEDED
    assert resumed.attempt == 2
    assert handler.worker_calls == 2


class ReplayGroupHandler:
    name = "group-replay.v1"

    def __init__(self):
        self.worker_calls = 0

    def execute(self, context):
        def worker(unit):
            self.worker_calls += 1
            return {"id": unit.unit_id}

        group = context.run_group(
            "group",
            (WorkUnit("a", {"x": 1}), WorkUnit("b", {"x": 2})),
            worker,
            max_workers=2,
            failure_mode=FailureMode.COLLECT,
        )
        if context.attempt == 1:
            return Paused(
                Awaiting(
                    ResumeReason.EXTERNAL_CONDITION,
                    "retry-later",
                    False,
                    details={"code": "provider_unavailable"},
                )
            )
        return Succeeded(
            context.artifacts.publish_json(
                "result", {"units": len(group.units)}
            )
        )


class RequestArtifactPauseHandler:
    name = "request-artifact-pause.v1"

    def __init__(self, *, remove_before_return=False):
        self.remove_before_return = remove_before_return

    def execute(self, context):
        request_ref = context.artifacts.publish_json(
            "pause/request", {"question": "continue?"}
        )
        if self.remove_before_return:
            (context.run_directory / request_ref.relative_path).unlink()
        return Paused(
            Awaiting(
                ResumeReason.INTERACTION_REQUIRED,
                "resume",
                True,
                request_ref,
                "example.response.v1",
            )
        )


def test_pause_request_artifact_is_verified_before_persistence(tmp_path):
    repository = RunRepository(tmp_path)
    handler = RequestArtifactPauseHandler(remove_before_return=True)

    snapshot = RunEngine(repository).execute(
        RunSpec("run-1", handler.name, {}), handler
    )

    assert snapshot.status is RunStatus.FAILED
    assert snapshot.awaiting is None
    assert snapshot.error is not None
    assert snapshot.error.code == "handler_unhandled_exception"
    assert repository.validate("run-1").ok


@pytest.mark.parametrize("mutation", ("missing", "corrupt"))
def test_validate_verifies_persisted_pause_request_artifact(tmp_path, mutation):
    repository = RunRepository(tmp_path)
    handler = RequestArtifactPauseHandler()
    snapshot = RunEngine(repository).execute(
        RunSpec("run-1", handler.name, {}), handler
    )
    assert snapshot.status is RunStatus.PAUSED
    assert snapshot.awaiting is not None
    assert snapshot.awaiting.request_ref is not None
    assert repository.validate("run-1").ok

    request_path = (
        repository.run_directory("run-1")
        / snapshot.awaiting.request_ref.relative_path
    )
    if mutation == "missing":
        request_path.unlink()
    else:
        request_path.write_bytes(b"corrupt")

    report = repository.validate("run-1")
    assert not report.ok
    assert report.issues


def test_validate_accepts_no_input_pause_without_request_artifact(tmp_path):
    repository = RunRepository(tmp_path)
    handler = ReplayGroupHandler()

    snapshot = RunEngine(repository).execute(
        RunSpec("run-1", handler.name, {}), handler
    )

    assert snapshot.status is RunStatus.PAUSED
    assert snapshot.awaiting is not None
    assert snapshot.awaiting.request_ref is None
    assert repository.validate("run-1").ok


def test_completed_group_units_replay_across_run_resume(tmp_path):
    handler = ReplayGroupHandler()
    engine = RunEngine(RunRepository(tmp_path))
    first = engine.execute(RunSpec("run-1", handler.name, {}), handler)
    assert first.status is RunStatus.PAUSED
    second = engine.resume("run-1", handler)
    assert second.status is RunStatus.SUCCEEDED
    assert handler.worker_calls == 2


def test_group_documents_use_v2_and_reject_legacy_v1(tmp_path):
    repository = RunRepository(tmp_path)
    handler = ReplayGroupHandler()
    snapshot = RunEngine(repository).execute(
        RunSpec("run-1", handler.name, {}), handler
    )
    assert snapshot.status is RunStatus.PAUSED
    group_directory = repository.run_directory("run-1") / "groups" / "group"
    state_path = group_directory / "state.json"
    unit_path = group_directory / "units" / "a.json"
    state = json.loads(state_path.read_text())
    unit = json.loads(unit_path.read_text())
    assert state["schema_version"] == "arc.jobs.group.v2"
    assert unit["schema_version"] == "arc.jobs.group_unit.v2"

    state["schema_version"] = "arc.jobs.group.v1"
    state_path.write_text(json.dumps(state))
    with pytest.raises(UnsupportedSchemaError):
        repository.inspect_group("run-1", "group")

    state["schema_version"] = "arc.jobs.group.v2"
    state_path.write_text(json.dumps(state))
    unit["schema_version"] = "arc.jobs.group_unit.v1"
    unit_path.write_text(json.dumps(unit))
    with pytest.raises(UnsupportedSchemaError):
        repository.inspect_group("run-1", "group")


def test_group_replay_reads_each_completed_unit_document_once(tmp_path, monkeypatch):
    handler = ReplayGroupHandler()
    engine = RunEngine(RunRepository(tmp_path))
    first = engine.execute(RunSpec("run-1", handler.name, {}), handler)
    assert first.status is RunStatus.PAUSED

    reads = []
    original = group_storage.read_json_object

    def record_unit_read(path):
        if path.parent.name == "units":
            reads.append(path.name)
        return original(path)

    monkeypatch.setattr(group_storage, "read_json_object", record_unit_read)
    second = engine.resume("run-1", handler)

    assert second.status is RunStatus.SUCCEEDED
    assert reads == ["a.json", "b.json"]


class FailFastGroupHandler:
    name = "fail-fast.v1"

    def __init__(self):
        self.called = []

    def execute(self, context):
        def worker(unit):
            self.called.append(unit.unit_id)
            if unit.unit_id == "a":
                return UnitResult(
                    "a",
                    "failed",
                    {"partial": True},
                    RunError("expected", "failed"),
                )
            return {"ok": True}

        result = context.run_group(
            "group",
            tuple(WorkUnit(name, {"name": name}) for name in ("a", "b", "c")),
            worker,
            max_workers=1,
            failure_mode=FailureMode.FAIL_FAST,
        )
        return Succeeded(
            context.artifacts.publish_json(
                "result", {"started": [unit.unit_id for unit in result.units]}
            )
        )


def test_fail_fast_stops_new_units_and_joins_started_work(tmp_path):
    handler = FailFastGroupHandler()
    repository = RunRepository(tmp_path)
    snapshot = RunEngine(repository).execute(
        RunSpec("run-1", handler.name, {}), handler
    )
    assert snapshot.status is RunStatus.SUCCEEDED
    assert handler.called == ["a"]
    before = {
        path: path.stat().st_mtime_ns
        for path in repository.run_directory("run-1").glob("groups/**/*")
        if path.is_file()
    }

    view = repository.inspect_group("run-1", "group")

    assert [unit.status for unit in view.units] == ["failed", "pending", "pending"]
    assert view.units[0].value == {"partial": True}
    assert view.units[0].error == RunError("expected", "failed")
    assert view.units[1].value is None
    assert repository.inspect_group("run-1", "group") == view
    assert {
        path: path.stat().st_mtime_ns
        for path in repository.run_directory("run-1").glob("groups/**/*")
        if path.is_file()
    } == before


def test_failed_group_unit_retry_requires_a_new_run(tmp_path):
    repository = RunRepository(tmp_path)
    engine = RunEngine(repository)
    handler = FailFastGroupHandler()
    spec = RunSpec("run-1", handler.name, {})

    first = engine.execute(spec, handler)
    replayed = engine.execute(spec, handler)

    assert replayed == first
    assert handler.called == ["a"]
    first_view = repository.inspect_group("run-1", "group")
    assert first_view.units[0].status == "failed"

    engine.execute(RunSpec("run-2", handler.name, {}), handler)

    assert handler.called == ["a", "a"]
    assert repository.inspect_group("run-1", "group") == first_view


class FailFastReplayHandler:
    name = "fail-fast-replay.v1"

    def __init__(self):
        self.called = []

    def execute(self, context):
        def worker(unit):
            self.called.append(unit.unit_id)
            if unit.unit_id == "a":
                return UnitResult("a", "failed", error=None)
            return {"ok": True}

        result = context.run_group(
            "group",
            (WorkUnit("a", {"x": 1}), WorkUnit("b", {"x": 2})),
            worker,
            max_workers=1,
            failure_mode=FailureMode.FAIL_FAST,
        )
        if context.attempt == 1:
            return Paused(
                Awaiting(
                    ResumeReason.EXTERNAL_CONDITION,
                    "retry",
                    False,
                    details={"code": "provider_unavailable"},
                )
            )
        return Succeeded(
            context.artifacts.publish_json(
                "result", {"units": [item.unit_id for item in result.units]}
            )
        )


def test_fail_fast_resume_does_not_schedule_after_durable_failure(tmp_path):
    handler = FailFastReplayHandler()
    engine = RunEngine(RunRepository(tmp_path))
    assert engine.execute(
        RunSpec("run-1", handler.name, {}), handler
    ).status is RunStatus.PAUSED
    assert engine.resume("run-1", handler).status is RunStatus.SUCCEEDED
    assert handler.called == ["a"]
