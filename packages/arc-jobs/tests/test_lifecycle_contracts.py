from __future__ import annotations

import threading
import json

import pytest

from arc_jobs import (
    Awaiting,
    EffectRequestDigest,
    EventWriter,
    FailureMode,
    Paused,
    ResumeReason,
    RunBusyError,
    RunEngine,
    RunRepository,
    RunSpec,
    RunStatus,
    RunContext,
    Succeeded,
    UnitResult,
    WorkUnit,
    validate_artifact_id,
    validate_simple_id,
)
from arc_jobs.lease import FileLease


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
    writer.emit("step", {"summary": "safe"})
    with pytest.raises(ValueError):
        writer.emit("step", {"summary": "x" * (256 * 1024)})
    with (tmp_path / "events.jsonl").open("ab") as handle:
        handle.write(b'{"incomplete":')
    assert len(writer.tail()) == 1
    writer.validate()  # A trailing crash fragment is ignored.
    writer.emit("next_step", {"summary": "recovered"})
    writer.validate()
    assert [item["sequence"] for item in writer.tail()] == [1, 2]
    with pytest.raises(ValueError):
        writer.emit("unsafe", {"nested": {"content": "body"}})


def test_event_validation_rejects_tampered_identity(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path, run_id="run-1")
    writer.emit("step", {"summary": "safe"})
    document = json.loads(path.read_text())
    document["event_id"] = "0" * 64
    path.write_text(json.dumps(document) + "\n")
    with pytest.raises(Exception):
        writer.validate()


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


def test_cancel_running_handler_is_visible_and_cooperative(tmp_path):
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

    view = repository.request_cancel("run-1", reason="stop")
    assert view.snapshot.status is RunStatus.RUNNING
    assert view.cancel_request is not None

    handler.release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result[0].status is RunStatus.CANCELLED


def test_resume_observes_preexisting_cancel_before_handler(tmp_path):
    repository = RunRepository(tmp_path)
    handler = ReplayGroupHandler()
    engine = RunEngine(repository)
    paused = engine.execute(RunSpec("run-1", handler.name, {}), handler)
    assert paused.status is RunStatus.PAUSED
    from arc_jobs import CancellationToken

    CancellationToken(repository.run_directory("run-1") / "cancel.json").request(
        reason="stop before resume"
    )
    cancelled = engine.resume("run-1", handler)
    assert cancelled.status is RunStatus.CANCELLED
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


def test_completed_group_units_replay_across_run_resume(tmp_path):
    handler = ReplayGroupHandler()
    engine = RunEngine(RunRepository(tmp_path))
    first = engine.execute(RunSpec("run-1", handler.name, {}), handler)
    assert first.status is RunStatus.PAUSED
    second = engine.resume("run-1", handler)
    assert second.status is RunStatus.SUCCEEDED
    assert handler.worker_calls == 2


class FailFastGroupHandler:
    name = "fail-fast.v1"

    def __init__(self):
        self.called = []

    def execute(self, context):
        def worker(unit):
            self.called.append(unit.unit_id)
            if unit.unit_id == "a":
                return UnitResult(
                    "a", "failed", {"partial": True}
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
    snapshot = RunEngine(RunRepository(tmp_path)).execute(
        RunSpec("run-1", handler.name, {}), handler
    )
    assert snapshot.status is RunStatus.SUCCEEDED
    assert handler.called == ["a"]


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


class UnsafeEffectHandler:
    name = "unsafe-effect.v1"

    def execute(self, context):
        context.effects.prepare(
            "call", effect_request_digest=EffectRequestDigest("a" * 64)
        )
        context.effects.mark_may_have_run("call")
        context.effects.recover("call")
        return Succeeded()


def test_uncertain_effect_pauses_for_supervision_without_retry(tmp_path):
    snapshot = RunEngine(RunRepository(tmp_path)).execute(
        RunSpec("run-1", "unsafe-effect.v1", {}), UnsafeEffectHandler()
    )
    assert snapshot.status is RunStatus.PAUSED
    assert snapshot.awaiting.reason is ResumeReason.SUPERVISION_REQUIRED
    assert snapshot.awaiting.details["code"] == "unsafe_effect_recovery"


def test_effect_recovery_revalidates_saved_artifact(tmp_path):
    repository = RunRepository(tmp_path)
    snapshot = repository.create(RunSpec("run-1", "example.v1", {}))
    context = RunContext(
        repository,
        snapshot,
        resume_input=None,
        execution_slice=None,
    )
    context.effects.prepare(
        "call", effect_request_digest=EffectRequestDigest("a" * 64)
    )
    context.effects.mark_may_have_run("call")
    output = context.artifacts.publish_bytes(
        "raw", b"safe", media_type="application/octet-stream"
    )
    context.effects.save_output("call", output)
    object_path = repository.run_directory("run-1") / output.relative_path
    object_path.write_bytes(b"tampered")
    with pytest.raises(Exception):
        context.effects.recover("call")
