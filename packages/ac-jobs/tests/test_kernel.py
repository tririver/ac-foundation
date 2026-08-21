from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import pytest

from ac_jobs import (
    ArtifactConflictError,
    ArtifactSourceRef,
    Awaiting,
    Failed,
    FailureMode,
    GroupResult,
    IdempotencyConflictError,
    ImmutableArtifactStore,
    Paused,
    ResumeInputConflictError,
    ResumeMismatchError,
    ResumeReason,
    RunEngine,
    RunError,
    RunRepository,
    RunSpec,
    RunStatus,
    Succeeded,
    UnitResult,
    WorkUnit,
)


class SuccessHandler:
    name = "example.v1"

    def __init__(self):
        self.calls = 0

    def execute(self, context):
        self.calls += 1
        ref = context.artifacts.publish_json("result/final", {"answer": 42})
        return Succeeded(ref)


def test_execute_replays_and_rejects_changed_semantics(tmp_path):
    repository = RunRepository(tmp_path)
    engine = RunEngine(repository)
    handler = SuccessHandler()
    spec = RunSpec("run-1", handler.name, {"question": "life"})

    first = engine.execute(spec, handler)
    second = engine.execute(spec, handler)

    assert first.status is RunStatus.SUCCEEDED
    assert second == first
    assert handler.calls == 1
    with pytest.raises(IdempotencyConflictError):
        engine.execute(
            RunSpec("run-1", handler.name, {"question": "different"}), handler
        )
    assert handler.calls == 1


class InteractiveHandler:
    name = "interactive.v1"

    def execute(self, context):
        if context.resume_input is None:
            request = context.artifacts.publish_json(
                "interaction/request", {"requests": [{"id": "one"}]}
            )
            return Paused(
                Awaiting(
                    ResumeReason.INTERACTION_REQUIRED,
                    "resume-1",
                    True,
                    request,
                    "example.resume.v1",
                    {"code": "operation_requests_pending"},
                )
            )
        return Succeeded(
            context.artifacts.publish_json(
                "result", {"choice": context.resume_input["choice"]}
            )
        )


def test_pause_resume_input_is_keyed_and_idempotent(tmp_path):
    repository = RunRepository(tmp_path)
    engine = RunEngine(repository)
    handler = InteractiveHandler()
    paused = engine.execute(RunSpec("run-1", handler.name, {"x": 1}), handler)
    assert paused.status is RunStatus.PAUSED

    with pytest.raises(ResumeMismatchError):
        engine.resume("run-1", handler, input={"resume_key": "wrong", "choice": 1})

    completed = engine.resume(
        "run-1", handler, input={"resume_key": "resume-1", "choice": 1}
    )
    replayed = engine.resume(
        "run-1", handler, input={"resume_key": "resume-1", "choice": 1}
    )
    assert completed.status is RunStatus.SUCCEEDED
    assert replayed == completed
    with pytest.raises(ResumeInputConflictError):
        engine.resume(
            "run-1", handler, input={"resume_key": "resume-1", "choice": 2}
        )


def test_stop_pending_run_is_immediately_paused(tmp_path):
    repository = RunRepository(tmp_path)
    repository.create(RunSpec("run-1", InteractiveHandler.name, {"x": 1}))

    view = repository.request_stop("run-1", reason="no longer needed")

    assert view.snapshot.status is RunStatus.PAUSED
    assert view.snapshot.attempt == 0
    assert view.stop_request is not None
    assert view.stop_request.reason == "no longer needed"
    assert view.snapshot.awaiting.reason is ResumeReason.EXECUTION_STOPPED
    assert repository.inspect("run-1") == view


def test_hierarchical_artifacts_and_adoption(tmp_path):
    repository = RunRepository(tmp_path)
    source_dir = repository.run_directory("source")
    target_dir = repository.run_directory("target")
    source = ImmutableArtifactStore(source_dir, repository_root=repository.root)
    target = ImmutableArtifactStore(target_dir, repository_root=repository.root)
    source_ref = source.publish_bytes(
        "loops/a/rounds/001/result", b"value", media_type="text/plain"
    )
    source_locator = ArtifactSourceRef(
        "source", source_ref.artifact_id, source_ref.digest
    )

    adopted = target.scoped("reuse").adopt(
        source_locator,
        artifact_id="result",
        expected_verified_digest=source_ref.digest,
    )

    assert target.read_bytes(adopted) == b"value"
    assert adopted.artifact_id == "reuse/result"
    assert target.find("reuse/result") == adopted
    assert target.find("reuse/missing") is None
    with pytest.raises(ArtifactConflictError):
        target.publish_bytes(
            adopted.artifact_id, b"different", media_type="text/plain"
        )


class GroupHandler:
    name = "group.v1"

    def __init__(self):
        self.calls = 0
        self.lock = Lock()

    def execute(self, context):
        def worker(unit):
            with self.lock:
                self.calls += 1
            if unit.unit_id == "b":
                return UnitResult(
                    "b",
                    "failed",
                    {"partial": True},
                    RunError("expected", "failed"),
                )
            return {"unit": unit.unit_id}

        result = context.run_group(
            "batch",
            (WorkUnit("a", {"x": 1}), WorkUnit("b", {"x": 2})),
            worker,
            max_workers=1,
            failure_mode=FailureMode.COLLECT,
        )
        assert isinstance(result, GroupResult)
        return Succeeded(context.artifacts.publish_json("group-result", {
            "statuses": [unit.status for unit in result.units]
        }))


def test_group_preserves_typed_failures_and_replays(tmp_path):
    handler = GroupHandler()
    snapshot = RunEngine(RunRepository(tmp_path)).execute(
        RunSpec("run-1", handler.name, {}), handler
    )
    assert snapshot.status is RunStatus.SUCCEEDED
    assert handler.calls == 2
