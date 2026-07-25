from __future__ import annotations

from pathlib import Path

import pytest

from arc_jobs import JsonValue, RunRepository, RunSpec, RunStatus
from arc_proposer_reviewer.handler import ProposerReviewerHandler
from arc_proposer_reviewer.identity import derive_batch_run_id
from arc_proposer_reviewer.models import (
    BATCH_SCHEMA_VERSION,
    BatchRequest,
    ExecutionOptions,
    LoopSpec,
    WorkerSpec,
)
from arc_proposer_reviewer.protocol import encode_batch_request
from arc_proposer_reviewer.runner import BatchRunner
from arc_proposer_reviewer.validation import RequestValidationError


SCHEMA = {"type": "object", "additionalProperties": True}


def _request() -> BatchRequest:
    return BatchRequest(
        BATCH_SCHEMA_VERSION,
        "batch-runner",
        (
            LoopSpec(
                "loop-a",
                {"question": "Q"},
                (WorkerSpec("proposer-a", "Propose.", SCHEMA),),
                WorkerSpec("reviewer-a", "Review.", SCHEMA),
                1,
            ),
        ),
    )


def test_prepare_derives_run_id_and_read_request_validates_closed_spec(
    tmp_path: Path,
) -> None:
    assert BATCH_SCHEMA_VERSION == "arc.proposer_reviewer.batch.v3"
    assert ProposerReviewerHandler.name == "arc.proposer_reviewer.batch.v3"
    runner = BatchRunner()
    request = _request()

    snapshot = runner.prepare(request, tmp_path)

    assert snapshot.run_id == derive_batch_run_id(request.batch_id)
    assert snapshot.status is RunStatus.PENDING
    spec = RunRepository(tmp_path).read_spec(snapshot.run_id)
    assert spec.handler == ProposerReviewerHandler.name
    assert spec.semantic_input == encode_batch_request(request)
    assert runner.read_request(tmp_path, snapshot.run_id) == request

    repository = RunRepository(tmp_path)
    repository.create(
        RunSpec(
            "wrong-handler",
            "another.handler",
            encode_batch_request(request),
        )
    )
    with pytest.raises(ValueError, match="not a proposer-reviewer"):
        runner.read_request(tmp_path, "wrong-handler")

    corrupt_request: dict[str, JsonValue] = encode_batch_request(request)
    corrupt_request["unexpected"] = True
    repository.create(
        RunSpec(
            "open-request",
            ProposerReviewerHandler.name,
            corrupt_request,
        )
    )
    with pytest.raises(RequestValidationError, match="unknown field"):
        runner.read_request(tmp_path, "open-request")


def test_run_and_resume_keep_operational_options_out_of_semantic_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arc_proposer_reviewer import runner as runner_module

    task_service = object()
    runner = BatchRunner(task_service)  # type: ignore[arg-type]
    request = _request()
    run_options = ExecutionOptions(
        max_concurrent_loops=2,
        max_concurrent_workers=3,
    )
    resume_options = ExecutionOptions(
        max_concurrent_loops=4,
        max_concurrent_workers=5,
    )
    calls: list[tuple[str, object, object]] = []

    def execute(engine, spec, handler):
        calls.append(("run", spec, handler))
        return engine.repository.inspect(spec.run_id).snapshot

    def resume(engine, run_id, handler, *, input=None):
        calls.append(("resume", input, handler))
        return engine.repository.inspect(run_id).snapshot

    monkeypatch.setattr(runner_module.RunEngine, "execute", execute)
    monkeypatch.setattr(runner_module.RunEngine, "resume", resume)

    first = runner.run(request, tmp_path, "run-a", run_options)
    resume_input: dict[str, JsonValue] = {
        "resume_key": "resume-a",
        "action": "continue",
    }
    second = runner.resume(
        tmp_path,
        "run-a",
        resume_input,
        resume_options,
    )

    assert first.run_id == second.run_id == "run-a"
    spec = RunRepository(tmp_path).read_spec("run-a")
    assert spec.semantic_input == encode_batch_request(request)
    assert "max_concurrent_loops" not in spec.semantic_input
    assert "max_concurrent_workers" not in spec.semantic_input
    run_handler = calls[0][2]
    resume_handler = calls[1][2]
    assert run_handler.options is run_options  # type: ignore[attr-defined]
    assert resume_handler.options is resume_options  # type: ignore[attr-defined]
    assert run_handler.service.llm is task_service  # type: ignore[attr-defined]
    assert resume_handler.service.llm is task_service  # type: ignore[attr-defined]
    assert calls[1][1] == resume_input


def test_inspect_stop_and_projection_are_read_only_of_llm_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arc_proposer_reviewer import runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "LLMTaskService",
        lambda: (_ for _ in ()).throw(AssertionError("queries must not create an LLM")),
    )
    runner = BatchRunner()
    snapshot = runner.prepare(_request(), tmp_path, "run-a")

    assert runner.inspect(tmp_path, "run-a").snapshot == snapshot
    projection = runner.projection(tmp_path, "run-a")
    assert projection.inspect().durable_lifecycle == "pending"
    stopped = runner.stop(tmp_path, "run-a", "operator request")

    assert stopped.snapshot.status is RunStatus.PAUSED
    assert stopped.snapshot.awaiting is not None
    assert stopped.snapshot.awaiting.details["reason"] == "operator request"
