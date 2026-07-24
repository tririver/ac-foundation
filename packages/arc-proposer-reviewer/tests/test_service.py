from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from arc_jobs import (
    ImmutableArtifactStore,
    RunContext,
    RunEngine,
    RunRepository,
    RunSpec,
    RunStatus,
)
from arc_llm import (
    InvalidRequestError,
    LLMCompleted,
    LLMFailed,
    LLMPaused,
    ResumeReason,
    SessionRef,
)
from arc_proposer_reviewer import (
    BatchFailurePolicy,
    BatchRequest,
    ExecutionOptions,
    LoopSpec,
    ProposerFailurePolicy,
    ProposerReviewerHandler,
    ProposerReviewerService,
    WorkerSpec,
)
from arc_proposer_reviewer.models import BATCH_SCHEMA_VERSION
from arc_proposer_reviewer.protocol import decode_batch_result, encode_batch_request


PROPOSAL_SCHEMA = {
    "type": "object",
    "required": ["proposal"],
    "properties": {"proposal": {"type": "string"}},
    "additionalProperties": False,
}
REVIEW_PAYLOAD_SCHEMA = {
    "type": "object",
    "required": ["score"],
    "properties": {"score": {"type": "integer"}},
    "additionalProperties": False,
}


class FakeLLM:
    def __init__(self, *, pause_once: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.pause_once = pause_once
        self.paused_task_id: str | None = None

    def execute(self, context, request, *, options):
        self.calls.append(("execute", request.task_id))
        if self.pause_once and self.paused_task_id is None and _is_proposer(request):
            self.paused_task_id = request.task_id
            return LLMPaused(
                ResumeReason.EXTERNAL_CONDITION,
                "provider-wait",
                {"code": "provider_unavailable"},
            )
        if "FAIL" in request.prompt:
            return LLMFailed(InvalidRequestError("deliberate fake failure"))
        return _completed(request)

    def resume(self, context, task_id, *, input, options):
        self.calls.append(("resume", task_id))
        assert task_id == self.paused_task_id
        return LLMCompleted(
            {"proposal": f"resumed:{task_id}"},
            "fake",
            "fake-model",
            None,
            None,
        )


def _is_proposer(request) -> bool:
    return "one proposer" in request.prompt


def _completed(request):
    if _is_proposer(request):
        return LLMCompleted(
            {"proposal": request.task_id},
            "fake",
            "fake-model",
            None,
            None,
        )
    feedback_schema = request.output.schema["properties"]["feedback"]
    active_ids = feedback_schema["required"]
    return LLMCompleted(
        {
            "schema_version": "arc.proposer_reviewer.review.v1",
            "action": "stop",
            "reason": "The current proposals are sufficient.",
            "feedback": {worker_id: "Keep the result." for worker_id in active_ids},
            "payload": {"score": 1},
        },
        "fake",
        "fake-model",
        None,
        None,
    )


def _worker(worker_id: str, instructions: str = "Produce a proposal.") -> WorkerSpec:
    return WorkerSpec(worker_id, instructions, PROPOSAL_SCHEMA)


def _loop(
    loop_id: str = "loop-a",
    *,
    proposers: tuple[WorkerSpec, ...] | None = None,
    max_rounds: int = 1,
    allow_early_stop: bool = True,
    policy: ProposerFailurePolicy = ProposerFailurePolicy.FAIL_LOOP,
) -> LoopSpec:
    return LoopSpec(
        loop_id=loop_id,
        context={"question": loop_id},
        proposers=proposers or (_worker(f"{loop_id}-p"),),
        reviewer=WorkerSpec(f"{loop_id}-r", "Review all proposals.", REVIEW_PAYLOAD_SCHEMA),
        max_rounds=max_rounds,
        allow_early_stop=allow_early_stop,
        on_proposer_failure=policy,
    )


def _request(
    *loops: LoopSpec,
    failure_policy: BatchFailurePolicy = BatchFailurePolicy.COLLECT,
) -> BatchRequest:
    return BatchRequest(BATCH_SCHEMA_VERSION, "batch-a", loops, failure_policy)


def _run(
    root: Path,
    request: BatchRequest,
    fake: FakeLLM,
    *,
    options: ExecutionOptions = ExecutionOptions(),
):
    repository = RunRepository(root)
    handler = ProposerReviewerHandler(
        ProposerReviewerService(fake),  # type: ignore[arg-type]
        options=options,
    )
    snapshot = RunEngine(repository).execute(
        RunSpec("run-a", handler.name, encode_batch_request(request)),
        handler,
    )
    return repository, handler, snapshot


def _result(repository: RunRepository, snapshot):
    assert snapshot.result_ref is not None
    raw = repository.inspect(snapshot.run_id)
    assert raw.snapshot.result_ref == snapshot.result_ref
    content = ImmutableArtifactStore(
        repository.run_directory(snapshot.run_id),
        repository_root=repository.root,
    ).read_bytes(snapshot.result_ref)
    return decode_batch_result(json.loads(content))


def test_one_proposer_one_reviewer_one_round_publishes_typed_result(
    tmp_path: Path,
) -> None:
    fake = FakeLLM()
    repository, _handler, snapshot = _run(tmp_path, _request(_loop()), fake)
    assert snapshot.status is RunStatus.SUCCEEDED
    result = _result(repository, snapshot)
    loop = result.loops[0]
    assert loop.termination.value == "reviewer_stop"
    assert loop.rounds_completed == 1
    assert tuple(loop.final_proposals) == ("loop-a-p",)
    assert isinstance(loop.final_review, Mapping)
    assert [kind for kind, _ in fake.calls] == ["execute", "execute"]

    replayed = RunEngine(repository).execute(
        RunSpec("run-a", _handler.name, encode_batch_request(_request(_loop()))),
        _handler,
    )
    assert replayed == snapshot
    assert len(fake.calls) == 2


def test_stop_is_recorded_but_ignored_when_early_stop_is_disabled(
    tmp_path: Path,
) -> None:
    fake = FakeLLM()
    repository, _handler, snapshot = _run(
        tmp_path,
        _request(_loop(max_rounds=2, allow_early_stop=False)),
        fake,
    )
    result = _result(repository, snapshot)
    assert result.loops[0].termination.value == "round_limit"
    assert result.loops[0].rounds_completed == 2
    assert len(fake.calls) == 4


def test_continue_if_any_excludes_failed_proposer_without_placeholder(
    tmp_path: Path,
) -> None:
    fake = FakeLLM()
    loop = _loop(
        proposers=(_worker("good"), _worker("bad", "FAIL")),
        policy=ProposerFailurePolicy.CONTINUE_IF_ANY,
    )
    repository, _handler, snapshot = _run(tmp_path, _request(loop), fake)
    result = _result(repository, snapshot).loops[0]
    assert result.error is None
    assert tuple(result.final_proposals) == ("good",)
    assert "bad" not in result.final_proposals
    assert set(result.final_review["feedback"]) == {"good"}  # type: ignore[index]


def test_all_proposers_failed_produces_failed_loop_not_fabricated_output(
    tmp_path: Path,
) -> None:
    fake = FakeLLM()
    loop = _loop(
        proposers=(_worker("bad-a", "FAIL"), _worker("bad-b", "FAIL")),
        policy=ProposerFailurePolicy.CONTINUE_IF_ANY,
    )
    repository, _handler, snapshot = _run(tmp_path, _request(loop), fake)
    result = _result(repository, snapshot).loops[0]
    assert result.termination.value == "failed"
    assert result.final_proposals == {}
    assert result.final_review is None
    assert result.error is not None
    assert result.error.code == "all_proposers_failed"
    assert len(fake.calls) == 2


def test_batch_fail_fast_preserves_request_order_and_marks_unstarted_loop(
    tmp_path: Path,
) -> None:
    fake = FakeLLM()
    first = _loop("first", proposers=(_worker("bad", "FAIL"),))
    second = _loop("second")
    repository, _handler, snapshot = _run(
        tmp_path,
        _request(first, second, failure_policy=BatchFailurePolicy.FAIL_FAST),
        fake,
        options=ExecutionOptions(max_concurrent_loops=1),
    )
    result = _result(repository, snapshot)
    assert [loop.loop_id for loop in result.loops] == ["first", "second"]
    assert result.loops[0].error.code == "proposer_failed"  # type: ignore[union-attr]
    assert result.loops[1].error.code == "fail_fast_skipped"  # type: ignore[union-attr]
    assert len(fake.calls) == 1


def test_collect_runs_multiple_loops_and_keeps_request_order(tmp_path: Path) -> None:
    fake = FakeLLM()
    repository, _handler, snapshot = _run(
        tmp_path,
        _request(_loop("z-loop"), _loop("a-loop")),
        fake,
        options=ExecutionOptions(max_concurrent_loops=2),
    )
    result = _result(repository, snapshot)
    assert [loop.loop_id for loop in result.loops] == ["z-loop", "a-loop"]
    assert all(loop.error is None for loop in result.loops)
    assert len(fake.calls) == 4


def test_invalid_reviewer_value_fails_without_fabricating_review(
    tmp_path: Path,
) -> None:
    class InvalidReviewer(FakeLLM):
        def execute(self, context, request, *, options):
            self.calls.append(("execute", request.task_id))
            if _is_proposer(request):
                return _completed(request)
            return LLMCompleted(
                {"action": "stop"},
                "fake",
                "fake-model",
                None,
                None,
            )

    fake = InvalidReviewer()
    repository, _handler, snapshot = _run(tmp_path, _request(_loop()), fake)
    result = _result(repository, snapshot).loops[0]
    assert result.termination.value == "failed"
    assert result.final_review is None
    assert result.error is not None


def test_sessions_continue_only_within_the_same_worker_lineage(
    tmp_path: Path,
) -> None:
    class SessionFake(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.proposer_sessions = []
            self.reviewer_sessions = []

        def execute(self, context, request, *, options):
            self.calls.append(("execute", request.task_id))
            if _is_proposer(request):
                self.proposer_sessions.append(request.session)
                return LLMCompleted(
                    {"proposal": request.task_id},
                    "fake",
                    "fake-model",
                    SessionRef("proposer-session", "a" * 64),
                    None,
                )
            self.reviewer_sessions.append(request.session)
            completed = _completed(request)
            return LLMCompleted(
                completed.value,
                completed.provider,
                completed.model,
                SessionRef("reviewer-session", "b" * 64),
                completed.usage,
            )

    fake = SessionFake()
    _repository, _handler, snapshot = _run(
        tmp_path,
        _request(_loop(max_rounds=2, allow_early_stop=False)),
        fake,
    )
    assert snapshot.status is RunStatus.SUCCEEDED
    assert fake.proposer_sessions[0] is None
    assert fake.proposer_sessions[1] == SessionRef("proposer-session", "a" * 64)
    assert fake.reviewer_sessions[0] is None
    assert fake.reviewer_sessions[1] == SessionRef("reviewer-session", "b" * 64)


def test_maximum_length_ids_use_bounded_runtime_locators(tmp_path: Path) -> None:
    loop_id = "l" * 128
    proposer_id = "p" * 128
    reviewer_id = "r" * 128
    loop = LoopSpec(
        loop_id,
        {"question": "Q"},
        (_worker(proposer_id),),
        WorkerSpec(reviewer_id, "Review.", REVIEW_PAYLOAD_SCHEMA),
        1,
    )
    repository, _handler, snapshot = _run(
        tmp_path,
        _request(loop),
        FakeLLM(),
    )
    assert snapshot.status is RunStatus.SUCCEEDED
    assert _result(repository, snapshot).loops[0].loop_id == loop_id


def test_pause_resume_returns_to_same_worker_task_identity(tmp_path: Path) -> None:
    fake = FakeLLM(pause_once=True)
    repository, handler, paused = _run(tmp_path, _request(_loop()), fake)
    assert paused.status is RunStatus.PAUSED
    assert paused.awaiting is not None
    assert paused.awaiting.details["worker_id"] == "loop-a-p"

    resumed = RunEngine(repository).resume("run-a", handler)
    assert resumed.status is RunStatus.SUCCEEDED
    execute_task = fake.calls[0][1]
    resume_task = next(task_id for kind, task_id in fake.calls if kind == "resume")
    assert resume_task == execute_task
    assert _result(repository, resumed).loops[0].rounds_completed == 1


def test_committed_round_replays_after_outer_unit_interruption(
    tmp_path: Path,
) -> None:
    fake = FakeLLM()
    repository = RunRepository(tmp_path)
    snapshot = repository.create(
        RunSpec("parent", "test.parent", {"case": "round-replay"})
    )
    context = RunContext(
        repository,
        snapshot,
        resume_input=None,
        execution_slice=None,
    )
    service = ProposerReviewerService(fake)  # type: ignore[arg-type]
    loop = _loop()
    artifacts = context.artifacts.scoped("proposer-reviewer")

    first = service._execute_loop(  # type: ignore[attr-defined]
        context,
        artifacts,
        loop,
        options=ExecutionOptions(),
    )
    assert len(fake.calls) == 2
    second = service._execute_loop(  # type: ignore[attr-defined]
        context,
        artifacts,
        loop,
        options=ExecutionOptions(),
    )
    assert second == first
    assert len(fake.calls) == 2
