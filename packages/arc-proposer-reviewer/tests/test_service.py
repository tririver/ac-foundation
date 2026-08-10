from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from arc_jobs import (
    AtomicStateStore,
    ArtifactDigest,
    ArtifactSourceRef,
    EventWriter,
    ImmutableArtifactStore,
    RunContext,
    RunEngine,
    RunRepository,
    RunSpec,
    RunStatus,
    Paused,
    Succeeded,
)
from arc_llm import (
    InvalidRequestError,
    JsonOutput,
    LLMCompleted,
    LLMFailed,
    LLMInputArtifact,
    LLMPaused,
    ModelSelection,
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
    RevisionContextMode,
    WorkerSpec,
)
from arc_proposer_reviewer.models import BATCH_SCHEMA_VERSION
from arc_proposer_reviewer.projection import read_batch_round, read_batch_trace
from arc_proposer_reviewer.protocol import decode_batch_result, encode_batch_request
from arc_proposer_reviewer.prompts import reviewer_envelope_schema
from arc_proposer_reviewer.state import (
    _LoopStateContract,
    batch_group_id,
    proposer_group_id,
    state_namespace,
)


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
            return LLMFailed(
                InvalidRequestError(
                    "deliberate fake failure",
                    details=(
                        {"fake_marker": "kept"}
                        if "FAIL_DETAILS" in request.prompt
                        else None
                    ),
                )
            )
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
    output_schema = request.output.schema
    feedback_schema = output_schema["properties"]["feedback"]
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
    review_final_round: bool = True,
    revision_context_mode: RevisionContextMode = RevisionContextMode.FEEDBACK_ONLY,
) -> LoopSpec:
    return LoopSpec(
        loop_id=loop_id,
        context={"question": loop_id},
        proposers=proposers or (_worker(f"{loop_id}-p"),),
        reviewer=WorkerSpec(f"{loop_id}-r", "Review all proposals.", REVIEW_PAYLOAD_SCHEMA),
        max_rounds=max_rounds,
        allow_early_stop=allow_early_stop,
        on_proposer_failure=policy,
        review_final_round=review_final_round,
        revision_context_mode=revision_context_mode,
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
    event_sink=None,
):
    repository = RunRepository(root)
    handler = ProposerReviewerHandler(
        ProposerReviewerService(fake),  # type: ignore[arg-type]
        options=options,
    )
    snapshot = RunEngine(repository).execute(
        RunSpec("run-a", handler.name, encode_batch_request(request)),
        handler,
        event_sink=event_sink,
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


def _replace_persisted_request(
    repository: RunRepository,
    request_document: Mapping[str, object],
) -> None:
    artifacts = ImmutableArtifactStore(
        repository.run_directory("run-a"),
        repository_root=repository.root,
    ).scoped("proposer-reviewer")
    logical_id = artifacts._logical_id("request")  # type: ignore[attr-defined]
    artifacts._manifest_path(logical_id).unlink()  # type: ignore[attr-defined]
    artifacts.publish_json("request", request_document)


def context_events(repository: RunRepository, run_id: str):
    return EventWriter(
        repository.run_directory(run_id) / "events.jsonl",
        run_id=run_id,
    ).tail()


def _direct_context(root: Path) -> tuple[RunRepository, RunContext]:
    repository = RunRepository(root)
    snapshot = repository.create(
        RunSpec("parent", "test.parent", {"case": "scope"})
    )
    return repository, RunContext(repository, snapshot, resume_input=None)


def _artifact_ids(repository: RunRepository) -> set[str]:
    manifests = (
        repository.run_directory("parent") / "artifacts" / "manifests"
    )
    return {
        str(json.loads(path.read_text(encoding="utf-8"))["artifact_id"])
        for path in manifests.glob("*.json")
    }


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


def test_default_execution_scope_preserves_existing_runtime_identities(
    tmp_path: Path,
) -> None:
    request = _request(_loop())
    legacy_repository, legacy_context = _direct_context(tmp_path / "legacy")
    explicit_repository, explicit_context = _direct_context(tmp_path / "explicit")
    legacy_fake = FakeLLM()
    explicit_fake = FakeLLM()

    legacy_outcome = ProposerReviewerService(legacy_fake).execute(  # type: ignore[arg-type]
        legacy_context,
        request,
        options=ExecutionOptions(),
    )
    explicit_outcome = ProposerReviewerService(explicit_fake).execute(  # type: ignore[arg-type]
        explicit_context,
        request,
        options=ExecutionOptions(),
        execution_scope=None,
    )

    assert isinstance(legacy_outcome, Succeeded)
    assert isinstance(explicit_outcome, Succeeded)
    assert legacy_fake.calls == explicit_fake.calls
    assert _artifact_ids(legacy_repository) == _artifact_ids(explicit_repository)
    loop = request.loops[0]
    assert state_namespace(loop.loop_id).startswith("pr-loop-")
    assert batch_group_id() == "batch.loops"
    assert proposer_group_id(loop.loop_id, 1).startswith("pr.")
    assert (
        legacy_repository.run_directory("parent")
        / "state"
        / f"{state_namespace(loop.loop_id)}.json"
    ).read_bytes() == (
        explicit_repository.run_directory("parent")
        / "state"
        / f"{state_namespace(loop.loop_id)}.json"
    ).read_bytes()
    legacy_events = [
        item["data"] for item in context_events(legacy_repository, "parent")
    ]
    explicit_events = [
        item["data"] for item in context_events(explicit_repository, "parent")
    ]
    assert legacy_events == explicit_events
    assert all("execution_scope" not in item for item in legacy_events)


def test_nondefault_execution_scope_isolates_artifacts_state_groups_tasks_and_events(
    tmp_path: Path,
) -> None:
    repository, context = _direct_context(tmp_path)
    fake = FakeLLM()
    service = ProposerReviewerService(fake)  # type: ignore[arg-type]
    request = _request(_loop())

    default_outcome = service.execute(context, request, options=ExecutionOptions())
    scoped_outcome = service.execute(
        context,
        request,
        options=ExecutionOptions(),
        execution_scope="editorial",
    )

    assert isinstance(default_outcome, Succeeded)
    assert isinstance(scoped_outcome, Succeeded)
    default_tasks = [task_id for _kind, task_id in fake.calls[:2]]
    scoped_tasks = [task_id for _kind, task_id in fake.calls[2:]]
    assert default_tasks != scoped_tasks
    assert all(task_id.startswith("pr-") for task_id in default_tasks)
    assert all("editorial" not in task_id for task_id in scoped_tasks)
    loop = request.loops[0]
    artifact_ids = _artifact_ids(repository)
    assert "proposer-reviewer/request" in artifact_ids
    assert "proposer-reviewer/scopes/editorial/request" in artifact_ids
    assert (
        repository.run_directory("parent")
        / "state"
        / f"{state_namespace(loop.loop_id)}.json"
    ).is_file()
    assert (
        repository.run_directory("parent")
        / "state"
        / f"{state_namespace(loop.loop_id, execution_scope='editorial')}.json"
    ).is_file()
    groups = repository.run_directory("parent") / "groups"
    assert (groups / batch_group_id()).is_dir()
    assert (groups / batch_group_id(execution_scope="editorial")).is_dir()
    assert (groups / proposer_group_id(loop.loop_id, 1)).is_dir()
    assert (
        groups
        / proposer_group_id(
            loop.loop_id, 1, execution_scope="editorial"
        )
    ).is_dir()
    scoped_events = [
        item["data"]
        for item in context_events(repository, "parent")
        if item["data"].get("execution_scope") == "editorial"
    ]
    assert scoped_events
    assert all(item["execution_scope"] == "editorial" for item in scoped_events)


def test_scoped_pause_resume_does_not_overwrite_default_loop_state(
    tmp_path: Path,
) -> None:
    repository, context = _direct_context(tmp_path)
    fake = FakeLLM(pause_once=True)
    service = ProposerReviewerService(fake)  # type: ignore[arg-type]
    request = _request(_loop())

    paused = service.execute(
        context,
        request,
        options=ExecutionOptions(),
        execution_scope="editorial",
    )
    default_outcome = service.execute(context, request, options=ExecutionOptions())
    resumed = service.execute(
        context,
        request,
        options=ExecutionOptions(),
        execution_scope="editorial",
    )

    assert isinstance(paused, Paused)
    assert isinstance(default_outcome, Succeeded)
    assert isinstance(resumed, Succeeded)
    loop = request.loops[0]
    default_state = context.state(
        state_namespace(loop.loop_id), _LoopStateContract()
    ).read()
    editorial_state = context.state(
        state_namespace(loop.loop_id, execution_scope="editorial"),
        _LoopStateContract(),
    ).read()
    assert default_state is not None and default_state.pauses == {}
    assert editorial_state is not None and editorial_state.pauses == {}
    assert default_state.termination is not None
    assert editorial_state.termination is not None
    assert fake.paused_task_id is not None
    assert fake.paused_task_id != fake.calls[1][1]


def test_event_sink_and_durable_events_include_task_correlation(
    tmp_path: Path,
) -> None:
    observed: list[Mapping[str, object]] = []
    fake = FakeLLM()
    repository, _handler, snapshot = _run(
        tmp_path,
        _request(_loop()),
        fake,
        event_sink=observed.append,
    )

    assert snapshot.status is RunStatus.SUCCEEDED
    callback_events = [
        item["event"]
        for item in observed
        if str(item["event"]).startswith("proposer_reviewer_")
    ]
    assert callback_events == [
        "proposer_reviewer_loop_started",
        "proposer_reviewer_round_started",
        "proposer_reviewer_worker_started",
        "proposer_reviewer_worker_finished",
        "proposer_reviewer_worker_started",
        "proposer_reviewer_worker_finished",
        "proposer_reviewer_round_committed",
        "proposer_reviewer_loop_finished",
    ]
    rendered = json.dumps(observed)
    assert "task_id" in rendered
    assert "session" not in rendered
    assert str(tmp_path) not in rendered
    durable = [
        event["event"]
        for event in context_events(repository, "run-a")
        if str(event["event"]).startswith("proposer_reviewer_")
    ]
    assert durable == callback_events


def test_event_sink_failure_does_not_fail_execution(tmp_path: Path) -> None:
    def fail_sink(_event) -> None:
        raise RuntimeError("terminal closed")

    _repository, _handler, snapshot = _run(
        tmp_path,
        _request(_loop()),
        FakeLLM(),
        event_sink=fail_sink,
    )

    assert snapshot.status is RunStatus.SUCCEEDED


def test_workers_receive_json_output_and_the_same_runtime_options(
    tmp_path: Path,
) -> None:
    class RecordingFake(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.requests = []
            self.options = []

        def execute(self, context, request, *, options):
            self.calls.append(("execute", request.task_id))
            self.requests.append(request)
            self.options.append(options)
            return _completed(request)

    fake = RecordingFake()
    configured = ExecutionOptions()
    workspace_input = LLMInputArtifact(
        "domain-markdown-001",
        ArtifactSourceRef(
            "run-a",
            "proposer-reviewer/inputs/source/0000-domain-markdown-001",
            ArtifactDigest("sha256", "a" * 64, 8),
        ),
        "text/markdown",
    )
    base_request = _request(_loop())
    request = BatchRequest(
        base_request.schema_version,
        base_request.batch_id,
        base_request.loops,
        base_request.failure_policy,
        (workspace_input,),
    )
    _repository, _handler, snapshot = _run(
        tmp_path, request, fake, options=configured
    )

    assert snapshot.status is RunStatus.SUCCEEDED
    assert all(isinstance(request.output, JsonOutput) for request in fake.requests)
    assert all(options is configured.llm for options in fake.options)
    assert all(request.inputs == (workspace_input,) for request in fake.requests)
    assert fake.requests[0].output.schema == PROPOSAL_SCHEMA
    assert fake.requests[1].output.schema == reviewer_envelope_schema(
        payload_schema=REVIEW_PAYLOAD_SCHEMA,
        active_proposer_ids=("loop-a-p",),
    )


def test_batch_inputs_reach_all_proposers_reviewers_and_rounds(
    tmp_path: Path,
) -> None:
    class RecordingFake(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.requests = []

        def execute(self, context, request, *, options):
            self.calls.append(("execute", request.task_id))
            self.requests.append(request)
            return _completed(request)

    fake = RecordingFake()
    workspace_input = LLMInputArtifact(
        "domain-markdown-001",
        ArtifactSourceRef(
            "run-a",
            "proposer-reviewer/inputs/source/0000-domain-markdown-001",
            ArtifactDigest("sha256", "a" * 64, 8),
        ),
        "text/markdown",
    )
    loop = _loop(
        proposers=(_worker("first"), _worker("second")),
        max_rounds=2,
        allow_early_stop=False,
    )
    base_request = _request(loop)
    request = BatchRequest(
        base_request.schema_version,
        base_request.batch_id,
        base_request.loops,
        base_request.failure_policy,
        (workspace_input,),
    )

    _repository, _handler, snapshot = _run(tmp_path, request, fake)

    assert snapshot.status is RunStatus.SUCCEEDED
    assert len(fake.requests) == 6
    assert sum(_is_proposer(request) for request in fake.requests) == 4
    assert sum(not _is_proposer(request) for request in fake.requests) == 2
    assert all(request.inputs == (workspace_input,) for request in fake.requests)


def test_loop_input_ids_scope_worker_requests_and_semantic_keys(
    tmp_path: Path,
) -> None:
    class RecordingFake(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.requests = []

        def execute(self, context, request, *, options):
            self.calls.append(("execute", request.task_id))
            self.requests.append(request)
            return _completed(request)

    fake = RecordingFake()
    inputs = tuple(
        LLMInputArtifact(
            input_id,
            ArtifactSourceRef(
                "run-a",
                f"proposer-reviewer/inputs/source/{index:04d}-{input_id}",
                ArtifactDigest("sha256", digest * 64, 8),
            ),
            "text/markdown",
        )
        for index, (input_id, digest) in enumerate(
            (("chapter-a", "a"), ("chapter-b", "b"))
        )
    )
    loops = (
        replace(_loop("loop-a"), input_ids=("chapter-a",)),
        replace(_loop("loop-b"), input_ids=("chapter-b",)),
    )
    request = BatchRequest(
        BATCH_SCHEMA_VERSION,
        "batch-a",
        loops,
        BatchFailurePolicy.COLLECT,
        inputs,
    )

    _repository, _handler, snapshot = _run(tmp_path, request, fake)

    assert snapshot.status is RunStatus.SUCCEEDED
    requests_by_loop = {
        loop_id: [
            item
            for item in fake.requests
            if f'"question":"{loop_id}"' in item.prompt
        ]
        for loop_id in ("loop-a", "loop-b")
    }
    assert requests_by_loop["loop-a"]
    assert requests_by_loop["loop-b"]
    assert all(
        item.inputs == (inputs[0],) for item in requests_by_loop["loop-a"]
    )
    assert all(
        item.inputs == (inputs[1],) for item in requests_by_loop["loop-b"]
    )


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


def test_failed_loop_preserves_typed_worker_error_as_private_cause(
    tmp_path: Path,
) -> None:
    fake = FakeLLM()
    loop = _loop(proposers=(_worker("bad", "FAIL_DETAILS"),))

    repository, _handler, snapshot = _run(tmp_path, _request(loop), fake)
    error = _result(repository, snapshot).loops[0].error

    assert error is not None
    assert error.code == "proposer_failed"
    assert error.message == "one or more proposers failed"
    assert error.details["causes"] == [
        {
            "worker_id": "bad",
            "code": "invalid_request",
            "message": "deliberate fake failure",
            "details": {"fake_marker": "kept"},
        }
    ]


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


def test_collect_keeps_failed_loop_from_corrupting_successful_loop(
    tmp_path: Path,
) -> None:
    fake = FakeLLM()
    failed = _loop("failed", proposers=(_worker("failed-p", "FAIL"),))
    successful = _loop("successful")
    repository, _handler, snapshot = _run(
        tmp_path,
        _request(failed, successful),
        fake,
        options=ExecutionOptions(max_concurrent_loops=1),
    )
    result = _result(repository, snapshot)
    assert [loop.loop_id for loop in result.loops] == ["failed", "successful"]
    assert result.loops[0].error is not None
    assert result.loops[0].final_proposals == {}
    assert result.loops[1].error is None
    assert tuple(result.loops[1].final_proposals) == ("successful-p",)


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


@pytest.mark.parametrize(
    "invalid_value",
    [
        "plain reviewer text",
        {"action": "stop"},
        {
            "schema_version": "arc.proposer_reviewer.review.v1",
            "action": "stop",
            "reason": "Too short.",
            "feedback": {},
            "payload": {"score": 1},
        },
    ],
)
def test_invalid_reviewer_shapes_fail_without_retry_or_fabrication(
    tmp_path: Path,
    invalid_value,
) -> None:
    class InvalidReviewer(FakeLLM):
        def execute(self, context, request, *, options):
            self.calls.append(("execute", request.task_id))
            if _is_proposer(request):
                return _completed(request)
            return LLMCompleted(
                invalid_value,
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
    assert len(fake.calls) == 2


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


def test_pause_resume_completes_the_paused_worker(tmp_path: Path) -> None:
    fake = FakeLLM(pause_once=True)
    repository, handler, paused = _run(tmp_path, _request(_loop()), fake)
    assert paused.status is RunStatus.PAUSED
    assert paused.awaiting is not None
    assert paused.awaiting.details["worker_id"] == "loop-a-p"

    resumed = RunEngine(repository).resume("run-a", handler)
    assert resumed.status is RunStatus.SUCCEEDED
    assert _result(repository, resumed).loops[0].rounds_completed == 1


def test_paused_v5_request_artifact_resumes_after_schema_upgrade(
    tmp_path: Path,
) -> None:
    fake = FakeLLM(pause_once=True)
    request = _request(_loop())
    legacy_request = encode_batch_request(request)
    legacy_request["schema_version"] = "arc.proposer_reviewer.batch.v5"
    for loop in legacy_request["loops"]:
        assert isinstance(loop, dict)
        loop.pop("revision_context_mode")
        loop.pop("input_ids")

    repository, handler, paused = _run(tmp_path, request, fake)
    assert paused.status is RunStatus.PAUSED

    # Emulate the immutable request artifact written by the v5 service before
    # this process was upgraded. The protocol tests separately cover decoding
    # the v5 run spec itself.
    _replace_persisted_request(repository, legacy_request)

    resumed = RunEngine(repository).resume("run-a", handler)
    assert resumed.status is RunStatus.SUCCEEDED
    assert _result(repository, resumed).loops[0].rounds_completed == 1


def test_schema_upgrade_still_rejects_a_different_persisted_request(
    tmp_path: Path,
) -> None:
    fake = FakeLLM(pause_once=True)
    request = _request(_loop())
    repository, handler, paused = _run(tmp_path, request, fake)
    assert paused.status is RunStatus.PAUSED

    different = encode_batch_request(request)
    different["schema_version"] = "arc.proposer_reviewer.batch.v5"
    different["batch_id"] = "different-batch"
    for loop in different["loops"]:
        assert isinstance(loop, dict)
        loop.pop("revision_context_mode")
        loop.pop("input_ids")
    _replace_persisted_request(repository, different)

    failed = RunEngine(repository).resume("run-a", handler)
    assert failed.status is RunStatus.FAILED
    assert failed.error is not None
    assert "persisted proposer-reviewer request differs" in failed.error.message


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


def test_run_root_and_logical_artifact_ids_are_explicit_and_stable(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "explicit-runs"
    repository, _handler, snapshot = _run(
        run_root,
        _request(_loop()),
        FakeLLM(),
    )
    assert snapshot.status is RunStatus.SUCCEEDED
    run_directory = repository.run_directory("run-a")
    assert run_directory.is_relative_to(run_root)
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (run_directory / "artifacts" / "manifests").glob("*.json")
    ]
    logical_ids = {item["artifact_id"] for item in manifests}
    assert {
        "proposer-reviewer/request",
        "proposer-reviewer/loops/loop-a/rounds/001/proposals/loop-a-p",
        "proposer-reviewer/loops/loop-a/rounds/001/reviews/loop-a-r",
        "proposer-reviewer/loops/loop-a/result",
        "proposer-reviewer/batch/result",
    } <= logical_ids
    assert all(not Path(item["relative_path"]).is_absolute() for item in manifests)


def test_targeted_feedback_and_worker_llm_contracts_reach_the_same_lineage(
    tmp_path: Path,
) -> None:
    class RoutingLLM(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.requests = []
            self.review_calls = 0

        def execute(self, context, request, *, options):
            self.calls.append(("execute", request.task_id))
            self.requests.append(request)
            if _is_proposer(request):
                return LLMCompleted(
                    {"proposal": request.task_id},
                    "fake",
                    "fake-model",
                    SessionRef(
                        (
                            "session-p-one"
                            if "Produce proposal one." in request.prompt
                            else "session-p-two"
                        ),
                        "a" * 64,
                    ),
                    None,
                )
            self.review_calls += 1
            completed = _completed(request)
            value = dict(completed.value)
            active_ids = tuple(value["feedback"])
            value["action"] = "continue" if self.review_calls == 1 else "stop"
            value["reason"] = "Route feedback by worker identity."
            value["feedback"] = {
                worker_id: f"feedback-for:{worker_id}"
                for worker_id in active_ids
            }
            value["payload"] = {"score": self.review_calls}
            return LLMCompleted(
                value,
                "fake",
                "fake-model",
                SessionRef("reviewer-session", "b" * 64),
                None,
            )

    first = WorkerSpec(
        "p-one",
        "Produce proposal one.",
        PROPOSAL_SCHEMA,
        ModelSelection(provider="codex", model="model-one"),
    )
    second = WorkerSpec(
        "p-two",
        "Produce proposal two.",
        PROPOSAL_SCHEMA,
        ModelSelection(provider="claude", model="model-two"),
    )
    fake = RoutingLLM()
    loop = _loop(
        proposers=(first, second),
        max_rounds=2,
        allow_early_stop=False,
    )
    repository, _handler, snapshot = _run(tmp_path, _request(loop), fake)
    assert snapshot.status is RunStatus.SUCCEEDED
    assert _result(repository, snapshot).loops[0].rounds_completed == 2

    proposer_requests = [item for item in fake.requests if _is_proposer(item)]
    assert len(proposer_requests) == 4
    first_round = proposer_requests[:2]
    assert {item.model for item in first_round} == {first.model, second.model}
    assert all(
        isinstance(item.output, JsonOutput)
        and item.output.schema == PROPOSAL_SCHEMA
        for item in proposer_requests
    )
    second_round = proposer_requests[2:]
    assert any(
        "Produce proposal one." in item.prompt
        and "feedback-for:p-one" in item.prompt
        and "feedback-for:p-two" not in item.prompt
        for item in second_round
    )
    assert any(
        "Produce proposal two." in item.prompt
        and "feedback-for:p-two" in item.prompt
        and "feedback-for:p-one" not in item.prompt
        for item in second_round
    )
    assert all("previous_review_envelope" not in item.prompt for item in second_round)
    assert proposer_requests[2].session is not None
    assert proposer_requests[3].session is not None


def test_full_review_envelope_reaches_delta_proposer_and_recovers_after_pause(
    tmp_path: Path,
) -> None:
    class PauseRevisionLLM(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.requests = []
            self.proposer_calls = 0
            self.review_calls = 0

        def execute(self, context, request, *, options):
            self.calls.append(("execute", request.task_id))
            self.requests.append(request)
            if _is_proposer(request):
                self.proposer_calls += 1
                if self.proposer_calls == 2:
                    self.paused_task_id = request.task_id
                    return LLMPaused(
                        ResumeReason.EXTERNAL_CONDITION,
                        "provider-wait",
                        {"code": "provider_unavailable"},
                    )
                return _completed(request)
            self.review_calls += 1
            completed = _completed(request)
            value = dict(completed.value)
            value["action"] = "continue" if self.review_calls == 1 else "stop"
            value["reason"] = "Revise the proposal once."
            value["feedback"] = {"loop-a-p": "Tighten the derivation."}
            value["payload"] = {"score": self.review_calls}
            return LLMCompleted(value, "fake", "fake-model", None, None)

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

    fake = PauseRevisionLLM()
    loop = _loop(
        max_rounds=2,
        allow_early_stop=False,
        revision_context_mode=RevisionContextMode.FULL_REVIEW_ENVELOPE,
    )
    repository, handler, paused = _run(tmp_path, _request(loop), fake)

    assert paused.status is RunStatus.PAUSED
    revision_request = fake.requests[2]
    round_task = json.loads(revision_request.prompt.split("## Round task\n", 1)[1])
    assert round_task["targeted_feedback"] == "Tighten the derivation."
    assert round_task["previous_review_envelope"] == {
        "schema_version": "arc.proposer_reviewer.review.v1",
        "action": "continue",
        "reason": "Revise the proposal once.",
        "feedback": {"loop-a-p": "Tighten the derivation."},
        "payload": {"score": 1},
    }

    resumed = RunEngine(repository).resume("run-a", handler)

    assert resumed.status is RunStatus.SUCCEEDED
    assert _result(repository, resumed).loops[0].rounds_completed == 2
    assert ("resume", revision_request.task_id) in fake.calls


def test_invalid_later_review_preserves_the_last_complete_round(
    tmp_path: Path,
) -> None:
    class InvalidSecondReview(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.review_calls = 0

        def execute(self, context, request, *, options):
            self.calls.append(("execute", request.task_id))
            if _is_proposer(request):
                return _completed(request)
            self.review_calls += 1
            if self.review_calls == 2:
                return LLMCompleted(
                    {"action": "stop"},
                    "fake",
                    "fake-model",
                    None,
                    None,
                )
            completed = _completed(request)
            value = dict(completed.value)
            value["action"] = "continue"
            value["reason"] = "One more independent round is required."
            value["feedback"] = {
                worker_id: "Recompute the complete proposal."
                for worker_id in value["feedback"]
            }
            return LLMCompleted(
                value,
                "fake",
                "fake-model",
                None,
                None,
            )

    fake = InvalidSecondReview()
    repository, _handler, snapshot = _run(
        tmp_path,
        _request(_loop(max_rounds=2, allow_early_stop=False)),
        fake,
    )
    result = _result(repository, snapshot).loops[0]
    assert fake.review_calls == 2
    assert result.termination.value == "failed"
    assert tuple(result.final_proposals) == ("loop-a-p",)
    assert result.final_review is not None
    assert result.final_review["payload"] == {"score": 1}
    assert result.error is not None


def test_terminal_proposer_round_skips_reviewer_and_replays_durably(
    tmp_path: Path,
) -> None:
    class ContinuingReviews(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.requests = []
            self.review_calls = 0

        def execute(self, context, request, *, options):
            self.calls.append(("execute", request.task_id))
            self.requests.append(request)
            completed = _completed(request)
            if _is_proposer(request):
                return completed
            self.review_calls += 1
            value = dict(completed.value)
            value["action"] = "continue"
            value["reason"] = "A complete revision is required."
            value["feedback"] = {
                worker_id: "Revise the complete proposal."
                for worker_id in value["feedback"]
            }
            value["payload"] = {"score": self.review_calls}
            return LLMCompleted(value, "fake", "fake-model", None, None)

    fake = ContinuingReviews()
    loop = _loop(
        max_rounds=3,
        allow_early_stop=True,
        review_final_round=False,
    )
    repository, handler, snapshot = _run(tmp_path, _request(loop), fake)

    assert snapshot.status is RunStatus.SUCCEEDED
    result = _result(repository, snapshot).loops[0]
    assert result.termination.value == "round_limit"
    assert result.rounds_completed == 3
    assert result.final_review is not None
    assert result.final_review["payload"] == {"score": 2}
    assert [
        "proposer" if _is_proposer(request) else "reviewer"
        for request in fake.requests
    ] == ["proposer", "reviewer", "proposer", "reviewer", "proposer"]
    assert fake.review_calls == 2

    trace = read_batch_trace(repository, "run-a")
    assert [round_ref.review_ref is None for round_ref in trace.loops[0].rounds] == [
        False,
        False,
        True,
    ]
    terminal = read_batch_round(repository, "run-a", loop.loop_id, 3)
    assert terminal.review is None
    assert terminal.review_ref is None
    assert len(terminal.transcript_refs) == 1
    state = AtomicStateStore(
        repository.run_directory("run-a")
        / "state"
        / f"{state_namespace(loop.loop_id)}.json",
        _LoopStateContract(),
    ).read()
    assert state is not None
    assert state.rounds_completed == 3
    assert state.review_ref is not None
    assert state.review_ref.artifact_id.endswith("rounds/002/reviews/loop-a-r")
    assert len(state.transcript_refs) == 5
    artifacts = ImmutableArtifactStore(
        repository.run_directory("run-a"), repository_root=repository.root
    )
    assert artifacts.find(
        "proposer-reviewer/loops/loop-a/rounds/003/proposals/loop-a-p"
    ) is not None
    assert artifacts.find(
        "proposer-reviewer/loops/loop-a/rounds/003/reviews/loop-a-r"
    ) is None
    assert artifacts.find(
        "proposer-reviewer/loops/loop-a/rounds/003/transcript/001"
    ) is not None

    replayed = RunEngine(repository).execute(
        RunSpec("run-a", handler.name, encode_batch_request(_request(loop))),
        handler,
    )
    assert replayed == snapshot
    assert len(fake.requests) == 5


def test_early_reviewer_stop_prevents_terminal_proposer_round(
    tmp_path: Path,
) -> None:
    fake = FakeLLM()
    loop = _loop(
        max_rounds=3,
        allow_early_stop=True,
        review_final_round=False,
    )
    repository, _handler, snapshot = _run(tmp_path, _request(loop), fake)

    assert snapshot.status is RunStatus.SUCCEEDED
    result = _result(repository, snapshot).loops[0]
    assert result.termination.value == "reviewer_stop"
    assert result.rounds_completed == 1
    assert len(fake.calls) == 2
    assert read_batch_trace(repository, "run-a").loops[0].rounds[0].review_ref is not None


def test_terminal_proposer_pause_resumes_without_a_terminal_reviewer(
    tmp_path: Path,
) -> None:
    class PauseTerminalProposal(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.proposer_calls = 0
            self.review_calls = 0
            self.requests = []

        def execute(self, context, request, *, options):
            self.calls.append(("execute", request.task_id))
            self.requests.append(request)
            if _is_proposer(request):
                self.proposer_calls += 1
                if self.proposer_calls == 3:
                    self.paused_task_id = request.task_id
                    return LLMPaused(
                        ResumeReason.EXTERNAL_CONDITION,
                        "terminal-provider-wait",
                        {"code": "provider_unavailable"},
                    )
                return _completed(request)
            self.review_calls += 1
            completed = _completed(request)
            value = dict(completed.value)
            value["action"] = "continue"
            value["reason"] = "One terminal revision remains."
            value["feedback"] = {
                worker_id: "Revise the complete proposal."
                for worker_id in value["feedback"]
            }
            return LLMCompleted(value, "fake", "fake-model", None, None)

        def resume(self, context, task_id, *, input, options):
            self.calls.append(("resume", task_id))
            assert task_id == self.paused_task_id
            return LLMCompleted(
                {"proposal": "terminal-resumed"},
                "fake",
                "fake-model",
                None,
                None,
            )

    fake = PauseTerminalProposal()
    loop = _loop(
        max_rounds=3,
        allow_early_stop=True,
        review_final_round=False,
    )
    repository, handler, paused = _run(tmp_path, _request(loop), fake)

    assert paused.status is RunStatus.PAUSED
    assert paused.awaiting is not None
    assert paused.awaiting.details["round"] == 3
    assert paused.awaiting.details["role"] == "proposer"
    assert fake.review_calls == 2

    resumed = RunEngine(repository).resume("run-a", handler)

    assert resumed.status is RunStatus.SUCCEEDED
    result = _result(repository, resumed).loops[0]
    assert result.rounds_completed == 3
    assert result.final_proposals == {"loop-a-p": {"proposal": "terminal-resumed"}}
    assert result.final_review is not None
    assert result.final_review["payload"] == {"score": 1}
    assert fake.review_calls == 2
    assert [kind for kind, _task_id in fake.calls] == [
        "execute",
        "execute",
        "execute",
        "execute",
        "execute",
        "resume",
    ]
    terminal = read_batch_round(repository, "run-a", loop.loop_id, 3)
    assert terminal.review is None
    assert terminal.proposals == {"loop-a-p": {"proposal": "terminal-resumed"}}
