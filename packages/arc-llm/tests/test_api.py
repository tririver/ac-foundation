from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import threading

import pytest

from arc_jobs import (
    ArtifactDigest,
    ArtifactSourceRef,
    RunContext,
    RunRepository,
    RunSpec,
    RunStatus,
)
from arc_jobs import EffectStage
from arc_llm import (
    AdoptionAuthorization,
    InteractiveJsonOutput,
    InteractionResponse,
    JsonOutput,
    LLMClient,
    LLMCompleted,
    LLMExecutionOptions,
    LLMFailed,
    LLMInputArtifact,
    LLMPaused,
    LLMRequest,
    LLMTaskService,
    ModelSelection,
    NativeResumeHandle,
    OperationContract,
    ProviderExecution,
    ProviderFailure,
    ProviderGateOptions,
    ProviderTerminalKind,
    DeliveryState,
    FailureCategory,
    ResumeAction,
    ResumeInput,
    SemanticKeyDigest,
    resume_input_matches,
)
from arc_llm.identity import semantic_key
from arc_llm.output import CandidateMaterial


def _request(task_id: str = "task") -> LLMRequest:
    return LLMRequest(
        task_id,
        "Return an object.",
        JsonOutput({"type": "object", "required": ["answer"]}),
        ModelSelection("codex"),
    )


def _completed(value: object, *, handle: str = "thread-1") -> ProviderExecution:
    return ProviderExecution(
        ProviderTerminalKind.COMPLETED,
        (CandidateMaterial(value=value, terminal=True),),
        NativeResumeHandle("codex", handle),
    )


def _source_input(
    repository: RunRepository,
    *,
    content: bytes = b"# paper\n",
    media_type: str = "text/markdown",
    run_id: str = "source-run",
    artifact_id: str = "paper/source",
) -> tuple[LLMInputArtifact, Path]:
    source_snapshot = repository.create(
        RunSpec(run_id, "test.source", {"case": "input"})
    )
    source_context = RunContext(
        repository,
        source_snapshot,
        resume_input=None,
        execution_slice=None,
    )
    ref = source_context.artifacts.publish_bytes(
        artifact_id,
        content,
        media_type=media_type,
    )
    return (
        LLMInputArtifact(
            "paper",
            ArtifactSourceRef(
                run_id,
                artifact_id,
                ArtifactDigest(
                    "sha256",
                    ref.digest.value,
                    ref.digest.size_bytes,
                ),
            ),
            media_type,
        ),
        repository.run_directory(run_id) / ref.relative_path,
    )


def _semantic_mutant(
    repository: RunRepository,
    request: LLMRequest,
    mutation: str,
) -> LLMRequest:
    if mutation == "prompt":
        return replace(request, prompt="Changed prompt.")
    if mutation == "output":
        return replace(
            request,
            output=JsonOutput({"type": "object", "required": ["changed"]}),
        )
    if mutation == "input":
        input_artifact, _ = _source_input(repository)
        return replace(request, inputs=(input_artifact,))
    raise AssertionError(f"unknown semantic mutation: {mutation}")


def test_client_generate_replays_accepted_result_without_provider_call(
    tmp_path: Path, adapter, registry
) -> None:
    adapter.steps.append(_completed({"answer": 42}))
    client = LLMClient(registry=registry)
    first = client.generate(_request(), run_root=tmp_path)
    second = client.generate(_request(), run_root=tmp_path)
    assert first.snapshot.status is RunStatus.SUCCEEDED
    assert isinstance(first.outcome, LLMCompleted)
    assert second.snapshot.result_ref == first.snapshot.result_ref
    assert isinstance(second.outcome, LLMCompleted)
    assert second.outcome.value == {"answer": 42}
    assert adapter.start_calls == 1


def test_same_task_id_with_changed_semantics_fails_before_provider(
    tmp_path: Path, adapter, registry
) -> None:
    adapter.steps.append(_completed({"answer": 42}))
    client = LLMClient(registry=registry)
    first = client.generate(_request(), run_root=tmp_path)
    changed = LLMRequest(
        "task",
        "Different prompt.",
        _request().output,
        ModelSelection("codex"),
    )
    second = client.generate(changed, run_root=tmp_path)
    assert first.snapshot.status is RunStatus.SUCCEEDED
    from arc_llm import LLMFailed

    assert isinstance(second.outcome, LLMFailed)
    assert second.outcome.error.code.value == "idempotency_conflict"
    assert adapter.start_calls == 1


def test_service_supports_multiple_tasks_in_one_parent_run_without_effect_collision(
    tmp_path: Path, adapter, registry
) -> None:
    adapter.steps.extend([_completed({"answer": 1}), _completed({"answer": 2})])
    repository = RunRepository(tmp_path)
    snapshot = repository.create(RunSpec("parent", "test.parent", {"case": "multi"}))
    context = RunContext(
        repository,
        snapshot,
        resume_input=None,
        execution_slice=None,
    )
    service = LLMTaskService(registry=registry)
    assert isinstance(service.execute(context, _request("one")), LLMCompleted)
    assert isinstance(service.execute(context, _request("two")), LLMCompleted)
    effect_files = tuple((tmp_path / "runs" / "parent" / "effects").glob("*.json"))
    assert len(effect_files) == 2
    assert effect_files[0].name != effect_files[1].name


def test_execute_or_resume_preserves_input_required_pause_without_input(
    tmp_path: Path,
    adapter,
    registry,
) -> None:
    contract = InteractiveJsonOutput(
        {"type": "object", "required": ["answer"]},
        {
            "lookup": OperationContract(
                {"type": "object", "required": ["query"]},
                {"type": "object", "required": ["value"]},
            )
        },
    )
    adapter.steps.append(
        _completed(
            {
                "schema_version": "arc.llm.interactive_turn.v1",
                "state": "interact",
                "result": None,
                "requests": [
                    {
                        "request_id": "req-1",
                        "operation": "lookup",
                        "arguments": {"query": "x"},
                    }
                ],
            }
        )
    )
    repository = RunRepository(tmp_path)
    parent = repository.create(
        RunSpec("parent", "test.parent", {"case": "preserve-child-pause"})
    )
    context = RunContext(repository, parent, resume_input=None, execution_slice=None)
    request = LLMRequest(
        "paused-child",
        "Solve.",
        contract,
        ModelSelection("codex"),
    )
    service = LLMTaskService(registry=registry)

    first = service.execute_or_resume(context, request)
    replayed = service.execute_or_resume(context, request)

    assert isinstance(first, LLMPaused)
    assert replayed == first
    resume_input = ResumeInput(first.resume_key, ResumeAction.CONTINUE)
    assert resume_input_matches(request, resume_input)
    assert not resume_input_matches(
        LLMRequest(
            "other-child",
            request.prompt,
            request.output,
            request.model,
        ),
        resume_input,
    )
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 0


@pytest.mark.parametrize("mutation", ("prompt", "output", "input"))
def test_execute_or_resume_rejects_semantic_conflict_at_input_required_pause(
    tmp_path: Path,
    adapter,
    registry,
    mutation: str,
) -> None:
    contract = InteractiveJsonOutput(
        {"type": "object", "required": ["answer"]},
        {
            "lookup": OperationContract(
                {"type": "object", "required": ["query"]},
                {"type": "object", "required": ["value"]},
            )
        },
    )
    adapter.steps.append(
        _completed(
            {
                "schema_version": "arc.llm.interactive_turn.v1",
                "state": "interact",
                "result": None,
                "requests": [
                    {
                        "request_id": "req-conflict",
                        "operation": "lookup",
                        "arguments": {"query": "x"},
                    }
                ],
            }
        )
    )
    repository = RunRepository(tmp_path)
    parent = repository.create(
        RunSpec("parent", "test.parent", {"case": "input-pause-conflict"})
    )
    context = RunContext(repository, parent, resume_input=None, execution_slice=None)
    request = LLMRequest(
        "paused-child",
        "Solve.",
        contract,
        ModelSelection("codex"),
    )
    service = LLMTaskService(registry=registry)

    paused = service.execute_or_resume(context, request)
    conflict = service.execute_or_resume(
        context,
        _semantic_mutant(repository, request, mutation),
    )

    assert isinstance(paused, LLMPaused)
    assert paused.input_required
    assert isinstance(conflict, LLMFailed)
    assert conflict.error.code.value == "idempotency_conflict"
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 0


@pytest.mark.parametrize("mutation", ("prompt", "output", "input"))
def test_execute_or_resume_rejects_semantic_conflict_at_no_input_pause(
    tmp_path: Path,
    adapter,
    registry,
    mutation: str,
) -> None:
    adapter.steps.extend(
        [
            ProviderFailure(
                "authentication unavailable",
                category=FailureCategory.AUTHENTICATION,
                delivery=DeliveryState.NOT_DELIVERED,
            ),
            _completed({"answer": 42}),
        ]
    )
    repository = RunRepository(tmp_path)
    parent = repository.create(
        RunSpec("parent", "test.parent", {"case": "no-input-pause-conflict"})
    )
    context = RunContext(repository, parent, resume_input=None, execution_slice=None)
    request = _request("paused-child")
    service = LLMTaskService(registry=registry)
    options = LLMExecutionOptions(gate=ProviderGateOptions(enabled=False))

    paused = service.execute_or_resume(context, request, options=options)
    conflict = service.execute_or_resume(
        context,
        _semantic_mutant(repository, request, mutation),
        options=options,
    )

    assert isinstance(paused, LLMPaused)
    assert not paused.input_required
    assert isinstance(conflict, LLMFailed)
    assert conflict.error.code.value == "idempotency_conflict"
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 0


def test_input_artifact_is_verified_and_materialized_before_provider_call(
    tmp_path: Path,
    adapter,
    registry,
) -> None:
    repository = RunRepository(tmp_path)
    input_artifact, _ = _source_input(repository)
    parent = repository.create(
        RunSpec("parent", "test.parent", {"case": "verified-input"})
    )
    context = RunContext(repository, parent, resume_input=None, execution_slice=None)
    adapter.steps.append(_completed({"answer": 42}))
    request = LLMRequest(
        "input-task",
        "Review the input.",
        _request().output,
        ModelSelection("codex"),
        inputs=(input_artifact,),
    )

    outcome = LLMTaskService(registry=registry).execute(context, request)

    assert isinstance(outcome, LLMCompleted)
    delivered = adapter.requests[0].inputs
    assert len(delivered) == 1
    assert delivered[0].path.is_file()
    assert delivered[0].path.read_bytes() == b"# paper\n"
    assert delivered[0].sha256 == input_artifact.source.expected_digest.value
    assert delivered[0].path.stat().st_mode & 0o777 == 0o400
    assert delivered[0].path.parent.stat().st_mode & 0o777 == 0o500
    object_paths = (
        path
        for path in (tmp_path / "runs" / "parent" / "artifacts" / "objects").rglob("*")
        if path.is_file()
    )
    recipe = next(path for path in object_paths if b'"input_delivery"' in path.read_bytes())
    assert b'"mode":"read_tool"' in recipe.read_bytes()


def test_relocated_input_source_replays_same_standalone_run(
    tmp_path: Path,
    adapter,
    registry,
) -> None:
    repository = RunRepository(tmp_path)
    first_input, _ = _source_input(repository)
    second_input, _ = _source_input(
        repository,
        run_id="other-source",
        artifact_id="other/paper",
    )
    adapter.steps.append(_completed({"answer": 42}))
    client = LLMClient(registry=registry)
    first_request = LLMRequest(
        "relocated",
        "Review.",
        _request().output,
        ModelSelection("codex"),
        inputs=(first_input,),
    )
    second_request = LLMRequest(
        "relocated",
        "Review.",
        _request().output,
        ModelSelection("codex"),
        inputs=(second_input,),
    )

    first = client.generate(first_request, run_root=tmp_path, run_id="llm-run")
    second = client.generate(second_request, run_root=tmp_path, run_id="llm-run")

    assert isinstance(first.outcome, LLMCompleted)
    assert isinstance(second.outcome, LLMCompleted)
    assert adapter.start_calls == 1


def test_standalone_resume_recovers_request_locators_before_task_state_exists(
    tmp_path: Path,
    adapter,
    registry,
    monkeypatch,
) -> None:
    repository = RunRepository(tmp_path)
    input_artifact, _ = _source_input(repository)
    request = LLMRequest(
        "bootstrap-crash",
        "Review.",
        _request().output,
        ModelSelection("codex"),
        inputs=(input_artifact,),
    )
    adapter.steps.append(_completed({"answer": 42}))
    client = LLMClient(registry=registry)
    original_execute = client.service._executor.execute

    def crash_before_task_state(*args, **kwargs):
        raise KeyboardInterrupt("simulated crash before task state publication")

    monkeypatch.setattr(client.service._executor, "execute", crash_before_task_state)
    with pytest.raises(KeyboardInterrupt):
        client.generate(request, run_root=tmp_path, run_id="llm-run")

    snapshot = repository.inspect("llm-run").snapshot
    context = RunContext(
        repository,
        snapshot,
        resume_input=None,
        execution_slice=None,
    )
    assert client.service._executor._task_store(context, request.task_id).read() is None

    monkeypatch.setattr(client.service._executor, "execute", original_execute)
    resumed = client.resume(run_root=tmp_path, run_id="llm-run")

    assert resumed.snapshot.status is RunStatus.SUCCEEDED
    assert isinstance(resumed.outcome, LLMCompleted)
    assert resumed.outcome.value == {"answer": 42}
    assert adapter.start_calls == 1
    assert adapter.requests[0].inputs[0].sha256 == (
        input_artifact.source.expected_digest.value
    )


def test_standalone_resume_reuses_canonical_input_published_before_task_state(
    tmp_path: Path,
    adapter,
    registry,
    monkeypatch,
) -> None:
    repository = RunRepository(tmp_path)
    input_artifact, source_path = _source_input(repository)
    request = LLMRequest(
        "canonical-bootstrap-crash",
        "Review.",
        _request().output,
        ModelSelection("codex"),
        inputs=(input_artifact,),
    )
    adapter.steps.append(_completed({"answer": 42}))
    client = LLMClient(registry=registry)
    executor = client.service._executor
    original_execution_document = executor._execution_document

    def crash_after_input_publication(*args, **kwargs):
        raise KeyboardInterrupt("simulated crash after canonical input publication")

    monkeypatch.setattr(
        executor,
        "_execution_document",
        crash_after_input_publication,
    )
    with pytest.raises(KeyboardInterrupt):
        client.generate(request, run_root=tmp_path, run_id="llm-run")

    snapshot = repository.inspect("llm-run").snapshot
    context = RunContext(
        repository,
        snapshot,
        resume_input=None,
        execution_slice=None,
    )
    assert executor._task_store(context, request.task_id).read() is None
    source_path.write_bytes(b"corrupt upstream source")

    monkeypatch.setattr(
        executor,
        "_execution_document",
        original_execution_document,
    )
    resumed = client.resume(run_root=tmp_path, run_id="llm-run")

    assert resumed.snapshot.status is RunStatus.SUCCEEDED
    assert isinstance(resumed.outcome, LLMCompleted)
    assert resumed.outcome.value == {"answer": 42}
    assert adapter.start_calls == 1


def test_unfinished_task_resumes_from_canonical_current_run_input(
    tmp_path: Path,
    adapter,
    registry,
) -> None:
    repository = RunRepository(tmp_path)
    input_artifact, source_path = _source_input(repository)
    request = LLMRequest(
        "canonical-resume",
        "Review.",
        _request().output,
        ModelSelection("codex"),
        inputs=(input_artifact,),
    )
    adapter.steps.extend(
        [
            ProviderFailure(
                "authentication unavailable",
                category=FailureCategory.AUTHENTICATION,
                delivery=DeliveryState.NOT_DELIVERED,
            ),
            _completed({"answer": 42}),
        ]
    )
    client = LLMClient(registry=registry)
    options = LLMExecutionOptions(
        gate=ProviderGateOptions(enabled=False),
    )

    paused = client.generate(
        request,
        run_root=tmp_path,
        run_id="llm-run",
        options=options,
    )
    assert paused.snapshot.status is RunStatus.PAUSED
    assert isinstance(paused.outcome, LLMPaused)
    source_path.write_bytes(b"corrupt upstream source")

    resumed = client.resume(
        run_root=tmp_path,
        run_id="llm-run",
        options=options,
    )

    assert resumed.snapshot.status is RunStatus.SUCCEEDED
    assert isinstance(resumed.outcome, LLMCompleted)
    assert resumed.outcome.value == {"answer": 42}
    assert adapter.start_calls == 2
    assert all(
        item.inputs[0].path.stat().st_mode & 0o777 == 0o400
        for item in adapter.requests
    )


def test_corrupt_input_fails_before_provider_invocation(
    tmp_path: Path,
    adapter,
    registry,
) -> None:
    repository = RunRepository(tmp_path)
    input_artifact, source_path = _source_input(repository)
    source_path.write_bytes(b"corrupt")
    parent = repository.create(
        RunSpec("parent", "test.parent", {"case": "corrupt-input"})
    )
    context = RunContext(repository, parent, resume_input=None, execution_slice=None)
    request = LLMRequest(
        "corrupt-input",
        "Review.",
        _request().output,
        ModelSelection("codex"),
        inputs=(input_artifact,),
    )

    outcome = LLMTaskService(registry=registry).execute(context, request)

    assert isinstance(outcome, LLMFailed)
    assert outcome.error.code.value == "invalid_request"
    assert outcome.error.details["code"] == "invalid_input_artifact"
    assert adapter.start_calls == 0


def test_explicit_provider_rejects_unsupported_media_before_provider_invocation(
    tmp_path: Path,
    adapter,
    registry,
) -> None:
    repository = RunRepository(tmp_path)
    input_artifact, _ = _source_input(
        repository,
        content=b"%PDF",
        media_type="application/pdf",
    )
    parent = repository.create(
        RunSpec("parent", "test.parent", {"case": "unsupported-input"})
    )
    context = RunContext(repository, parent, resume_input=None, execution_slice=None)
    request = LLMRequest(
        "unsupported-input",
        "Review.",
        _request().output,
        ModelSelection("codex"),
        inputs=(input_artifact,),
    )

    outcome = LLMTaskService(registry=registry).execute(context, request)

    assert isinstance(outcome, LLMFailed)
    assert outcome.error.code.value == "invalid_request"
    assert outcome.error.details["code"] == "unsupported_input_media"
    assert adapter.start_calls == 0


def test_same_semantic_task_is_single_flight_and_replays_to_concurrent_caller(
    tmp_path: Path,
    adapter,
    registry,
) -> None:
    repository = RunRepository(tmp_path)
    snapshot = repository.create(
        RunSpec("parent", "test.parent", {"case": "same-task-single-flight"})
    )
    context = RunContext(
        repository,
        snapshot,
        resume_input=None,
        execution_slice=None,
    )
    service = LLMTaskService(registry=registry)
    request = _request("shared-task")
    provider_entered = threading.Event()
    release_provider = threading.Event()

    def blocking_start(request, observer, cancel):
        adapter.start_calls += 1
        adapter.requests.append(request)
        observer.before_delivery()
        provider_entered.set()
        assert release_provider.wait(timeout=5)
        result = _completed({"answer": 42})
        observer.native_handle(result.native_handle)
        return result

    adapter.start = blocking_start
    worker_label = threading.local()
    replay_caller_waiting = threading.Event()
    original_checkpoint = context.checkpoint

    def checkpoint() -> None:
        if getattr(worker_label, "value", None) == "replay":
            replay_caller_waiting.set()
        original_checkpoint()

    context.checkpoint = checkpoint

    def execute(label: str):
        worker_label.value = label
        return service.execute(context, request)

    with ThreadPoolExecutor(max_workers=2) as pool:
        provider_call = pool.submit(execute, "provider")
        assert provider_entered.wait(timeout=5)
        replay_call = pool.submit(execute, "replay")
        assert replay_caller_waiting.wait(timeout=5)
        assert adapter.start_calls == 1
        release_provider.set()
        provider_outcome = provider_call.result(timeout=5)
        replay_outcome = replay_call.result(timeout=5)

    assert isinstance(provider_outcome, LLMCompleted)
    assert isinstance(replay_outcome, LLMCompleted)
    assert replay_outcome == provider_outcome
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 0


def test_same_session_prefix_allows_only_one_concurrent_paid_sibling(
    tmp_path: Path, adapter, registry
) -> None:
    adapter.steps.append(_completed({"answer": 1}, handle="thread-root"))
    repository = RunRepository(tmp_path)
    snapshot = repository.create(
        RunSpec("parent", "test.parent", {"case": "session-siblings"})
    )
    context = RunContext(
        repository,
        snapshot,
        resume_input=None,
        execution_slice=None,
    )
    service = LLMTaskService(registry=registry)
    root = service.execute(context, _request("root"))
    assert isinstance(root, LLMCompleted)
    assert root.session is not None

    provider_entered = threading.Event()
    release_provider = threading.Event()

    def blocking_resume(handle, request, observer, cancel):
        adapter.resume_calls += 1
        adapter.requests.append(request)
        observer.before_delivery()
        provider_entered.set()
        assert release_provider.wait(timeout=5)
        result = _completed({"answer": 2}, handle="thread-child")
        observer.native_handle(result.native_handle)
        return result

    adapter.resume = blocking_resume
    sibling_a = LLMRequest(
        "sibling-a",
        "Return an object.",
        _request().output,
        ModelSelection("codex"),
        session=root.session,
    )
    sibling_b = LLMRequest(
        "sibling-b",
        "Return an object.",
        _request().output,
        ModelSelection("codex"),
        session=root.session,
    )

    worker_label = threading.local()
    sibling_b_waiting = threading.Event()
    original_checkpoint = context.checkpoint

    def checkpoint() -> None:
        if getattr(worker_label, "value", None) == "b":
            sibling_b_waiting.set()
        original_checkpoint()

    context.checkpoint = checkpoint

    def execute_sibling(label: str, request: LLMRequest):
        worker_label.value = label
        return service.execute(context, request)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(execute_sibling, "a", sibling_a)
        assert provider_entered.wait(timeout=5)
        second = pool.submit(execute_sibling, "b", sibling_b)
        assert sibling_b_waiting.wait(timeout=5)
        assert adapter.resume_calls == 1
        release_provider.set()
        outcomes = (first.result(timeout=5), second.result(timeout=5))

    assert sum(isinstance(item, LLMCompleted) for item in outcomes) == 1
    conflicts = [item for item in outcomes if isinstance(item, LLMFailed)]
    assert len(conflicts) == 1
    assert conflicts[0].error.code.value == "invalid_request"
    assert adapter.resume_calls == 1


def test_host_driven_interaction_pauses_and_resumes_same_native_session(
    tmp_path: Path, adapter, registry
) -> None:
    contract = InteractiveJsonOutput(
        {"type": "object", "required": ["answer"]},
        {
            "lookup": OperationContract(
                {"type": "object", "required": ["query"]},
                {"type": "object", "required": ["value"]},
            )
        },
    )
    adapter.steps.extend(
        [
            _completed(
                {
                    "schema_version": "arc.llm.interactive_turn.v1",
                    "state": "interact",
                    "result": None,
                    "requests": [
                        {
                            "request_id": "req-1",
                            "operation": "lookup",
                            "arguments": {"query": "x"},
                        }
                    ],
                }
            ),
            _completed(
                {
                    "schema_version": "arc.llm.interactive_turn.v1",
                    "state": "complete",
                    "result": {"answer": 7},
                    "requests": [],
                }
            ),
        ]
    )
    request = LLMRequest("interactive", "Solve.", contract, ModelSelection("codex"))
    client = LLMClient(registry=registry)
    paused = client.generate(request, run_root=tmp_path)
    assert paused.snapshot.status is RunStatus.PAUSED
    assert isinstance(paused.outcome, LLMPaused)
    assert paused.outcome.resume_key.startswith(
        f"resume-{semantic_key(request).sha256[:24]}-"
    )
    resume = ResumeInput(
        paused.outcome.resume_key,
        ResumeAction.CONTINUE,
        (InteractionResponse("req-1", result={"value": "found"}),),
    )
    completed = client.resume(
        run_root=tmp_path,
        run_id=paused.snapshot.run_id,
        input=resume,
    )
    assert completed.snapshot.status is RunStatus.SUCCEEDED
    assert isinstance(completed.outcome, LLMCompleted)
    assert completed.outcome.value == {"answer": 7}
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 1
    assert json.loads(
        (
                tmp_path
                / "runs"
                / paused.snapshot.run_id
                / "resume-inputs"
                / next(
                    (
                        tmp_path
                        / "runs"
                        / paused.snapshot.run_id
                        / "resume-inputs"
                    ).iterdir()
                ).name
        ).read_text()
    )["input"]["resume_key"] == paused.outcome.resume_key


def test_adoption_proves_source_identity_and_requires_cross_semantic_authorization(
    tmp_path: Path, adapter, registry
) -> None:
    adapter.steps.append(_completed({"answer": 42}))
    client = LLMClient(registry=registry)
    source_request = _request("source")
    source = client.generate(
        source_request,
        run_root=tmp_path,
        run_id="source-run",
    )
    assert source.snapshot.result_ref is not None
    source_ref = ArtifactSourceRef(
        source.snapshot.run_id,
        source.snapshot.result_ref.artifact_id,
        source.snapshot.result_ref.digest,
    )

    same = client.adopt(
        source_request,
        source_ref,
        run_root=tmp_path,
        run_id="same-adoption",
    )
    assert isinstance(same.outcome, LLMCompleted)
    assert same.outcome.provider is None
    assert adapter.start_calls == 1

    target_request = LLMRequest(
        "target",
        source_request.prompt,
        source_request.output,
        source_request.model,
    )
    denied = client.adopt(
        target_request,
        source_ref,
        run_root=tmp_path,
        run_id="denied-adoption",
    )
    assert isinstance(denied.outcome, LLMFailed)
    assert denied.outcome.error.code.value == "adoption_not_authorized"

    wrong = client.adopt(
        target_request,
        source_ref,
        run_root=tmp_path,
        run_id="wrong-adoption",
        authorization=AdoptionAuthorization(
            semantic_key(source_request),
            SemanticKeyDigest("0" * 64),
            "Reviewed reuse",
        ),
    )
    assert isinstance(wrong.outcome, LLMFailed)
    assert wrong.outcome.error.code.value == "adoption_not_authorized"

    accepted = client.adopt(
        target_request,
        source_ref,
        run_root=tmp_path,
        run_id="authorized-adoption",
        authorization=AdoptionAuthorization(
            semantic_key(source_request),
            semantic_key(target_request),
            "Reviewed reuse",
        ),
    )
    assert isinstance(accepted.outcome, LLMCompleted)
    assert accepted.outcome.value == {"answer": 42}


def test_validated_artifact_and_session_record_commit_acceptance_before_native_resume(
    tmp_path: Path, adapter, registry
) -> None:
    adapter.steps.extend(
        [
            _completed({"answer": 1}, handle="thread"),
            _completed({"answer": 2}, handle="thread"),
        ]
    )
    repository = RunRepository(tmp_path)
    snapshot = repository.create(RunSpec("parent", "test.parent", {"case": "session"}))
    context = RunContext(
        repository,
        snapshot,
        resume_input=None,
        execution_slice=None,
    )
    service = LLMTaskService(registry=registry)
    first = service.execute(context, _request("first"))
    assert isinstance(first, LLMCompleted)
    assert first.session is not None
    second_request = LLMRequest(
        "second",
        "Return another object.",
        _request().output,
        ModelSelection("codex"),
        session=first.session,
    )
    second = service.execute(context, second_request)
    assert isinstance(second, LLMCompleted)
    assert second.value == {"answer": 2}
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 1
    assert second.session is not None
    assert second.session.accepted_prefix_sha256 != first.session.accepted_prefix_sha256
    replayed_first = service.execute(context, _request("first"))
    assert isinstance(replayed_first, LLMCompleted)
    assert replayed_first.session == first.session
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 1
    session = service._executor._session_store(context, "first").read()
    assert session is not None
    assert session.accepted_turns == 2
    assert len(session.accepted_turn_records) == 2


def test_session_commit_closes_hard_crash_window_before_task_accepted_cas(
    tmp_path: Path, adapter, registry, monkeypatch
) -> None:
    adapter.steps.extend(
        [
            _completed({"answer": 1}, handle="thread"),
            _completed({"answer": 2}, handle="thread"),
        ]
    )
    repository = RunRepository(tmp_path)
    snapshot = repository.create(
        RunSpec("parent", "test.parent", {"case": "session-repair"})
    )
    context = RunContext(repository, snapshot, resume_input=None, execution_slice=None)
    service = LLMTaskService(registry=registry)
    executor = service._executor
    root = service.execute(context, _request("root-before-crash"))
    assert isinstance(root, LLMCompleted)
    assert root.session is not None

    original = executor._advance_session
    calls = 0

    def crash_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        committed_session = original(*args, **kwargs)
        if calls == 1:
            raise OSError("simulated crash after session acceptance commit")
        return committed_session

    monkeypatch.setattr(executor, "_advance_session", crash_once)
    crashing_request = LLMRequest(
        "crash-after-session-commit",
        "Return another object.",
        _request().output,
        ModelSelection("codex"),
        session=root.session,
    )
    failed = service.execute(context, crashing_request)
    assert isinstance(failed, LLMFailed)
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 1

    task = executor._task_store(context, crashing_request.task_id).read()
    assert task is not None
    assert task.accepted is None
    session_after_crash = executor._session_store(
        context, root.session.session_key
    ).read()
    assert session_after_crash is not None
    assert session_after_crash.accepted_turns == 2

    sibling = service.execute(
        context,
        LLMRequest(
            "old-prefix-sibling",
            "Return a sibling object.",
            _request().output,
            ModelSelection("codex"),
            session=root.session,
        ),
    )
    assert isinstance(sibling, LLMFailed)
    assert sibling.error.code.value == "invalid_request"
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 1

    replayed = service.execute(context, crashing_request)
    assert isinstance(replayed, LLMCompleted)
    assert replayed.session is not None
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 1

    replayed_again = service.execute(context, crashing_request)
    assert isinstance(replayed_again, LLMCompleted)
    assert replayed_again.session == replayed.session
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 1
    session = executor._session_store(context, root.session.session_key).read()
    assert session is not None
    assert session.accepted_turns == 2
    task = executor._task_store(context, crashing_request.task_id).read()
    assert task is not None
    assert task.accepted is not None
    effect = context.effects.read(task.current.effect_id)
    assert effect is not None
    assert effect.stage is EffectStage.COMMITTED


def test_interaction_limit_pause_carries_request_and_can_resume(
    tmp_path: Path, adapter, registry
) -> None:
    contract = InteractiveJsonOutput(
        {"type": "object", "required": ["answer"]},
        {
            "lookup": OperationContract(
                {"type": "object", "required": ["query"]},
                {"type": "object", "required": ["value"]},
            )
        },
        max_interaction_turns=1,
    )
    adapter.steps.extend(
        [
            _completed(
                {
                    "schema_version": "arc.llm.interactive_turn.v1",
                    "state": "interact",
                    "result": None,
                    "requests": [
                        {
                            "request_id": "req-limit",
                            "operation": "lookup",
                            "arguments": {"query": "x"},
                        }
                    ],
                }
            ),
            _completed(
                {
                    "schema_version": "arc.llm.interactive_turn.v1",
                    "state": "complete",
                    "result": {"answer": 8},
                    "requests": [],
                }
            ),
        ]
    )
    client = LLMClient(registry=registry)
    paused = client.generate(
        LLMRequest("interaction-limit", "Solve.", contract, ModelSelection("codex")),
        run_root=tmp_path,
    )
    assert isinstance(paused.outcome, LLMPaused)
    assert paused.outcome.reason.value == "execution_budget_exhausted"
    assert paused.outcome.input_required
    assert paused.outcome.request_ref is not None
    assert paused.outcome.response_contract == "arc.llm.resume_input.v1"

    resumed = client.resume(
        run_root=tmp_path,
        run_id=paused.snapshot.run_id,
        input=ResumeInput(
            paused.outcome.resume_key,
            ResumeAction.CONTINUE,
            (InteractionResponse("req-limit", result={"value": "found"}),),
        ),
    )
    assert isinstance(resumed.outcome, LLMCompleted)
    assert resumed.outcome.value == {"answer": 8}
    assert adapter.resume_calls == 1


def test_uncertain_delivery_with_saved_handle_uses_one_native_resume(
    tmp_path: Path, adapter, registry
) -> None:
    def uncertain_start(request, observer, cancel):
        adapter.start_calls += 1
        observer.before_delivery()
        observer.native_handle(NativeResumeHandle("codex", "recoverable-thread"))
        raise ProviderFailure(
            "transport disconnected",
            category=FailureCategory.TRANSPORT,
            delivery=DeliveryState.MAY_HAVE_RUN,
        )

    adapter.start = uncertain_start
    adapter.steps.append(_completed({"answer": 9}, handle="recoverable-thread"))
    result = LLMClient(registry=registry).generate(_request(), run_root=tmp_path)
    assert isinstance(result.outcome, LLMCompleted)
    assert result.outcome.value == {"answer": 9}
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 1


def test_stale_resume_key_fails_before_resolver_or_provider(
    tmp_path: Path, adapter, registry
) -> None:
    contract = InteractiveJsonOutput(
        {"type": "object"},
        {
            "lookup": OperationContract(
                {"type": "object"},
                {"type": "object"},
            )
        },
    )
    adapter.steps.append(
        _completed(
            {
                "schema_version": "arc.llm.interactive_turn.v1",
                "state": "interact",
                "result": None,
                "requests": [
                    {
                        "request_id": "req-1",
                        "operation": "lookup",
                        "arguments": {},
                    }
                ],
            }
        )
    )
    client = LLMClient(registry=registry)
    paused = client.generate(
        LLMRequest("stale", "p", contract, ModelSelection("codex")),
        run_root=tmp_path,
    )
    before = adapter.resume_calls
    stale = client.resume(
        run_root=tmp_path,
        run_id=paused.snapshot.run_id,
        input=ResumeInput(
            "resume-stale",
            ResumeAction.CONTINUE,
            (InteractionResponse("req-1", result={}),),
        ),
    )
    assert isinstance(stale.outcome, LLMFailed)
    assert stale.outcome.error.code.value == "resume_key_mismatch"
    assert adapter.resume_calls == before
