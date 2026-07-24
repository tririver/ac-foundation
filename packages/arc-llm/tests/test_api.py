from __future__ import annotations

import json
from pathlib import Path

from arc_jobs import ArtifactSourceRef, RunContext, RunRepository, RunSpec, RunStatus
from arc_llm import (
    AdoptionAuthorization,
    InteractiveJsonOutput,
    InteractionResponse,
    JsonOutput,
    LLMClient,
    LLMCompleted,
    LLMFailed,
    LLMPaused,
    LLMRequest,
    LLMTaskService,
    ModelSelection,
    NativeResumeHandle,
    OperationContract,
    ProviderExecution,
    ProviderFailure,
    ProviderTerminalKind,
    DeliveryState,
    FailureCategory,
    ResumeAction,
    ResumeInput,
    SemanticKeyDigest,
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


def test_session_advances_only_after_acceptance_and_next_task_uses_native_resume(
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
