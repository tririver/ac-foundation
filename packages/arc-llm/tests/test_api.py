from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import threading

import pytest

import arc_llm.api as api_module
from arc_jobs import (
    ArtifactDigest,
    ArtifactSourceRef,
    RunContext,
    RunEngine,
    RunRepository,
    ResumeReason,
    RunSpec,
    RunStatus,
    StoppedError,
)
from arc_jobs import EffectStage
from arc_jobs.lease import FileLease
from arc_llm import (
    AdoptionAuthorization,
    ExecutionLimits,
    HostAuthority,
    InvalidRequestError,
    JsonOutput,
    LLMClient,
    LLMCompleted,
    LLMExecutionOptions,
    LLMFailed,
    LLMInputArtifact,
    LLMPaused,
    LLMRequest,
    LLMStopped,
    LLMTaskService,
    ModelSelection,
    NativeResumeHandle,
    ProviderExecution,
    ProviderFailure,
    ProviderDiagnostic,
    ProviderGateOptions,
    ProviderTerminalKind,
    ProviderUsage,
    DeliveryState,
    FailureCategory,
    ResumeAction,
    ResumeInput,
    SemanticKeyDigest,
    request_to_document,
    resume_input_matches,
)
from arc_llm.identity import canonical_json_bytes, semantic_key
from arc_llm.output import CandidateMaterial
from arc_llm.executor import HANDLER_NAME, LLMTaskExecutor
from arc_llm.recovery import effect_id_for


def _request(task_id: str = "task", *, repair: str = "format") -> LLMRequest:
    return LLMRequest(
        task_id,
        "Return an object.",
        JsonOutput({"type": "object", "required": ["answer"]}, repair=repair),
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


def test_provider_prompt_uses_workspace_control(
    tmp_path: Path, adapter, registry
) -> None:
    prompt = "Exact prompt bytes.\nKeep this spacing.  "
    adapter.steps.append(_completed({"answer": 42}))
    result = LLMClient(registry=registry).generate(
        LLMRequest(
            "exact-provider-prompt",
            prompt,
            JsonOutput({"type": "object", "required": ["answer"]}),
            ModelSelection("codex"),
        ),
        run_root=tmp_path,
    )
    assert isinstance(result.outcome, LLMCompleted)
    provider_request = adapter.requests[0]
    assert provider_request.prompt == LLMTaskExecutor._workspace_prompt()
    control = json.loads(
        (provider_request.workspace / "host" / "control.json").read_text()
    )
    assert control["prompt"] == prompt
    assert control["inputs"] == []


def test_replay_commit_checkpoint_returns_stopped_outcome(
    tmp_path: Path, adapter, registry, monkeypatch
) -> None:
    adapter.steps.append(_completed({"answer": 42}))
    repository = RunRepository(tmp_path)
    snapshot = repository.create(
        RunSpec("replay-stop", "test.parent", {"case": "replay-stop"})
    )
    context = RunContext(
        repository,
        snapshot,
        resume_input=None,
        execution_slice=None,
    )
    service = LLMTaskService(registry=registry)
    first = service.execute(context, _request())
    assert isinstance(first, LLMCompleted)

    def stop_at_commit(effect_id: str) -> None:
        raise StoppedError(f"stopped before committing {effect_id}")

    monkeypatch.setattr(context.effects, "commit", stop_at_commit)
    replayed = service.execute(context, _request())

    assert isinstance(replayed, LLMStopped)
    assert adapter.start_calls == 1


@pytest.mark.parametrize(
    ("replacement_limit", "replaces"),
    (
        (1, True),
        (0, False),
    ),
)
def test_live_invalid_output_replacement_or_pause_matrix(
    tmp_path: Path,
    adapter,
    registry,
    replacement_limit: int,
    replaces: bool,
) -> None:
    adapter.steps.append(_completed({"not_answer": True}))
    if replacement_limit:
        adapter.steps.append(_completed({"answer": 42}, handle="replacement"))

    result = LLMClient(registry=registry).generate(
        _request("live-invalid", repair="local"),
        run_root=tmp_path,
        options=LLMExecutionOptions(
            limits=ExecutionLimits(automatic_replacement_limit=replacement_limit)
        ),
    )

    if replaces:
        assert isinstance(result.outcome, LLMCompleted)
        assert result.outcome.value == {"answer": 42}
        assert adapter.start_calls == 2
    else:
        assert isinstance(result.outcome, LLMPaused)
        assert result.outcome.details["code"] == "output_invalid"
        assert adapter.start_calls == 1


def test_content_rich_invalid_output_uses_formatter_without_worker_replacement(
    tmp_path: Path, adapter, registry
) -> None:
    adapter.steps.extend(
        (
            _completed({"answer_text": "the complete answer is present"}),
            replace(
                _completed(
                    {
                        "action": "format",
                        "reason": "required content is present",
                        "formatted_output": {
                            "answer": "the complete answer is present"
                        },
                    },
                    handle="formatter",
                ),
                usage=ProviderUsage(11, 7, 3),
            ),
        )
    )

    result = LLMClient(registry=registry).generate(
        _request("format-rich"),
        run_root=tmp_path,
        options=LLMExecutionOptions(
            limits=ExecutionLimits(automatic_replacement_limit=3)
        ),
    )

    assert isinstance(result.outcome, LLMCompleted)
    assert result.outcome.value == {"answer": "the complete answer is present"}
    assert adapter.start_calls == 2
    formatter_request = adapter.requests[1]
    assert formatter_request.model == adapter.requests[0].model
    assert formatter_request.capabilities["internet"] is True
    assert formatter_request.capabilities["effective_host_mode"] == "brokered"
    assert "inherit_host_config" not in formatter_request.capabilities
    assert "allowed_tools" not in formatter_request.capabilities
    records = []
    for path in tmp_path.rglob("*"):
        if not path.is_file() or path.name.endswith(".lock"):
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (
            isinstance(document, dict)
            and document.get("schema_version") == "arc.llm.output_formatting.v1"
        ):
            records.append(document)
    assert len(records) == 1
    record = records[0]
    assert record["status"] == "formatted"
    assert record["child_accepted_ref"] is not None
    assert record["formatter_usage"] == {
        "input_tokens": 11,
        "output_tokens": 7,
        "cached_input_tokens": 3,
    }


def test_formatter_insufficient_allows_worker_replacement(
    tmp_path: Path, adapter, registry
) -> None:
    adapter.steps.extend(
        (
            _completed({"partial_answer": "substantive but incomplete material"}),
            _completed(
                {
                    "action": "insufficient",
                    "reason": "required answer is absent",
                    "formatted_output": None,
                },
                handle="formatter",
            ),
            _completed({"answer": "replacement answer"}, handle="replacement"),
        )
    )

    result = LLMClient(registry=registry).generate(
        _request("format-insufficient"),
        run_root=tmp_path,
    )

    assert isinstance(result.outcome, LLMCompleted)
    assert result.outcome.value == {"answer": "replacement answer"}
    assert adapter.start_calls == 3


def test_invalid_formatter_result_pauses_without_worker_replacement(
    tmp_path: Path, adapter, registry
) -> None:
    adapter.steps.extend(
        (
            _completed({"answer_text": "the complete answer is present"}),
            _completed(
                {
                    "action": "format",
                    "reason": "bad formatting",
                    "formatted_output": {"wrong": "shape"},
                },
                handle="formatter",
            ),
        )
    )

    result = LLMClient(registry=registry).generate(
        _request("format-invalid"),
        run_root=tmp_path,
        options=LLMExecutionOptions(
            limits=ExecutionLimits(automatic_replacement_limit=3)
        ),
    )

    assert isinstance(result.outcome, LLMPaused)
    assert result.outcome.details["code"] == "output_formatting_failed"
    assert adapter.start_calls == 2


def test_malformed_formatter_envelope_pauses_without_worker_replacement(
    tmp_path: Path, adapter, registry
) -> None:
    adapter.steps.extend(
        (
            _completed({"answer_text": "the complete answer is present"}),
            _completed({"not_a_formatter_decision": True}, handle="formatter"),
        )
    )

    result = LLMClient(registry=registry).generate(
        _request("format-envelope-invalid"),
        run_root=tmp_path,
        options=LLMExecutionOptions(
            limits=ExecutionLimits(automatic_replacement_limit=3)
        ),
    )

    assert isinstance(result.outcome, LLMPaused)
    assert result.outcome.details["code"] == "output_formatting_failed"
    assert adapter.start_calls == 2


def test_formatter_stop_propagates_without_worker_replacement(
    tmp_path: Path, adapter, registry
) -> None:
    adapter.steps.extend(
        (
            _completed({"answer_text": "the complete answer is present"}),
            ProviderExecution(ProviderTerminalKind.STOPPED),
        )
    )

    result = LLMClient(registry=registry).generate(
        _request("format-stopped"),
        run_root=tmp_path,
        options=LLMExecutionOptions(
            limits=ExecutionLimits(automatic_replacement_limit=3)
        ),
    )

    assert isinstance(result.outcome, LLMStopped)
    assert adapter.start_calls == 2


def test_formatter_external_pause_resumes_without_replaying_worker(
    tmp_path: Path, adapter, registry, monkeypatch
) -> None:
    available = True

    def doctor():
        return ProviderDiagnostic("codex", available, "fake-codex")

    monkeypatch.setattr(adapter, "doctor", doctor)
    adapter.steps.append(
        _completed({"answer_text": "the complete answer is present"})
    )
    client = LLMClient(registry=registry)

    available = False
    # Admit the original call before making only the formatter unavailable.
    original_doctor = adapter.doctor
    doctor_calls = 0

    def staged_doctor():
        nonlocal doctor_calls
        doctor_calls += 1
        if doctor_calls == 1:
            return ProviderDiagnostic("codex", True, "fake-codex")
        return original_doctor()

    monkeypatch.setattr(adapter, "doctor", staged_doctor)
    paused = client.generate(
        _request("format-external-pause"),
        run_root=tmp_path,
        run_id="format-external-pause-run",
    )

    assert isinstance(paused.outcome, LLMPaused)
    assert paused.outcome.reason is ResumeReason.EXTERNAL_CONDITION
    assert not paused.outcome.input_required
    assert adapter.start_calls == 1

    available = True
    adapter.steps.append(
        _completed(
            {
                "action": "format",
                "reason": "required content is present",
                "formatted_output": {"answer": "the complete answer is present"},
            },
            handle="formatter",
        )
    )
    resumed = client.resume(
        run_root=tmp_path,
        run_id="format-external-pause-run",
    )

    assert isinstance(resumed.outcome, LLMCompleted)
    assert resumed.outcome.value == {"answer": "the complete answer is present"}
    assert adapter.start_calls == 2


def test_formatter_completion_replays_after_outer_acceptance_crash(
    tmp_path: Path, adapter, registry, monkeypatch
) -> None:
    adapter.steps.extend(
        (
            _completed({"answer_text": "the complete answer is present"}),
            _completed(
                {
                    "action": "format",
                    "reason": "required content is present",
                    "formatted_output": {"answer": "the complete answer is present"},
                },
                handle="formatter",
            ),
        )
    )
    client = LLMClient(registry=registry)
    executor = client.service._executor
    original_accept = executor._accept

    def crash_before_outer_accept(*args, **kwargs):
        raise KeyboardInterrupt("simulated crash after formatter completion")

    monkeypatch.setattr(executor, "_accept", crash_before_outer_accept)
    with pytest.raises(KeyboardInterrupt):
        client.generate(
            _request("format-replay"),
            run_root=tmp_path,
            run_id="format-replay-run",
        )
    assert adapter.start_calls == 2

    monkeypatch.setattr(executor, "_accept", original_accept)
    resumed = client.resume(run_root=tmp_path, run_id="format-replay-run")

    assert isinstance(resumed.outcome, LLMCompleted)
    assert resumed.outcome.value == {"answer": "the complete answer is present"}
    assert adapter.start_calls == 2


@pytest.mark.parametrize(
    ("case", "execution", "llm_request", "reason", "code"),
    (
        (
            "conflict",
            ProviderExecution(
                ProviderTerminalKind.COMPLETED,
                (
                    CandidateMaterial(value={"answer": 1}),
                    CandidateMaterial(value={"answer": 2}),
                ),
                NativeResumeHandle("codex", "saved-conflict"),
            ),
            _request("saved-conflict"),
            ResumeReason.SUPERVISION_REQUIRED,
            "candidate_selection_required",
        ),
        (
            "invalid",
            _completed({"not_answer": True}, handle="saved-invalid"),
            _request("saved-invalid", repair="local"),
            ResumeReason.SUPERVISION_REQUIRED,
            "output_invalid",
        ),
    ),
)
def test_saved_output_recovery_pauses_without_provider_replay(
    tmp_path: Path,
    adapter,
    registry,
    monkeypatch,
    case: str,
    execution: ProviderExecution,
    llm_request: LLMRequest,
    reason: ResumeReason,
    code: str,
) -> None:
    adapter.steps.append(execution)
    client = LLMClient(registry=registry)
    executor = client.service._executor
    original_consume_candidates = executor._consume_candidates

    def crash_after_raw_output(*args, **kwargs):
        raise KeyboardInterrupt("simulated crash after saving raw provider output")

    monkeypatch.setattr(executor, "_consume_candidates", crash_after_raw_output)
    run_id = f"saved-output-{case}"
    with pytest.raises(KeyboardInterrupt):
        client.generate(
            llm_request,
            run_root=tmp_path,
            run_id=run_id,
            options=LLMExecutionOptions(
                limits=ExecutionLimits(automatic_replacement_limit=3)
            ),
        )

    repository = RunRepository(tmp_path)
    snapshot = repository.inspect(run_id).snapshot
    context = RunContext(repository, snapshot, resume_input=None, execution_slice=None)
    state = executor._task_store(context, llm_request.task_id).read()
    assert state is not None
    effect = context.effects.read(state.current.effect_id)
    assert effect is not None
    assert effect.stage is EffectStage.OUTPUT_SAVED

    monkeypatch.setattr(executor, "_consume_candidates", original_consume_candidates)
    recovered = client.resume(
        run_root=tmp_path,
        run_id=run_id,
        options=LLMExecutionOptions(
            limits=ExecutionLimits(automatic_replacement_limit=3)
        ),
    )

    assert isinstance(recovered.outcome, LLMPaused)
    assert recovered.outcome.reason is reason
    assert recovered.outcome.details["code"] == code
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 0


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


def test_input_artifact_is_verified_and_copied_to_workspace_before_provider_call(
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
    workspace = adapter.requests[0].workspace
    control = json.loads((workspace / "host" / "control.json").read_text())
    assert control["inputs"] == [
        {
            "input_id": "paper",
            "media_type": "text/markdown",
            "sha256": input_artifact.source.expected_digest.value,
            "size_bytes": len(b"# paper\n"),
            "path": "inputs/0000-paper.md",
        }
    ]
    assert (workspace / control["inputs"][0]["path"]).read_bytes() == b"# paper\n"
    assert all(not Path(item).is_absolute() for item in (control["inputs"][0]["path"], control["work_directory"]))
    object_paths = (
        path
        for path in (tmp_path / "runs" / "parent" / "artifacts" / "objects").rglob("*")
        if path.is_file()
    )
    recipe = next(path for path in object_paths if b'"input_transport"' in path.read_bytes())
    assert b'"input_transport":"workspace_control.v1"' in recipe.read_bytes()


def test_relocated_input_source_conflicts_with_immutable_standalone_invocation(
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
    assert isinstance(second.outcome, LLMFailed)
    assert second.outcome.error.code.value == "idempotency_conflict"
    assert semantic_key(first_request) == semantic_key(second_request)
    assert adapter.start_calls == 1


def test_new_standalone_run_writes_only_fixed_closed_invocation_v2(
    tmp_path: Path,
    adapter,
    registry,
) -> None:
    adapter.steps.append(_completed({"answer": 42}))
    request = _request("fixed-invocation")

    result = LLMClient(registry=registry).generate(
        request,
        run_root=tmp_path,
        run_id="llm-run",
    )

    assert isinstance(result.outcome, LLMCompleted)
    invocation_path = (
        tmp_path
        / "runs"
        / "llm-run"
        / "state"
        / "llm-standalone-invocation.json"
    )
    document = json.loads(invocation_path.read_text(encoding="utf-8"))
    assert set(document) == {
        "schema_version",
        "contract_schema_version",
        "revision",
        "value",
    }
    assert document["schema_version"] == "arc.jobs.state.v1"
    assert (
        document["contract_schema_version"]
        == "arc.llm.standalone_invocation.v2"
    )
    assert document["revision"] == 0
    assert set(document["value"]) == {
        "revision",
        "mode",
        "request",
        "adoption",
    }
    assert document["value"]["revision"] == 0
    assert document["value"]["mode"] == "generate"
    assert document["value"]["request"] == request_to_document(request)
    assert document["value"]["adoption"] is None
    assert not list(invocation_path.parent.glob("llm-bootstrap-*.json"))


def test_provider_runs_after_standalone_invocation_lock_is_released(
    tmp_path: Path,
    adapter,
    registry,
) -> None:
    invocation_path = (
        tmp_path
        / "runs"
        / "llm-run"
        / "state"
        / "llm-standalone-invocation.json"
    )
    original_start = adapter.start

    def start(request, observer, stop):
        FileLease(
            invocation_path.with_suffix(
                f"{invocation_path.suffix}.lock"
            )
        ).acquire().release()
        return original_start(request, observer, stop)

    adapter.start = start
    adapter.steps.append(_completed({"answer": 42}))

    result = LLMClient(registry=registry).generate(
        _request("released-invocation-lock"),
        run_root=tmp_path,
        run_id="llm-run",
    )

    assert isinstance(result.outcome, LLMCompleted)
    assert adapter.start_calls == 1


def test_conflicting_request_does_not_persist_into_existing_invocation(
    tmp_path: Path,
    adapter,
    registry,
) -> None:
    adapter.steps.append(_completed({"answer": 42}))
    client = LLMClient(registry=registry)
    original = _request("bound-task")
    conflicting = replace(original, prompt="A prompt that must not be persisted.")

    first = client.generate(original, run_root=tmp_path, run_id="llm-run")
    second = client.generate(
        conflicting,
        run_root=tmp_path,
        run_id="llm-run",
    )

    assert isinstance(first.outcome, LLMCompleted)
    assert isinstance(second.outcome, LLMFailed)
    assert second.outcome.error.code.value == "idempotency_conflict"
    invocation_path = (
        tmp_path
        / "runs"
        / "llm-run"
        / "state"
        / "llm-standalone-invocation.json"
    )
    invocation_bytes = invocation_path.read_bytes()
    assert original.prompt.encode() in invocation_bytes
    assert conflicting.prompt.encode() not in invocation_bytes
    assert not list(invocation_path.parent.glob("llm-bootstrap-*.json"))
    assert adapter.start_calls == 1


def test_closed_invocation_bound_to_different_spec_fails_as_corrupt(
    tmp_path: Path,
    adapter,
    registry,
) -> None:
    adapter.steps.append(_completed({"answer": 42}))
    client = LLMClient(registry=registry)
    original = _request("bound-task")
    completed = client.generate(
        original,
        run_root=tmp_path,
        run_id="llm-run",
    )
    assert isinstance(completed.outcome, LLMCompleted)
    invocation_path = (
        tmp_path
        / "runs"
        / "llm-run"
        / "state"
        / "llm-standalone-invocation.json"
    )
    document = json.loads(invocation_path.read_text(encoding="utf-8"))
    document["value"]["request"]["prompt"] = "Wrong durable prompt."
    invocation_path.write_text(json.dumps(document), encoding="utf-8")

    result = client.generate(
        replace(original, prompt="Caller conflict."),
        run_root=tmp_path,
        run_id="llm-run",
    )

    assert isinstance(result.outcome, LLMFailed)
    assert result.outcome.error.code.value == "corrupt_state"
    assert (
        result.outcome.error.details["code"]
        == "standalone_invocation_corrupt"
    )
    assert adapter.start_calls == 1


def test_concurrent_conflicting_invocations_publish_one_lineage(
    tmp_path: Path,
    adapter,
    registry,
) -> None:
    adapter.steps.append(_completed({"answer": 42}))
    client = LLMClient(registry=registry)
    requests = (
        _request("concurrent-task"),
        replace(
            _request("concurrent-task"),
            prompt="Concurrent conflicting prompt.",
        ),
    )
    ready = threading.Barrier(2)

    def invoke(request):
        ready.wait(timeout=5)
        return request, client.generate(
            request,
            run_root=tmp_path,
            run_id="llm-run",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(invoke, requests))

    completed = [
        (request, result)
        for request, result in results
        if isinstance(result.outcome, LLMCompleted)
    ]
    conflicts = [
        result
        for _, result in results
        if isinstance(result.outcome, LLMFailed)
    ]
    assert len(completed) == 1
    assert len(conflicts) == 1
    assert conflicts[0].outcome.error.code.value == "idempotency_conflict"
    invocation_path = (
        tmp_path
        / "runs"
        / "llm-run"
        / "state"
        / "llm-standalone-invocation.json"
    )
    invocation = json.loads(
        invocation_path.read_text(encoding="utf-8")
    )["value"]
    assert invocation["request"] == request_to_document(completed[0][0])
    assert adapter.start_calls == 1


def test_standalone_adopt_invocation_recovers_source_and_authorization(
    tmp_path: Path,
    adapter,
    registry,
    monkeypatch,
) -> None:
    adapter.steps.append(_completed({"answer": 42}))
    client = LLMClient(registry=registry)
    source_request = _request("source")
    source_result = client.generate(
        source_request,
        run_root=tmp_path,
        run_id="source-run",
    )
    assert source_result.snapshot.result_ref is not None
    source = ArtifactSourceRef(
        "source-run",
        source_result.snapshot.result_ref.artifact_id,
        source_result.snapshot.result_ref.digest,
    )
    target_request = replace(source_request, task_id="target")
    authorization = AdoptionAuthorization(
        semantic_key(source_request),
        semantic_key(target_request),
        "Reviewed recovery reuse",
    )
    original_adopt = client.service.adopt_and_revalidate

    def crash_before_task_state(*args, **kwargs):
        raise KeyboardInterrupt("simulated adoption crash")

    monkeypatch.setattr(
        client.service,
        "adopt_and_revalidate",
        crash_before_task_state,
    )
    with pytest.raises(KeyboardInterrupt, match="adoption crash"):
        client.adopt(
            target_request,
            source,
            run_root=tmp_path,
            run_id="adopt-run",
            authorization=authorization,
        )

    invocation_path = (
        tmp_path
        / "runs"
        / "adopt-run"
        / "state"
        / "llm-standalone-invocation.json"
    )
    invocation = json.loads(
        invocation_path.read_text(encoding="utf-8")
    )["value"]
    assert invocation["mode"] == "adopt"
    assert invocation["adoption"] == {
        "source_run_id": source.source_run_id,
        "source_artifact_id": source.source_artifact_id,
        "expected_digest": {
            "algorithm": "sha256",
            "value": source.expected_digest.value,
            "size_bytes": source.expected_digest.size_bytes,
        },
        "authorization": {
            "source_semantic_key_sha256": (
                authorization.source_semantic_key.sha256
            ),
            "target_semantic_key_sha256": (
                authorization.target_semantic_key.sha256
            ),
            "reason": authorization.reason,
        },
    }

    monkeypatch.setattr(
        client.service,
        "adopt_and_revalidate",
        original_adopt,
    )
    resumed = client.resume(run_root=tmp_path, run_id="adopt-run")

    assert resumed.snapshot.status is RunStatus.SUCCEEDED
    assert isinstance(resumed.outcome, LLMCompleted)
    assert resumed.outcome.value == {"answer": 42}
    assert adapter.start_calls == 1


def test_standalone_mode_change_returns_typed_conflict(
    tmp_path: Path,
    adapter,
    registry,
) -> None:
    adapter.steps.append(_completed({"answer": 42}))
    client = LLMClient(registry=registry)
    request = _request("mode-conflict")
    generated = client.generate(
        request,
        run_root=tmp_path,
        run_id="llm-run",
    )
    assert generated.snapshot.result_ref is not None
    source = ArtifactSourceRef(
        generated.snapshot.run_id,
        generated.snapshot.result_ref.artifact_id,
        generated.snapshot.result_ref.digest,
    )

    conflict = client.adopt(
        request,
        source,
        run_root=tmp_path,
        run_id="llm-run",
    )

    assert isinstance(conflict.outcome, LLMFailed)
    assert conflict.outcome.error.code.value == "idempotency_conflict"
    assert adapter.start_calls == 1


def test_standalone_adoption_source_and_authorization_are_immutable(
    tmp_path: Path,
    adapter,
    registry,
) -> None:
    adapter.steps.extend(
        (
            _completed({"answer": 1}, handle="source-1"),
            _completed({"answer": 2}, handle="source-2"),
        )
    )
    client = LLMClient(registry=registry)
    first_request = _request("source-1")
    second_request = _request("source-2")
    first_result = client.generate(
        first_request,
        run_root=tmp_path,
        run_id="source-run-1",
    )
    second_result = client.generate(
        second_request,
        run_root=tmp_path,
        run_id="source-run-2",
    )
    assert first_result.snapshot.result_ref is not None
    assert second_result.snapshot.result_ref is not None
    first_source = ArtifactSourceRef(
        first_result.snapshot.run_id,
        first_result.snapshot.result_ref.artifact_id,
        first_result.snapshot.result_ref.digest,
    )
    second_source = ArtifactSourceRef(
        second_result.snapshot.run_id,
        second_result.snapshot.result_ref.artifact_id,
        second_result.snapshot.result_ref.digest,
    )
    target_request = _request("target")
    first_authorization = AdoptionAuthorization(
        semantic_key(first_request),
        semantic_key(target_request),
        "First reviewed reuse",
    )
    adopted = client.adopt(
        target_request,
        first_source,
        run_root=tmp_path,
        run_id="adopt-run",
        authorization=first_authorization,
    )

    source_conflict = client.adopt(
        target_request,
        second_source,
        run_root=tmp_path,
        run_id="adopt-run",
        authorization=AdoptionAuthorization(
            semantic_key(second_request),
            semantic_key(target_request),
            "Second reviewed reuse",
        ),
    )
    authorization_conflict = client.adopt(
        target_request,
        first_source,
        run_root=tmp_path,
        run_id="adopt-run",
        authorization=AdoptionAuthorization(
            semantic_key(first_request),
            semantic_key(target_request),
            "Changed reason",
        ),
    )

    assert isinstance(adopted.outcome, LLMCompleted)
    for conflict in (source_conflict, authorization_conflict):
        assert isinstance(conflict.outcome, LLMFailed)
        assert conflict.outcome.error.code.value == "idempotency_conflict"
    assert adapter.start_calls == 2


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
    control = json.loads(
        (adapter.requests[0].workspace / "host" / "control.json").read_text()
    )
    assert control["inputs"][0]["sha256"] == input_artifact.source.expected_digest.value


def test_standalone_run_without_task_or_invocation_fails_typed_recovery(
    tmp_path: Path,
    adapter,
    registry,
) -> None:
    request = _request("missing-invocation")
    repository = RunRepository(tmp_path)

    class CrashHandler:
        name = HANDLER_NAME

        def execute(self, context):
            raise KeyboardInterrupt("missing invocation fixture")

    with pytest.raises(KeyboardInterrupt, match="missing invocation fixture"):
        RunEngine(repository).execute(
            RunSpec(
                "missing-invocation-run",
                HANDLER_NAME,
                api_module.semantic_document(request),
            ),
            CrashHandler(),
        )

    resumed = LLMClient(registry=registry).resume(
        run_root=tmp_path,
        run_id="missing-invocation-run",
    )

    assert resumed.snapshot.status is RunStatus.FAILED
    assert isinstance(resumed.outcome, LLMFailed)
    assert resumed.outcome.error.code.value == "corrupt_state"
    assert (
        resumed.outcome.error.details["code"]
        == "standalone_invocation_missing"
    )
    repeated = LLMClient(registry=registry).resume(
        run_root=tmp_path,
        run_id="missing-invocation-run",
    )
    assert repeated.snapshot.status is RunStatus.FAILED
    assert isinstance(repeated.outcome, LLMFailed)
    assert repeated.outcome.error.code.value == "corrupt_state"
    assert (
        repeated.outcome.error.details["code"]
        == "standalone_invocation_missing"
    )
    assert adapter.start_calls == 0


def test_terminal_resume_requires_fixed_invocation_despite_task_state(
    tmp_path: Path,
    adapter,
    registry,
) -> None:
    adapter.steps.append(_completed({"answer": 42}))
    client = LLMClient(registry=registry)
    generated = client.generate(
        _request("missing-terminal-invocation"),
        run_root=tmp_path,
        run_id="missing-terminal-invocation-run",
    )
    assert generated.snapshot.status is RunStatus.SUCCEEDED
    invocation_path = (
        tmp_path
        / "runs"
        / "missing-terminal-invocation-run"
        / "state"
        / "llm-standalone-invocation.json"
    )
    invocation_path.unlink()

    resumed = client.resume(
        run_root=tmp_path,
        run_id="missing-terminal-invocation-run",
    )

    assert resumed.snapshot.status is RunStatus.SUCCEEDED
    assert isinstance(resumed.outcome, LLMFailed)
    assert resumed.outcome.error.code.value == "corrupt_state"
    assert (
        resumed.outcome.error.details["code"]
        == "standalone_invocation_missing"
    )
    assert adapter.start_calls == 1


def test_corrupt_standalone_invocation_fails_closed_without_provider(
    tmp_path: Path,
    adapter,
    registry,
    monkeypatch,
) -> None:
    request = _request("corrupt-invocation")
    client = LLMClient(registry=registry)
    original_execute = client.service._executor.execute

    def crash_before_task_state(*args, **kwargs):
        raise KeyboardInterrupt("invocation corruption fixture")

    monkeypatch.setattr(
        client.service._executor,
        "execute",
        crash_before_task_state,
    )
    with pytest.raises(KeyboardInterrupt, match="corruption fixture"):
        client.generate(
            request,
            run_root=tmp_path,
            run_id="corrupt-run",
        )
    invocation_path = (
        tmp_path
        / "runs"
        / "corrupt-run"
        / "state"
        / "llm-standalone-invocation.json"
    )
    document = json.loads(invocation_path.read_text(encoding="utf-8"))
    document["value"]["unknown"] = True
    invocation_path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(
        client.service._executor,
        "execute",
        original_execute,
    )

    resumed = client.resume(run_root=tmp_path, run_id="corrupt-run")

    assert resumed.snapshot.status is RunStatus.FAILED
    assert isinstance(resumed.outcome, LLMFailed)
    assert resumed.outcome.error.code.value == "corrupt_state"
    assert (
        resumed.outcome.error.details["code"]
        == "standalone_invocation_corrupt"
    )
    repeated = client.resume(run_root=tmp_path, run_id="corrupt-run")
    assert repeated.snapshot.status is RunStatus.FAILED
    assert isinstance(repeated.outcome, LLMFailed)
    assert repeated.outcome.error.code.value == "corrupt_state"
    assert (
        repeated.outcome.error.details["code"]
        == "standalone_invocation_corrupt"
    )
    assert adapter.start_calls == 0


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
        (item.workspace / "inputs" / "0000-paper.md").read_bytes() == b"# paper\n"
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


def test_workspace_transport_copies_unknown_media_without_provider_filtering(
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
    adapter.steps.append(_completed({"answer": 42}))
    request = LLMRequest(
        "unsupported-input",
        "Review.",
        _request().output,
        ModelSelection("codex"),
        inputs=(input_artifact,),
    )

    outcome = LLMTaskService(registry=registry).execute(context, request)

    assert isinstance(outcome, LLMCompleted)
    assert adapter.start_calls == 1
    control = json.loads(
        (adapter.requests[0].workspace / "host" / "control.json").read_text()
    )
    assert control["inputs"][0]["path"] == "inputs/0000-paper.bin"
    assert (adapter.requests[0].workspace / "inputs" / "0000-paper.bin").read_bytes() == b"%PDF"


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
    direct = LLMExecutionOptions(host_authority=HostAuthority.UNRESTRICTED)
    provider_entered = threading.Event()
    release_provider = threading.Event()

    def blocking_start(request, observer, stop):
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
        return service.execute(context, request, options=direct)

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
    direct = LLMExecutionOptions(host_authority=HostAuthority.UNRESTRICTED)
    root = service.execute(context, _request("root"), options=direct)
    assert isinstance(root, LLMCompleted)
    assert root.session is not None

    provider_entered = threading.Event()
    release_provider = threading.Event()

    def blocking_resume(handle, request, observer, stop):
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
        return service.execute(context, request, options=direct)

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


@pytest.mark.parametrize("safe_retry_limit", (0, 1))
def test_exhausted_safe_retries_pause_without_replacement_and_manual_resume(
    tmp_path: Path,
    adapter,
    registry,
    safe_retry_limit: int,
) -> None:
    failures = safe_retry_limit + 1
    adapter.steps.extend(
        [
            *(
                ProviderFailure(
                    "request was not delivered",
                    category=FailureCategory.TRANSPORT,
                    delivery=DeliveryState.NOT_DELIVERED,
                )
                for _ in range(failures)
            ),
            _completed({"answer": 42}),
        ]
    )

    def start_before_delivery(request, observer, stop):
        adapter.start_calls += 1
        adapter.requests.append(request)
        result = adapter.steps.popleft()
        if isinstance(result, ProviderFailure):
            raise result
        observer.before_delivery()
        if result.native_handle is not None:
            observer.native_handle(result.native_handle)
        return result

    adapter.start = start_before_delivery
    client = LLMClient(registry=registry)
    request = _request(f"safe-retry-{safe_retry_limit}")
    options = LLMExecutionOptions(
        limits=ExecutionLimits(safe_retry_limit=safe_retry_limit),
        gate=ProviderGateOptions(enabled=False),
        host_authority=HostAuthority.UNRESTRICTED,
    )

    paused = client.generate(request, run_root=tmp_path, options=options)

    assert isinstance(paused.outcome, LLMPaused)
    assert paused.outcome.reason is ResumeReason.SUPERVISION_REQUIRED
    assert paused.outcome.details["code"] == "recovery_limit_reached"
    assert paused.outcome.input_required
    assert paused.outcome.request_ref is not None
    assert adapter.start_calls == failures
    repository = RunRepository(tmp_path)
    snapshot = repository.inspect(paused.snapshot.run_id).snapshot
    context = RunContext(repository, snapshot, resume_input=None, execution_slice=None)
    state = client.service._executor._task_store(context, request.task_id).read()
    assert state is not None
    assert len(state.generations) == 1
    assert state.current.safe_retries == safe_retry_limit
    assert state.current.replacement_of is None
    assert not state.current.possible_duplicate_execution
    assert context.effects.read(state.current.effect_id).stage is EffectStage.PREPARED

    resumed = client.resume(
        run_root=tmp_path,
        run_id=paused.snapshot.run_id,
        input=ResumeInput(paused.outcome.resume_key, ResumeAction.CONTINUE),
        options=options,
    )

    assert isinstance(resumed.outcome, LLMCompleted)
    assert resumed.outcome.value == {"answer": 42}
    assert adapter.start_calls == failures + 1


def test_pre_delivery_local_io_failure_stops_without_recovery_or_circuit(
    tmp_path: Path,
    adapter,
    registry,
) -> None:
    def fail_before_delivery(request, observer, stop):
        adapter.start_calls += 1
        adapter.requests.append(request)
        raise ProviderFailure(
            "durable observer write failed",
            category=FailureCategory.LOCAL_IO,
            delivery=DeliveryState.NOT_DELIVERED,
        )

    adapter.start = fail_before_delivery
    client = LLMClient(registry=registry)
    request = _request("local-io-before-delivery")

    result = client.generate(request, run_root=tmp_path)

    assert isinstance(result.outcome, LLMFailed)
    assert result.outcome.error.code.value == "local_io"
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 0
    repository = RunRepository(tmp_path)
    snapshot = repository.inspect(result.snapshot.run_id).snapshot
    context = RunContext(repository, snapshot, resume_input=None, execution_slice=None)
    state = client.service._executor._task_store(context, request.task_id).read()
    assert state is not None
    assert len(state.generations) == 1
    assert state.current.replacement_of is None
    assert not state.current.possible_duplicate_execution
    circuits = tmp_path / "operational" / "llm" / "circuits"
    assert not circuits.exists() or list(circuits.glob("*.json")) == []


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


def test_uncertain_delivery_with_saved_handle_uses_one_native_resume(
    tmp_path: Path, adapter, registry
) -> None:
    def uncertain_start(request, observer, stop):
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
    client = LLMClient(registry=registry)
    result = client.generate(_request(), run_root=tmp_path)
    assert isinstance(result.outcome, LLMCompleted)
    assert result.outcome.value == {"answer": 9}
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 1


def test_uncertain_delivery_without_handle_requires_supervision(
    tmp_path: Path, adapter, registry
) -> None:
    adapter.steps.append(
        ProviderFailure(
            "transport disconnected",
            category=FailureCategory.TRANSPORT,
            delivery=DeliveryState.MAY_HAVE_RUN,
        )
    )

    client = LLMClient(registry=registry)
    result = client.generate(_request(), run_root=tmp_path)

    assert isinstance(result.outcome, LLMPaused)
    assert result.outcome.reason is ResumeReason.SUPERVISION_REQUIRED
    assert result.outcome.input_required
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 0
    repository = RunRepository(tmp_path)
    context = RunContext(
        repository,
        repository.inspect(result.snapshot.run_id).snapshot,
        resume_input=None,
        execution_slice=None,
    )
    state = client.service._executor._task_store(context, "task").read()
    assert state is not None
    assert len(state.generations) == 1
    assert not state.current.possible_duplicate_execution


def test_provider_stop_pauses_the_outer_llm_run(
    tmp_path: Path, adapter, registry
) -> None:
    adapter.steps.append(ProviderExecution(ProviderTerminalKind.STOPPED))

    result = LLMClient(registry=registry).generate(_request(), run_root=tmp_path)

    assert result.snapshot.status is RunStatus.PAUSED
    assert isinstance(result.outcome, LLMStopped)
    assert result.snapshot.awaiting is not None
    assert result.snapshot.awaiting.reason is ResumeReason.EXECUTION_STOPPED
