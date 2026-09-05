from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from ac_jobs import ArtifactSourceRef, ResumeReason, RunContext, RunRepository
from ac_llm import (
    AcRuntimeEnvironment,
    EffectiveHostMode,
    ErrorCode,
    FailureCategory,
    HostAuthority,
    HostRequest,
    HostResponse,
    HostResponseStatus,
    InvalidRequestError,
    JsonOutput,
    LLMClient,
    LLMCompleted,
    LLMExecutionOptions,
    LLMFailed,
    LLMPaused,
    LLMRequest,
    ModelSelection,
    NativeResumeHandle,
    ProviderExecution,
    ProviderFailure,
    ProviderGateOptions,
    ProviderTerminalKind,
    ResumeAction,
    ResumeInput,
    TextOutput,
)
from ac_llm.host import effective_host_mode
from ac_llm.output import CandidateMaterial
from jsonschema import Draft202012Validator

ANSWER_SCHEMA = {
    "type": "object",
    "required": ["answer"],
    "properties": {"answer": {"type": "integer"}},
    "additionalProperties": False,
}


def _request(task_id: str) -> LLMRequest:
    return LLMRequest(
        task_id,
        "Return an answer.",
        JsonOutput(ANSWER_SCHEMA),
        ModelSelection("codex"),
    )


def _host_request(
    request_id: str = "request-1",
    instruction: str = "Read the supplied local source.",
    purpose: str = "verify one premise",
) -> ProviderExecution:
    return ProviderExecution(
        ProviderTerminalKind.COMPLETED,
        (
            CandidateMaterial(
                value={
                    "schema_version": "ac.llm.host_turn.v1",
                    "state": "request_host",
                    "result": None,
                    "host_request": {
                        "request_id": request_id,
                        "instruction": instruction,
                        "purpose": purpose,
                    },
                },
                terminal=True,
            ),
        ),
        NativeResumeHandle("codex", "thread-host"),
    )


def _host_complete(value: dict[str, Any]) -> ProviderExecution:
    return ProviderExecution(
        ProviderTerminalKind.COMPLETED,
        (
            CandidateMaterial(
                value={
                    "schema_version": "ac.llm.host_turn.v1",
                    "state": "complete",
                    "result": value,
                    "host_request": None,
                },
                terminal=True,
            ),
        ),
        NativeResumeHandle("codex", "thread-host"),
    )


def _host_complete_text(value: str) -> ProviderExecution:
    return ProviderExecution(
        ProviderTerminalKind.COMPLETED,
        (
            CandidateMaterial(
                value={
                    "schema_version": "ac.llm.host_turn.v1",
                    "state": "complete",
                    "result": value,
                    "host_request": None,
                },
                terminal=True,
            ),
        ),
        NativeResumeHandle("codex", "thread-text"),
    )


def _direct_complete(value: dict[str, int]) -> ProviderExecution:
    return ProviderExecution(
        ProviderTerminalKind.COMPLETED,
        (CandidateMaterial(value=value, terminal=True),),
        NativeResumeHandle("codex", "thread-direct"),
    )


def _duplicate_recovery_document(
    tmp_path: Path,
    client: LLMClient,
    result: Any,
    task_id: str,
) -> dict[str, Any]:
    context = RunContext(RunRepository(tmp_path), result.snapshot, resume_input=None)
    executor = client.service._executor
    state = executor._task_store(context, task_id).read()
    assert state is not None
    ref = executor._artifacts(context, state.semantic_key).find(
        executor._duplicate_recovery_artifact_id(state)
    )
    assert ref is not None
    return json.loads(context.artifacts.read_bytes(ref).decode("utf-8"))


def test_unknown_authority_is_brokered_and_missing_broker_pauses(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    adapter.steps.append(_host_request())

    result = LLMClient(registry=registry).generate(
        _request("broker-required"), run_root=tmp_path
    )

    assert isinstance(result.outcome, LLMPaused)
    assert result.outcome.input_required
    assert result.outcome.details == {"code": "host_broker_required"}
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 0
    assert adapter.requests[0].capabilities["effective_host_mode"] == "brokered"
    assert adapter.requests[0].capabilities["internet"] is True


def test_manual_host_response_resumes_a_paused_host_turn(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    adapter.steps.extend((_host_request(), _host_complete({"answer": 11})))
    client = LLMClient(registry=registry)
    paused = client.generate(_request("manual-host-turn"), run_root=tmp_path)

    assert isinstance(paused.outcome, LLMPaused)
    resumed = client.resume(
        run_root=tmp_path,
        run_id=paused.snapshot.run_id,
        input=ResumeInput(
            paused.outcome.resume_key,
            ResumeAction.CONTINUE,
            host_response=HostResponse(
                HostResponseStatus.COMPLETED,
                result={"verified": True},
            ),
        ),
    )

    assert isinstance(resumed.outcome, LLMCompleted)
    assert resumed.outcome.value == {"answer": 11}
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 1


def test_brokered_text_output_uses_the_same_host_turn_envelope(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    adapter.steps.append(_host_complete_text("complete text"))
    result = LLMClient(registry=registry).generate(
        LLMRequest(
            "text-host-turn",
            "Return text.",
            TextOutput(),
            ModelSelection("codex"),
        ),
        run_root=tmp_path,
    )

    assert isinstance(result.outcome, LLMCompleted)
    assert result.outcome.value == "complete text"


def test_brokered_json_output_keeps_definitions_at_envelope_root(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    choice = {"paper_id": "paper-1"}
    adapter.steps.append(_host_complete({"selected_foundation": choice}))
    paper_choice_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["paper_id"],
        "properties": {"paper_id": {"type": "string"}},
    }
    result_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["selected_foundation"],
        "properties": {
            "selected_foundation": {"$ref": "#/$defs/paper_choice"},
        },
        "$defs": {"paper_choice": paper_choice_schema},
    }

    result = LLMClient(registry=registry).generate(
        LLMRequest(
            "foundation-selection",
            "Select one foundation.",
            JsonOutput(result_schema),
            ModelSelection("codex"),
        ),
        run_root=tmp_path,
    )

    assert isinstance(result.outcome, LLMCompleted)
    provider_schema = dict(adapter.requests[0].output_schema)
    assert provider_schema["$defs"] == {"paper_choice": paper_choice_schema}
    embedded = provider_schema["properties"]["result"]["anyOf"][0]
    assert embedded["$defs"] == {"paper_choice": paper_choice_schema}
    Draft202012Validator(provider_schema).validate(
        {
            "schema_version": "ac.llm.host_turn.v1",
            "state": "complete",
            "result": {"selected_foundation": choice},
            "host_request": None,
        }
    )


class _Broker:
    execution_identity: ClassVar[dict[str, object]] = {
        "kind": "test-broker",
        "revision": 1,
    }

    def __init__(self) -> None:
        self.calls: list[tuple[HostRequest, Path]] = []

    def execute(self, request: HostRequest, *, workspace: Path) -> HostResponse:
        self.calls.append((request, workspace))
        result_file = workspace / "work" / "broker-result.txt"
        result_file.write_text("verified", encoding="utf-8")
        return HostResponse(
            HostResponseStatus.COMPLETED,
            result={"verified": True},
            files=("work/broker-result.txt",),
        )


def test_brokered_host_turn_resumes_natively_and_reports_internet_warning(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    adapter.steps.extend((_host_request(), _host_complete({"answer": 7})))
    broker = _Broker()
    options = LLMExecutionOptions(host_broker=broker, internet=True)

    result = LLMClient(registry=registry).generate(
        _request("brokered-complete"), run_root=tmp_path, options=options
    )

    assert isinstance(result.outcome, LLMCompleted)
    assert result.outcome.value == {"answer": 7}
    assert result.outcome.warnings[0]["code"] == "internet_best_effort"
    assert [request.request_id for request, _workspace in broker.calls] == ["request-1"]
    assert (broker.calls[0][1] / "work" / "broker-result.txt").is_file()
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 1
    response_files = list(tmp_path.rglob("host/continuation.json"))
    assert len(response_files) == 1
    document = json.loads(response_files[0].read_text(encoding="utf-8"))
    assert document["schema_version"] == "ac.llm.host_continuation.v1"
    assert document["request_id"] == "request-1"
    assert document["response"]["status"] == "completed"


def test_duplicate_host_request_replays_persisted_completion_once(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    adapter.steps.extend(
        (
            _host_request(),
            _host_request(),
            _host_complete({"answer": 13}),
        )
    )
    broker = _Broker()

    result = LLMClient(registry=registry).generate(
        _request("duplicate-completed"),
        run_root=tmp_path,
        options=LLMExecutionOptions(host_broker=broker),
    )

    assert isinstance(result.outcome, LLMCompleted)
    assert result.outcome.value == {"answer": 13}
    assert [item.request_id for item, _workspace in broker.calls] == ["request-1"]
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 2
    document = _duplicate_recovery_document(
        tmp_path, LLMClient(registry=registry), result, "duplicate-completed"
    )
    assert document["action"] == "replay"
    assert document["continuation"]["response"]["status"] == "completed"


def test_duplicate_host_request_replays_persisted_refusal_once(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    class RefusingBroker:
        execution_identity: ClassVar[dict[str, object]] = {
            "kind": "test-refusing-broker",
            "revision": 1,
        }

        def __init__(self) -> None:
            self.calls: list[HostRequest] = []

        def execute(self, request: HostRequest, *, workspace: Path) -> HostResponse:
            del workspace
            self.calls.append(request)
            return HostResponse(
                HostResponseStatus.REFUSED,
                reason_code="not_allowed",
                reason="The host declined this request.",
                retryable=False,
                retry_condition="new evidence",
            )

    adapter.steps.extend(
        (
            _host_request(),
            _host_request(),
            _host_complete({"answer": 14}),
        )
    )
    broker = RefusingBroker()

    result = LLMClient(registry=registry).generate(
        _request("duplicate-refused"),
        run_root=tmp_path,
        options=LLMExecutionOptions(host_broker=broker),
    )

    assert isinstance(result.outcome, LLMCompleted)
    assert result.outcome.value == {"answer": 14}
    assert [item.request_id for item in broker.calls] == ["request-1"]
    document = _duplicate_recovery_document(
        tmp_path, LLMClient(registry=registry), result, "duplicate-refused"
    )
    assert document["action"] == "replay"
    assert document["continuation"]["response"]["status"] == "refused"


def test_duplicate_host_request_with_changed_instruction_is_a_conflict(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    adapter.steps.extend(
        (
            _host_request(),
            _host_request(instruction="Read a different source."),
        )
    )
    broker = _Broker()

    result = LLMClient(registry=registry).generate(
        _request("duplicate-conflict"),
        run_root=tmp_path,
        options=LLMExecutionOptions(host_broker=broker),
    )

    assert isinstance(result.outcome, LLMFailed)
    assert result.outcome.error.code is ErrorCode.HOST_REQUEST_ID_CONFLICT
    assert result.outcome.error.details == {
        "code": "host_request_id_conflict",
        "request_id": "request-1",
        "mismatch": "instruction",
        "expected_instruction": "Read the supplied local source.",
        "received_instruction": "Read a different source.",
    }
    assert [item.request_id for item, _workspace in broker.calls] == ["request-1"]
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 1


def test_unique_host_request_ids_continue_normally_in_one_task(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    adapter.steps.extend(
        (
            _host_request("request-1"),
            _host_request(
                "request-2",
                "Read the second supplied local source.",
                "verify a second premise",
            ),
            _host_complete({"answer": 15}),
        )
    )
    broker = _Broker()

    result = LLMClient(registry=registry).generate(
        _request("unique-multi-turn"),
        run_root=tmp_path,
        options=LLMExecutionOptions(host_broker=broker),
    )

    assert isinstance(result.outcome, LLMCompleted)
    assert [item.request_id for item, _workspace in broker.calls] == [
        "request-1",
        "request-2",
    ]
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 2


def test_same_host_request_id_is_scoped_to_one_task(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    adapter.steps.extend(
        (
            _host_request(),
            _host_complete({"answer": 16}),
            _host_request(),
            _host_complete({"answer": 17}),
        )
    )
    broker = _Broker()
    client = LLMClient(registry=registry)

    first = client.generate(
        _request("first-task"),
        run_root=tmp_path,
        options=LLMExecutionOptions(host_broker=broker),
    )
    second = client.generate(
        _request("second-task"),
        run_root=tmp_path,
        options=LLMExecutionOptions(host_broker=broker),
    )

    assert isinstance(first.outcome, LLMCompleted)
    assert isinstance(second.outcome, LLMCompleted)
    assert [item.request_id for item, _workspace in broker.calls] == [
        "request-1",
        "request-1",
    ]


def test_duplicate_host_request_recovery_is_bounded(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    adapter.steps.extend((_host_request(), _host_request(), _host_request()))
    broker = _Broker()

    result = LLMClient(registry=registry).generate(
        _request("duplicate-exhausted"),
        run_root=tmp_path,
        options=LLMExecutionOptions(host_broker=broker),
    )

    assert isinstance(result.outcome, LLMFailed)
    assert result.outcome.error.code is ErrorCode.DUPLICATE_HOST_REQUEST_ID
    assert result.outcome.error.details == {
        "code": "duplicate_host_request_recovery_exhausted",
        "request_id": "request-1",
        "recovery_exhausted": True,
    }
    assert [item.request_id for item, _workspace in broker.calls] == ["request-1"]
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 2


def test_duplicate_host_request_recovery_retries_after_rate_limit_pause(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    adapter.steps.extend(
        (
            _host_request(),
            _host_request(),
            ProviderFailure("try again later", category=FailureCategory.RATE_LIMIT),
            _host_complete({"answer": 22}),
        )
    )
    broker = _Broker()
    client = LLMClient(registry=registry)
    options = LLMExecutionOptions(
        host_broker=broker,
        gate=ProviderGateOptions(enabled=False),
    )

    paused = client.generate(
        _request("duplicate-rate-limit"),
        run_root=tmp_path,
        run_id="duplicate-rate-limit-run",
        options=options,
    )

    assert isinstance(paused.outcome, LLMPaused)
    assert paused.outcome.details["code"] == FailureCategory.RATE_LIMIT.value
    recovered = client.resume(
        run_root=tmp_path,
        run_id="duplicate-rate-limit-run",
        options=options,
    )

    assert isinstance(recovered.outcome, LLMCompleted)
    assert recovered.outcome.value == {"answer": 22}
    assert [item.request_id for item, _workspace in broker.calls] == ["request-1"]
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 3


def test_duplicate_host_request_recovery_retries_after_crash_before_resume(
    tmp_path: Path,
    adapter: Any,
    registry: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter.steps.extend(
        (
            _host_request(),
            _host_request(),
            _host_complete({"answer": 24}),
        )
    )
    broker = _Broker()
    client = LLMClient(registry=registry)
    executor = client.service._executor
    original_resume = executor._resume_duplicate_host_turn

    def crash_before_resume(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt("crash after duplicate recovery publication")

    monkeypatch.setattr(executor, "_resume_duplicate_host_turn", crash_before_resume)
    with pytest.raises(KeyboardInterrupt):
        client.generate(
            _request("duplicate-resume-window"),
            run_root=tmp_path,
            run_id="duplicate-resume-window-run",
            options=LLMExecutionOptions(host_broker=broker),
        )

    monkeypatch.setattr(executor, "_resume_duplicate_host_turn", original_resume)
    recovered = client.resume(
        run_root=tmp_path,
        run_id="duplicate-resume-window-run",
        options=LLMExecutionOptions(host_broker=broker),
    )

    assert isinstance(recovered.outcome, LLMCompleted)
    assert recovered.outcome.value == {"answer": 24}
    assert [item.request_id for item, _workspace in broker.calls] == ["request-1"]
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 2


def test_duplicate_recovery_attempt_crash_uses_the_standard_crash_guard(
    tmp_path: Path,
    adapter: Any,
    registry: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter.steps.extend((_host_request(), _host_request()))
    broker = _Broker()
    client = LLMClient(registry=registry)
    executor = client.service._executor
    original_resume = adapter.resume

    def crash_after_provider_acceptance(
        handle: Any,
        provider_request: Any,
        observer: Any,
        stop: Any,
    ) -> Any:
        if adapter.resume_calls == 1:
            adapter.resume_calls += 1
            adapter.requests.append(provider_request)
            raise KeyboardInterrupt("crash after duplicate continuation acceptance")
        return original_resume(handle, provider_request, observer, stop)

    monkeypatch.setattr(adapter, "resume", crash_after_provider_acceptance)
    request = _request("duplicate-attempt-crash")
    options = LLMExecutionOptions(host_broker=broker)
    with pytest.raises(KeyboardInterrupt):
        client.generate(
            request,
            run_root=tmp_path,
            run_id="duplicate-attempt-crash-run",
            options=options,
        )

    monkeypatch.setattr(adapter, "resume", original_resume)
    repository = RunRepository(tmp_path)
    snapshot = repository.inspect("duplicate-attempt-crash-run").snapshot
    context = RunContext(repository, snapshot, resume_input=None)
    store = executor._task_store(context, request.task_id)
    state = store.read()
    assert state is not None
    assert state.current.raw_response is None
    assert state.current.attempt_started

    guarded = executor._drive(
        context,
        request,
        state,
        store,
        options,
        crash_retry_available=False,
    )

    assert isinstance(guarded, LLMPaused)
    assert guarded.reason is ResumeReason.EXECUTION_INTERRUPTED
    assert guarded.details == {"code": "provider_crash_retry_exhausted"}
    assert [item.request_id for item, _workspace in broker.calls] == ["request-1"]
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 2


def test_duplicate_recovery_raw_publish_replaces_stale_duplicate_raw_after_crashes(
    tmp_path: Path,
    adapter: Any,
    registry: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter.steps.extend(
        (
            _host_request(),
            _host_request(),
            _host_complete({"answer": 26}),
        )
    )
    broker = _Broker()
    client = LLMClient(registry=registry)
    executor = client.service._executor
    original_update = executor._update_current
    crash_stage = 0

    def crash_in_combined_window(state: Any, **changes: Any) -> Any:
        nonlocal crash_stage
        if (
            crash_stage == 0
            and state.current.raw_response is not None
            and changes.get("raw_response") is None
            and changes.get("attempt_started") is False
        ):
            crash_stage = 1
            raise KeyboardInterrupt("crash after duplicate recovery publication")
        recovery_raw = changes.get("raw_response")
        if (
            crash_stage == 1
            and recovery_raw is not None
            and recovery_raw.artifact_id.endswith("-duplicate-recovery-1.json")
        ):
            crash_stage = 2
            raise KeyboardInterrupt("crash after dedicated recovery raw publication")
        return original_update(state, **changes)

    monkeypatch.setattr(executor, "_update_current", crash_in_combined_window)
    with pytest.raises(KeyboardInterrupt):
        client.generate(
            _request("duplicate-combined-window"),
            run_root=tmp_path,
            run_id="duplicate-combined-window-run",
            options=LLMExecutionOptions(host_broker=broker),
        )
    with pytest.raises(KeyboardInterrupt):
        client.resume(
            run_root=tmp_path,
            run_id="duplicate-combined-window-run",
            options=LLMExecutionOptions(host_broker=broker),
        )

    monkeypatch.setattr(executor, "_update_current", original_update)
    recovered = client.resume(
        run_root=tmp_path,
        run_id="duplicate-combined-window-run",
        options=LLMExecutionOptions(host_broker=broker),
    )

    assert isinstance(recovered.outcome, LLMCompleted)
    assert recovered.outcome.value == {"answer": 26}
    assert [item.request_id for item, _workspace in broker.calls] == ["request-1"]
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 2


def test_host_request_identity_crash_after_pending_turn_commit_brokers_on_resume(
    tmp_path: Path,
    adapter: Any,
    registry: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter.steps.extend((_host_request(), _host_complete({"answer": 23})))
    broker = _Broker()
    client = LLMClient(registry=registry)
    executor = client.service._executor
    original_publish = executor._publish_host_request_identity

    def crash_after_identity(*args: Any, **kwargs: Any) -> Any:
        original_publish(*args, **kwargs)
        raise KeyboardInterrupt("crash after host request identity publication")

    monkeypatch.setattr(
        executor,
        "_publish_host_request_identity",
        crash_after_identity,
    )
    with pytest.raises(KeyboardInterrupt):
        client.generate(
            _request("identity-commit-window"),
            run_root=tmp_path,
            run_id="identity-commit-window-run",
            options=LLMExecutionOptions(host_broker=broker),
        )

    monkeypatch.setattr(executor, "_publish_host_request_identity", original_publish)
    recovered = client.resume(
        run_root=tmp_path,
        run_id="identity-commit-window-run",
        options=LLMExecutionOptions(host_broker=broker),
    )

    assert isinstance(recovered.outcome, LLMCompleted)
    assert recovered.outcome.value == {"answer": 23}
    assert [item.request_id for item, _workspace in broker.calls] == ["request-1"]
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 1


def test_duplicate_host_request_uses_one_synthetic_refusal_without_continuation(
    tmp_path: Path, adapter: Any, registry: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter.steps.extend(
        (
            _host_request(),
            _host_request(),
            _host_complete({"answer": 18}),
        )
    )
    broker = _Broker()
    client = LLMClient(registry=registry)
    monkeypatch.setattr(
        client.service._executor,
        "_persisted_continuation_prompt",
        lambda *_args: None,
    )

    result = client.generate(
        _request("duplicate-synthetic-refusal"),
        run_root=tmp_path,
        options=LLMExecutionOptions(host_broker=broker),
    )

    assert isinstance(result.outcome, LLMCompleted)
    assert [item.request_id for item, _workspace in broker.calls] == ["request-1"]
    document = _duplicate_recovery_document(
        tmp_path, client, result, "duplicate-synthetic-refusal"
    )
    assert document["action"] == "synthetic_refusal"
    assert document["continuation"]["response"]["reason_code"] == (
        "duplicate_host_request_id"
    )


def test_fresh_generation_retains_seen_host_request_ids(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    adapter.steps.extend(
        (
            _host_request(),
            ProviderFailure("lost", category=FailureCategory.TRANSPORT),
            _host_request(),
            _host_complete({"answer": 19}),
        )
    )
    broker = _Broker()
    client = LLMClient(registry=registry)

    result = client.generate(
        _request("duplicate-after-fresh-generation"),
        run_root=tmp_path,
        options=LLMExecutionOptions(host_broker=broker),
    )

    assert isinstance(result.outcome, LLMCompleted)
    assert [item.request_id for item, _workspace in broker.calls] == ["request-1"]
    assert adapter.start_calls == 2
    assert adapter.resume_calls == 2
    context = RunContext(RunRepository(tmp_path), result.snapshot, resume_input=None)
    state = client.service._executor._task_store(
        context, "duplicate-after-fresh-generation"
    ).read()
    assert state is not None
    assert state.current.generation == 2
    assert state.seen_host_request_ids == ("request-1",)


def test_duplicate_recovery_raw_publish_crash_replays_without_new_broker_call(
    tmp_path: Path,
    adapter: Any,
    registry: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter.steps.extend(
        (
            _host_request(),
            _host_request(),
            _host_complete({"answer": 20}),
        )
    )
    broker = _Broker()
    client = LLMClient(registry=registry)
    executor = client.service._executor
    original_update = executor._update_current
    raw_publications = 0

    def crash_after_recovery_raw(state: Any, **changes: Any) -> Any:
        nonlocal raw_publications
        if changes.get("raw_response") is not None:
            raw_publications += 1
            if raw_publications == 3:
                raise KeyboardInterrupt("crash after duplicate recovery raw publication")
        return original_update(state, **changes)

    monkeypatch.setattr(executor, "_update_current", crash_after_recovery_raw)
    with pytest.raises(KeyboardInterrupt):
        client.generate(
            _request("duplicate-raw-window"),
            run_root=tmp_path,
            run_id="duplicate-raw-window-run",
            options=LLMExecutionOptions(host_broker=broker),
        )

    monkeypatch.setattr(executor, "_update_current", original_update)
    recovered = client.resume(
        run_root=tmp_path,
        run_id="duplicate-raw-window-run",
        options=LLMExecutionOptions(host_broker=broker),
    )

    assert isinstance(recovered.outcome, LLMCompleted)
    assert recovered.outcome.value == {"answer": 20}
    assert [item.request_id for item, _workspace in broker.calls] == ["request-1"]
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 2


def test_broker_side_effect_crash_requires_reconciliation_not_reexecution(
    tmp_path: Path,
    adapter: Any,
    registry: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter.steps.extend((_host_request(), _host_complete({"answer": 21})))
    broker = _Broker()
    client = LLMClient(registry=registry)
    executor = client.service._executor

    def crash_after_broker(*_args: Any, **_kwargs: Any) -> None:
        raise KeyboardInterrupt("crash after broker side effect")

    monkeypatch.setattr(executor, "_record_host_broker_completion", crash_after_broker)
    with pytest.raises(KeyboardInterrupt):
        client.generate(
            _request("broker-side-effect-window"),
            run_root=tmp_path,
            run_id="broker-side-effect-window-run",
            options=LLMExecutionOptions(host_broker=broker),
        )

    monkeypatch.undo()
    reconciled = client.resume(
        run_root=tmp_path,
        run_id="broker-side-effect-window-run",
        options=LLMExecutionOptions(host_broker=broker),
    )

    assert isinstance(reconciled.outcome, LLMPaused)
    assert reconciled.outcome.details == {
        "code": "host_broker_reconciliation_required",
        "broker_invocation_started": True,
    }
    assert reconciled.outcome.input_required
    assert [item.request_id for item, _workspace in broker.calls] == ["request-1"]

    completed = client.resume(
        run_root=tmp_path,
        run_id="broker-side-effect-window-run",
        input=ResumeInput(
            reconciled.outcome.resume_key,
            ResumeAction.CONTINUE,
            host_response=HostResponse(
                HostResponseStatus.COMPLETED,
                result={"reconciled": True},
            ),
        ),
        options=LLMExecutionOptions(host_broker=broker),
    )

    assert isinstance(completed.outcome, LLMCompleted)
    assert completed.outcome.value == {"answer": 21}
    assert [item.request_id for item, _workspace in broker.calls] == ["request-1"]


def test_host_response_rejects_non_json_result() -> None:
    with pytest.raises(InvalidRequestError, match="JSON-compatible"):
        HostResponse(HostResponseStatus.COMPLETED, result={"value": object()})


def test_runtime_authority_and_environment_are_explicit_runtime_state() -> None:
    environment = AcRuntimeEnvironment(
        {
            "AC_HOME": "/ac-home",
            "AC_RUNTIME_HOME": "/runtime",
            "AC_DOCUMENT_CACHE": "/documents",
            "PATH": "/bin",
        }
    )
    options = LLMExecutionOptions(
        host_authority=HostAuthority.UNRESTRICTED,
        runtime_environment=environment,
    )

    assert effective_host_mode(options.host_authority) is EffectiveHostMode.DIRECT
    assert environment.execution_document()["AC_DOCUMENT_CACHE"] == "/documents"
    applied = environment.apply_to({"UNRELATED": "kept", "AC_HOME": "old"})
    assert applied == {
        "UNRELATED": "kept",
        "AC_HOME": "/ac-home",
        "AC_RUNTIME_HOME": "/runtime",
        "AC_DOCUMENT_CACHE": "/documents",
        "PATH": "/bin",
    }


def test_internet_warning_is_preserved_when_an_accepted_task_replays(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    adapter.steps.append(_host_complete({"answer": 3}))
    client = LLMClient(registry=registry)
    options = LLMExecutionOptions(internet=True)
    request = _request("replay-warning")

    first = client.generate(request, run_root=tmp_path, options=options)
    replay = client.generate(request, run_root=tmp_path, options=options)

    assert isinstance(first.outcome, LLMCompleted)
    assert isinstance(replay.outcome, LLMCompleted)
    assert replay.outcome.warnings[0]["code"] == "internet_best_effort"
    assert adapter.start_calls == 1


def test_client_resume_replay_uses_the_supplied_runtime_options(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    adapter.steps.append(_direct_complete({"answer": 5}))
    client = LLMClient(registry=registry)
    request = _request("resume-options")
    options = LLMExecutionOptions(
        host_authority=HostAuthority.UNRESTRICTED,
        internet=False,
        gate=ProviderGateOptions(minimum_available_memory_fraction=None),
    )
    generated = client.generate(request, run_root=tmp_path, options=options)

    replayed = client.resume(
        run_root=tmp_path,
        run_id=generated.snapshot.run_id,
        options=options,
    )

    assert isinstance(replayed.outcome, LLMCompleted)
    assert replayed.outcome.warnings == ()


def test_client_adopt_preserves_the_supplied_runtime_options(
    tmp_path: Path, adapter: Any, registry: Any
) -> None:
    adapter.steps.append(_direct_complete({"answer": 6}))
    client = LLMClient(registry=registry)
    request = _request("adopt-options")
    options = LLMExecutionOptions(
        host_authority=HostAuthority.UNRESTRICTED,
        internet=False,
    )
    source_result = client.generate(request, run_root=tmp_path, options=options)
    assert source_result.snapshot.result_ref is not None
    source = ArtifactSourceRef(
        source_result.snapshot.run_id,
        source_result.snapshot.result_ref.artifact_id,
        source_result.snapshot.result_ref.digest,
    )

    adopted = client.adopt(
        request,
        source,
        run_root=tmp_path,
        run_id="adopt-options-run",
        options=options,
    )

    assert isinstance(adopted.outcome, LLMCompleted)
    assert adopted.outcome.warnings == ()
