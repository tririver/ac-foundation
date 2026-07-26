from __future__ import annotations

from pathlib import Path

import pytest

from arc_jobs import RunContext, RunRepository
from arc_llm import (
    FailureCategory,
    HostAuthority,
    JsonOutput,
    LLMClient,
    LLMCompleted,
    LLMExecutionOptions,
    LLMPaused,
    LLMStopped,
    LLMRequest,
    ModelSelection,
    NativeResumeHandle,
    ProviderExecution,
    ProviderFailure,
    ProviderTerminalKind,
    ResumeReason,
)
from arc_llm.host import HostResponse, HostResponseStatus
from arc_llm.output import CandidateMaterial
from arc_llm.providers._cli import EventAccumulator


def _request(task_id: str) -> LLMRequest:
    return LLMRequest(
        task_id,
        "Return an answer.",
        JsonOutput({"type": "object", "required": ["answer"]}),
        ModelSelection("codex"),
    )


def _completed(value: object) -> ProviderExecution:
    return ProviderExecution(
        ProviderTerminalKind.COMPLETED,
        (CandidateMaterial(value=value, terminal=True),),
    )


def _direct() -> LLMExecutionOptions:
    return LLMExecutionOptions(host_authority=HostAuthority.UNRESTRICTED)


def test_transport_crash_gets_one_fresh_generation_then_completes(
    tmp_path: Path, adapter, registry
) -> None:
    adapter.steps.extend(
        (
            ProviderFailure("lost", category=FailureCategory.TRANSPORT),
            _completed({"answer": 42}),
        )
    )
    client = LLMClient(registry=registry)

    result = client.generate(_request("one-retry"), run_root=tmp_path, options=_direct())

    assert isinstance(result.outcome, LLMCompleted)
    assert adapter.start_calls == 2
    assert adapter.resume_calls == 0
    context = RunContext(
        RunRepository(tmp_path), result.snapshot, resume_input=None, execution_slice=None
    )
    state = client.service._executor._task_store(context, "one-retry").read()
    assert state is not None
    assert state.current.generation == 2
    assert state.current.native_handle is None


def test_second_crash_pauses_then_explicit_resume_gets_fresh_allowance(
    tmp_path: Path, adapter, registry
) -> None:
    adapter.steps.extend(
        (
            ProviderFailure("lost one", category=FailureCategory.TRANSPORT),
            ProviderFailure("lost two", category=FailureCategory.TIMEOUT),
            _completed({"answer": 7}),
        )
    )
    client = LLMClient(registry=registry)
    first = client.generate(_request("retry-budget"), run_root=tmp_path, options=_direct())

    assert isinstance(first.outcome, LLMPaused)
    assert first.outcome.reason is ResumeReason.EXECUTION_INTERRUPTED
    assert first.outcome.details["code"] == "provider_crash_retry_exhausted"
    assert not first.outcome.input_required
    assert adapter.start_calls == 2

    resumed = client.resume(
        run_root=tmp_path, run_id=first.snapshot.run_id, options=_direct()
    )
    assert isinstance(resumed.outcome, LLMCompleted)
    assert adapter.start_calls == 3
    assert adapter.resume_calls == 0


def test_published_raw_response_is_recovered_after_pre_cas_crash(
    tmp_path: Path, adapter, registry, monkeypatch
) -> None:
    adapter.steps.append(_completed({"answer": "saved"}))
    client = LLMClient(registry=registry)
    executor = client.service._executor
    original_update = executor._update_current
    crash_once = True

    def crash_between_publish_and_cas(state, **changes):
        nonlocal crash_once
        if crash_once and "raw_response" in changes:
            crash_once = False
            raise KeyboardInterrupt("crash after raw publication")
        return original_update(state, **changes)

    monkeypatch.setattr(executor, "_update_current", crash_between_publish_and_cas)
    with pytest.raises(KeyboardInterrupt):
        client.generate(
            _request("raw-window"),
            run_root=tmp_path,
            run_id="raw-window-run",
            options=_direct(),
        )

    monkeypatch.setattr(executor, "_update_current", original_update)
    recovered = client.resume(
        run_root=tmp_path, run_id="raw-window-run", options=_direct()
    )
    assert isinstance(recovered.outcome, LLMCompleted)
    assert recovered.outcome.value == {"answer": "saved"}
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 0


def test_process_restart_mid_provider_call_freshens_the_started_generation(
    tmp_path: Path, adapter, registry
) -> None:
    client = LLMClient(registry=registry)
    original_start = adapter.start

    def crash_process(request, observer, stop):
        adapter.start_calls += 1
        adapter.requests.append(request)
        raise KeyboardInterrupt("simulated process termination")

    adapter.start = crash_process
    with pytest.raises(KeyboardInterrupt):
        client.generate(
            _request("process-restart"),
            run_root=tmp_path,
            run_id="process-restart-run",
            options=_direct(),
        )

    adapter.start = original_start
    adapter.steps.append(_completed({"answer": "after restart"}))
    resumed = client.resume(
        run_root=tmp_path, run_id="process-restart-run", options=_direct()
    )
    assert isinstance(resumed.outcome, LLMCompleted)
    assert resumed.outcome.value == {"answer": "after restart"}
    assert adapter.start_calls == 2
    assert adapter.resume_calls == 0
    context = RunContext(
        RunRepository(tmp_path), resumed.snapshot, resume_input=None, execution_slice=None
    )
    state = client.service._executor._task_store(context, "process-restart").read()
    assert state is not None
    assert state.current.generation == 2


def test_workspace_preparation_failure_never_marks_a_provider_attempt(
    tmp_path: Path, adapter, registry, monkeypatch
) -> None:
    from arc_jobs import RunSpec
    from arc_llm import LLMFailed

    repository = RunRepository(tmp_path)
    snapshot = repository.create(RunSpec("workspace-failure", "test", {}))
    context = RunContext(repository, snapshot, resume_input=None, execution_slice=None)
    client = LLMClient(registry=registry)
    executor = client.service._executor
    original_prepare = executor._prepare_workspace

    def fail_workspace(*_args, **_kwargs):
        raise OSError("workspace unavailable")

    monkeypatch.setattr(executor, "_prepare_workspace", fail_workspace)
    request = _request("workspace-failure")
    failed = client.service.execute(context, request, options=_direct())
    assert isinstance(failed, LLMFailed)
    assert adapter.start_calls == 0

    state = executor._task_store(context, request.task_id).read()
    assert state is not None
    assert not state.current.attempt_started
    assert state.current.generation == 1

    monkeypatch.setattr(executor, "_prepare_workspace", original_prepare)
    adapter.steps.append(_completed({"answer": "prepared"}))
    completed = client.service.execute(context, request, options=_direct())
    assert isinstance(completed, LLMCompleted)
    assert adapter.start_calls == 1
    state = executor._task_store(context, request.task_id).read()
    assert state is not None
    assert state.current.generation == 1


@pytest.mark.parametrize(
    ("category", "outcome_type", "reason"),
    (
        (FailureCategory.AUTHENTICATION, LLMPaused, ResumeReason.EXTERNAL_CONDITION),
        (FailureCategory.QUOTA, LLMPaused, ResumeReason.EXTERNAL_CONDITION),
        (FailureCategory.RATE_LIMIT, LLMPaused, ResumeReason.EXTERNAL_CONDITION),
        (FailureCategory.UNAVAILABLE, LLMPaused, ResumeReason.EXTERNAL_CONDITION),
        (FailureCategory.INVALID_REQUEST, type(None), None),
        (FailureCategory.SCHEMA, type(None), None),
        (FailureCategory.LOCAL_IO, type(None), None),
    ),
)
def test_non_crash_provider_categories_do_not_consume_the_retry(
    tmp_path: Path, adapter, registry, category, outcome_type, reason
) -> None:
    from arc_llm import LLMFailed

    adapter.steps.extend(
        (
            ProviderFailure("not a crash", category=category),
            _completed({"answer": "must remain unused"}),
        )
    )
    result = LLMClient(registry=registry).generate(
        _request(f"non-crash-{category.value}"), run_root=tmp_path, options=_direct()
    )

    if outcome_type is LLMPaused:
        assert isinstance(result.outcome, LLMPaused)
        assert result.outcome.reason is reason
    else:
        assert isinstance(result.outcome, LLMFailed)
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 0
    assert len(adapter.steps) == 1


def test_gate_backoff_does_not_consume_the_retry(
    tmp_path: Path, adapter, registry, monkeypatch
) -> None:
    class BlockedGate:
        def acquire(self, *_args, **_kwargs):
            raise ProviderFailure(
                "circuit open",
                category=FailureCategory.UNAVAILABLE,
                details={"code": "provider_circuit_open"},
            )

    client = LLMClient(registry=registry)
    monkeypatch.setattr(client.service._executor, "_provider_gate", lambda *_: BlockedGate())
    result = client.generate(_request("gate-backoff"), run_root=tmp_path, options=_direct())

    assert isinstance(result.outcome, LLMPaused)
    assert result.outcome.reason is ResumeReason.EXTERNAL_CONDITION
    assert adapter.start_calls == 0


def test_provider_stop_does_not_consume_the_retry(tmp_path: Path, adapter, registry) -> None:
    adapter.steps.extend(
        (ProviderExecution(ProviderTerminalKind.STOPPED), _completed({"answer": 1}))
    )
    result = LLMClient(registry=registry).generate(
        _request("provider-stop"), run_root=tmp_path, options=_direct()
    )
    assert isinstance(result.outcome, LLMStopped)
    assert adapter.start_calls == 1
    assert len(adapter.steps) == 1


def test_successful_broker_continuation_keeps_its_generation_and_native_resume(
    tmp_path: Path, adapter, registry
) -> None:
    class RefusingBroker:
        execution_identity = {"kind": "test"}

        def execute(self, _request, *, workspace):
            return HostResponse(
                HostResponseStatus.REFUSED,
                reason_code="not_allowed",
                reason="The host declined this request.",
                retryable=False,
                retry_condition="new evidence",
            )

    adapter.steps.extend(
        (
            ProviderExecution(
                ProviderTerminalKind.COMPLETED,
                (
                    CandidateMaterial(
                        value={
                            "schema_version": "arc.llm.host_turn.v1",
                            "state": "request_host",
                            "result": None,
                            "host_request": {
                                "request_id": "look-up",
                                "instruction": "Look up a paper.",
                                "purpose": "evidence",
                            },
                        },
                        terminal=True,
                    ),
                ),
                NativeResumeHandle("codex", "host-thread"),
            ),
            _completed({"answer": "continued"}),
        )
    )
    client = LLMClient(registry=registry)
    result = client.generate(
        _request("broker-refusal"),
        run_root=tmp_path,
        options=LLMExecutionOptions(host_broker=RefusingBroker()),
    )

    assert isinstance(result.outcome, LLMCompleted)
    assert adapter.start_calls == 1
    assert adapter.resume_calls == 1
    context = RunContext(
        RunRepository(tmp_path), result.snapshot, resume_input=None, execution_slice=None
    )
    state = client.service._executor._task_store(context, "broker-refusal").read()
    assert state is not None
    assert state.current.generation == 1


def test_incomplete_terminal_closure_is_a_transport_crash() -> None:
    class Observer:
        def native_handle(self, _handle):
            pass

        def raw_event(self, _event):
            pass

        def progress(self, _kind, _data):
            pass

    accumulator = EventAccumulator("test", Observer(), lambda _event: (None, None, None))
    accumulator.feed(b'{"kind":"partial"}\n')
    with pytest.raises(ProviderFailure) as raised:
        accumulator.finish()
    assert raised.value.category is FailureCategory.TRANSPORT
    assert raised.value.details["code"] == "incomplete_terminal_closure"
