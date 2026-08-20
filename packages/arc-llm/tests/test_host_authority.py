from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from arc_jobs import ArtifactSourceRef

from arc_llm import (
    ArcRuntimeEnvironment,
    EffectiveHostMode,
    HostAuthority,
    HostRequest,
    HostResponse,
    HostResponseStatus,
    InvalidRequestError,
    JsonOutput,
    LLMClient,
    LLMCompleted,
    LLMExecutionOptions,
    LLMPaused,
    LLMRequest,
    ModelSelection,
    NativeResumeHandle,
    ProviderExecution,
    ProviderTerminalKind,
    ResumeAction,
    ResumeInput,
    TextOutput,
)
from arc_llm.host import effective_host_mode
from arc_llm.output import CandidateMaterial


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


def _host_request() -> ProviderExecution:
    return ProviderExecution(
        ProviderTerminalKind.COMPLETED,
        (
            CandidateMaterial(
                value={
                    "schema_version": "arc.llm.host_turn.v1",
                    "state": "request_host",
                    "result": None,
                    "host_request": {
                        "request_id": "request-1",
                        "instruction": "Read the supplied local source.",
                        "purpose": "verify one premise",
                    },
                },
                terminal=True,
            ),
        ),
        NativeResumeHandle("codex", "thread-host"),
    )


def _host_complete(value: dict[str, int]) -> ProviderExecution:
    return ProviderExecution(
        ProviderTerminalKind.COMPLETED,
        (
            CandidateMaterial(
                value={
                    "schema_version": "arc.llm.host_turn.v1",
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
                    "schema_version": "arc.llm.host_turn.v1",
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


class _Broker:
    execution_identity = {"kind": "test-broker", "revision": 1}

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
    assert document["schema_version"] == "arc.llm.host_continuation.v1"
    assert document["request_id"] == "request-1"
    assert document["response"]["status"] == "completed"


def test_host_response_rejects_non_json_result() -> None:
    with pytest.raises(InvalidRequestError, match="JSON-compatible"):
        HostResponse(HostResponseStatus.COMPLETED, result={"value": object()})


def test_runtime_authority_and_environment_are_explicit_runtime_state() -> None:
    environment = ArcRuntimeEnvironment(
        {
            "ARC_HOME": "/arc-home",
            "ARC_RUNTIME_HOME": "/runtime",
            "ARC_DOCUMENT_CACHE": "/documents",
            "ARC_PAPER_CACHE": "/cache",
            "PATH": "/bin",
        }
    )
    options = LLMExecutionOptions(
        host_authority=HostAuthority.UNRESTRICTED,
        runtime_environment=environment,
    )

    assert effective_host_mode(options.host_authority) is EffectiveHostMode.DIRECT
    assert environment.execution_document()["ARC_PAPER_CACHE"] == "/cache"
    applied = environment.apply_to({"UNRELATED": "kept", "ARC_HOME": "old"})
    assert applied == {
        "UNRELATED": "kept",
        "ARC_HOME": "/arc-home",
        "ARC_RUNTIME_HOME": "/runtime",
        "ARC_DOCUMENT_CACHE": "/documents",
        "ARC_PAPER_CACHE": "/cache",
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
