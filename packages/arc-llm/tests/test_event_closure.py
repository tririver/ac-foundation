"""Provider stream-closure contracts independent of delivery accounting."""

from __future__ import annotations

from pathlib import Path

import pytest

from arc_jobs import StoppedError
from arc_llm import FailureCategory, ProviderFailure, ProviderTerminalKind
from arc_llm.output import CandidateMaterial
from arc_llm.providers._cli import EventAccumulator, run_cli
from arc_llm.providers.base import ProviderExecution
from arc_llm.providers.codex import _parse_event as parse_codex_event
from arc_llm.providers.process import ProcessResult


class _Observer:
    def native_handle(self, _handle) -> None:
        pass

    def raw_event(self, _event) -> None:
        pass

    def progress(self, _kind, _data) -> None:
        pass


def _candidate(event):
    if event.get("kind") == "terminal":
        return CandidateMaterial(value={"answer": event["answer"]}, terminal=True), None, None
    return None, None, None


def test_duplicate_terminal_events_are_schema_failures() -> None:
    accumulator = EventAccumulator("test", _Observer(), _candidate)
    accumulator.feed(b'{"kind":"terminal","answer":1}\n')
    accumulator.feed(b'{"kind":"terminal","answer":2}\n')

    with pytest.raises(ProviderFailure) as raised:
        accumulator.finish()
    assert raised.value.category is FailureCategory.SCHEMA
    assert raised.value.details["code"] == "invalid_terminal_closure"


def test_single_terminal_event_closes_cleanly() -> None:
    accumulator = EventAccumulator("test", _Observer(), _candidate)
    accumulator.feed(b'{"kind":"terminal","answer":1}\n')
    accumulator.finish()
    assert len(accumulator.candidates) == 1


def test_codex_agent_messages_are_not_terminal_responses() -> None:
    accumulator = EventAccumulator("codex", _Observer(), parse_codex_event)
    accumulator.feed(
        b'{"type":"item.completed","item":{"type":"agent_message","text":"draft"}}\n'
    )
    with pytest.raises(ProviderFailure) as raised:
        accumulator.finish()
    assert raised.value.category is FailureCategory.TRANSPORT
    assert raised.value.details["code"] == "incomplete_terminal_closure"


def test_nonzero_exit_precedes_stream_closure_validation(tmp_path: Path) -> None:
    class Runner:
        def run(self, *_args, **_kwargs):
            return ProcessResult(1, b"", b"authentication failed")

    class Stop:
        def raise_if_requested(self) -> None:
            pass

    execution = run_cli(
        provider="test",
        argv=("test",),
        prompt="prompt",
        observer=_Observer(),
        stop=Stop(),
        timeout=1,
        parse_event=_candidate,
        runner=Runner(),
        env={},
        cwd=tmp_path,
    )
    assert isinstance(execution, ProviderExecution)
    assert execution.terminal_kind is ProviderTerminalKind.FAILED
    assert execution.failure is not None
    assert execution.failure.category is FailureCategory.AUTHENTICATION
    assert execution.failure.details.get("code") != "incomplete_terminal_closure"


def test_runner_transport_failure_is_normalized_with_partial_diagnostics(
    tmp_path: Path,
) -> None:
    class Runner:
        def run(self, *_args, **kwargs):
            kwargs["on_stdout"](b'{"kind":"partial"}\n')
            raise ProviderFailure(
                "pipe failed",
                category=FailureCategory.TRANSPORT,
                details={"code": "pipe_failed"},
            )

    class Stop:
        def raise_if_requested(self) -> None:
            pass

    execution = run_cli(
        provider="test",
        argv=("test",),
        prompt="prompt",
        observer=_Observer(),
        stop=Stop(),
        timeout=1,
        parse_event=_candidate,
        runner=Runner(),
        env={},
        cwd=tmp_path,
    )

    assert execution.terminal_kind is ProviderTerminalKind.FAILED
    assert execution.failure is not None
    assert execution.failure.details["code"] == "pipe_failed"
    assert execution.diagnostics["returncode"] is None
    assert execution.diagnostics["runner_failure"] is True


def test_runner_stop_still_propagates(tmp_path: Path) -> None:
    class Runner:
        def run(self, *_args, **_kwargs):
            raise StoppedError("stopped")

    class Stop:
        def raise_if_requested(self) -> None:
            pass

    with pytest.raises(StoppedError):
        run_cli(
            provider="test",
            argv=("test",),
            prompt="prompt",
            observer=_Observer(),
            stop=Stop(),
            timeout=1,
            parse_event=_candidate,
            runner=Runner(),
            env={},
            cwd=tmp_path,
        )
