from __future__ import annotations

import json

import pytest

from arc_llm import DeliveryState, FailureCategory, ProviderFailure
from arc_llm.output import CandidateMaterial
from arc_llm.providers._cli import EventAccumulator, run_cli
from arc_llm.providers.base import ProviderTerminalKind
from arc_llm.providers.codex import _parse_event as parse_codex_event
from arc_llm.providers.process import ProcessResult


class _Observer:
    def before_delivery(self) -> None:
        pass

    def native_handle(self, handle) -> None:
        pass

    def raw_event(self, event) -> None:
        pass


class _Cancel:
    def raise_if_requested(self) -> None:
        pass


def _encoded(*events: dict[str, object]) -> bytes:
    return b"".join(json.dumps(event).encode() + b"\n" for event in events)


def test_event_stream_requires_exactly_one_terminal_candidate() -> None:
    def parse_terminal(event):
        text = event.get("text")
        return (
            CandidateMaterial(text=text, terminal=bool(event.get("terminal")))
            if isinstance(text, str)
            else None,
            None,
            None,
        )

    incomplete = EventAccumulator("test", _Observer(), parse_terminal)
    incomplete.feed(
        _encoded(
            {
                "text": "draft",
            }
        )
    )
    with pytest.raises(ProviderFailure) as missing:
        incomplete.finish()
    assert missing.value.category is FailureCategory.SCHEMA
    assert missing.value.delivery is DeliveryState.MAY_HAVE_RUN
    assert missing.value.details["code"] == "incomplete_terminal_closure"

    duplicate = EventAccumulator("test", _Observer(), parse_terminal)
    duplicate.feed(
        _encoded(
            {
                "text": "one",
                "terminal": True,
            },
            {
                "text": "two",
                "terminal": True,
            },
        )
    )
    with pytest.raises(ProviderFailure) as multiple:
        duplicate.finish()
    assert multiple.value.details["code"] == "invalid_terminal_closure"

    complete = EventAccumulator("test", _Observer(), parse_terminal)
    complete.feed(
        _encoded(
            {"text": "done", "terminal": True},
        )
    )
    complete.finish()
    assert len(complete.candidates) == 1
    assert complete.candidates[0].terminal


def test_codex_events_are_not_terminal_candidates() -> None:
    accumulator = EventAccumulator("codex", _Observer(), parse_codex_event)
    accumulator.feed(
        _encoded(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "one"},
            },
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "two"},
            },
        )
    )
    accumulator.finish()
    assert accumulator.candidates == []


@pytest.mark.parametrize(
    "stdout",
    [
        _encoded(
            {
                "type": "item.updated",
                "item": {"type": "agent_message", "text": "draft"},
            }
        ),
        _encoded(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "one"},
            },
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "two"},
            },
        ),
    ],
)
def test_nonzero_exit_precedes_missing_or_multiple_terminal_validation(
    stdout: bytes,
) -> None:
    class _NonzeroRunner:
        def run(self, argv, **kwargs):
            kwargs["before_stdin"]()
            kwargs["on_stdout"](stdout)
            return ProcessResult(19, stdout, b"connection reset")

    result = run_cli(
        provider="codex",
        argv=("fake",),
        prompt="prompt",
        observer=_Observer(),
        cancel=_Cancel(),
        timeout=1,
        parse_event=parse_codex_event,
        runner=_NonzeroRunner(),
        env={},
    )
    assert result.terminal_kind is ProviderTerminalKind.FAILED
    assert result.failure is not None
    assert result.failure.category is FailureCategory.TRANSPORT
    assert result.diagnostics["returncode"] == 19
