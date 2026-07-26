from __future__ import annotations

from typing import Any

from arc_llm import NativeResumeHandle
from arc_llm.progress import DurableProviderObserver, message_preview


def _context(calls: list[tuple[str, object]], *, fail: bool = False):
    class Events:
        def emit(self, kind: str, data: object) -> None:
            if fail:
                raise OSError("event log unavailable")
            calls.append((kind, data))

    return type("Context", (), {"events": Events()})()


def _observer(
    calls: list[tuple[str, object]],
    *,
    fail: bool = False,
) -> DurableProviderObserver:
    return DurableProviderObserver(
        context=_context(calls, fail=fail),
        on_handle=lambda _handle: None,
        task_id="task",
        provider="codex",
        generation=2,
        host_turn_round=1,
    )


def test_observer_projects_handles_metadata_and_arbitrary_progress_body() -> None:
    calls: list[tuple[str, object]] = []
    handles: list[NativeResumeHandle] = []
    observer = DurableProviderObserver(
        context=_context(calls),
        on_handle=handles.append,
        task_id="task",
        provider="codex",
        generation=2,
        host_turn_round=1,
    )

    handle = NativeResumeHandle("codex", "thread")
    observer.native_handle(handle)
    assert handles == [handle]
    observer.progress(
        "phase",
        {"nested": {"content": "useful body"}, "prompt": "logical prompt"},
    )
    assert calls == [
        (
            "phase",
            {
                "task_id": "task",
                "provider": "codex",
                "generation": 2,
                "host_turn_round": 1,
                "nested": {"content": "useful body"},
                "prompt": "logical prompt",
            },
        )
    ]


def test_observer_failure_is_diagnostic_only() -> None:
    observer = _observer([], fail=True)
    observer.progress("phase", {"content": "paid output"})
    assert observer.observation_errors == ["OSError"]


def test_message_preview_redacts_normalizes_and_counts_unicode_codepoints() -> None:
    value = "  Bearer secret-token\n" + ("界" * 105)
    preview = message_preview(value)
    assert preview["preview"].startswith("Bearer [REDACTED] ")
    assert len(preview["preview"]) == 100
    assert preview["truncated"] is True


def test_provider_stream_emits_message_without_raw_event_duplication() -> None:
    from arc_llm.providers._cli import EventAccumulator

    calls: list[tuple[str, object]] = []
    observer = _observer(calls)
    accumulator = EventAccumulator(
        "codex",
        observer,
        lambda _event: (None, None, None),
        extract_message=lambda event: (
            event.get("body") if isinstance(event.get("body"), str) else None
        ),
    )

    accumulator.feed(b'{"type":"progress","body":"assistant answer"}\n')

    assert calls == [
        (
            "llm_message",
            {
                "task_id": "task",
                "provider": "codex",
                "generation": 2,
                "host_turn_round": 1,
                "direction": "response",
                "message_kind": "assistant",
                "preview": "assistant answer",
                "truncated": False,
            },
        )
    ]
    assert accumulator.diagnostics()["event_count"] == 1
