from __future__ import annotations

import pytest

from arc_llm import NativeResumeHandle
from arc_llm.progress import DurableProviderObserver


def _context(calls: list[tuple[str, object]]):
    class Events:
        def emit(self, kind: str, data: object) -> None:
            calls.append((kind, data))

    return type("Context", (), {"events": Events()})()


def test_observer_projects_handles_bounded_raw_diagnostics_and_progress() -> None:
    calls: list[tuple[str, object]] = []
    handles: list[NativeResumeHandle] = []
    observer = DurableProviderObserver(
        context=_context(calls), on_handle=handles.append, raw_limit_bytes=12
    )

    handle = NativeResumeHandle("codex", "thread")
    observer.native_handle(handle)
    assert handles == [handle]
    observer.raw_event("small")
    observer.raw_event("this diagnostic is too large")
    assert observer.raw_events == ["small"]
    assert observer.truncated

    observer.progress("phase", {"round": 1, "status": "running"})
    assert calls == [("phase", {"round": 1, "status": "running"})]
    with pytest.raises(ValueError):
        observer.progress("unsafe", {"nested": {"content": "paid output"}})


@pytest.mark.parametrize(
    "key",
    ("text", "token", "content", "output", "delta", "prompt", "candidate", "result"),
)
def test_observer_rejects_nested_progress_body_keys(key: str) -> None:
    observer = DurableProviderObserver(context=_context([]), on_handle=lambda _handle: None)
    with pytest.raises(ValueError):
        observer.progress("unsafe", {"nested": [{key.capitalize(): "body"}]})


def test_observer_rejects_stringifiable_progress_body_key() -> None:
    class BodyKey:
        def __str__(self) -> str:
            return "ConTent"

    observer = DurableProviderObserver(context=_context([]), on_handle=lambda _handle: None)
    with pytest.raises(ValueError):
        observer.progress("unsafe", {BodyKey(): "body"})
