from __future__ import annotations

import pytest

from arc_llm import NativeResumeHandle
from arc_llm.progress import DurableProviderObserver


def test_observer_contract_matrix_is_bounded_body_free_and_delivery_ordered() -> None:
    calls: list[tuple[str, object]] = []

    class Effects:
        def mark_may_have_run(self, effect_id: str) -> None:
            calls.append(("delivery", effect_id))

        def save_output(self, effect_id: str, ref: object) -> None:
            calls.append(("output", (effect_id, ref)))

    class Events:
        def emit(self, kind: str, data: object) -> None:
            calls.append(("progress", (kind, data)))

    context = type("Context", (), {"effects": Effects(), "events": Events()})()
    handles: list[NativeResumeHandle] = []
    observer = DurableProviderObserver(
        context=context,
        effect_id="effect",
        on_handle=handles.append,
        raw_limit_bytes=12,
    )
    observer.before_delivery()
    observer.before_delivery()
    assert calls == [("delivery", "effect")]

    handle = NativeResumeHandle("codex", "thread")
    observer.native_handle(handle)
    assert handles == [handle]
    observer.raw_event("small")
    observer.raw_event("this diagnostic is too large")
    assert observer.raw_events == ["small"]
    assert observer.truncated

    observer.progress("phase", {"round": 1, "status": "running"})
    assert calls[-1][0] == "progress"
    with pytest.raises(ValueError):
        observer.progress("unsafe", {"nested": {"content": "paid output"}})
    marker = object()
    observer.response_saved(marker)
    assert calls[-1] == ("output", ("effect", marker))


@pytest.mark.parametrize(
    "key",
    ("text", "token", "content", "output", "delta", "prompt", "candidate", "result"),
)
def test_observer_uses_shared_progress_body_validator(key: str) -> None:
    class Effects:
        def mark_may_have_run(self, effect_id: str) -> None:
            pass

    class Events:
        def emit(self, kind: str, data: object) -> None:
            pass

    context = type("Context", (), {"effects": Effects(), "events": Events()})()
    observer = DurableProviderObserver(
        context=context,
        effect_id="effect",
        on_handle=lambda handle: None,
    )

    with pytest.raises(ValueError):
        observer.progress("unsafe", {"nested": [{key.capitalize(): "body"}]})


def test_observer_rejects_stringifiable_progress_body_key() -> None:
    class BodyKey:
        def __str__(self) -> str:
            return "ConTent"

    class Effects:
        def mark_may_have_run(self, effect_id: str) -> None:
            pass

    class Events:
        def emit(self, kind: str, data: object) -> None:
            pass

    context = type("Context", (), {"effects": Effects(), "events": Events()})()
    observer = DurableProviderObserver(
        context=context,
        effect_id="effect",
        on_handle=lambda handle: None,
    )

    with pytest.raises(ValueError):
        observer.progress("unsafe", {BodyKey(): "body"})
