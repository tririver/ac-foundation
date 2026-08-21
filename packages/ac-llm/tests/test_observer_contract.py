from __future__ import annotations

from typing import Any

from ac_llm import NativeResumeHandle
from ac_llm.progress import DurableProviderObserver, message_preview


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


def test_native_handle_observer_failure_is_diagnostic_only() -> None:
    def fail_handle(_handle: NativeResumeHandle) -> None:
        raise OSError("state temporarily unavailable")

    observer = DurableProviderObserver(
        context=_context([]),
        on_handle=fail_handle,
        task_id="task",
        provider="codex",
        generation=2,
        host_turn_round=1,
    )

    observer.native_handle(NativeResumeHandle("codex", "thread"))

    assert observer.observation_errors == ["OSError"]


def test_message_preview_redacts_normalizes_and_counts_unicode_codepoints() -> None:
    value = (
        "  Bearer secret-token\n"
        "password=hunter2 x-api-key: private-key "
        "github_pat_abcdefghijklmnopqrstuvwxyz "
        + ("界" * 105)
    )
    preview = message_preview(value)
    assert preview["preview"].startswith("Bearer [REDACTED] ")
    assert "hunter2" not in preview["preview"]
    assert "private-key" not in preview["preview"]
    assert "github_pat_" not in preview["preview"]
    assert len(preview["preview"]) == 100
    assert preview["truncated"] is True


def test_provider_stream_emits_message_without_raw_event_duplication() -> None:
    from ac_llm.providers._cli import EventAccumulator

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


def test_provider_raw_diagnostics_redact_secret_keys_and_embedded_credentials() -> None:
    from ac_llm.providers._cli import EventAccumulator

    accumulator = EventAccumulator(
        "codex",
        _observer([]),
        lambda _event: (None, None, None),
    )

    accumulator.feed(
        b'{"type":"error","access_token":"private-token",'
        b'"message":"Cookie: session-secret password=hunter2"}\n'
    )

    raw_event = accumulator.diagnostics()["raw_events"][0]
    assert raw_event["access_token"] == "[REDACTED]"
    assert "session-secret" not in raw_event["message"]
    assert "hunter2" not in raw_event["message"]
    assert raw_event["message"] == (
        "Cookie: [REDACTED] password=[REDACTED]"
    )


def test_free_text_redaction_covers_quoted_env_and_basic_credentials() -> None:
    from ac_llm.diagnostics import redact_text

    value = (
        '{"api_key":"json-secret"} '
        "OPENAI_API_KEY=environment-secret "
        "Authorization: Basic Zm9vOmJhcg== "
        "'refresh_token': 'quoted secret'"
    )

    redacted = redact_text(value)

    for secret in (
        "json-secret",
        "environment-secret",
        "Zm9vOmJhcg==",
        "quoted secret",
    ):
        assert secret not in redacted
