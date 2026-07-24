from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from arc_llm import (
    ModelSelection,
    NativeResumeHandle,
    ProviderRequest,
    ProviderResumeRequest,
    ProviderTerminalKind,
)
from arc_llm.config import detect_host, resolve_model_selection
from arc_llm.errors import ProviderFailure
from arc_llm.progress import DurableProviderObserver
from arc_llm.providers.claude import ClaudeAdapter
from arc_llm.providers.codex import CodexAdapter
from arc_llm.providers.kimi import KimiAdapter, _parse_event as parse_kimi_event
from arc_llm.providers._cli import classify_cli_failure
from arc_llm.providers.process import ProcessResult


class Observer:
    def __init__(self) -> None:
        self.deliveries = 0
        self.handles: list[NativeResumeHandle] = []
        self.raw: list[Any] = []

    def before_delivery(self) -> None:
        self.deliveries += 1

    def native_handle(self, handle: NativeResumeHandle) -> None:
        self.handles.append(handle)

    def raw_event(self, event: Any) -> None:
        self.raw.append(event)

    def progress(self, kind: str, data: Any) -> None:
        pass

    def response_saved(self, ref: Any) -> None:
        pass


class Cancel:
    def raise_if_requested(self) -> None:
        pass


class FakeRunner:
    def __init__(self, stdout: bytes) -> None:
        self.stdout = stdout
        self.calls: list[dict[str, Any]] = []

    def run(self, argv: Any, **kwargs: Any) -> ProcessResult:
        kwargs["before_stdin"]()
        self.calls.append({"argv": list(argv), **kwargs})
        kwargs["on_stdout"](self.stdout)
        return ProcessResult(0, self.stdout, b"")


@pytest.mark.parametrize(
    ("adapter_type", "event", "handle"),
    [
        (
            CodexAdapter,
            {
                "type": "thread.started",
                "thread_id": "codex-thread",
            },
            "codex-thread",
        ),
        (
            ClaudeAdapter,
            {
                "type": "result",
                "session_id": "claude-session",
                "structured_output": {"ok": True},
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
            "claude-session",
        ),
        (
            KimiAdapter,
            {
                "session_id": "kimi-session",
                "result": {"content": '{"ok":true}'},
            },
            "kimi-session",
        ),
    ],
)
def test_provider_adapters_share_start_resume_contract(
    adapter_type: type, event: dict[str, Any], handle: str
) -> None:
    terminal = (
        {"type": "item.completed", "item": {"type": "agent_message", "text": '{"ok":true}'}}
        if adapter_type is CodexAdapter
        else None
    )
    lines = [event] + ([] if terminal is None else [terminal])
    runner = FakeRunner(
        b"".join(json.dumps(item).encode() + b"\n" for item in lines)
    )
    adapter = adapter_type(binary="fake", runner=runner, env={})
    observer = Observer()
    request = ProviderRequest(
        "prompt",
        "model",
        {"type": "object"},
        {},
        2,
    )
    started = adapter.start(request, observer, Cancel())
    assert started.terminal_kind is ProviderTerminalKind.COMPLETED
    assert started.native_handle is not None
    assert started.native_handle.value == handle
    assert observer.deliveries == 1
    assert observer.handles[-1].value == handle
    assert runner.calls[-1]["stdin"]

    resumed = adapter.resume(
        NativeResumeHandle(adapter.name, handle),
        ProviderResumeRequest("follow-up", {"type": "object"}, {}, 2),
        Observer(),
        Cancel(),
    )
    assert resumed.terminal_kind is ProviderTerminalKind.COMPLETED


def test_provider_schema_is_transported_by_each_adapter() -> None:
    codex_runner = FakeRunner(
        b'{"type":"item.completed","item":{"type":"agent_message","text":"{}"}}\n'
    )
    CodexAdapter(binary="fake", runner=codex_runner, env={}).start(
        ProviderRequest("p", "m", {"type": "object"}, {}, 2),
        Observer(),
        Cancel(),
    )
    assert "--output-schema" in codex_runner.calls[0]["argv"]
    assert "--ignore-user-config" in codex_runner.calls[0]["argv"]
    assert codex_runner.calls[0]["argv"][
        codex_runner.calls[0]["argv"].index("--sandbox") + 1
    ] == "read-only"

    claude_runner = FakeRunner(
        b'{"type":"result","session_id":"s","structured_output":{}}\n'
    )
    ClaudeAdapter(binary="fake", runner=claude_runner, env={}).start(
        ProviderRequest("p", "m", {"type": "object"}, {}, 2),
        Observer(),
        Cancel(),
    )
    assert "--json-schema" in claude_runner.calls[0]["argv"]

    kimi_runner = FakeRunner(
        b'{"session_id":"s","result":{"content":"{}"}}\n'
    )
    KimiAdapter(binary="fake", runner=kimi_runner, env={}).start(
        ProviderRequest("p", "m", {"type": "object"}, {}, 2),
        Observer(),
        Cancel(),
    )
    assert b"JSON Schema" in kimi_runner.calls[0]["stdin"]


def test_kimi_denies_reverse_permission_and_filesystem_requests() -> None:
    for method in (
        "session/request_permission",
        "fs/read_text_file",
        "fs/write_text_file",
    ):
        with pytest.raises(ProviderFailure) as caught:
            parse_kimi_event({"jsonrpc": "2.0", "id": 1, "method": method})
        assert caught.value.details["code"] == "reverse_operation_denied"


def test_auto_provider_and_model_resolution_is_host_deterministic() -> None:
    assert detect_host(env={"CODEX_THREAD_ID": "x"}).host == "codex"
    resolved = resolve_model_selection(
        ModelSelection(),
        env={"CLAUDE_CODE": "1"},
        available=("codex", "claude"),
    )
    assert resolved.provider == "claude"
    assert resolved.model


def test_observer_bounds_raw_events_and_rejects_nested_response_body() -> None:
    class Effects:
        def mark_may_have_run(self, effect_id: str) -> None:
            pass

    class Events:
        def emit(self, kind: str, data: Any) -> None:
            pass

    context = type("Context", (), {"effects": Effects(), "events": Events()})()
    observer = DurableProviderObserver(
        context=context,
        effect_id="effect",
        on_handle=lambda handle: None,
        raw_limit_bytes=8,
    )
    observer.raw_event("0123456789")
    assert observer.truncated
    assert observer.raw_events == []
    with pytest.raises(ValueError):
        observer.progress("safe", {"nested": {"content": "secret"}})


def test_provider_diagnostic_redacts_credentials() -> None:
    failure = classify_cli_failure(
        "unauthorized Authorization: Bearer secret-token api sk-1234567890ABCDEF"
    )
    diagnostic = failure.details["diagnostic"]
    assert "secret-token" not in diagnostic
    assert "1234567890ABCDEF" not in diagnostic
