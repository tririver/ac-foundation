from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from arc_llm import (
    DeliveryState,
    ErrorCode,
    FailureCategory,
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
from arc_llm.providers.claude import _parse_event as parse_claude_event
from arc_llm.providers.codex import CodexAdapter
from arc_llm.providers.codex import _parse_event as parse_codex_event
from arc_llm.providers.kimi import KimiAdapter, _parse_event as parse_kimi_event
from arc_llm.providers._cli import classify_cli_failure, run_cli
from arc_llm.providers.base import UsageAvailability
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
        self.output_schemas: list[Any] = []

    def run(self, argv: Any, **kwargs: Any) -> ProcessResult:
        kwargs["before_stdin"]()
        argv = list(argv)
        if "--output-schema" in argv:
            schema_path = Path(argv[argv.index("--output-schema") + 1])
            self.output_schemas.append(json.loads(schema_path.read_text()))
        self.calls.append({"argv": argv, **kwargs})
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
        assert caught.value.category is FailureCategory.INVALID_REQUEST
        assert caught.value.delivery is DeliveryState.MAY_HAVE_RUN
        assert caught.value.details["code"] == "reverse_operation_denied"
        assert caught.value.details["method"] == method


def test_codex_start_and_resume_preserve_native_history_and_schema_contract() -> None:
    schema = {
        "type": "object",
        "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
        "additionalProperties": False,
    }
    runner = FakeRunner(
        b'{"type":"thread.started","thread_id":"thread-1"}\n'
        b'{"type":"item.completed","item":'
        b'{"type":"agent_message","text":"{\\"ok\\":true}"}}\n'
    )
    adapter = CodexAdapter(binary="fake-codex", runner=runner, env={"PATH": "/tools"})

    started = adapter.start(
        ProviderRequest("large private prompt", "gpt-test", schema, {}, 17),
        Observer(),
        Cancel(),
    )
    start_argv = runner.calls[0]["argv"]
    schema_path = start_argv[start_argv.index("--output-schema") + 1]
    assert start_argv == [
        "fake-codex",
        "exec",
        "--json",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--model",
        "gpt-test",
        "--output-schema",
        schema_path,
        "-",
    ]
    assert runner.calls[0]["stdin"] == b"large private prompt"
    assert "large private prompt" not in start_argv
    assert runner.calls[0]["env"] == {"PATH": "/tools"}
    assert runner.calls[0]["idle_timeout_seconds"] == 17
    assert runner.output_schemas == [schema]
    assert not Path(schema_path).exists()
    assert started.native_handle == NativeResumeHandle("codex", "thread-1")

    adapter.resume(
        started.native_handle,
        ProviderResumeRequest("delta", schema, {}, 19),
        Observer(),
        Cancel(),
    )
    resume_argv = runner.calls[1]["argv"]
    resume_schema_path = resume_argv[resume_argv.index("--output-schema") + 1]
    assert resume_argv == [
        "fake-codex",
        "exec",
        "resume",
        "--json",
        "--ignore-user-config",
        "--ignore-rules",
        "-c",
        'sandbox_mode="read-only"',
        "thread-1",
        "--output-schema",
        resume_schema_path,
        "-",
    ]
    assert runner.calls[1]["stdin"] == b"delta"
    assert runner.calls[1]["idle_timeout_seconds"] == 19
    assert runner.output_schemas == [schema, schema]
    assert not Path(resume_schema_path).exists()


def test_claude_start_resume_and_structured_output_preference() -> None:
    schema = {
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "type": "object",
    }
    runner = FakeRunner(
        b'{"type":"result","session_id":"session-1",'
        b'"result":"not json","structured_output":{"ok":true},'
        b'"usage":{"input_tokens":4,"output_tokens":2}}\n'
    )
    adapter = ClaudeAdapter(binary="fake-claude", runner=runner, env={})

    started = adapter.start(
        ProviderRequest("prompt", "claude-test", schema, {}, 11),
        Observer(),
        Cancel(),
    )
    encoded_schema = json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert runner.calls[0]["argv"] == [
        "fake-claude",
        "--print",
        "--verbose",
        "--output-format",
        "stream-json",
        "--model",
        "claude-test",
        "--json-schema",
        encoded_schema,
    ]
    assert runner.calls[0]["stdin"] == b"prompt"
    assert started.native_handle == NativeResumeHandle("claude", "session-1")
    assert len(started.candidates) == 1
    assert started.candidates[0].has_value
    assert started.candidates[0].value == {"ok": True}
    assert started.usage is not None
    assert (started.usage.input_tokens, started.usage.output_tokens) == (4, 2)

    adapter.resume(
        started.native_handle,
        ProviderResumeRequest("continue", schema, {}, 13),
        Observer(),
        Cancel(),
    )
    assert runner.calls[1]["argv"] == [
        "fake-claude",
        "--print",
        "--verbose",
        "--output-format",
        "stream-json",
        "--resume",
        "session-1",
        "--json-schema",
        encoded_schema,
    ]
    assert runner.calls[1]["stdin"] == b"continue"
    assert "--no-session-persistence" not in runner.calls[0]["argv"]
    assert "--no-session-persistence" not in runner.calls[1]["argv"]


def test_kimi_resume_uses_session_without_creation_or_model_and_has_null_usage() -> None:
    schema = {
        "type": "object",
        "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
        "additionalProperties": False,
    }
    runner = FakeRunner(
        b'{"session_id":"kimi-1","result":{"content":"{\\"ok\\":true}"}}\n'
    )
    adapter = KimiAdapter(binary="fake-kimi", runner=runner, env={})

    started = adapter.start(
        ProviderRequest("private prompt", "kimi-test", schema, {}, 7),
        Observer(),
        Cancel(),
    )
    assert runner.calls[0]["argv"] == [
        "fake-kimi",
        "--acp",
        "--model",
        "kimi-test",
    ]
    expected_schema = json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    sent = runner.calls[0]["stdin"].decode()
    assert sent == (
        "private prompt\n\n"
        "Return exactly one JSON value satisfying this JSON Schema. "
        "Do not add prose or code fences.\n"
        f"JSON Schema:\n{expected_schema}"
    )
    assert "private prompt" not in runner.calls[0]["argv"]
    assert started.native_handle == NativeResumeHandle("kimi", "kimi-1")
    assert started.usage is None

    resumed = adapter.resume(
        started.native_handle,
        ProviderResumeRequest("follow-up", schema, {}, 9),
        Observer(),
        Cancel(),
    )
    assert runner.calls[1]["argv"] == [
        "fake-kimi",
        "--acp",
        "--session",
        "kimi-1",
    ]
    assert "--model" not in runner.calls[1]["argv"]
    assert runner.calls[1]["stdin"].decode().endswith(
        f"JSON Schema:\n{expected_schema}"
    )
    assert resumed.usage is None


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


@pytest.mark.parametrize(
    ("category", "code"),
    [
        (FailureCategory.AUTHENTICATION, ErrorCode.PROVIDER_AUTHENTICATION),
        (FailureCategory.QUOTA, ErrorCode.PROVIDER_QUOTA),
        (FailureCategory.RATE_LIMIT, ErrorCode.PROVIDER_RATE_LIMIT),
        (FailureCategory.UNAVAILABLE, ErrorCode.PROVIDER_UNAVAILABLE),
        (FailureCategory.INVALID_REQUEST, ErrorCode.PROVIDER_INVALID_REQUEST),
        (FailureCategory.SCHEMA, ErrorCode.INVALID_SCHEMA),
        (FailureCategory.TRANSPORT, ErrorCode.PROVIDER_TRANSPORT),
        (FailureCategory.TIMEOUT, ErrorCode.PROVIDER_TIMEOUT),
        (FailureCategory.CANCELLED, ErrorCode.CANCELLED),
        (FailureCategory.LOCAL_IO, ErrorCode.LOCAL_IO),
    ],
)
def test_provider_failure_categories_have_stable_machine_codes(
    category: FailureCategory, code: ErrorCode
) -> None:
    failure = ProviderFailure(
        "normalized",
        category=category,
        delivery=DeliveryState.NOT_DELIVERED,
    )
    assert failure.code is code


@pytest.mark.parametrize(
    ("diagnostic", "category", "retryable", "retry_after"),
    [
        ("401 unauthorized", FailureCategory.AUTHENTICATION, False, None),
        ("quota insufficient", FailureCategory.QUOTA, False, None),
        ("429 rate limit retry-after: 12", FailureCategory.RATE_LIMIT, True, 12.0),
        ("invalid request schema", FailureCategory.INVALID_REQUEST, False, None),
        ("connection reset", FailureCategory.TRANSPORT, True, None),
    ],
)
def test_cli_failure_classification_matrix(
    diagnostic: str,
    category: FailureCategory,
    retryable: bool,
    retry_after: float | None,
) -> None:
    failure = classify_cli_failure(diagnostic)
    assert failure.category is category
    assert failure.retryable is retryable
    assert failure.retry_after_seconds == retry_after


def test_provider_nonzero_exit_wins_over_structured_terminal_material() -> None:
    class NonzeroRunner:
        def run(self, argv, **kwargs):
            kwargs["before_stdin"]()
            stdout = (
                b'{"type":"item.completed","item":'
                b'{"type":"agent_message","text":"done"}}\n'
            )
            kwargs["on_stdout"](stdout)
            return ProcessResult(17, stdout, b"provider failed")

    result = run_cli(
        provider="codex",
        argv=("fake",),
        prompt="p",
        observer=Observer(),
        cancel=Cancel(),
        timeout=1,
        parse_event=parse_codex_event,
        runner=NonzeroRunner(),
        env={},
    )
    assert result.terminal_kind is ProviderTerminalKind.FAILED
    assert result.failure is not None
    assert result.diagnostics["returncode"] == 17
    assert result.candidates


def test_provider_usage_normalization_is_explicit() -> None:
    _, _, codex_usage = parse_codex_event(
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "cached_input_tokens": 3,
            },
        }
    )
    _, _, claude_usage = parse_claude_event(
        {
            "type": "result",
            "result": "done",
            "usage": {
                "input_tokens": 13,
                "output_tokens": 5,
                "cache_read_input_tokens": 2,
            },
        }
    )
    assert codex_usage is not None
    assert (
        codex_usage.input_tokens,
        codex_usage.output_tokens,
        codex_usage.cached_input_tokens,
    ) == (11, 7, 3)
    assert claude_usage is not None
    assert (
        claude_usage.input_tokens,
        claude_usage.output_tokens,
        claude_usage.cached_input_tokens,
    ) == (13, 5, 2)
    assert KimiAdapter().capabilities().usage is UsageAvailability.UNAVAILABLE


def test_provider_doctor_reports_availability_and_provider_warning(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "arc_llm.providers._cli.shutil.which",
        lambda binary: f"/tools/{binary}" if binary == "codex" else None,
    )
    codex = CodexAdapter().doctor()
    kimi = KimiAdapter().doctor()
    assert (codex.provider, codex.available, codex.executable) == (
        "codex",
        True,
        "/tools/codex",
    )
    assert not kimi.available
    assert kimi.executable is None
    assert kimi.details["warning"] == "provider_configuration_is_inherited"
