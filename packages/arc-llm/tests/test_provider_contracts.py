from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from arc_llm import (
    FailureCategory,
    NativeResumeHandle,
    OutputInvalidError,
    ProviderFailure,
    ProviderRequest,
    ProviderResumeRequest,
    ProviderTerminalKind,
)
from arc_llm.output import select_output
from arc_llm.request import JsonOutput
from arc_llm.providers.claude import ClaudeAdapter
from arc_llm.providers._cli import classify_cli_failure
from arc_llm.providers.codex import CodexAdapter
from arc_llm.providers.kimi import KimiAdapter
from arc_llm.providers.process import ProcessResult
from arc_llm.providers.registry import default_registry


class Observer:
    def __init__(self) -> None:
        self.handles: list[NativeResumeHandle] = []
        self.progress_events: list[tuple[str, Any]] = []

    def native_handle(self, handle: NativeResumeHandle) -> None:
        self.handles.append(handle)

    def raw_event(self, _event: Any) -> None:
        pass

    def progress(self, kind: str, data: Any) -> None:
        self.progress_events.append((kind, data))



class Stop:
    def raise_if_requested(self) -> None:
        pass


class FakeRunner:
    def __init__(
        self,
        stdout: bytes,
        *,
        last_message: bytes | None = None,
        returncode: int = 0,
        stderr: bytes = b"",
    ) -> None:
        self.stdout = stdout
        self.last_message = last_message
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[dict[str, Any]] = []
        self.output_schemas: list[Any] = []

    def run(self, argv: Any, **kwargs: Any) -> ProcessResult:
        argv = list(argv)
        if "--output-schema" in argv:
            path = Path(argv[argv.index("--output-schema") + 1])
            if not path.is_absolute():
                path = kwargs["cwd"] / path
            self.output_schemas.append(json.loads(path.read_text(encoding="utf-8")))
        if "--output-last-message" in argv and self.last_message is not None:
            path = Path(argv[argv.index("--output-last-message") + 1])
            if not path.is_absolute():
                path = kwargs["cwd"] / path
            path.write_bytes(self.last_message)
        self.calls.append({"argv": argv, **kwargs})
        kwargs["on_stdout"](self.stdout)
        return ProcessResult(self.returncode, self.stdout, self.stderr)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    for name in ("inputs", "work", "host"):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    return workspace


def test_codex_uses_workspace_cwd_without_attachments_or_readonly_sandbox(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    runner = FakeRunner(
        b'{"type":"thread.started","thread_id":"thread-1"}\n',
        last_message=b'{"ok":true}',
    )
    adapter = CodexAdapter(binary="fake-codex", runner=runner, env={})
    observer = Observer()
    result = adapter.start(
        ProviderRequest("Read host/control.json", "model", {"type": "object"}, {}, 3, workspace),
        observer,
        Stop(),
    )

    assert result.terminal_kind is ProviderTerminalKind.COMPLETED
    assert result.native_handle == NativeResumeHandle("codex", "thread-1")
    call = runner.calls[0]
    assert call["cwd"] == workspace
    assert "--image" not in call["argv"]
    assert "--sandbox" not in call["argv"]
    assert call["stdin"] == b"Read host/control.json"
    assert "--output-schema" in call["argv"]
    assert runner.output_schemas == [{"type": "object"}]


def test_codex_projects_native_schema_but_selects_only_last_message(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    schema = {
        "type": "object",
        "required": ["item"],
        "properties": {
            "item": {"const": "guide"},
            "values": {"type": "array", "uniqueItems": True},
        },
    }
    runner = FakeRunner(
        b'{"type":"thread.started","thread_id":"thread-1"}\n'
        b'{"type":"item.completed","item":{"type":"agent_message","text":"{\\"item\\":\\"draft\\"}"}}\n',
        last_message=b'{"item":"guide","values":[1,1]}',
    )
    observer = Observer()
    result = CodexAdapter(binary="fake-codex", runner=runner, env={}).start(
        ProviderRequest("Read host/control.json", "model", schema, {}, 3, workspace),
        observer,
        Stop(),
    )

    native = runner.output_schemas[0]
    assert native["properties"]["item"]["type"] == "string"
    assert "uniqueItems" not in native["properties"]["values"]
    assert len(result.candidates) == 1
    assert result.candidates[0].text == '{"item":"guide","values":[1,1]}'
    assert (
        "llm_message",
        {
            "direction": "response",
            "message_kind": "assistant",
            "preview": '{"item":"draft"}',
            "truncated": False,
        },
    ) in observer.progress_events
    with pytest.raises(OutputInvalidError):
        select_output(result.candidates, JsonOutput(schema))


def test_codex_structured_invalid_request_is_not_delivered(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    event = {
        "type": "error",
        "error": {
            "code": "invalid_json_schema",
            "message": "unsupported schema",
            "param": "text.format.schema",
        },
    }
    runner = FakeRunner(
        json.dumps(event).encode() + b"\n",
        returncode=1,
    )
    execution = CodexAdapter(binary="fake-codex", runner=runner, env={}).start(
        ProviderRequest("Read host/control.json", "model", {"type": "object"}, {}, 3, workspace),
        Observer(),
        Stop(),
    )

    assert execution.terminal_kind is ProviderTerminalKind.FAILED
    assert execution.failure is not None
    assert execution.failure.details["provider_code"] == "invalid_json_schema"


def test_codex_accepts_fresh_completed_output_after_nonzero_exit(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    runner = FakeRunner(
        b'{"type":"thread.started","thread_id":"thread-1"}\n'
        b'{"type":"turn.completed"}\n',
        last_message=b'{"ok":true}',
        returncode=1,
        stderr=b"provider transport closed late",
    )

    execution = CodexAdapter(binary="fake-codex", runner=runner, env={}).start(
        ProviderRequest(
            "Read host/control.json",
            "model",
            {"type": "object", "properties": {"ok": {"type": "boolean"}}},
            {},
            3,
            workspace,
        ),
        Observer(),
        Stop(),
    )

    assert execution.terminal_kind is ProviderTerminalKind.COMPLETED
    assert execution.failure is None
    assert execution.candidates[0].text == '{"ok":true}'
    assert execution.diagnostics["warnings"] == [
        {
            "code": "provider_nonzero_exit_with_valid_output",
            "message": (
                "The provider returned a nonzero exit after writing a complete "
                "terminal response."
            ),
            "provider": "codex",
            "returncode": 1,
        }
    ]


def test_codex_nonzero_completed_turn_does_not_accept_stale_output(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    output = workspace / "host" / "codex-last-message.txt"
    output.write_text('{"stale":true}', encoding="utf-8")
    runner = FakeRunner(
        b'{"type":"turn.completed"}\n',
        returncode=1,
    )

    execution = CodexAdapter(binary="fake-codex", runner=runner, env={}).start(
        ProviderRequest("prompt", "model", {"type": "object"}, {}, 3, workspace),
        Observer(),
        Stop(),
    )

    assert execution.terminal_kind is ProviderTerminalKind.FAILED
    assert execution.diagnostics["last_message"] == "unavailable"
    assert not output.exists()


def test_codex_terminal_failure_overrides_completed_final_file(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    runner = FakeRunner(
        b'{"type":"turn.completed"}\n'
        b'{"type":"turn.failed","error":{"message":"late failure"}}\n',
        last_message=b'{"ok":true}',
        returncode=1,
    )

    execution = CodexAdapter(binary="fake-codex", runner=runner, env={}).start(
        ProviderRequest("prompt", "model", {"type": "object"}, {}, 3, workspace),
        Observer(),
        Stop(),
    )

    assert execution.terminal_kind is ProviderTerminalKind.FAILED
    assert execution.failure is not None
    assert execution.diagnostics["terminal_event_types"] == [
        "turn.completed",
        "turn.failed",
    ]


def test_codex_accepts_completion_after_an_earlier_error(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    runner = FakeRunner(
        b'{"type":"error","message":"transient stream error"}\n'
        b'{"type":"turn.completed"}\n',
        last_message=b'{"ok":true}',
        returncode=1,
    )

    execution = CodexAdapter(binary="fake-codex", runner=runner, env={}).start(
        ProviderRequest("prompt", "model", {"type": "object"}, {}, 3, workspace),
        Observer(),
        Stop(),
    )

    assert execution.terminal_kind is ProviderTerminalKind.COMPLETED
    assert execution.failure is None


def test_codex_accepts_configured_timeout_after_completed_file(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)

    class FailingRunner(FakeRunner):
        def run(self, argv: Any, **kwargs: Any) -> ProcessResult:
            super().run(argv, **kwargs)
            raise ProviderFailure(
                "idle timeout",
                category=FailureCategory.TIMEOUT,
            )

    runner = FailingRunner(
        b'{"type":"turn.completed"}\n',
        last_message=b'{"ok":true}',
    )
    execution = CodexAdapter(binary="fake-codex", runner=runner, env={}).start(
        ProviderRequest("prompt", "model", {"type": "object"}, {}, 3, workspace),
        Observer(),
        Stop(),
    )

    assert execution.terminal_kind is ProviderTerminalKind.COMPLETED
    assert execution.failure is None
    assert execution.diagnostics["runner_failure"] is True
    assert execution.diagnostics["terminal_event_types"] == ["turn.completed"]
    assert execution.diagnostics["warnings"][0]["code"] == (
        "provider_idle_timeout_with_valid_output"
    )


def test_claude_uses_workspace_cwd_and_keeps_prompt_free_of_artifact_paths(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    runner = FakeRunner(
        b'{"type":"result","session_id":"claude-1","structured_output":{"ok":true}}\n'
    )
    adapter = ClaudeAdapter(binary="fake-claude", runner=runner, env={})
    result = adapter.start(
        ProviderRequest("Read host/control.json", "model", {"type": "object"}, {}, 3, workspace),
        Observer(),
        Stop(),
    )

    assert result.terminal_kind is ProviderTerminalKind.COMPLETED
    call = runner.calls[0]
    assert call["cwd"] == workspace
    assert call["stdin"] == b"Read host/control.json"
    assert str(workspace) not in call["stdin"].decode()
    assert "--json-schema" in call["argv"]


def test_claude_accepts_successful_result_after_earlier_error_and_nonzero_exit(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    runner = FakeRunner(
        b'{"type":"result","is_error":true,"subtype":"error","result":"retry"}\n'
        b'{"type":"result","subtype":"success","result":"complete"}\n',
        returncode=1,
        stderr=b"provider transport closed late",
    )

    execution = ClaudeAdapter(
        binary="fake-claude", runner=runner, env={}
    ).start(
        ProviderRequest("prompt", "model", None, {}, 3, workspace),
        Observer(),
        Stop(),
    )

    assert execution.terminal_kind is ProviderTerminalKind.COMPLETED
    assert execution.candidates[-1].text == "complete"
    assert execution.diagnostics["warnings"][0]["code"] == (
        "provider_nonzero_exit_with_valid_output"
    )


def test_claude_late_error_result_overrides_successful_candidate(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    runner = FakeRunner(
        b'{"type":"result","subtype":"success","result":"draft"}\n'
        b'{"type":"result","is_error":true,"subtype":"error","result":"failed"}\n',
    )

    execution = ClaudeAdapter(
        binary="fake-claude", runner=runner, env={}
    ).start(
        ProviderRequest("prompt", "model", None, {}, 3, workspace),
        Observer(),
        Stop(),
    )

    assert execution.terminal_kind is ProviderTerminalKind.FAILED
    assert execution.failure is not None
    assert execution.failure.details["code"] == "claude_unsuccessful_result"


def test_claude_error_result_text_is_never_a_candidate(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    runner = FakeRunner(
        b'{"type":"result","is_error":true,"subtype":"error","result":"looks valid"}\n'
    )

    execution = ClaudeAdapter(
        binary="fake-claude", runner=runner, env={}
    ).start(
        ProviderRequest("prompt", "model", None, {}, 3, workspace),
        Observer(),
        Stop(),
    )

    assert execution.terminal_kind is ProviderTerminalKind.FAILED
    assert execution.candidates == ()


def test_kimi_print_mode_uses_prompt_stream_json_and_clean_session_resume(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    runner = FakeRunner(
        b'{"role":"assistant","content":"{\\"ok\\":true}","session_id":"kimi-1"}\n'
    )
    adapter = KimiAdapter(binary="fake-kimi", runner=runner, env={})
    started = adapter.start(
        ProviderRequest("Read host/control.json", "kimi-model", None, {}, 3, workspace),
        Observer(),
        Stop(),
    )

    start = runner.calls[0]
    assert start["cwd"] == workspace
    assert start["stdin"] == b""
    assert start["argv"][:5] == [
        "fake-kimi", "-p", "Read host/control.json", "--output-format", "stream-json"
    ]
    assert "--auto" not in start["argv"]
    assert started.native_handle == NativeResumeHandle("kimi", "kimi-1")

    adapter.resume(
        started.native_handle,
        ProviderResumeRequest("Read host/control.json", None, {}, 3, workspace),
        Observer(),
        Stop(),
    )
    resume = runner.calls[1]["argv"]
    assert resume[:5] == ["fake-kimi", "-S", "kimi-1", "-p", "Read host/control.json"]
    assert "--auto" not in resume


def test_kimi_accepts_final_message_after_nonzero_exit(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    runner = FakeRunner(
        b'{"role":"assistant","content":"complete","session_id":"kimi-1"}\n',
        returncode=1,
        stderr=b"provider transport closed late",
    )

    execution = KimiAdapter(binary="fake-kimi", runner=runner, env={}).start(
        ProviderRequest("prompt", "model", None, {}, 3, workspace),
        Observer(),
        Stop(),
    )

    assert execution.terminal_kind is ProviderTerminalKind.COMPLETED
    assert execution.candidates[0].text == "complete"
    assert "--final-message-only" in runner.calls[0]["argv"]


def test_direct_authority_enables_only_documented_provider_permission_flags(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    runtime = {"effective_host_mode": "direct"}

    codex_runner = FakeRunner(
        b'{"type":"thread.started","thread_id":"codex-1"}\n',
        last_message=b'{"ok":true}',
    )
    CodexAdapter(binary="fake-codex", runner=codex_runner, env={}).start(
        ProviderRequest("Read host/control.json", "model", None, runtime, 3, workspace),
        Observer(),
        Stop(),
    )
    codex_argv = codex_runner.calls[0]["argv"]
    assert "--dangerously-bypass-approvals-and-sandbox" in codex_argv
    assert codex_argv[codex_argv.index("-C") + 1] == str(workspace)

    claude_runner = FakeRunner(
        b'{"type":"result","session_id":"claude-1","structured_output":{"ok":true}}\n'
    )
    ClaudeAdapter(binary="fake-claude", runner=claude_runner, env={}).start(
        ProviderRequest("Read host/control.json", "model", None, runtime, 3, workspace),
        Observer(),
        Stop(),
    )
    assert "--dangerously-skip-permissions" in claude_runner.calls[0]["argv"]

    kimi_runner = FakeRunner(
        b'{"role":"assistant","content":"ok","session_id":"kimi-1"}\n'
    )
    KimiAdapter(binary="fake-kimi", runner=kimi_runner, env={}).start(
        ProviderRequest("Read host/control.json", "model", None, runtime, 3, workspace),
        Observer(),
        Stop(),
    )
    assert "--auto" in kimi_runner.calls[0]["argv"]


def test_default_registry_has_no_media_delivery_selection_surface() -> None:
    registry = default_registry()
    assert registry.names() == ("claude", "codex", "kimi")
    assert not hasattr(registry, "delivery_modes")
    assert not hasattr(registry, "supporting")


@pytest.mark.parametrize(
    ("stderr", "category"),
    (
        ("authentication failed", FailureCategory.AUTHENTICATION),
        ("HTTP 403: forbidden", FailureCategory.AUTHENTICATION),
        ("quota exceeded", FailureCategory.QUOTA),
        ("HTTP 429: retry-after: 0.1", FailureCategory.RATE_LIMIT),
        ("invalid request", FailureCategory.INVALID_REQUEST),
        ("insufficient evidence in research", FailureCategory.TRANSPORT),
        ("schema analysis completed", FailureCategory.TRANSPORT),
        ("author biography unavailable", FailureCategory.TRANSPORT),
    ),
)
def test_cli_failure_classification_uses_explicit_provider_diagnostics(
    stderr: str,
    category: FailureCategory,
) -> None:
    assert classify_cli_failure(stderr).category is category


def test_retry_after_is_clamped_for_operational_cooldown() -> None:
    assert classify_cli_failure(
        "HTTP 429; retry-after: 0.1"
    ).retry_after_seconds == 1
    assert classify_cli_failure(
        "HTTP 429; retry-after: 99999"
    ).retry_after_seconds == 3600
