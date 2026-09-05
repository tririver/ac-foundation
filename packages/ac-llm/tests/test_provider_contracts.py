from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ac_llm import (
    FailureCategory,
    NativeResumeHandle,
    OutputInvalidError,
    ProviderFailure,
    ProviderInputFile,
    ProviderRequest,
    ProviderResumeRequest,
    ProviderTerminalKind,
)
from ac_llm.host import host_turn_schema
from ac_llm.output import select_output
from ac_llm.request import JsonOutput
from ac_llm.providers.claude import (
    ClaudeAdapter,
    _extract_failure as extract_claude_failure,
)
from ac_llm.providers._cli import classify_cli_failure
from ac_llm.providers.codex import (
    CodexAdapter,
    _extract_failure as extract_codex_failure,
)
from ac_llm.providers.kimi import KimiAdapter
from ac_llm.providers.process import ProcessResult
from ac_llm.providers.registry import default_registry


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


def test_codex_direct_start_and_resume_keep_workspace_cwd_but_only_start_uses_cwd_flag(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    runner = FakeRunner(
        b'{"type":"thread.started","thread_id":"thread-1"}\n',
        last_message=b'{"ok":true}',
    )
    adapter = CodexAdapter(binary="fake-codex", runner=runner, env={})
    started = adapter.start(
        ProviderRequest(
            "Read host/control.json",
            "model",
            None,
            {"effective_host_mode": "direct"},
            3,
            workspace,
        ),
        Observer(),
        Stop(),
    )
    adapter.resume(
        started.native_handle,
        ProviderResumeRequest(
            "Continue.",
            None,
            {"effective_host_mode": "direct"},
            3,
            workspace,
        ),
        Observer(),
        Stop(),
    )

    start = runner.calls[0]
    assert start["cwd"] == workspace
    assert "--skip-git-repo-check" in start["argv"]
    assert "-C" in start["argv"]
    assert start["argv"][start["argv"].index("-C") + 1] == str(workspace)

    resume = runner.calls[1]
    assert resume["cwd"] == workspace
    assert "-C" not in resume["argv"]
    assert resume["argv"][:4] == [
        "fake-codex",
        "exec",
        "resume",
        "--json",
    ]
    assert "--skip-git-repo-check" in resume["argv"]
    assert "--dangerously-bypass-approvals-and-sandbox" in resume["argv"]
    assert "thread-1" in resume["argv"]


def test_codex_start_and_resume_pin_model_reasoning_effort(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    runner = FakeRunner(
        b'{"type":"thread.started","thread_id":"thread-1"}\n',
        last_message=b'{"ok":true}',
    )
    adapter = CodexAdapter(binary="fake-codex", runner=runner, env={})
    started = adapter.start(
        ProviderRequest(
            "Translate.",
            "gpt-5.6-terra",
            None,
            {},
            3,
            workspace,
            reasoning_effort="high",
        ),
        Observer(),
        Stop(),
    )
    assert started.native_handle is not None
    adapter.resume(
        started.native_handle,
        ProviderResumeRequest(
            "Continue.",
            None,
            {},
            3,
            workspace,
            model="gpt-5.6-terra",
            reasoning_effort="high",
        ),
        Observer(),
        Stop(),
    )

    for call in runner.calls:
        argv = call["argv"]
        assert argv[argv.index("--model") + 1] == "gpt-5.6-terra"
        assert "model_reasoning_effort=\"high\"" in argv


def test_codex_bounded_profile_attaches_images_without_shell_or_host_bypass(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    image = Path("inputs/0000-page.png")
    (workspace / image).write_bytes(b"png")
    runner = FakeRunner(
        b'{"type":"thread.started","thread_id":"thread-1"}\n',
        last_message=b'{"ok":true}',
    )
    adapter = CodexAdapter(binary="fake-codex", runner=runner, env={})

    started = adapter.start(
        ProviderRequest(
            "Review page.",
            "model",
            None,
            {"effective_host_mode": "direct", "execution_profile": "bounded"},
            3,
            workspace,
            inputs=(ProviderInputFile("page", "image/png", image),),
        ),
        Observer(),
        Stop(),
    )
    assert started.native_handle is not None
    adapter.resume(
        started.native_handle,
        ProviderResumeRequest(
            "Review page again.",
            None,
            {"effective_host_mode": "direct", "execution_profile": "bounded"},
            3,
            workspace,
            inputs=(ProviderInputFile("page", "image/png", image),),
        ),
        Observer(),
        Stop(),
    )

    argv = runner.calls[0]["argv"]
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    assert "--ignore-user-config" in argv
    assert "--ignore-rules" in argv
    assert "--skip-git-repo-check" in argv
    assert argv[argv.index("--sandbox") + 1] == "danger-full-access"
    assert "default_tools_enabled=false" in argv
    assert [argv[index + 1] for index, item in enumerate(argv) if item == "--disable"] == [
        "multi_agent",
    ]
    assert argv[argv.index("--image") + 1] == str(image)
    resume_argv = runner.calls[1]["argv"]
    assert "--dangerously-bypass-approvals-and-sandbox" not in resume_argv
    assert "--skip-git-repo-check" in resume_argv
    assert 'sandbox_mode="danger-full-access"' in resume_argv
    assert "default_tools_enabled=false" in resume_argv
    assert [
        resume_argv[index + 1]
        for index, item in enumerate(resume_argv)
        if item == "--disable"
    ] == ["multi_agent"]
    assert resume_argv[resume_argv.index("--image") + 1] == str(image)
    assert adapter.capabilities().config_isolation.value == "inherited"


def test_codex_rejects_output_when_required_image_cannot_be_loaded(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    image = Path("inputs/0000-page.png")
    (workspace / image).write_bytes(b"png")
    runner = FakeRunner(
        b'{"type":"thread.started","thread_id":"thread-1"}\n'
        b'{"type":"turn.completed"}\n',
        last_message=b'{"checks":"passed"}',
        stderr=(
            b"ERROR unable to locate image at `inputs/0000-page.png`: "
            b"fs sandbox helper failed\n"
        ),
    )

    execution = CodexAdapter(binary="fake-codex", runner=runner, env={}).start(
        ProviderRequest(
            "Review page.",
            "model",
            None,
            {"execution_profile": "bounded"},
            3,
            workspace,
            inputs=(ProviderInputFile("page", "image/png", image),),
        ),
        Observer(),
        Stop(),
    )

    assert execution.terminal_kind is ProviderTerminalKind.FAILED
    assert execution.candidates == ()
    assert execution.failure is not None
    assert execution.failure.category is FailureCategory.LOCAL_IO
    assert execution.failure.details["code"] == "input_attachment_unavailable"


def test_codex_projects_native_schema_but_selects_only_last_message(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    schema = {
        "type": "object",
        "required": ["item"],
        "properties": {
            "item": {
                "const": "guide",
                "minLength": 1,
                "pattern": "guide",
            },
            "values": {
                "type": "array",
                "uniqueItems": True,
                "minItems": 1,
                "maxItems": 2,
            },
        },
        "allOf": [{"properties": {"item": {"minLength": 1}}}],
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
    serialized_native = json.dumps(native)
    for unsupported in (
        "allOf",
        "maxItems",
        "minItems",
        "minLength",
        "pattern",
        "uniqueItems",
    ):
        assert f'"{unsupported}"' not in serialized_native
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


def test_codex_projects_nested_one_of_to_supported_any_of(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    part_variants = [
        {
            "type": "object",
            "properties": {
                "kind": {"const": "text"},
                "text": {"type": "string"},
            },
            "required": ["kind", "text"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "atom"},
                "atom_id": {"type": "string"},
            },
            "required": ["kind", "atom_id"],
            "additionalProperties": False,
        },
    ]
    schema = {
        "type": "object",
        "properties": {
            "parts": {
                "type": "array",
                "items": {"oneOf": part_variants},
            }
        },
        "required": ["parts"],
        "additionalProperties": False,
    }
    runner = FakeRunner(
        b'{"type":"thread.started","thread_id":"thread-1"}\n'
        b'{"type":"turn.completed"}\n',
        last_message=b'{"parts":[{"kind":"text","text":"translated"}]}',
    )

    result = CodexAdapter(binary="fake-codex", runner=runner, env={}).start(
        ProviderRequest("Translate.", "model", schema, {}, 3, workspace),
        Observer(),
        Stop(),
    )

    native_items = runner.output_schemas[0]["properties"]["parts"]["items"]
    assert "oneOf" not in native_items
    assert [item["properties"]["kind"] for item in native_items["anyOf"]] == [
        {"const": "text", "type": "string"},
        {"const": "atom", "type": "string"},
    ]
    assert all(
        item["type"] == "object"
        and item["additionalProperties"] is False
        and set(item["required"]) == set(item["properties"])
        for item in native_items["anyOf"]
    )
    assert select_output(result.candidates, JsonOutput(schema)) == {
        "parts": [{"kind": "text", "text": "translated"}]
    }


def test_codex_native_host_turn_schema_keeps_root_definitions(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    paper_choice_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["paper_id"],
        "properties": {"paper_id": {"type": "string"}},
    }
    schema = host_turn_schema(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["selected_foundation"],
            "properties": {
                "selected_foundation": {"$ref": "#/$defs/paper_choice"},
            },
            "$defs": {"paper_choice": paper_choice_schema},
        }
    )
    runner = FakeRunner(
        b'{"type":"thread.started","thread_id":"thread-1"}\n'
        b'{"type":"turn.completed"}\n',
        last_message=(
            b'{"schema_version":"ac.llm.host_turn.v1","state":"complete",'
            b'"result":{"selected_foundation":{"paper_id":"paper-1"}},'
            b'"host_request":null}'
        ),
    )

    CodexAdapter(binary="fake-codex", runner=runner, env={}).start(
        ProviderRequest("Select one foundation.", "model", schema, {}, 3, workspace),
        Observer(),
        Stop(),
    )

    native = runner.output_schemas[0]
    assert native["$defs"] == {"paper_choice": paper_choice_schema}
    selected = native["properties"]["result"]["anyOf"][0]["properties"][
        "selected_foundation"
    ]
    assert selected == {"$ref": "#/$defs/paper_choice"}


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


@pytest.mark.parametrize(
    "category",
    [FailureCategory.RATE_LIMIT, FailureCategory.UNAVAILABLE],
)
def test_later_success_clears_retryable_provider_failure(
    category: FailureCategory,
) -> None:
    from ac_llm.providers._cli import EventAccumulator

    accumulator = EventAccumulator(
        "test",
        Observer(),
        lambda event: (
            None,
            None,
            None,
        ),
        extract_failure=lambda event: (
            ProviderFailure("retryable", category=category, retryable=True)
            if event["type"] == "error"
            else None
        ),
    )

    accumulator.feed(
        b'{"type":"error"}\n{"type":"turn.completed"}\n'
    )

    assert accumulator.has_success_evidence
    assert accumulator.failure is None


def test_codex_definitive_error_is_not_cleared_by_later_completion(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    runner = FakeRunner(
        b'{"type":"error","error":{"code":"invalid_request","message":"bad input"}}\n'
        b'{"type":"turn.completed"}\n',
        last_message=b'{"ok":true}',
    )

    execution = CodexAdapter(binary="fake-codex", runner=runner, env={}).start(
        ProviderRequest("prompt", "model", {"type": "object"}, {}, 3, workspace),
        Observer(),
        Stop(),
    )

    assert execution.terminal_kind is ProviderTerminalKind.FAILED
    assert execution.failure is not None
    assert execution.failure.category is FailureCategory.INVALID_REQUEST


def test_typed_provider_failure_outranks_runner_timeout(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    class TimeoutRunner(FakeRunner):
        def run(self, argv: Any, **kwargs: Any) -> ProcessResult:
            super().run(argv, **kwargs)
            raise ProviderFailure(
                "idle timeout",
                category=FailureCategory.TIMEOUT,
            )

    runner = TimeoutRunner(
        b'{"type":"error","error":{"code":"invalid_request","message":"bad input"}}\n'
    )
    execution = CodexAdapter(binary="fake-codex", runner=runner, env={}).start(
        ProviderRequest("prompt", "model", {"type": "object"}, {}, 3, workspace),
        Observer(),
        Stop(),
    )

    assert execution.terminal_kind is ProviderTerminalKind.FAILED
    assert execution.failure is not None
    assert execution.failure.category is FailureCategory.INVALID_REQUEST


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
    assert registry.names() == ("claude", "codex", "dsh", "kimi")
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


@pytest.mark.parametrize(
    ("code", "message", "category"),
    (
        (
            "authentication_error",
            "HTTP 401 authentication failed",
            FailureCategory.AUTHENTICATION,
        ),
        (
            "insufficient_quota",
            "quota exceeded",
            FailureCategory.QUOTA,
        ),
        (
            "rate_limit_exceeded",
            "HTTP 429 retry-after: 61",
            FailureCategory.RATE_LIMIT,
        ),
    ),
)
def test_codex_structured_failures_preserve_typed_category(
    code: str,
    message: str,
    category: FailureCategory,
) -> None:
    failure = extract_codex_failure(
        {
            "type": "turn.failed",
            "error": {"code": code, "message": message},
        }
    )

    assert failure is not None
    assert failure.category is category
    if category is FailureCategory.RATE_LIMIT:
        assert failure.retry_after_seconds == 61


def test_claude_structured_failure_preserves_rate_limit_retry_after() -> None:
    failure = extract_claude_failure(
        {
            "type": "result",
            "is_error": True,
            "subtype": "error_during_execution",
            "result": "HTTP 429 rate_limit_exceeded; retry-after: 45",
        }
    )

    assert failure is not None
    assert failure.category is FailureCategory.RATE_LIMIT
    assert failure.retry_after_seconds == 45
