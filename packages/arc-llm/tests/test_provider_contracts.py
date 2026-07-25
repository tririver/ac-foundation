from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from arc_jobs import StoppedError

from arc_llm import (
    DeliveryState,
    ErrorCode,
    ExecutionLimits,
    FailureCategory,
    JsonOutput,
    LLMClient,
    LLMExecutionOptions,
    LLMPaused,
    LLMRequest,
    ModelSelection,
    NativeResumeHandle,
    ProviderExecution,
    ProviderInput,
    ProviderRequest,
    ProviderResumeRequest,
    ProviderTerminalKind,
    OutputInvalidError,
)
from arc_llm.output import CandidateMaterial, select_output
from arc_llm.config import detect_host, resolve_model_selection
from arc_llm.errors import ProviderFailure
from arc_llm.progress import DurableProviderObserver
from arc_llm.providers.claude import ClaudeAdapter
from arc_llm.providers.claude import _parse_event as parse_claude_event
from arc_llm.providers.codex import CodexAdapter
from arc_llm.providers.codex import _parse_event as parse_codex_event
from arc_llm.providers.acp import _ACPClient
from arc_llm.providers.acp import OfficialACPRunner
from arc_llm.providers.base import InputDeliveryMode
from arc_llm.providers.kimi import KimiAdapter
from arc_llm.providers._cli import classify_cli_failure, run_cli
from arc_llm.providers.base import UsageAvailability
from arc_llm.providers.process import ProcessResult
from arc_llm.providers.registry import ProviderRegistry, default_registry


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
        kwargs["before_stdin"]()
        argv = list(argv)
        if "--output-schema" in argv:
            schema_path = Path(argv[argv.index("--output-schema") + 1])
            self.output_schemas.append(json.loads(schema_path.read_text()))
        if "--output-last-message" in argv and self.last_message is not None:
            output_path = Path(argv[argv.index("--output-last-message") + 1])
            output_path.write_bytes(self.last_message)
        self.calls.append({"argv": argv, **kwargs})
        kwargs["on_stdout"](self.stdout)
        return ProcessResult(self.returncode, self.stdout, self.stderr)


class FakeACPRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> ProviderExecution:
        self.calls.append(kwargs)
        kwargs["observer"].before_delivery()
        session_id = kwargs["session_id"] or "kimi-session"
        handle = NativeResumeHandle("kimi", session_id)
        kwargs["observer"].native_handle(handle)
        return ProviderExecution(
            ProviderTerminalKind.COMPLETED,
            candidates=(CandidateMaterial(text='{"ok":true}', terminal=True),),
            native_handle=handle,
        )


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
    started = adapter.start(request, observer, Stop())
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
        Stop(),
    )
    assert resumed.terminal_kind is ProviderTerminalKind.COMPLETED


def test_provider_schema_is_transported_by_each_adapter() -> None:
    codex_runner = FakeRunner(
        b'{"type":"item.completed","item":{"type":"agent_message","text":"{}"}}\n'
    )
    CodexAdapter(binary="fake", runner=codex_runner, env={}).start(
        ProviderRequest("p", "m", {"type": "object"}, {}, 2),
        Observer(),
        Stop(),
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
        Stop(),
    )
    assert "--json-schema" in claude_runner.calls[0]["argv"]

    kimi_runner = FakeACPRunner()
    KimiAdapter(binary="fake", acp_runner=kimi_runner, env={}).start(
        ProviderRequest("p", "m", {"type": "object"}, {}, 2),
        Observer(),
        Stop(),
    )
    assert "JSON Schema" in kimi_runner.calls[0]["prompt"]


def test_kimi_acp_client_denies_reverse_permission_and_filesystem_requests() -> None:
    from acp import RequestError

    client = _ACPClient("kimi", [], Observer())
    for operation in (
        client.request_permission,
        client.read_text_file,
        client.write_text_file,
    ):
        with pytest.raises(RequestError):
            asyncio.run(operation())


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
        b'{"type":"agent_message","text":"{\\"ok\\":true}"}}\n',
        last_message=b'{"ok":true}',
    )
    adapter = CodexAdapter(binary="fake-codex", runner=runner, env={"PATH": "/tools"})

    started = adapter.start(
        ProviderRequest("large private prompt", "gpt-test", schema, {}, 17),
        Observer(),
        Stop(),
    )
    start_argv = runner.calls[0]["argv"]
    schema_path = start_argv[start_argv.index("--output-schema") + 1]
    output_path = start_argv[start_argv.index("--output-last-message") + 1]
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
        "--output-last-message",
        output_path,
        "-",
    ]
    assert runner.calls[0]["stdin"] == b"large private prompt"
    assert "large private prompt" not in start_argv
    assert runner.calls[0]["env"] == {"PATH": "/tools"}
    assert runner.calls[0]["idle_timeout_seconds"] == 17
    assert runner.output_schemas == [schema]
    assert not Path(schema_path).exists()
    assert not Path(output_path).exists()
    assert started.native_handle == NativeResumeHandle("codex", "thread-1")

    adapter.resume(
        started.native_handle,
        ProviderResumeRequest("delta", schema, {}, 19),
        Observer(),
        Stop(),
    )
    resume_argv = runner.calls[1]["argv"]
    resume_schema_path = resume_argv[resume_argv.index("--output-schema") + 1]
    resume_output_path = resume_argv[
        resume_argv.index("--output-last-message") + 1
    ]
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
        "--output-last-message",
        resume_output_path,
        "-",
    ]
    assert runner.calls[1]["stdin"] == b"delta"
    assert runner.calls[1]["idle_timeout_seconds"] == 19
    assert runner.output_schemas == [schema, schema]
    assert not Path(resume_schema_path).exists()
    assert not Path(resume_output_path).exists()


def test_codex_uses_last_message_as_its_only_terminal_candidate() -> None:
    stdout = (
        b'{"type":"thread.started","thread_id":"thread-1"}\n'
        b'{"type":"item.completed","item":'
        b'{"type":"agent_message","text":"{\\"ok\\":false}"}}\n'
        b'{"type":"item.completed","item":'
        b'{"type":"agent_message","text":"{\\"ok\\":false}"}}\n'
        b'{"type":"turn.completed","usage":'
        b'{"input_tokens":7,"output_tokens":3,"cached_input_tokens":2}}\n'
    )
    runner = FakeRunner(stdout, last_message=b'{"ok":true}')
    observer = Observer()

    result = CodexAdapter(binary="fake-codex", runner=runner, env={}).start(
        ProviderRequest("prompt", "model", {"type": "object"}, {}, 2),
        observer,
        Stop(),
    )

    assert CodexAdapter.compatibility_version == "codex-jsonl.v3"
    assert result.native_handle == NativeResumeHandle("codex", "thread-1")
    assert result.usage is not None
    assert result.usage.input_tokens == 7
    assert result.candidates == (CandidateMaterial(text='{"ok":true}', terminal=True),)
    assert result.diagnostics["last_message"] == "present"
    assert len(result.diagnostics["raw_events"]) == 4
    assert len(observer.raw) == 4
    assert select_output(result.candidates, JsonOutput({"type": "object"})) == {
        "ok": True
    }


@pytest.mark.parametrize(
    ("last_message", "diagnostic"),
    (
        (None, "empty"),
        (b'{"other":true}', "present"),
        (b"\xff", "unavailable"),
    ),
)
def test_codex_missing_or_invalid_last_message_uses_output_invalid_path(
    last_message: bytes | None,
    diagnostic: str,
) -> None:
    runner = FakeRunner(
        b'{"type":"item.completed","item":'
        b'{"type":"agent_message","text":"{\\"ok\\":true}"}}\n',
        last_message=last_message,
    )

    result = CodexAdapter(binary="fake-codex", runner=runner, env={}).start(
        ProviderRequest("prompt", "model", {"type": "object"}, {}, 2),
        Observer(),
        Stop(),
    )

    assert result.terminal_kind is ProviderTerminalKind.COMPLETED
    assert result.diagnostics["last_message"] == diagnostic
    with pytest.raises(OutputInvalidError):
        select_output(
            result.candidates,
            JsonOutput(
                {
                    "type": "object",
                    "required": ["ok"],
                    "properties": {"ok": {"type": "boolean"}},
                    "additionalProperties": False,
                }
            ),
        )


def test_codex_nonzero_exit_wins_over_last_message_file() -> None:
    runner = FakeRunner(
        b'{"type":"thread.started","thread_id":"thread-1"}\n',
        last_message=b'{"ok":true}',
        returncode=17,
        stderr=b"connection reset",
    )

    result = CodexAdapter(binary="fake-codex", runner=runner, env={}).start(
        ProviderRequest("prompt", "model", {"type": "object"}, {}, 2),
        Observer(),
        Stop(),
    )

    assert result.terminal_kind is ProviderTerminalKind.FAILED
    assert result.failure is not None
    assert result.failure.category is FailureCategory.TRANSPORT
    assert result.diagnostics["returncode"] == 17
    assert result.candidates == ()


def test_codex_invalid_last_message_is_durable_before_output_recovery(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(
        b'{"type":"thread.started","thread_id":"thread-1"}\n'
        b'{"type":"turn.completed","usage":'
        b'{"input_tokens":7,"output_tokens":3,"cached_input_tokens":2}}\n'
    )
    adapter = CodexAdapter(binary=sys.executable, runner=runner, env={})
    registry = ProviderRegistry()
    registry.register("codex", lambda: adapter)
    request = LLMRequest(
        "durable-invalid-last-message",
        "prompt",
        JsonOutput(
            {
                "type": "object",
                "required": ["ok"],
                "properties": {"ok": {"type": "boolean"}},
                "additionalProperties": False,
            }
        ),
        ModelSelection("codex"),
    )

    result = LLMClient(registry=registry).generate(
        request,
        run_root=tmp_path,
        options=LLMExecutionOptions(
            limits=ExecutionLimits(automatic_replacement_limit=0)
        ),
    )

    assert isinstance(result.outcome, LLMPaused)
    assert result.outcome.details["code"] == "output_invalid"
    raw_documents = []
    for path in tmp_path.rglob("*"):
        if not path.is_file() or path.name.endswith(".lock"):
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (
            isinstance(document, dict)
            and document.get("schema_version") == "arc.llm.provider_material.v1"
        ):
            raw_documents.append(document)
    assert len(raw_documents) == 1
    raw = raw_documents[0]
    assert raw["candidates"] == []
    assert raw["native_handle"] == {"provider": "codex", "value": "thread-1"}
    assert raw["usage"] == {
        "input_tokens": 7,
        "output_tokens": 3,
        "cached_input_tokens": 2,
    }
    assert raw["diagnostics"]["last_message"] == "empty"
    assert len(raw["diagnostics"]["raw_events"]) == 2


def test_codex_start_and_resume_deliver_native_image_and_read_context(
    tmp_path: Path,
) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"\x89PNG\r\n")
    markdown = tmp_path / "paper.md"
    markdown.write_text("# paper", encoding="utf-8")
    inputs = (
        ProviderInput(
            "page",
            "image/png",
            "a" * 64,
            image.stat().st_size,
            image,
            InputDeliveryMode.NATIVE_ATTACHMENT,
        ),
        ProviderInput(
            "paper",
            "text/markdown",
            "b" * 64,
            markdown.stat().st_size,
            markdown,
            InputDeliveryMode.READ_TOOL,
        ),
    )
    runner = FakeRunner(
        b'{"type":"thread.started","thread_id":"thread-1"}\n'
        b'{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\n'
    )
    adapter = CodexAdapter(binary="fake-codex", runner=runner, env={})

    adapter.start(
        ProviderRequest("Review.", "model", None, {}, 2, inputs),
        Observer(),
        Stop(),
    )
    adapter.resume(
        NativeResumeHandle("codex", "thread-1"),
        ProviderResumeRequest("Continue.", None, {}, 2, inputs),
        Observer(),
        Stop(),
    )

    for call in runner.calls:
        assert call["argv"][call["argv"].index("--image") + 1] == str(image)
        assert str(markdown).encode() in call["stdin"]
        assert str(image).encode() not in call["stdin"]


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
    assert ClaudeAdapter.compatibility_version == "claude-stream-json.v2"

    started = adapter.start(
        ProviderRequest("prompt", "claude-test", schema, {}, 11),
        Observer(),
        Stop(),
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
        Stop(),
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


def test_claude_rejects_plain_stdout_as_missing_structured_terminal() -> None:
    observer = Observer()
    adapter = ClaudeAdapter(
        binary="fake-claude",
        runner=FakeRunner(b'{"ok":true}\nnot-json output\n'),
        env={},
    )

    with pytest.raises(ProviderFailure) as caught:
        adapter.start(
            ProviderRequest("prompt", "claude-test", {"type": "object"}, {}, 2),
            observer,
            Stop(),
        )

    assert caught.value.category is FailureCategory.SCHEMA
    assert caught.value.details["code"] == "incomplete_terminal_closure"
    assert observer.raw == [
        {"ok": True},
        {"kind": "unparsed", "text": "not-json output"},
    ]


def test_codex_ignores_removed_underscored_thread_event_alias() -> None:
    candidate, handle, usage = parse_codex_event(
        {"type": "thread_started", "thread_id": "old-thread"}
    )

    assert candidate is None
    assert handle is None
    assert usage is None


def test_claude_start_and_resume_deliver_inputs_through_read_paths(
    tmp_path: Path,
) -> None:
    image = tmp_path / "page.jpg"
    image.write_bytes(b"\xff\xd8\xff")
    context = tmp_path / "context.json"
    context.write_text("{}", encoding="utf-8")
    inputs = (
        ProviderInput(
            "page",
            "image/jpeg",
            "a" * 64,
            image.stat().st_size,
            image,
            InputDeliveryMode.READ_TOOL,
        ),
        ProviderInput(
            "context",
            "application/json",
            "b" * 64,
            context.stat().st_size,
            context,
            InputDeliveryMode.READ_TOOL,
        ),
    )
    runner = FakeRunner(
        b'{"type":"result","session_id":"session-1","result":"ok"}\n'
    )
    adapter = ClaudeAdapter(binary="fake-claude", runner=runner, env={})

    adapter.start(
        ProviderRequest("Review.", "model", None, {}, 2, inputs),
        Observer(),
        Stop(),
    )
    adapter.resume(
        NativeResumeHandle("claude", "session-1"),
        ProviderResumeRequest("Continue.", None, {}, 2, inputs),
        Observer(),
        Stop(),
    )

    for call in runner.calls:
        assert str(image).encode() in call["stdin"]
        assert str(context).encode() in call["stdin"]


def test_kimi_resume_uses_session_without_creation_or_model_and_has_null_usage() -> None:
    schema = {
        "type": "object",
        "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
        "additionalProperties": False,
    }
    runner = FakeACPRunner()
    adapter = KimiAdapter(binary="fake-kimi", acp_runner=runner, env={})

    started = adapter.start(
        ProviderRequest("private prompt", "kimi-test", schema, {}, 7),
        Observer(),
        Stop(),
    )
    assert runner.calls[0]["binary"] == "fake-kimi"
    assert runner.calls[0]["model"] == "kimi-test"
    assert runner.calls[0]["session_id"] is None
    expected_schema = json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    sent = runner.calls[0]["prompt"]
    assert sent == (
        "private prompt\n\n"
        "Return exactly one JSON value satisfying this JSON Schema. "
        "Do not add prose or code fences.\n"
        f"JSON Schema:\n{expected_schema}"
    )
    assert started.native_handle == NativeResumeHandle("kimi", "kimi-session")
    assert started.usage is None

    resumed = adapter.resume(
        NativeResumeHandle("kimi", "kimi-1"),
        ProviderResumeRequest("follow-up", schema, {}, 9),
        Observer(),
        Stop(),
    )
    assert runner.calls[1]["session_id"] == "kimi-1"
    assert runner.calls[1]["model"] is None
    assert runner.calls[1]["prompt"].endswith(
        f"JSON Schema:\n{expected_schema}"
    )
    assert resumed.usage is None


def test_official_acp_runner_negotiates_media_and_resumes_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from acp import text_block
    from acp.schema import (
        AgentMessageChunk,
        ImageContentBlock,
        TextContentBlock,
    )

    image = tmp_path / "page.png"
    image.write_bytes(b"\x89PNG\r\n")
    markdown = tmp_path / "paper.md"
    markdown.write_text("# paper", encoding="utf-8")
    inputs = (
        ProviderInput(
            "page",
            "image/png",
            "a" * 64,
            image.stat().st_size,
            image,
            InputDeliveryMode.ACP_CONTENT,
        ),
        ProviderInput(
            "paper",
            "text/markdown",
            "b" * 64,
            markdown.stat().st_size,
            markdown,
            InputDeliveryMode.ACP_CONTENT,
        ),
    )
    calls: list[Any] = []

    class Connection:
        async def initialize(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                agent_capabilities=SimpleNamespace(
                    prompt_capabilities=SimpleNamespace(
                        image=True,
                        embedded_context=True,
                    )
                )
            )

        async def new_session(self, **kwargs: Any) -> Any:
            calls.append(("new", kwargs))
            return SimpleNamespace(session_id="kimi-new")

        async def resume_session(self, **kwargs: Any) -> Any:
            calls.append(("resume", kwargs))

        async def set_config_option(self, *args: Any) -> Any:
            calls.append(("model", args))

        async def prompt(self, **kwargs: Any) -> Any:
            calls.append(("prompt", kwargs))
            await client.session_update(
                kwargs["session_id"],
                AgentMessageChunk(
                    sessionUpdate="agent_message_chunk",
                    content=text_block('{"ok":true}'),
                ),
            )
            return SimpleNamespace(stop_reason="end_turn", usage=None)

    client: Any

    @asynccontextmanager
    async def spawn(fake_client: Any, *args: Any, **kwargs: Any):
        nonlocal client
        client = fake_client
        yield Connection(), SimpleNamespace()

    monkeypatch.setattr("acp.spawn_agent_process", spawn)
    observer = Observer()
    started = OfficialACPRunner().run(
        provider="kimi",
        binary="fake-kimi",
        model="vision-model",
        prompt="Review.",
        inputs=inputs,
        session_id=None,
        idle_timeout_seconds=2,
        observer=observer,
        stop=Stop(),
        env={},
    )
    prompt_blocks = next(item[1]["prompt"] for item in calls if item[0] == "prompt")
    assert isinstance(prompt_blocks[0], TextContentBlock)
    assert isinstance(prompt_blocks[1], ImageContentBlock)
    assert prompt_blocks[1].mime_type == "image/png"
    assert prompt_blocks[2].resource.mime_type == "text/markdown"
    assert started.native_handle == NativeResumeHandle("kimi", "kimi-new")
    assert started.candidates[0].text == '{"ok":true}'
    assert ("model", ("model", "kimi-new", "vision-model")) in calls

    calls.clear()
    resumed = OfficialACPRunner().run(
        provider="kimi",
        binary="fake-kimi",
        model=None,
        prompt="Continue.",
        inputs=inputs,
        session_id="kimi-existing",
        idle_timeout_seconds=2,
        observer=Observer(),
        stop=Stop(),
        env={},
    )
    assert any(
        item[0] == "resume" and item[1]["session_id"] == "kimi-existing"
        for item in calls
    )
    assert resumed.native_handle == NativeResumeHandle("kimi", "kimi-existing")


def test_official_acp_runner_classifies_initialize_failure_as_not_delivered(
    monkeypatch,
) -> None:
    class Connection:
        async def initialize(self, **kwargs: Any) -> Any:
            raise OSError("initialize failed")

    @asynccontextmanager
    async def spawn(*args: Any, **kwargs: Any):
        yield Connection(), SimpleNamespace()

    monkeypatch.setattr("acp.spawn_agent_process", spawn)
    observer = Observer()

    with pytest.raises(ProviderFailure) as caught:
        OfficialACPRunner().run(
            provider="kimi",
            binary="fake-kimi",
            model="default",
            prompt="Review.",
            inputs=(),
            session_id=None,
            idle_timeout_seconds=2,
            observer=observer,
            stop=Stop(),
            env={},
        )

    assert caught.value.delivery is DeliveryState.NOT_DELIVERED
    assert observer.deliveries == 0


def test_official_acp_runner_classifies_prompt_failure_as_may_have_run(
    monkeypatch,
) -> None:
    class Connection:
        async def initialize(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                agent_capabilities=SimpleNamespace(
                    prompt_capabilities=SimpleNamespace(image=True)
                )
            )

        async def new_session(self, **kwargs: Any) -> Any:
            return SimpleNamespace(session_id="kimi-new")

        async def prompt(self, **kwargs: Any) -> Any:
            raise OSError("prompt transport failed")

    @asynccontextmanager
    async def spawn(*args: Any, **kwargs: Any):
        yield Connection(), SimpleNamespace()

    monkeypatch.setattr("acp.spawn_agent_process", spawn)
    observer = Observer()

    with pytest.raises(ProviderFailure) as caught:
        OfficialACPRunner().run(
            provider="kimi",
            binary="fake-kimi",
            model="default",
            prompt="Review.",
            inputs=(),
            session_id=None,
            idle_timeout_seconds=2,
            observer=observer,
            stop=Stop(),
            env={},
        )

    assert caught.value.delivery is DeliveryState.MAY_HAVE_RUN
    assert observer.deliveries == 1


@pytest.mark.parametrize("callback", ("native_handle", "before_delivery"))
def test_official_acp_runner_classifies_durable_observer_failures_as_local(
    monkeypatch,
    callback: str,
) -> None:
    prompt_calls: list[object] = []
    cleanup: list[bool] = []

    class Connection:
        async def initialize(self, **kwargs: Any) -> Any:
            return SimpleNamespace(agent_capabilities=SimpleNamespace())

        async def new_session(self, **kwargs: Any) -> Any:
            return SimpleNamespace(session_id="durable-observer")

        async def prompt(self, **kwargs: Any) -> Any:
            prompt_calls.append(kwargs)
            raise AssertionError("observer failure reached ACP prompt")

    class FailingObserver(Observer):
        def native_handle(self, handle: NativeResumeHandle) -> None:
            if callback == "native_handle":
                raise OSError("task-state write failed")
            super().native_handle(handle)

        def before_delivery(self) -> None:
            if callback == "before_delivery":
                raise OSError("effect write failed")
            super().before_delivery()

    @asynccontextmanager
    async def spawn(*args: Any, **kwargs: Any):
        try:
            yield Connection(), SimpleNamespace()
        finally:
            cleanup.append(True)

    monkeypatch.setattr("acp.spawn_agent_process", spawn)

    with pytest.raises(ProviderFailure) as caught:
        OfficialACPRunner().run(
            provider="kimi",
            binary="fake-kimi",
            model="default",
            prompt="Review.",
            inputs=(),
            session_id=None,
            idle_timeout_seconds=2,
            observer=FailingObserver(),
            stop=Stop(),
            env={},
        )

    assert caught.value.category is FailureCategory.LOCAL_IO
    assert caught.value.delivery is DeliveryState.NOT_DELIVERED
    assert not caught.value.retryable
    assert isinstance(caught.value.__cause__, OSError)
    assert prompt_calls == []
    assert cleanup == [True]


@pytest.mark.parametrize("callback", ("native_handle", "before_delivery"))
def test_official_acp_runner_preserves_stop_from_durable_observer(
    monkeypatch,
    callback: str,
) -> None:
    prompt_calls: list[object] = []
    cleanup: list[bool] = []

    class Connection:
        async def initialize(self, **kwargs: Any) -> Any:
            return SimpleNamespace(agent_capabilities=SimpleNamespace())

        async def new_session(self, **kwargs: Any) -> Any:
            return SimpleNamespace(session_id="stop-observer")

        async def prompt(self, **kwargs: Any) -> Any:
            prompt_calls.append(kwargs)
            raise AssertionError("observer stop reached ACP prompt")

    class StoppingObserver(Observer):
        def native_handle(self, handle: NativeResumeHandle) -> None:
            if callback == "native_handle":
                raise StoppedError("stopped while saving handle")
            super().native_handle(handle)

        def before_delivery(self) -> None:
            if callback == "before_delivery":
                raise StoppedError("stopped while marking delivery")
            super().before_delivery()

    @asynccontextmanager
    async def spawn(*args: Any, **kwargs: Any):
        try:
            yield Connection(), SimpleNamespace()
        finally:
            cleanup.append(True)

    monkeypatch.setattr("acp.spawn_agent_process", spawn)
    result = OfficialACPRunner().run(
        provider="kimi",
        binary="fake-kimi",
        model="default",
        prompt="Review.",
        inputs=(),
        session_id=None,
        idle_timeout_seconds=2,
        observer=StoppingObserver(),
        stop=Stop(),
        env={},
    )

    assert result.terminal_kind is ProviderTerminalKind.STOPPED
    assert result.native_handle == (
        None
        if callback == "native_handle"
        else NativeResumeHandle("kimi", "stop-observer")
    )
    assert prompt_calls == []
    assert cleanup == [True]


@pytest.mark.parametrize(
    "media_type",
    ("text/markdown", "application/json", "text/plain"),
)
def test_official_acp_runner_rejects_missing_embedded_context_capability(
    tmp_path: Path,
    monkeypatch,
    media_type: str,
) -> None:
    markdown = tmp_path / "paper.md"
    markdown.write_text("# paper", encoding="utf-8")
    provider_input = ProviderInput(
        "paper",
        media_type,
        "a" * 64,
        markdown.stat().st_size,
        markdown,
        InputDeliveryMode.ACP_CONTENT,
    )

    class Connection:
        async def initialize(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                agent_capabilities=SimpleNamespace(
                    prompt_capabilities=SimpleNamespace(
                        image=True,
                        embedded_context=False,
                    )
                )
            )

        async def new_session(self, **kwargs: Any) -> Any:
            raise AssertionError("unsupported input reached session creation")

    @asynccontextmanager
    async def spawn(*args: Any, **kwargs: Any):
        yield Connection(), SimpleNamespace()

    monkeypatch.setattr("acp.spawn_agent_process", spawn)
    observer = Observer()

    with pytest.raises(ProviderFailure) as caught:
        OfficialACPRunner().run(
            provider="kimi",
            binary="fake-kimi",
            model="default",
            prompt="Review.",
            inputs=(provider_input,),
            session_id=None,
            idle_timeout_seconds=2,
            observer=observer,
            stop=Stop(),
            env={},
        )

    assert caught.value.category is FailureCategory.INVALID_REQUEST
    assert caught.value.delivery is DeliveryState.NOT_DELIVERED
    assert caught.value.details["code"] == "unsupported_input_media"
    assert observer.deliveries == 0


def test_official_acp_runner_session_activity_resets_idle_timeout(
    monkeypatch,
) -> None:
    client: Any
    cancellations: list[str] = []

    class Connection:
        async def initialize(self, **kwargs: Any) -> Any:
            return SimpleNamespace(agent_capabilities=SimpleNamespace())

        async def new_session(self, **kwargs: Any) -> Any:
            return SimpleNamespace(session_id="active-session")

        async def prompt(self, **kwargs: Any) -> Any:
            for _ in range(6):
                await asyncio.sleep(0.01)
                await client.session_update(
                    kwargs["session_id"],
                    SimpleNamespace(kind="tool_activity"),
                )
            return SimpleNamespace(stop_reason="end_turn", usage=None)

        async def cancel(self, *, session_id: str) -> None:
            cancellations.append(session_id)

    @asynccontextmanager
    async def spawn(fake_client: Any, *args: Any, **kwargs: Any):
        nonlocal client
        client = fake_client
        yield Connection(), SimpleNamespace()

    monkeypatch.setattr("acp.spawn_agent_process", spawn)

    result = OfficialACPRunner().run(
        provider="kimi",
        binary="fake-kimi",
        model="default",
        prompt="Review.",
        inputs=(),
        session_id=None,
        idle_timeout_seconds=0.03,
        observer=Observer(),
        stop=Stop(),
        env={},
    )

    assert result.terminal_kind is ProviderTerminalKind.COMPLETED
    assert cancellations == []


def test_official_acp_runner_true_idle_timeout_cancels_prompt(
    monkeypatch,
) -> None:
    cancellations: list[str] = []

    class Connection:
        async def initialize(self, **kwargs: Any) -> Any:
            return SimpleNamespace(agent_capabilities=SimpleNamespace())

        async def new_session(self, **kwargs: Any) -> Any:
            return SimpleNamespace(session_id="idle-session")

        async def prompt(self, **kwargs: Any) -> Any:
            await asyncio.Event().wait()

        async def cancel(self, *, session_id: str) -> None:
            cancellations.append(session_id)

    @asynccontextmanager
    async def spawn(*args: Any, **kwargs: Any):
        yield Connection(), SimpleNamespace()

    monkeypatch.setattr("acp.spawn_agent_process", spawn)

    with pytest.raises(ProviderFailure) as caught:
        OfficialACPRunner().run(
            provider="kimi",
            binary="fake-kimi",
            model="default",
            prompt="Review.",
            inputs=(),
            session_id=None,
            idle_timeout_seconds=0.02,
            observer=Observer(),
            stop=Stop(),
            env={},
        )

    assert caught.value.category is FailureCategory.TIMEOUT
    assert caught.value.delivery is DeliveryState.MAY_HAVE_RUN
    assert cancellations == ["idle-session"]


def test_official_acp_runner_cooperatively_cancels_active_prompt(
    monkeypatch,
) -> None:
    cancellations: list[str] = []

    class StopDuringPrompt:
        def __init__(self) -> None:
            self.checks = 0

        def raise_if_requested(self) -> None:
            self.checks += 1
            if self.checks >= 3:
                raise StoppedError("requested")

    class Connection:
        async def initialize(self, **kwargs: Any) -> Any:
            return SimpleNamespace(agent_capabilities=SimpleNamespace())

        async def new_session(self, **kwargs: Any) -> Any:
            return SimpleNamespace(session_id="cancel-session")

        async def prompt(self, **kwargs: Any) -> Any:
            await asyncio.Event().wait()

        async def cancel(self, *, session_id: str) -> None:
            cancellations.append(session_id)

    @asynccontextmanager
    async def spawn(*args: Any, **kwargs: Any):
        yield Connection(), SimpleNamespace()

    monkeypatch.setattr("acp.spawn_agent_process", spawn)
    observer = Observer()

    result = OfficialACPRunner().run(
        provider="kimi",
        binary="fake-kimi",
        model="default",
        prompt="Review.",
        inputs=(),
        session_id=None,
        idle_timeout_seconds=1,
        observer=observer,
        stop=StopDuringPrompt(),
        env={},
    )

    assert result.terminal_kind is ProviderTerminalKind.STOPPED
    assert result.native_handle == NativeResumeHandle("kimi", "cancel-session")
    assert cancellations == ["cancel-session"]
    assert observer.deliveries == 1


def test_auto_provider_and_model_resolution_is_host_deterministic() -> None:
    assert detect_host(env={"CODEX_THREAD_ID": "x"}).host == "codex"
    resolved = resolve_model_selection(
        ModelSelection(),
        env={"CLAUDE_CODE": "1"},
        available=("codex", "claude"),
    )
    assert resolved.provider == "claude"
    assert resolved.model


def test_default_registry_filters_auto_candidates_by_all_input_media() -> None:
    registry = default_registry()
    assert registry.supporting(("image/png", "text/markdown")) == (
        "claude",
        "codex",
        "kimi",
    )
    assert registry.supporting(("application/pdf",)) == ()


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
        (FailureCategory.STOPPED, ErrorCode.STOPPED),
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
        stop=Stop(),
        timeout=1,
        parse_event=parse_codex_event,
        runner=NonzeroRunner(),
        env={},
    )
    assert result.terminal_kind is ProviderTerminalKind.FAILED
    assert result.failure is not None
    assert result.diagnostics["returncode"] == 17
    assert result.candidates == ()


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
    assert KimiAdapter().capabilities().usage is UsageAvailability.PARTIAL


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
    assert kimi.details["media_capability_scope"] == "acp_prompt_capability_only"
    assert (
        kimi.details["model_media_capability"]
        == "not_exposed_by_acp_session_config"
    )
