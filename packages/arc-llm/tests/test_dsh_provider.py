from __future__ import annotations

import hashlib
import json
import socket
import threading
from pathlib import Path
from typing import Any

import pytest

from arc_llm import ProviderRequest, ProviderTerminalKind
from arc_llm.errors import FailureCategory, ProviderFailure
from arc_llm.providers.dsh import (
    DshAdapter,
    EVENT_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    _decode_event,
    _workspace_bridge_prompt,
)


class Observer:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def progress(self, kind: str, data: Any) -> None:
        self.events.append((kind, data))

    def native_handle(self, _handle: Any) -> None:
        raise AssertionError("DSH bridge must not expose native resume handles")


class Stop:
    def raise_if_requested(self) -> None:
        return


def _write_control(
    workspace: Path,
    *,
    prompt: str = "prompt",
    inputs: list[dict[str, Any]] | None = None,
) -> None:
    host = workspace / "host"
    host.mkdir(parents=True, exist_ok=True)
    (host / "control.json").write_text(
        json.dumps(
            {
                "schema_version": "arc.llm.workspace_control.v1",
                "task_id": "test-task",
                "prompt": prompt,
                "output_contract": {"kind": "text"},
                "runtime": {},
                "inputs": inputs or [],
                "work_directory": "work",
                "continuation_response": None,
                "provider_instructions": None,
            }
        ),
        encoding="utf-8",
    )


def test_dsh_adapter_consumes_authenticated_ndjson_stream(tmp_path: Path) -> None:
    _write_control(tmp_path)
    socket_path = tmp_path / "arc-llm.sock"
    token_path = tmp_path / "arc-llm.token"
    token_path.write_text("test-token\n", encoding="utf-8")
    token_path.chmod(0o600)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)
    received: dict[str, Any] = {}

    def serve() -> None:
        connection, _ = server.accept()
        with connection:
            raw = b""
            while b"\n" not in raw:
                raw += connection.recv(4096)
            received.update(json.loads(raw.split(b"\n", 1)[0]))
            events = [
                {
                    "schema_version": EVENT_SCHEMA_VERSION,
                    "type": "started",
                    "provider": "fake-provider",
                    "model": "fake-model",
                },
                {
                    "schema_version": EVENT_SCHEMA_VERSION,
                    "type": "text-delta",
                    "text": "answer",
                },
                {
                    "schema_version": EVENT_SCHEMA_VERSION,
                    "type": "usage",
                    "usage": {"input_tokens": 4, "output_tokens": 2},
                },
                {
                    "schema_version": EVENT_SCHEMA_VERSION,
                    "type": "finish",
                    "reason": "stop",
                },
            ]
            connection.sendall(
                b"".join((json.dumps(event) + "\n").encode("utf-8") for event in events)
            )
        server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    adapter = DshAdapter(
        socket_path=socket_path,
        token_path=token_path,
        provider_route="fake-provider",
    )
    result = adapter.start(
        ProviderRequest("prompt", "fake-model", None, {}, 3, tmp_path),
        Observer(),
        Stop(),
    )
    thread.join(timeout=3)

    assert result.terminal_kind is ProviderTerminalKind.COMPLETED
    assert result.candidates[0].text == "answer"
    assert result.usage is not None
    assert result.usage.input_tokens == 4
    assert result.usage.output_tokens == 2
    bridge_prompt = received.pop("prompt")
    document = json.loads(bridge_prompt.split("\n\n", 1)[1])
    assert document["task"] == "prompt"
    assert document["output_contract"] == {"kind": "text"}
    assert document["inputs"] == []
    assert received == {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "token": "test-token",
        "op": "generate",
        "provider": "fake-provider",
        "model": "fake-model",
    }


def test_dsh_doctor_reports_missing_bridge_as_unavailable(tmp_path: Path) -> None:
    adapter = DshAdapter(
        socket_path=tmp_path / "missing.sock",
        token_path=tmp_path / "missing.token",
    )
    diagnostic = adapter.doctor()
    assert diagnostic.available is False
    assert diagnostic.details["credential_owner"] == "deepseek-harness"


def test_dsh_adapter_rejects_stream_without_terminal_event(tmp_path: Path) -> None:
    _write_control(tmp_path)
    socket_path = tmp_path / "arc-llm.sock"
    token_path = tmp_path / "arc-llm.token"
    token_path.write_text("test-token\n", encoding="utf-8")
    token_path.chmod(0o600)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)

    def serve() -> None:
        connection, _ = server.accept()
        with connection:
            while b"\n" not in connection.recv(4096):
                pass
            events = [
                {
                    "schema_version": EVENT_SCHEMA_VERSION,
                    "type": "started",
                    "provider": "fake-provider",
                    "model": "fake-model",
                },
                {
                    "schema_version": EVENT_SCHEMA_VERSION,
                    "type": "text-delta",
                    "text": "partial",
                },
            ]
            connection.sendall(
                b"".join((json.dumps(event) + "\n").encode("utf-8") for event in events)
            )
        server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    result = DshAdapter(
        socket_path=socket_path,
        token_path=token_path,
        provider_route="fake-provider",
    ).start(
        ProviderRequest("prompt", "fake-model", None, {}, 3, tmp_path),
        Observer(),
        Stop(),
    )
    thread.join(timeout=3)

    assert result.terminal_kind is ProviderTerminalKind.FAILED
    assert result.failure is not None
    assert result.failure.category is FailureCategory.TRANSPORT
    assert result.failure.details["code"] == "bridge_incomplete_stream"


def test_dsh_adapter_rejects_insecure_token_file(tmp_path: Path) -> None:
    token_path = tmp_path / "arc-llm.token"
    token_path.write_text("test-token\n", encoding="utf-8")
    token_path.chmod(0o644)
    adapter = DshAdapter(
        socket_path=tmp_path / "arc-llm.sock",
        token_path=token_path,
    )

    result = adapter.start(
        ProviderRequest("prompt", "fake-model", None, {}, 3, tmp_path),
        Observer(),
        Stop(),
    )

    assert result.terminal_kind is ProviderTerminalKind.FAILED
    assert result.failure is not None
    assert result.failure.category is FailureCategory.UNAVAILABLE
    assert result.failure.details["code"] == "bridge_token_unavailable"


def test_dsh_adapter_rejects_unknown_event_schema() -> None:
    raw = json.dumps(
        {"schema_version": "arc.dsh-llm.event.v999", "type": "started"}
    ).encode("utf-8")

    with pytest.raises(ProviderFailure) as caught:
        _decode_event(raw)

    assert caught.value.category is FailureCategory.SCHEMA
    assert caught.value.details["code"] == "bridge_event_schema_mismatch"


def test_dsh_workspace_prompt_embeds_verified_text_input(tmp_path: Path) -> None:
    content = "verified paper excerpt"
    input_path = tmp_path / "inputs" / "0000-paper.md"
    input_path.parent.mkdir(parents=True)
    input_path.write_text(content, encoding="utf-8")
    encoded = content.encode("utf-8")
    _write_control(
        tmp_path,
        prompt="Summarize the supplied excerpt.",
        inputs=[
            {
                "input_id": "paper",
                "media_type": "text/markdown",
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "size_bytes": len(encoded),
                "path": "inputs/0000-paper.md",
            }
        ],
    )

    bridge_prompt = _workspace_bridge_prompt(tmp_path)
    document = json.loads(bridge_prompt.split("\n\n", 1)[1])

    assert document["task"] == "Summarize the supplied excerpt."
    assert document["inputs"][0]["content"] == content
