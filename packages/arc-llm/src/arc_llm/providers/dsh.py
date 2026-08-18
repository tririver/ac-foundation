"""Native DeepSeek Harness model bridge adapter.

The adapter deliberately owns only the ARC-side transport.  Provider
credentials, model routing, retries, and streaming remain inside DSH's native
``ctx.llm`` service; this process authenticates to the local bridge with a
0600 token and exchanges versioned NDJSON events over a Unix socket.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import time
from pathlib import Path
from typing import Any, Mapping

from ..errors import FailureCategory, ProviderFailure
from ..output import CandidateMaterial
from .base import (
    IsolationMode,
    ProviderCapabilities,
    ProviderDiagnostic,
    ProviderExecution,
    ProviderRequest,
    ProviderResumeRequest,
    ProviderTerminalKind,
    ProviderUsage,
    StructuredOutputMode,
    UsageAvailability,
)

REQUEST_SCHEMA_VERSION = "arc.dsh-llm.request.v1"
EVENT_SCHEMA_VERSION = "arc.dsh-llm.event.v1"
DEFAULT_PROVIDER_ROUTE = "deepseek-official"
_FINISH_REASONS = {"stop", "max-tokens", "tool-calls", "error", "aborted"}
_MAX_BRIDGE_PROMPT_BYTES = 12 * 1024 * 1024
_TEXT_APPLICATION_TYPES = {
    "application/json",
    "application/tex",
    "application/x-latex",
    "application/x-tex",
    "application/xml",
}


class DshAdapter:
    name = "dsh"
    compatibility_version = "dsh-native-llm-bridge.v1"

    def __init__(
        self,
        *,
        socket_path: str | Path | None = None,
        token_path: str | Path | None = None,
        provider_route: str | None = None,
        connect_timeout_seconds: float = 3.0,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.socket_path = Path(socket_path) if socket_path is not None else None
        self.token_path = Path(token_path) if token_path is not None else None
        self.provider_route = provider_route
        self.connect_timeout_seconds = connect_timeout_seconds
        self.environment = environment

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            native_resume=False,
            structured_output=StructuredOutputMode.PROMPT,
            usage=UsageAvailability.PARTIAL,
            config_isolation=IsolationMode.EXPLICIT,
            tool_isolation=IsolationMode.EXPLICIT,
            cooperative_stop=True,
            provider_persistence=False,
        )

    def doctor(self) -> ProviderDiagnostic:
        socket_path, token_path, provider_route = self._paths_and_route()
        available = (
            socket_path.is_socket()
            and token_path.is_file()
            and bool(_read_token(token_path))
        )
        return ProviderDiagnostic(
            self.name,
            available,
            str(socket_path) if socket_path.exists() else None,
            details={
                "socket_path": str(socket_path),
                "token_path": str(token_path),
                "provider_route": provider_route,
                "credential_owner": "deepseek-harness",
            },
        )

    def start(
        self,
        request: ProviderRequest,
        observer: Any,
        stop: Any,
    ) -> ProviderExecution:
        return self._call(request, observer, stop)

    def resume(
        self,
        handle: Any,
        request: ProviderResumeRequest,
        observer: Any,
        stop: Any,
    ) -> ProviderExecution:
        return ProviderExecution(
            ProviderTerminalKind.FAILED,
            failure=ProviderFailure(
                "The DSH bridge does not expose native provider resume; ARC must start a fresh generation.",
                category=FailureCategory.UNAVAILABLE,
                details={"code": "native_resume_unavailable"},
            ),
            diagnostics={"provider": self.name, "native_resume": False},
        )

    def _call(
        self, request: ProviderRequest, observer: Any, stop: Any
    ) -> ProviderExecution:
        socket_path, token_path, provider_route = self._paths_and_route(
            request.environment
        )
        token = _read_token(token_path)
        if not token:
            return _failed(
                "The DSH ARC bridge token is missing, empty, or not private.",
                FailureCategory.UNAVAILABLE,
                details={
                    "code": "bridge_token_unavailable",
                    "token_path": str(token_path),
                },
            )

        try:
            bridge_prompt = _workspace_bridge_prompt(request.workspace)
        except ProviderFailure as exc:
            return ProviderExecution(
                ProviderTerminalKind.FAILED,
                failure=exc,
                diagnostics={"provider": self.name, **exc.details},
            )

        payload = {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "token": token,
            "op": "generate",
            "provider": provider_route,
            "model": request.model,
            "prompt": bridge_prompt,
        }
        if request.capabilities.get("dsh_system_prompt"):
            payload["system"] = request.capabilities["dsh_system_prompt"]

        started = time.monotonic()
        text_parts: list[str] = []
        event_count = 0
        reasoning_chars = 0
        usage: ProviderUsage | None = None
        finish_reason: str | None = None
        failure: ProviderFailure | None = None
        bridge_started = False
        observer.progress(
            "llm_provider_started", {"bridge": "dsh", "provider_route": provider_route}
        )
        try:
            with self._connect(socket_path) as connection:
                connection.sendall(
                    (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
                )
                buffer = b""
                while True:
                    stop.raise_if_requested()
                    try:
                        chunk = connection.recv(64 * 1024)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        raw, buffer = buffer.split(b"\n", 1)
                        if not raw.strip():
                            continue
                        event_count += 1
                        event = _decode_event(raw)
                        kind = event.get("type")
                        if kind == "started":
                            if bridge_started or event_count != 1:
                                raise _schema_failure(
                                    "The DSH bridge returned an out-of-order start event.",
                                    "bridge_started_order",
                                )
                            if event.get("provider") != provider_route:
                                raise _schema_failure(
                                    "The DSH bridge started a different provider route.",
                                    "bridge_provider_mismatch",
                                )
                            if event.get("model") != request.model:
                                raise _schema_failure(
                                    "The DSH bridge started a different model.",
                                    "bridge_model_mismatch",
                                )
                            bridge_started = True
                        elif kind == "text-delta":
                            _require_bridge_started(bridge_started)
                            text = event.get("text")
                            if not isinstance(text, str):
                                raise _schema_failure(
                                    "The DSH bridge returned an invalid text delta.",
                                    "bridge_invalid_text_delta",
                                )
                            text_parts.append(text)
                        elif kind == "reasoning-delta":
                            _require_bridge_started(bridge_started)
                            text = event.get("text")
                            if not isinstance(text, str):
                                raise _schema_failure(
                                    "The DSH bridge returned an invalid reasoning delta.",
                                    "bridge_invalid_reasoning_delta",
                                )
                            reasoning_chars += len(text)
                        elif kind == "usage":
                            _require_bridge_started(bridge_started)
                            usage = _usage(event.get("usage"))
                        elif kind == "finish":
                            finish_reason = event.get("reason")
                            if finish_reason not in _FINISH_REASONS:
                                raise _schema_failure(
                                    "The DSH bridge returned an unknown finish reason.",
                                    "bridge_invalid_finish_reason",
                                )
                            if not bridge_started and finish_reason not in {
                                "error",
                                "aborted",
                            }:
                                _require_bridge_started(bridge_started)
                            if finish_reason in {"error", "aborted"}:
                                failure = _event_failure(event)
                            break
                        else:
                            raise _schema_failure(
                                "The DSH bridge returned an unknown event type.",
                                "bridge_unknown_event",
                            )
                    if finish_reason is not None:
                        break
                if buffer.strip() and finish_reason is None:
                    raise _schema_failure(
                        "The DSH bridge returned an unterminated NDJSON event.",
                        "bridge_invalid_event_stream",
                    )
                if finish_reason is None:
                    raise ProviderFailure(
                        "The DSH bridge closed before a terminal event.",
                        category=FailureCategory.TRANSPORT,
                        retryable=True,
                        details={"code": "bridge_incomplete_stream"},
                    )
        except ProviderFailure as exc:
            failure = exc
        except OSError as exc:
            failure = ProviderFailure(
                f"Could not reach the DSH ARC bridge: {exc}",
                category=FailureCategory.TRANSPORT,
                retryable=True,
                details={"code": "bridge_transport", "socket_path": str(socket_path)},
            )

        diagnostics = {
            "socket_path": str(socket_path),
            "provider_route": provider_route,
            "event_count": event_count,
            "reasoning_chars": reasoning_chars,
            "finish_reason": finish_reason,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        if failure is not None:
            observer.progress(
                "llm_provider_failed", {"category": failure.category.value}
            )
            return ProviderExecution(
                ProviderTerminalKind.FAILED,
                usage=usage,
                failure=failure,
                diagnostics=diagnostics,
            )
        text = "".join(text_parts)
        if not text.strip():
            failure = ProviderFailure(
                "The DSH bridge completed without visible model text.",
                category=FailureCategory.TRANSPORT,
                details={"code": "bridge_empty_output"},
            )
            observer.progress(
                "llm_provider_failed", {"category": failure.category.value}
            )
            return ProviderExecution(
                ProviderTerminalKind.FAILED,
                usage=usage,
                failure=failure,
                diagnostics=diagnostics,
            )
        observer.progress(
            "llm_provider_finished",
            {"bridge": "dsh", "finish_reason": finish_reason},
        )
        return ProviderExecution(
            ProviderTerminalKind.COMPLETED,
            candidates=(CandidateMaterial(text=text, terminal=True),),
            usage=usage,
            diagnostics=diagnostics,
        )

    def _connect(self, socket_path: Path) -> socket.socket:
        deadline = time.monotonic() + self.connect_timeout_seconds
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(0.25)
            try:
                connection.connect(str(socket_path))
                return connection
            except OSError as exc:
                last_error = exc
                connection.close()
                time.sleep(0.05)
        if last_error is None:
            last_error = OSError("connection timeout")
        raise last_error

    def _paths_and_route(
        self,
        environment: Mapping[str, str] | None = None,
    ) -> tuple[Path, Path, str]:
        source = dict(os.environ if environment is None else environment)
        if self.environment is not None:
            source = {**source, **self.environment}
        socket_path = self.socket_path or Path(
            source.get("ARC_DSH_LLM_SOCKET")
            or source.get("DSH_ARC_LLM_SOCKET")
            or (Path.home() / ".dsh" / "runtime" / "arc-llm.sock")
        )
        token_path = self.token_path or Path(
            source.get("ARC_DSH_LLM_TOKEN_FILE")
            or source.get("DSH_ARC_LLM_TOKEN_FILE")
            or f"{socket_path}.token"
        )
        provider_route = self.provider_route or source.get(
            "ARC_DSH_PROVIDER", DEFAULT_PROVIDER_ROUTE
        )
        return socket_path, token_path, provider_route


def _workspace_bridge_prompt(workspace: Path) -> str:
    control_path = workspace / "host" / "control.json"
    try:
        control = json.loads(control_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProviderFailure(
            "The DSH provider could not read ARC's workspace control document.",
            category=FailureCategory.LOCAL_IO,
            details={"code": "workspace_control_unavailable"},
        ) from exc
    if not isinstance(control, Mapping) or control.get("schema_version") != (
        "arc.llm.workspace_control.v1"
    ):
        raise _schema_failure(
            "The DSH provider received an unsupported workspace control document.",
            "workspace_control_schema_mismatch",
        )
    prompt = control.get("prompt")
    output_contract = control.get("output_contract")
    inputs = control.get("inputs")
    if not isinstance(prompt, str) or not prompt.strip():
        raise _schema_failure(
            "The DSH provider received an invalid workspace task prompt.",
            "workspace_prompt_invalid",
        )
    if not isinstance(output_contract, Mapping) or not isinstance(inputs, list):
        raise _schema_failure(
            "The DSH provider received an invalid workspace contract.",
            "workspace_contract_invalid",
        )

    input_documents = [_workspace_text_input(workspace, item) for item in inputs]
    continuation = None
    continuation_path = control.get("continuation_response")
    if continuation_path is not None:
        if not isinstance(continuation_path, str):
            raise _schema_failure(
                "The DSH provider received an invalid continuation path.",
                "workspace_continuation_invalid",
            )
        continuation = _read_workspace_text(workspace, continuation_path)

    document = {
        "task": prompt,
        "output_contract": dict(output_contract),
        "provider_instructions": control.get("provider_instructions"),
        "continuation_response": continuation,
        "inputs": input_documents,
    }
    serialized = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    bridge_prompt = (
        "You are ARC's native text-generation backend. The JSON document below "
        "contains the complete verified task context. Follow its task and output "
        "contract, use only the supplied input contents, and return only the final "
        f"response.\n\n{serialized}"
    )
    if len(bridge_prompt.encode("utf-8")) > _MAX_BRIDGE_PROMPT_BYTES:
        raise ProviderFailure(
            "The ARC workspace is too large for the DSH bridge request.",
            category=FailureCategory.INVALID_REQUEST,
            details={"code": "bridge_prompt_too_large"},
        )
    return bridge_prompt


def _workspace_text_input(workspace: Path, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _schema_failure(
            "The DSH provider received invalid workspace input metadata.",
            "workspace_input_invalid",
        )
    input_id = value.get("input_id")
    media_type = value.get("media_type")
    relative_path = value.get("path")
    expected_sha256 = value.get("sha256")
    expected_size = value.get("size_bytes")
    if (
        not isinstance(input_id, str)
        or not isinstance(media_type, str)
        or not isinstance(relative_path, str)
        or not isinstance(expected_sha256, str)
        or not isinstance(expected_size, int)
    ):
        raise _schema_failure(
            "The DSH provider received incomplete workspace input metadata.",
            "workspace_input_invalid",
        )
    base_media_type = media_type.split(";", 1)[0].strip().lower()
    if not (
        base_media_type.startswith("text/")
        or base_media_type in _TEXT_APPLICATION_TYPES
    ):
        raise ProviderFailure(
            f"The DSH native bridge does not support input media type {media_type}.",
            category=FailureCategory.INVALID_REQUEST,
            details={
                "code": "bridge_input_media_unsupported",
                "input_id": input_id,
                "media_type": media_type,
            },
        )
    content = _read_workspace_text(workspace, relative_path)
    encoded = content.encode("utf-8")
    if len(encoded) != expected_size or hashlib.sha256(encoded).hexdigest() != (
        expected_sha256
    ):
        raise ProviderFailure(
            "A DSH bridge workspace input failed digest verification.",
            category=FailureCategory.LOCAL_IO,
            details={"code": "workspace_input_digest_mismatch", "input_id": input_id},
        )
    return {
        "input_id": input_id,
        "media_type": media_type,
        "sha256": expected_sha256,
        "content": content,
    }


def _read_workspace_text(workspace: Path, relative_path: str) -> str:
    root = workspace.resolve()
    path = (workspace / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ProviderFailure(
            "The DSH provider refused a workspace path outside the generation root.",
            category=FailureCategory.LOCAL_IO,
            details={"code": "workspace_path_escape"},
        ) from exc
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ProviderFailure(
            "The DSH provider could not read a verified text input.",
            category=FailureCategory.LOCAL_IO,
            details={"code": "workspace_text_unavailable"},
        ) from exc


def _read_token(path: Path) -> str:
    try:
        if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) & 0o077:
            return ""
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return ""


def _decode_event(raw: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderFailure(
            "The DSH bridge returned invalid JSON.",
            category=FailureCategory.SCHEMA,
            details={"code": "bridge_invalid_json"},
        ) from exc
    if not isinstance(value, Mapping):
        raise ProviderFailure(
            "The DSH bridge returned a non-object event.",
            category=FailureCategory.SCHEMA,
            details={"code": "bridge_event_not_object"},
        )
    if value.get("schema_version") != EVENT_SCHEMA_VERSION:
        raise _schema_failure(
            "The DSH bridge returned an unsupported event schema.",
            "bridge_event_schema_mismatch",
        )
    if not isinstance(value.get("type"), str):
        raise _schema_failure(
            "The DSH bridge event type is missing or invalid.",
            "bridge_event_type_invalid",
        )
    return value


def _usage(value: Any) -> ProviderUsage | None:
    if not isinstance(value, Mapping):
        raise _schema_failure(
            "The DSH bridge returned invalid usage data.",
            "bridge_invalid_usage",
        )
    parsed = tuple(
        _optional_nonnegative_int(value, key)
        for key in ("input_tokens", "output_tokens", "cached_input_tokens")
    )
    if all(item is None for item in parsed):
        raise _schema_failure(
            "The DSH bridge returned empty usage data.",
            "bridge_empty_usage",
        )
    return ProviderUsage(
        parsed[0],
        parsed[1],
        parsed[2],
    )


def _optional_nonnegative_int(value: Mapping[str, Any], key: str) -> int | None:
    if key not in value:
        return None
    candidate = value[key]
    if (
        isinstance(candidate, int)
        and not isinstance(candidate, bool)
        and candidate >= 0
    ):
        return candidate
    raise _schema_failure(
        "The DSH bridge returned invalid usage data.",
        "bridge_invalid_usage",
    )


def _event_failure(event: Mapping[str, Any]) -> ProviderFailure:
    failure = event.get("failure")
    message = (
        failure.get("message")
        if isinstance(failure, Mapping) and isinstance(failure.get("message"), str)
        else "The DSH native model call failed."
    )
    code = (
        failure.get("code")
        if isinstance(failure, Mapping) and isinstance(failure.get("code"), str)
        else "bridge_provider_error"
    )
    normalized = code.upper().replace("-", "_")
    category = {
        "AUTH": FailureCategory.AUTHENTICATION,
        "AUTHENTICATION": FailureCategory.AUTHENTICATION,
        "RATE_LIMIT": FailureCategory.RATE_LIMIT,
        "QUOTA": FailureCategory.QUOTA,
        "NO_ADAPTER": FailureCategory.UNAVAILABLE,
        "INVALID_REQUEST": FailureCategory.INVALID_REQUEST,
        "ABORTED": FailureCategory.STOPPED,
    }.get(normalized, FailureCategory.TRANSPORT)
    return ProviderFailure(
        message,
        category=category,
        retryable=category in {FailureCategory.RATE_LIMIT, FailureCategory.TRANSPORT},
        details={"code": code},
    )


def _require_bridge_started(started: bool) -> None:
    if not started:
        raise _schema_failure(
            "The DSH bridge returned generation data before its start event.",
            "bridge_missing_started",
        )


def _schema_failure(message: str, code: str) -> ProviderFailure:
    return ProviderFailure(
        message,
        category=FailureCategory.SCHEMA,
        details={"code": code},
    )


def _failed(
    message: str,
    category: FailureCategory,
    *,
    details: Mapping[str, Any],
) -> ProviderExecution:
    failure = ProviderFailure(message, category=category, details=details)
    return ProviderExecution(
        ProviderTerminalKind.FAILED, failure=failure, diagnostics=dict(details)
    )
