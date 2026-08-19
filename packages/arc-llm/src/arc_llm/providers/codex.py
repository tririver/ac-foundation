"""Codex CLI adapter."""

from __future__ import annotations

from copy import deepcopy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..errors import FailureCategory, ProviderFailure
from ..output import CandidateMaterial
from ..diagnostics import redact_text
from ._cli import (
    classify_provider_failure_evidence,
    executable_diagnostic,
    run_cli,
)
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
from .process import ProcessRunner


class CodexAdapter:
    name = "codex"
    compatibility_version = "codex-jsonl.v5-ordered-terminal"

    def __init__(
        self,
        *,
        binary: str = "codex",
        runner: ProcessRunner | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.binary = binary
        self.runner = runner or ProcessRunner()
        self.env = env

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            native_resume=True,
            structured_output=StructuredOutputMode.NATIVE,
            usage=UsageAvailability.COMPLETE,
            config_isolation=IsolationMode.INHERITED,
            tool_isolation=IsolationMode.EXPLICIT,
            cooperative_stop=True,
            provider_persistence=True,
        )

    def doctor(self) -> ProviderDiagnostic:
        available, path = executable_diagnostic(self.name, self.binary)
        return ProviderDiagnostic(self.name, available, path)

    def start(self, request: ProviderRequest, observer: Any, stop: Any) -> ProviderExecution:
        argv = [self.binary, "exec", "--json"]
        if request.capabilities.get("execution_profile") == "bounded":
            argv.extend(
                [
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--sandbox",
                    "danger-full-access",
                    "--disable",
                    "shell_tool",
                    "--disable",
                    "multi_agent",
                ]
            )
            for item in request.inputs:
                if item.media_type.startswith("image/"):
                    argv.extend(["--image", str(item.path)])
        elif request.capabilities.get("effective_host_mode") == "direct":
            argv.extend(
                ["--dangerously-bypass-approvals-and-sandbox", "-C", str(request.workspace)]
            )
        argv.extend(["--model", request.model, "-"])
        return self._run(
            argv,
            request.prompt,
            request.output_schema,
            request.idle_timeout_seconds,
            request.workspace,
            request.environment,
            observer,
            stop,
            has_image_inputs=any(
                item.media_type.startswith("image/") for item in request.inputs
            ),
        )

    def _run(
        self,
        argv: list[str],
        prompt: str,
        output_schema: Mapping[str, Any] | None,
        timeout: float | None,
        workspace: Path,
        environment: Mapping[str, str] | None,
        observer: Any,
        stop: Any,
        *,
        has_image_inputs: bool = False,
    ) -> ProviderExecution:
        schema_path: Path | None = None
        output_path: Path | None = None
        if output_schema is not None:
            native_schema = _codex_native_schema(output_schema)
            schema_path = workspace / "host" / "codex-output-schema.json"
            schema_path.write_text(
                json.dumps(native_schema, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            argv = [
                *argv[:-1],
                "--output-schema",
                str(schema_path.relative_to(workspace)),
                argv[-1],
            ]
        output_path = workspace / "host" / "codex-last-message.txt"
        argv = [
            *argv[:-1],
            "--output-last-message",
            str(output_path.relative_to(workspace)),
            argv[-1],
        ]
        try:
            output_path.unlink(missing_ok=True)
        except OSError as exc:
            raise ProviderFailure(
                "Unable to clear the prior Codex final-message file.",
                category=FailureCategory.LOCAL_IO,
                details={"code": "codex_last_message_clear_failed"},
            ) from exc
        try:
            execution = run_cli(
                provider=self.name,
                argv=argv,
                prompt=prompt,
                observer=observer,
                stop=stop,
                timeout=timeout,
                parse_event=_parse_event,
                runner=self.runner,
                env=environment if environment is not None else self.env,
                cwd=workspace,
                validate_terminal=False,
                extract_failure=_extract_failure,
                extract_message=_extract_message,
            )
            # Codex JSONL may contain several completed agent-message items.
            # They are progress/diagnostic material rather than an unambiguous
            # final response.  `--output-last-message` is the CLI's stable
            # final-response contract, so only its content participates in
            # output selection.
            final_message = _read_last_message(output_path)
            diagnostics = {
                **execution.diagnostics,
                "last_message": final_message[1],
            }
            if has_image_inputs and _image_input_unavailable(diagnostics):
                return replace(
                    execution,
                    terminal_kind=ProviderTerminalKind.FAILED,
                    candidates=(),
                    failure=ProviderFailure(
                        "Codex could not access a required image input.",
                        category=FailureCategory.LOCAL_IO,
                        details={"code": "input_attachment_unavailable"},
                    ),
                    diagnostics=diagnostics,
                )
            candidates = (
                ()
                if final_message[0] is None
                else (CandidateMaterial(text=final_message[0], terminal=True),)
            )
            if execution.terminal_kind is not ProviderTerminalKind.COMPLETED:
                return replace(execution, diagnostics=diagnostics)
            if final_message[0] is None:
                return replace(
                    execution,
                    terminal_kind=ProviderTerminalKind.FAILED,
                    candidates=(),
                    failure=ProviderFailure(
                        "Codex completed without a fresh final-message file.",
                        category=FailureCategory.TRANSPORT,
                        details={"code": "incomplete_terminal_closure"},
                    ),
                    diagnostics=diagnostics,
                )
            return replace(execution, candidates=candidates, diagnostics=diagnostics)
        finally:
            # These files are generation evidence in the controlled workspace.
            # Leave them in place for durable local inspection.
            pass

    def resume(
        self,
        handle: Any,
        request: ProviderResumeRequest,
        observer: Any,
        stop: Any,
    ) -> ProviderExecution:
        argv = [self.binary, "exec", "resume", "--json"]
        if request.capabilities.get("execution_profile") == "bounded":
            argv.extend(
                [
                    "--ignore-user-config",
                    "--ignore-rules",
                    "-c",
                    'sandbox_mode="danger-full-access"',
                    "--disable",
                    "shell_tool",
                    "--disable",
                    "multi_agent",
                ]
            )
            for item in request.inputs:
                if item.media_type.startswith("image/"):
                    argv.extend(["--image", str(item.path)])
        elif request.capabilities.get("effective_host_mode") == "direct":
            # `codex exec resume` does not accept `-C`.  `_run` already
            # launches the process in the controlled workspace, so the resume
            # invocation has the same working directory as `start` without
            # passing an unsupported CLI option.
            argv.append("--dangerously-bypass-approvals-and-sandbox")
        argv.extend([handle.value, "-"])
        return self._run(
            argv,
            request.prompt,
            request.output_schema,
            request.idle_timeout_seconds,
            request.workspace,
            request.environment,
            observer,
            stop,
            has_image_inputs=any(
                item.media_type.startswith("image/") for item in request.inputs
            ),
        )


def _parse_event(
    event: Mapping[str, Any],
) -> tuple[CandidateMaterial | None, str | None, ProviderUsage | None]:
    kind = event.get("type")
    handle = event.get("thread_id") if kind == "thread.started" else None
    usage_doc = event.get("usage")
    usage = None
    if isinstance(usage_doc, Mapping):
        usage = ProviderUsage(
            _integer(usage_doc.get("input_tokens")),
            _integer(usage_doc.get("output_tokens")),
            _integer(usage_doc.get("cached_input_tokens")),
        )
    # Agent-message JSONL entries are intentionally not candidates.  Codex
    # can emit several completed items in one turn; the final-message file is
    # the sole terminal response used for output selection.
    return None, handle if isinstance(handle, str) else None, usage


def _extract_message(event: Mapping[str, Any]) -> str | None:
    if event.get("type") != "item.completed":
        return None
    item = event.get("item")
    if not isinstance(item, Mapping) or item.get("type") != "agent_message":
        return None
    text = item.get("text")
    return text if isinstance(text, str) and text else None


def _extract_failure(event: Mapping[str, Any]) -> ProviderFailure | None:
    """Normalize Codex's terminal JSONL errors before exit-code fallback.

    Codex writes request-validation failures to its JSONL stdout stream and may
    leave stderr empty.  These events are stronger evidence than a nonzero
    process exit: an invalid output schema was rejected before the request was
    delivered to the model.
    """

    if event.get("type") not in {"error", "turn.failed"}:
        return None
    payload = _structured_error_payload(event) or event
    code = _error_string(payload, event, "code")
    message = _error_string(payload, event, "message")
    parameter = _error_string(payload, event, "param") or _error_string(
        payload, event, "field"
    )
    diagnostic = " ".join(value for value in (code, message) if value)
    if not diagnostic:
        diagnostic = "Codex returned a terminal error event."
    classified = classify_provider_failure_evidence(
        diagnostic,
        message="Codex reported a failed turn.",
    )
    if (
        classified.category is FailureCategory.TRANSPORT
        and _is_invalid_request(code, message)
    ):
        category = FailureCategory.INVALID_REQUEST
        retryable = False
        retry_after_seconds = None
        failure_message = "Codex rejected the request."
    else:
        category = classified.category
        retryable = classified.retryable
        retry_after_seconds = classified.retry_after_seconds
        failure_message = str(classified)
    details: dict[str, Any] = dict(classified.details)
    if code:
        details["provider_code"] = redact_text(code[:256])
    if parameter:
        details["param"] = redact_text(parameter[:1024])
    return ProviderFailure(
        failure_message,
        category=category,
        retryable=retryable,
        retry_after_seconds=retry_after_seconds,
        details=details,
    )


_MAX_ERROR_JSON_DEPTH = 3
_MAX_ERROR_JSON_CHARS = 16 * 1024


def _structured_error_payload(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return Codex's innermost structured error payload when present.

    Codex 0.145.0 wraps its HTTP error object as a JSON string both in an
    ``error`` event's top-level ``message`` and in ``turn.failed.error``'s
    ``message``.  Decode only those error-message wrappers and cap both depth
    and size so provider diagnostics cannot turn into an unbounded parser.
    """

    return _structured_error_payload_at(event, depth=_MAX_ERROR_JSON_DEPTH)


def _structured_error_payload_at(
    value: Mapping[str, Any], *, depth: int
) -> Mapping[str, Any] | None:
    if depth <= 0:
        return None
    error = value.get("error")
    if isinstance(error, Mapping):
        nested = _structured_error_payload_at(error, depth=depth - 1)
        if nested is not None:
            return nested
        return error
    for key in ("error", "message"):
        decoded = _decode_error_json(value.get(key))
        if decoded is None:
            continue
        nested = _structured_error_payload_at(decoded, depth=depth - 1)
        if nested is not None:
            return nested
        return decoded
    return None


def _decode_error_json(value: Any) -> Mapping[str, Any] | None:
    if (
        not isinstance(value, str)
        or len(value) > _MAX_ERROR_JSON_CHARS
    ):
        return None
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _error_string(
    payload: Mapping[str, Any], event: Mapping[str, Any], key: str
) -> str | None:
    value = payload.get(key)
    if key == "code" and not isinstance(value, str):
        value = payload.get("type")
    if not isinstance(value, str):
        value = event.get(key)
    return value if isinstance(value, str) and value else None


def _is_invalid_request(code: str | None, message: str | None) -> bool:
    normalized_code = code.lower().replace("-", "_") if code else ""
    if normalized_code.startswith("invalid_"):
        return True
    diagnostic = " ".join(value for value in (code, message) if value).lower()
    return "schema" in diagnostic and (
        "invalid" in diagnostic or "unsupported" in diagnostic
    )


_SCHEMA_SINGLE_CHILDREN = frozenset(
    {
        "additionalItems",
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_SCHEMA_ARRAY_CHILDREN = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SCHEMA_MAP_CHILDREN = frozenset(
    {
        "$defs",
        "definitions",
        "dependentSchemas",
        "patternProperties",
        "properties",
    }
)


def _codex_native_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Project a JSON Schema into the subset accepted by Codex native output.

    The original schema remains the durable output contract and is validated
    locally.  This projection is intentionally private to the Codex adapter.
    """

    projected = _project_schema_node(schema)
    assert isinstance(projected, dict)
    return projected


def _project_schema_node(schema: Any) -> Any:
    if not isinstance(schema, Mapping):
        return deepcopy(schema)
    projected = {
        key: deepcopy(value)
        for key, value in schema.items()
        if key not in {"$schema", "default", "uniqueItems"}
    }
    for key in _SCHEMA_SINGLE_CHILDREN:
        if key not in schema:
            continue
        value = schema[key]
        if key == "items" and isinstance(value, list):
            projected[key] = [_project_schema_node(item) for item in value]
        else:
            projected[key] = _project_schema_node(value)
    for key in _SCHEMA_ARRAY_CHILDREN:
        value = schema.get(key)
        if isinstance(value, list):
            projected[key] = [_project_schema_node(item) for item in value]
    for key in _SCHEMA_MAP_CHILDREN:
        value = schema.get(key)
        if isinstance(value, Mapping):
            projected[key] = {
                child_key: _project_schema_node(child)
                for child_key, child in value.items()
            }
    dependencies = schema.get("dependencies")
    if isinstance(dependencies, Mapping):
        projected["dependencies"] = {
            key: (
                _project_schema_node(value)
                if isinstance(value, Mapping)
                else deepcopy(value)
            )
            for key, value in dependencies.items()
        }
    if "const" in schema and "type" not in schema:
        value_type = _const_json_type(schema["const"])
        if value_type is not None:
            projected["type"] = value_type
    if "enum" in schema and "type" not in schema:
        enum_values = schema["enum"]
        if isinstance(enum_values, list) and enum_values:
            enum_types = {_const_json_type(value) for value in enum_values}
            if len(enum_types) == 1 and None not in enum_types:
                projected["type"] = enum_types.pop()
    raw_type = projected.get("type")
    if isinstance(raw_type, list):
        projected.pop("type")
        projected["anyOf"] = [{"type": value} for value in raw_type]
    properties = projected.get("properties")
    if projected.get("type") == "object" and isinstance(properties, Mapping):
        projected["required"] = list(properties)
        projected["additionalProperties"] = False
    return projected


def _const_json_type(value: Any) -> str | None:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return None


def _read_last_message(path: Path) -> tuple[str | None, str]:
    try:
        message = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None, "unavailable"
    if not message.strip():
        return None, "empty"
    return message, "present"


def _image_input_unavailable(diagnostics: Mapping[str, Any]) -> bool:
    stderr = diagnostics.get("stderr_tail")
    return isinstance(stderr, str) and "unable to locate image at" in stderr.lower()


def _integer(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )
