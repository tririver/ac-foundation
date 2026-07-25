"""Codex CLI adapter."""

from __future__ import annotations

from copy import deepcopy
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..errors import DeliveryState, FailureCategory, ProviderFailure
from ..output import CandidateMaterial
from ._cli import _redact_text, executable_diagnostic, run_cli
from .base import (
    InputDeliveryMode,
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
    compatibility_version = "codex-jsonl.v3"

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
            config_isolation=IsolationMode.ISOLATED,
            tool_isolation=IsolationMode.EXPLICIT,
            cooperative_stop=True,
            provider_persistence=True,
            input_delivery={
                "image/png": InputDeliveryMode.NATIVE_ATTACHMENT,
                "image/jpeg": InputDeliveryMode.NATIVE_ATTACHMENT,
                "text/markdown": InputDeliveryMode.READ_TOOL,
                "application/json": InputDeliveryMode.READ_TOOL,
            },
        )

    def doctor(self) -> ProviderDiagnostic:
        available, path = executable_diagnostic(self.name, self.binary)
        return ProviderDiagnostic(self.name, available, path)

    def start(self, request: ProviderRequest, observer: Any, stop: Any) -> ProviderExecution:
        argv = [self.binary, "exec", "--json"]
        if not request.capabilities.get("inherit_host_config", False):
            argv.extend(["--ignore-user-config", "--ignore-rules"])
        for item in request.inputs:
            if item.delivery_mode is InputDeliveryMode.NATIVE_ATTACHMENT:
                argv.extend(["--image", str(item.path)])
        argv.extend(["--sandbox", "read-only", "--model", request.model, "-"])
        return self._run(
            argv,
            _prompt_with_read_inputs(request.prompt, request.inputs),
            request.output_schema,
            request.idle_timeout_seconds,
            observer,
            stop,
        )

    def _run(
        self,
        argv: list[str],
        prompt: str,
        output_schema: Mapping[str, Any] | None,
        timeout: float,
        observer: Any,
        stop: Any,
    ) -> ProviderExecution:
        schema_path: Path | None = None
        output_path: Path | None = None
        if output_schema is not None:
            native_schema = _codex_native_schema(output_schema)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".json",
                delete=False,
            ) as handle:
                json.dump(native_schema, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                schema_path = Path(handle.name)
            argv = [*argv[:-1], "--output-schema", str(schema_path), argv[-1]]
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as handle:
            output_path = Path(handle.name)
        argv = [
            *argv[:-1],
            "--output-last-message",
            str(output_path),
            argv[-1],
        ]
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
                env=self.env,
                validate_terminal=False,
                extract_failure=_extract_failure,
            )
            # Codex JSONL may contain several completed agent-message items.
            # They are progress/diagnostic material rather than an unambiguous
            # final response.  `--output-last-message` is the CLI's stable
            # final-response contract, so only its content participates in
            # output selection.
            if execution.terminal_kind is not ProviderTerminalKind.COMPLETED:
                return execution
            final_message = _read_last_message(output_path)
            diagnostics = {
                **execution.diagnostics,
                "last_message": final_message[1],
            }
            candidates = (
                ()
                if final_message[0] is None
                else (CandidateMaterial(text=final_message[0], terminal=True),)
            )
            return replace(execution, candidates=candidates, diagnostics=diagnostics)
        finally:
            if schema_path is not None:
                schema_path.unlink(missing_ok=True)
            if output_path is not None:
                output_path.unlink(missing_ok=True)

    def resume(
        self,
        handle: Any,
        request: ProviderResumeRequest,
        observer: Any,
        stop: Any,
    ) -> ProviderExecution:
        argv = [self.binary, "exec", "resume", "--json"]
        if not request.capabilities.get("inherit_host_config", False):
            argv.extend(["--ignore-user-config", "--ignore-rules"])
        for item in request.inputs:
            if item.delivery_mode is InputDeliveryMode.NATIVE_ATTACHMENT:
                argv.extend(["--image", str(item.path)])
        argv.extend(["-c", 'sandbox_mode="read-only"', handle.value, "-"])
        return self._run(
            argv,
            _prompt_with_read_inputs(request.prompt, request.inputs),
            request.output_schema,
            request.idle_timeout_seconds,
            observer,
            stop,
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


def _extract_failure(event: Mapping[str, Any]) -> ProviderFailure | None:
    """Normalize Codex's terminal JSONL errors before exit-code fallback.

    Codex writes request-validation failures to its JSONL stdout stream and may
    leave stderr empty.  These events are stronger evidence than a nonzero
    process exit: an invalid output schema was rejected before the request was
    delivered to the model.
    """

    if event.get("type") not in {"error", "turn.failed"}:
        return None
    error = event.get("error")
    payload = error if isinstance(error, Mapping) else event
    code = _error_string(payload, event, "code")
    message = _error_string(payload, event, "message")
    parameter = _error_string(payload, event, "param") or _error_string(
        payload, event, "field"
    )
    diagnostic = " ".join(value for value in (code, message) if value)
    if not diagnostic:
        diagnostic = "Codex returned a terminal error event."
    if _is_invalid_request(code, message):
        category = FailureCategory.INVALID_REQUEST
        delivery = DeliveryState.NOT_DELIVERED
        retryable = False
        failure_message = "Codex rejected the request."
    else:
        category = FailureCategory.TRANSPORT
        delivery = DeliveryState.MAY_HAVE_RUN
        retryable = True
        failure_message = "Codex reported a failed turn."
    details: dict[str, Any] = {"diagnostic": _redact_text(diagnostic[:4096])}
    if code:
        details["provider_code"] = _redact_text(code[:256])
    if parameter:
        details["param"] = _redact_text(parameter[:1024])
    return ProviderFailure(
        failure_message,
        category=category,
        delivery=delivery,
        retryable=retryable,
        details=details,
    )


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
        if key != "uniqueItems"
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


def _integer(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def _prompt_with_read_inputs(prompt: str, inputs: tuple[Any, ...]) -> str:
    readable = [
        item for item in inputs if item.delivery_mode is InputDeliveryMode.READ_TOOL
    ]
    if not readable:
        return prompt
    lines = [
        prompt,
        "",
        "Read these verified, read-only input artifacts before answering:",
    ]
    lines.extend(
        f"- {item.input_id} ({item.media_type}): {item.path}" for item in readable
    )
    return "\n".join(lines)
