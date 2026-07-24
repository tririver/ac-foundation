"""Kimi Code adapter with a normalized JSON-event boundary."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..errors import DeliveryState, FailureCategory, ProviderFailure
from ..output import CandidateMaterial
from ._cli import executable_diagnostic, run_cli
from .base import (
    IsolationMode,
    ProviderCapabilities,
    ProviderDiagnostic,
    ProviderExecution,
    ProviderRequest,
    ProviderResumeRequest,
    ProviderUsage,
    StructuredOutputMode,
    UsageAvailability,
)
from .process import ProcessRunner


class KimiAdapter:
    name = "kimi"
    compatibility_version = "kimi-acp.v1"

    def __init__(
        self,
        *,
        binary: str = "kimi",
        runner: ProcessRunner | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.binary = binary
        self.runner = runner or ProcessRunner()
        self.env = env

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            native_resume=True,
            structured_output=StructuredOutputMode.PROMPT,
            usage=UsageAvailability.UNAVAILABLE,
            config_isolation=IsolationMode.INHERITED,
            tool_isolation=IsolationMode.EXPLICIT,
            cooperative_cancel=True,
            provider_persistence=True,
        )

    def doctor(self) -> ProviderDiagnostic:
        available, path = executable_diagnostic(self.name, self.binary)
        return ProviderDiagnostic(
            self.name,
            available,
            path,
            details={"warning": "provider_configuration_is_inherited"},
        )

    def start(self, request: ProviderRequest, observer: Any, cancel: Any) -> ProviderExecution:
        argv = [self.binary, "--acp", "--model", request.model]
        return self._run(
            argv,
            _prompt_contract(request.prompt, request.output_schema),
            request.idle_timeout_seconds,
            observer,
            cancel,
        )

    def resume(
        self,
        handle: Any,
        request: ProviderResumeRequest,
        observer: Any,
        cancel: Any,
    ) -> ProviderExecution:
        argv = [self.binary, "--acp", "--session", handle.value]
        return self._run(
            argv,
            _prompt_contract(request.prompt, request.output_schema),
            request.idle_timeout_seconds,
            observer,
            cancel,
        )

    def _run(
        self, argv: list[str], prompt: str, timeout: float, observer: Any, cancel: Any
    ) -> ProviderExecution:
        return run_cli(
            provider=self.name,
            argv=argv,
            prompt=prompt,
            observer=observer,
            cancel=cancel,
            timeout=timeout,
            parse_event=_parse_event,
            runner=self.runner,
            env=self.env,
        )


def _parse_event(
    event: Mapping[str, Any],
) -> tuple[CandidateMaterial | None, str | None, ProviderUsage | None]:
    method = event.get("method")
    if method in {
        "session/request_permission",
        "fs/read_text_file",
        "fs/write_text_file",
    }:
        raise ProviderFailure(
            "Kimi ACP requested a denied reverse operation.",
            category=FailureCategory.INVALID_REQUEST,
            delivery=DeliveryState.MAY_HAVE_RUN,
            details={"code": "reverse_operation_denied", "method": str(method)},
        )
    handle = event.get("session_id") or event.get("sessionId")
    candidate = None
    result = event.get("result")
    if isinstance(result, Mapping):
        content = result.get("content") or result.get("text")
        if isinstance(content, str):
            candidate = CandidateMaterial(text=content, terminal=True)
    params = event.get("params")
    if isinstance(params, Mapping):
        content = params.get("content") or params.get("text")
        if isinstance(content, str):
            candidate = CandidateMaterial(text=content, terminal=bool(params.get("terminal")))
    return candidate, handle if isinstance(handle, str) else None, None


def _prompt_contract(
    prompt: str, output_schema: Mapping[str, Any] | None
) -> str:
    if output_schema is None:
        return prompt
    schema = json.dumps(
        output_schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"{prompt}\n\nReturn exactly one JSON value satisfying this JSON Schema. "
        f"Do not add prose or code fences.\nJSON Schema:\n{schema}"
    )
