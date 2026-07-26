"""Kimi CLI print-mode adapter.

Kimi's local CLI is invoked in its non-interactive print mode.  The executor
places the task contract and verified inputs in the generation workspace; this
adapter never serializes artifacts into a provider-specific protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

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
    compatibility_version = "kimi-print-stream-json.v1-workspace"

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
            usage=UsageAvailability.PARTIAL,
            config_isolation=IsolationMode.INHERITED,
            tool_isolation=IsolationMode.EXPLICIT,
            cooperative_stop=True,
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

    def start(
        self,
        request: ProviderRequest,
        observer: Any,
        stop: Any,
    ) -> ProviderExecution:
        argv = [
            self.binary,
            "-p",
            request.prompt,
            "--output-format",
            "stream-json",
            "--final-message-only",
        ]
        if request.model:
            argv.extend(["--model", request.model])
        return self._run(
            argv,
            request.idle_timeout_seconds,
            request.workspace,
            observer,
            stop,
        )

    def resume(
        self,
        handle: Any,
        request: ProviderResumeRequest,
        observer: Any,
        stop: Any,
    ) -> ProviderExecution:
        # ``-S`` is intentionally limited to an existing clean CLI session.
        # ARC does not enable Kimi's direct-mode permission plumbing here, so
        # this command deliberately omits ``--auto``.
        argv = [
            self.binary,
            "-S",
            handle.value,
            "-p",
            request.prompt,
            "--output-format",
            "stream-json",
            "--final-message-only",
        ]
        return self._run(
            argv,
            request.idle_timeout_seconds,
            request.workspace,
            observer,
            stop,
        )

    def _run(
        self,
        argv: list[str],
        timeout: float,
        workspace: Path,
        observer: Any,
        stop: Any,
    ) -> ProviderExecution:
        return run_cli(
            provider=self.name,
            argv=argv,
            # The prompt is passed with Kimi's documented ``-p`` switch;
            # stdin remains empty so no second request can be delivered.
            prompt="",
            observer=observer,
            stop=stop,
            timeout=timeout,
            parse_event=_parse_event,
            runner=self.runner,
            env=self.env,
            cwd=workspace,
        )


def _parse_event(
    event: Mapping[str, Any],
) -> tuple[CandidateMaterial | None, str | None, ProviderUsage | None]:
    role = event.get("role")
    content = event.get("content")
    candidate = (
        CandidateMaterial(text=content, terminal=True)
        if role == "assistant" and isinstance(content, str)
        else None
    )
    handle = event.get("session_id")
    usage_doc = event.get("usage")
    usage = None
    if isinstance(usage_doc, Mapping):
        usage = ProviderUsage(
            _integer(usage_doc.get("input_tokens")),
            _integer(usage_doc.get("output_tokens")),
            _integer(usage_doc.get("cached_input_tokens")),
        )
    return candidate, handle if isinstance(handle, str) else None, usage


def _integer(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )
