"""Codex CLI adapter."""

from __future__ import annotations

import json
import tempfile
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


class CodexAdapter:
    name = "codex"
    compatibility_version = "codex-jsonl.v1"

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
            cooperative_cancel=True,
            provider_persistence=True,
        )

    def doctor(self) -> ProviderDiagnostic:
        available, path = executable_diagnostic(self.name, self.binary)
        return ProviderDiagnostic(self.name, available, path)

    def start(self, request: ProviderRequest, observer: Any, cancel: Any) -> ProviderExecution:
        argv = [self.binary, "exec", "--json"]
        if not request.capabilities.get("inherit_host_config", False):
            argv.extend(["--ignore-user-config", "--ignore-rules"])
        argv.extend(["--sandbox", "read-only", "--model", request.model, "-"])
        return self._run(
            argv,
            request.prompt,
            request.output_schema,
            request.idle_timeout_seconds,
            observer,
            cancel,
        )

    def _run(
        self,
        argv: list[str],
        prompt: str,
        output_schema: Mapping[str, Any] | None,
        timeout: float,
        observer: Any,
        cancel: Any,
    ) -> ProviderExecution:
        schema_path: Path | None = None
        if output_schema is not None:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".json",
                delete=False,
            ) as handle:
                json.dump(output_schema, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                schema_path = Path(handle.name)
            argv = [*argv[:-1], "--output-schema", str(schema_path), argv[-1]]
        try:
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
        finally:
            if schema_path is not None:
                schema_path.unlink(missing_ok=True)

    def resume(
        self,
        handle: Any,
        request: ProviderResumeRequest,
        observer: Any,
        cancel: Any,
    ) -> ProviderExecution:
        argv = [self.binary, "exec", "resume", "--json"]
        if not request.capabilities.get("inherit_host_config", False):
            argv.extend(["--ignore-user-config", "--ignore-rules"])
        argv.extend(["-c", 'sandbox_mode="read-only"', handle.value, "-"])
        return self._run(
            argv,
            request.prompt,
            request.output_schema,
            request.idle_timeout_seconds,
            observer,
            cancel,
        )


def _parse_event(
    event: Mapping[str, Any],
) -> tuple[CandidateMaterial | None, str | None, ProviderUsage | None]:
    kind = event.get("type")
    handle = event.get("thread_id") if kind in {"thread.started", "thread_started"} else None
    candidate = None
    item = event.get("item")
    if isinstance(item, Mapping) and item.get("type") in {"agent_message", "message"}:
        text = item.get("text") or item.get("content")
        if isinstance(text, str):
            candidate = CandidateMaterial(text=text, terminal=kind == "item.completed")
    if kind in {"message", "assistant"} and isinstance(event.get("text"), str):
        candidate = CandidateMaterial(text=event["text"], terminal=bool(event.get("terminal")))
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
    return value if isinstance(value, int) and not isinstance(value, bool) else None
