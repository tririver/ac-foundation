"""Claude Code CLI adapter."""

from __future__ import annotations

import json
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


class ClaudeAdapter:
    name = "claude"
    compatibility_version = "claude-stream-json.v1"

    def __init__(
        self,
        *,
        binary: str = "claude",
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
            config_isolation=IsolationMode.EXPLICIT,
            tool_isolation=IsolationMode.EXPLICIT,
            cooperative_cancel=True,
            provider_persistence=True,
        )

    def doctor(self) -> ProviderDiagnostic:
        available, path = executable_diagnostic(self.name, self.binary)
        return ProviderDiagnostic(self.name, available, path)

    def start(self, request: ProviderRequest, observer: Any, cancel: Any) -> ProviderExecution:
        argv = [
            self.binary,
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--model",
            request.model,
        ]
        if request.output_schema is not None:
            argv.extend(
                [
                    "--json-schema",
                    json.dumps(
                        request.output_schema,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ]
            )
        return self._run(argv, request.prompt, request.idle_timeout_seconds, observer, cancel)

    def resume(
        self,
        handle: Any,
        request: ProviderResumeRequest,
        observer: Any,
        cancel: Any,
    ) -> ProviderExecution:
        argv = [
            self.binary,
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--resume",
            handle.value,
        ]
        if request.output_schema is not None:
            argv.extend(
                [
                    "--json-schema",
                    json.dumps(
                        request.output_schema,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ]
            )
        return self._run(argv, request.prompt, request.idle_timeout_seconds, observer, cancel)

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
    kind = event.get("type")
    handle = event.get("session_id") if kind in {"system", "result"} else None
    candidate = None
    if kind == "result":
        if "structured_output" in event:
            candidate = CandidateMaterial(
                value=event["structured_output"],
                terminal=True,
            )
        elif isinstance(event.get("result"), str):
            candidate = CandidateMaterial(text=event["result"], terminal=True)
    usage_doc = event.get("usage")
    usage = None
    if isinstance(usage_doc, Mapping):
        usage = ProviderUsage(
            _integer(usage_doc.get("input_tokens")),
            _integer(usage_doc.get("output_tokens")),
            _integer(usage_doc.get("cache_read_input_tokens")),
        )
    return candidate, handle if isinstance(handle, str) else None, usage


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
