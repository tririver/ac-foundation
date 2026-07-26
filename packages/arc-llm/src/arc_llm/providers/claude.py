"""Claude Code CLI adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..errors import FailureCategory, ProviderFailure
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
    compatibility_version = "claude-stream-json.v4-ordered-terminal"

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
            cooperative_stop=True,
            provider_persistence=True,
        )

    def doctor(self) -> ProviderDiagnostic:
        available, path = executable_diagnostic(self.name, self.binary)
        return ProviderDiagnostic(self.name, available, path)

    def start(self, request: ProviderRequest, observer: Any, stop: Any) -> ProviderExecution:
        argv = [
            self.binary,
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--model",
            request.model,
        ]
        if request.capabilities.get("effective_host_mode") == "direct":
            argv.append("--dangerously-skip-permissions")
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
        return self._run(
            argv,
            request.prompt,
            request.idle_timeout_seconds,
            request.workspace,
            request.environment,
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
        argv = [
            self.binary,
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--resume",
            handle.value,
        ]
        if request.capabilities.get("effective_host_mode") == "direct":
            argv.append("--dangerously-skip-permissions")
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
        return self._run(
            argv,
            request.prompt,
            request.idle_timeout_seconds,
            request.workspace,
            request.environment,
            observer,
            stop,
        )

    def _run(
        self,
        argv: list[str],
        prompt: str,
        timeout: float | None,
        workspace: Path,
        environment: Mapping[str, str] | None,
        observer: Any,
        stop: Any,
    ) -> ProviderExecution:
        return run_cli(
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
            extract_failure=_extract_failure,
            extract_message=_extract_message,
        )


def _parse_event(
    event: Mapping[str, Any],
) -> tuple[CandidateMaterial | None, str | None, ProviderUsage | None]:
    kind = event.get("type")
    handle = event.get("session_id") if kind in {"system", "result"} else None
    candidate = None
    successful_result = (
        kind == "result"
        and event.get("is_error") is not True
        and event.get("subtype") in {None, "success"}
    )
    if successful_result:
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


def _extract_failure(event: Mapping[str, Any]) -> ProviderFailure | None:
    if event.get("type") != "result":
        return None
    subtype = event.get("subtype")
    if event.get("is_error") is not True and subtype in {None, "success"}:
        return None
    return ProviderFailure(
        "Claude reported an unsuccessful terminal result.",
        category=FailureCategory.TRANSPORT,
        retryable=True,
        details={
            "code": "claude_unsuccessful_result",
            "provider_subtype": (
                subtype[:256] if isinstance(subtype, str) else None
            ),
        },
    )


def _extract_message(event: Mapping[str, Any]) -> str | None:
    if event.get("type") != "assistant":
        return None
    message = event.get("message")
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    if not isinstance(content, list):
        return None
    texts = [
        item.get("text")
        for item in content
        if isinstance(item, Mapping)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
    ]
    return "\n".join(texts) if texts else None


def _integer(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )
