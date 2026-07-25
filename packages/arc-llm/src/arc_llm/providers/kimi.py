"""Kimi Code adapter backed by the official ACP Python SDK."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ._cli import executable_diagnostic
from .acp import ACPRunner, OfficialACPRunner
from .base import (
    InputDeliveryMode,
    IsolationMode,
    ProviderCapabilities,
    ProviderDiagnostic,
    ProviderExecution,
    ProviderRequest,
    ProviderResumeRequest,
    StructuredOutputMode,
    UsageAvailability,
)


class KimiAdapter:
    name = "kimi"
    compatibility_version = "kimi-acp-sdk.v2"

    def __init__(
        self,
        *,
        binary: str = "kimi",
        acp_runner: ACPRunner | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.binary = binary
        self.runner = acp_runner or OfficialACPRunner()
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
            input_delivery={
                "image/png": InputDeliveryMode.ACP_CONTENT,
                "image/jpeg": InputDeliveryMode.ACP_CONTENT,
                "text/markdown": InputDeliveryMode.ACP_CONTENT,
                "application/json": InputDeliveryMode.ACP_CONTENT,
            },
        )

    def doctor(self) -> ProviderDiagnostic:
        available, path = executable_diagnostic(self.name, self.binary)
        return ProviderDiagnostic(
            self.name,
            available,
            path,
            details={
                "warning": "provider_configuration_is_inherited",
                "media_capability_scope": "acp_prompt_capability_only",
                "model_media_capability": "not_exposed_by_acp_session_config",
            },
        )

    def start(
        self,
        request: ProviderRequest,
        observer: Any,
        stop: Any,
    ) -> ProviderExecution:
        return self.runner.run(
            provider=self.name,
            binary=self.binary,
            model=request.model,
            prompt=_prompt_contract(request.prompt, request.output_schema),
            inputs=request.inputs,
            session_id=None,
            idle_timeout_seconds=request.idle_timeout_seconds,
            observer=observer,
            stop=stop,
            env=self.env,
        )

    def resume(
        self,
        handle: Any,
        request: ProviderResumeRequest,
        observer: Any,
        stop: Any,
    ) -> ProviderExecution:
        return self.runner.run(
            provider=self.name,
            binary=self.binary,
            model=None,
            prompt=_prompt_contract(request.prompt, request.output_schema),
            inputs=request.inputs,
            session_id=handle.value,
            idle_timeout_seconds=request.idle_timeout_seconds,
            observer=observer,
            stop=stop,
            env=self.env,
        )


def _prompt_contract(
    prompt: str,
    output_schema: Mapping[str, Any] | None,
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
