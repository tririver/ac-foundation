from __future__ import annotations

from collections import deque
from typing import Any

import pytest

from arc_llm import (
    FailureCategory,
    IsolationMode,
    NativeResumeHandle,
    ProviderCapabilities,
    ProviderDiagnostic,
    ProviderExecution,
    ProviderFailure,
    ProviderRegistry,
    StructuredOutputMode,
    UsageAvailability,
)


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "live_provider_smoke: bounded opt-in smoke against a real provider",
    )


class ScriptedAdapter:
    name = "codex"
    compatibility_version = "fake-v1"

    def __init__(self) -> None:
        self.steps: deque[ProviderExecution | ProviderFailure] = deque()
        self.start_calls = 0
        self.resume_calls = 0
        self.requests: list[Any] = []

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            native_resume=True,
            structured_output=StructuredOutputMode.NATIVE,
            usage=UsageAvailability.COMPLETE,
            config_isolation=IsolationMode.ISOLATED,
            tool_isolation=IsolationMode.ISOLATED,
            cooperative_stop=True,
            provider_persistence=True,
        )

    def doctor(self) -> ProviderDiagnostic:
        return ProviderDiagnostic(self.name, True, "fake-codex")

    def start(self, request: Any, observer: Any, stop: Any) -> ProviderExecution:
        self.start_calls += 1
        self.requests.append(request)
        return self._next(request, observer)

    def resume(
        self, handle: NativeResumeHandle, request: Any, observer: Any, stop: Any
    ) -> ProviderExecution:
        self.resume_calls += 1
        self.requests.append(request)
        return self._next(request, observer)

    def _next(self, request: Any, observer: Any) -> ProviderExecution:
        if not self.steps:
            raise AssertionError("fake provider script exhausted")
        value = self.steps.popleft()
        if isinstance(value, ProviderFailure):
            raise value
        if value.native_handle is not None:
            observer.native_handle(value.native_handle)
        if not _uses_host_turn(request):
            return value
        return ProviderExecution(
            value.terminal_kind,
            tuple(_host_turn_candidate(candidate) for candidate in value.candidates),
            value.native_handle,
            value.usage,
            value.failure,
            value.diagnostics,
        )


def _uses_host_turn(request: Any) -> bool:
    schema = request.output_schema
    if not isinstance(schema, dict):
        return False
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return False
    version = properties.get("schema_version")
    return isinstance(version, dict) and version.get("const") == "arc.llm.host_turn.v1"


def _host_turn_candidate(candidate: Any) -> Any:
    if candidate.has_value and _is_host_turn(candidate.value):
        return candidate
    result = candidate.value if candidate.has_value else candidate.text
    return type(candidate)(
        value={
            "schema_version": "arc.llm.host_turn.v1",
            "state": "complete",
            "result": result,
            "host_request": None,
        },
        terminal=candidate.terminal,
    )


def _is_host_turn(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("schema_version") == "arc.llm.host_turn.v1"
    )


@pytest.fixture
def adapter() -> ScriptedAdapter:
    return ScriptedAdapter()


@pytest.fixture
def registry(adapter: ScriptedAdapter) -> ProviderRegistry:
    result = ProviderRegistry()
    result.register("codex", lambda: adapter)
    return result


def may_have_run_failure(message: str = "connection lost") -> ProviderFailure:
    return ProviderFailure(
        message,
        category=FailureCategory.TRANSPORT,
    )
