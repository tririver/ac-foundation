from __future__ import annotations

from collections import deque
from typing import Any

import pytest

from arc_llm import (
    DeliveryState,
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
        observer.before_delivery()
        return self._next(observer)

    def resume(
        self, handle: NativeResumeHandle, request: Any, observer: Any, stop: Any
    ) -> ProviderExecution:
        self.resume_calls += 1
        self.requests.append(request)
        observer.before_delivery()
        return self._next(observer)

    def _next(self, observer: Any) -> ProviderExecution:
        if not self.steps:
            raise AssertionError("fake provider script exhausted")
        value = self.steps.popleft()
        if isinstance(value, ProviderFailure):
            raise value
        if value.native_handle is not None:
            observer.native_handle(value.native_handle)
        return value


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
        delivery=DeliveryState.MAY_HAVE_RUN,
    )
