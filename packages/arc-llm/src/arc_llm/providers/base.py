"""Provider boundary types.

Adapters normalize their private wire protocols into these values.  They do
not own run state, retry policy, or artifact publication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Protocol

from arc_jobs import StopToken

from ..errors import ProviderFailure
from ..output import CandidateMaterial


class StructuredOutputMode(StrEnum):
    NATIVE = "native"
    PROMPT = "prompt"
    NONE = "none"


class UsageAvailability(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class IsolationMode(StrEnum):
    ISOLATED = "isolated"
    EXPLICIT = "explicit"
    INHERITED = "inherited"


class ProviderTerminalKind(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True)
class ProviderCapabilities:
    native_resume: bool
    structured_output: StructuredOutputMode
    usage: UsageAvailability
    config_isolation: IsolationMode
    tool_isolation: IsolationMode
    cooperative_stop: bool
    provider_persistence: bool


@dataclass(frozen=True)
class ProviderDiagnostic:
    provider: str
    available: bool
    executable: str | None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NativeResumeHandle:
    provider: str
    value: str


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None


@dataclass(frozen=True)
class ProviderRequest:
    prompt: str
    model: str
    output_schema: Mapping[str, Any] | None
    capabilities: Mapping[str, Any]
    idle_timeout_seconds: float
    workspace: Path
    environment: Mapping[str, str] | None = None


@dataclass(frozen=True)
class ProviderResumeRequest:
    prompt: str
    output_schema: Mapping[str, Any] | None
    capabilities: Mapping[str, Any]
    idle_timeout_seconds: float
    workspace: Path
    environment: Mapping[str, str] | None = None


@dataclass(frozen=True)
class ProviderExecution:
    terminal_kind: ProviderTerminalKind
    candidates: tuple[CandidateMaterial, ...] = ()
    native_handle: NativeResumeHandle | None = None
    usage: ProviderUsage | None = None
    failure: ProviderFailure | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.terminal_kind is ProviderTerminalKind.FAILED and self.failure is None:
            raise ValueError("A failed provider execution requires a normalized failure.")
        if self.terminal_kind is ProviderTerminalKind.COMPLETED and self.failure is not None:
            raise ValueError("A completed provider execution cannot contain a failure.")


class ProviderObserver(Protocol):
    def native_handle(self, handle: NativeResumeHandle) -> None: ...

    def raw_event(self, event: Mapping[str, Any] | str) -> None: ...

    def progress(self, kind: str, data: Mapping[str, Any]) -> None: ...



class ProviderAdapter(Protocol):
    name: str
    compatibility_version: str

    def capabilities(self) -> ProviderCapabilities: ...

    def doctor(self) -> ProviderDiagnostic: ...

    def start(
        self,
        request: ProviderRequest,
        observer: ProviderObserver,
        stop: StopToken,
    ) -> ProviderExecution: ...

    def resume(
        self,
        handle: NativeResumeHandle,
        request: ProviderResumeRequest,
        observer: ProviderObserver,
        stop: StopToken,
    ) -> ProviderExecution: ...
