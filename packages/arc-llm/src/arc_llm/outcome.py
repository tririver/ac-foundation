"""Typed in-process outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TypeAlias

from arc_jobs import ArtifactRef, JsonValue, ResumeReason

from .errors import ArcLLMError
from .providers.base import ProviderUsage
from .request import SessionRef


@dataclass(frozen=True)
class LLMCompleted:
    value: Any
    provider: str | None
    model: str | None
    session: SessionRef | None
    usage: ProviderUsage | None


@dataclass(frozen=True)
class LLMPaused:
    reason: ResumeReason
    resume_key: str
    details: Mapping[str, JsonValue] = field(default_factory=dict)
    request_ref: ArtifactRef | None = None
    input_required: bool = False
    response_contract: str | None = None


@dataclass(frozen=True)
class LLMFailed:
    error: ArcLLMError


@dataclass(frozen=True)
class LLMCancelled:
    pass


LLMTaskOutcome: TypeAlias = LLMCompleted | LLMPaused | LLMFailed | LLMCancelled
