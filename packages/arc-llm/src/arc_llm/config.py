"""Host detection and deterministic provider/model resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Sequence

from .errors import InvalidRequestError
from .request import ModelSelection

PROVIDERS = ("codex", "claude", "kimi", "dsh")

DEFAULT_MODELS: Mapping[str, Mapping[str, str]] = {
    "codex": {
        "low": "gpt-5.6-luna",
        "medium": "gpt-5.6-luna",
        "high": "gpt-5.6-sol",
        "xhigh": "gpt-5.6-sol",
    },
    "claude": {
        "low": "haiku",
        "medium": "sonnet",
        "high": "opus",
        "xhigh": "opus",
    },
    "kimi": {
        "low": "default",
        "medium": "default",
        "high": "default",
        "xhigh": "default",
    },
    "dsh": {
        "low": "deepseek-v4-flash",
        "medium": "deepseek-v4-flash",
        "high": "deepseek-v4-flash",
        "xhigh": "deepseek-v4-flash",
    },
}


@dataclass(frozen=True)
class HostDetection:
    host: str | None
    evidence: str | None


@dataclass(frozen=True)
class ResolvedModelSelection:
    provider: str
    model: str
    tier: str


def detect_host(
    *,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
) -> HostDetection:
    values = os.environ if env is None else env
    explicit = values.get("ARC_AGENT_HOST")
    if explicit in PROVIDERS:
        return HostDetection(explicit, "ARC_AGENT_HOST")
    indicators = (
        ("CODEX_THREAD_ID", "codex"),
        ("CLAUDE_CODE", "claude"),
        ("CLAUDECODE", "claude"),
        ("KIMI_CODE", "kimi"),
        ("DSH_SESSION_ID", "dsh"),
        ("DSH_ARC_LLM_SOCKET", "dsh"),
    )
    for key, provider in indicators:
        if values.get(key):
            return HostDetection(provider, key)
    chain = tuple(item.lower() for item in (process_chain or ()))
    for item in chain:
        if "codex" in item:
            return HostDetection("codex", "process")
        if "claude" in item:
            return HostDetection("claude", "process")
        if "kimi" in item:
            return HostDetection("kimi", "process")
    return HostDetection(None, None)


def resolve_model_selection(
    selection: ModelSelection,
    *,
    env: Mapping[str, str] | None = None,
    process_chain: Sequence[str] | None = None,
    available: Sequence[str] = PROVIDERS,
) -> ResolvedModelSelection:
    provider = selection.provider
    if provider == "auto":
        detected = detect_host(env=env, process_chain=process_chain).host
        if detected in available:
            provider = detected
        else:
            provider = next((item for item in PROVIDERS if item in available), "")
    if provider not in PROVIDERS:
        raise InvalidRequestError(f"Unknown provider: {provider or 'auto'}")
    if provider not in available:
        raise InvalidRequestError(f"Provider is unavailable: {provider}")
    model = selection.model or DEFAULT_MODELS[provider][selection.tier]
    return ResolvedModelSelection(provider, model, selection.tier)
