from __future__ import annotations

import pytest

from ac_llm import InvalidRequestError, ModelSelection, ProviderGateOptions
from ac_llm.config import DEFAULT_MODELS, detect_host, resolve_model_selection


@pytest.mark.parametrize(
    ("env", "chain", "expected"),
    [
        ({"AC_AGENT_HOST": "kimi", "CODEX_THREAD_ID": "x"}, (), "kimi"),
        ({"CODEX_THREAD_ID": "x"}, (), "codex"),
        ({"CLAUDE_CODE": "1"}, (), "claude"),
        ({"KIMI_CODE": "1"}, (), "kimi"),
        ({}, ("launcher", "codex"), "codex"),
        ({}, ("claude-code",), "claude"),
        ({}, ("kimi",), "kimi"),
    ],
)
def test_host_detection_precedence_matrix(env, chain, expected) -> None:
    assert detect_host(env=env, process_chain=chain).host == expected


@pytest.mark.parametrize("provider", ["codex", "claude", "kimi"])
@pytest.mark.parametrize("tier", ["low", "medium", "high", "xhigh"])
def test_provider_model_resolution_matrix(provider: str, tier: str) -> None:
    resolved = resolve_model_selection(
        ModelSelection(provider, tier=tier),
        env={"AC_LLM_PROVIDER": "ignored"},
        available=("codex", "claude", "kimi"),
    )
    assert resolved.provider == provider
    assert resolved.tier == tier
    assert resolved.model == DEFAULT_MODELS[provider][tier]


def test_auto_selection_and_gate_config_fail_closed() -> None:
    resolved = resolve_model_selection(
        ModelSelection(),
        env={},
        available=("claude",),
    )
    assert resolved.provider == "claude"
    with pytest.raises(InvalidRequestError):
        resolve_model_selection(ModelSelection("codex"), available=("claude",))
    configured = ProviderGateOptions(
        global_limit=200,
        provider_limits={"codex": 200},
        circuit_failure_threshold=200,
    )
    assert configured.global_limit == 200
    assert configured.circuit_failure_threshold == 200
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(InvalidRequestError):
            ProviderGateOptions(global_limit=invalid)  # type: ignore[arg-type]
        with pytest.raises(InvalidRequestError):
            ProviderGateOptions(  # type: ignore[arg-type]
                circuit_failure_threshold=invalid
            )
    with pytest.raises(InvalidRequestError):
        ProviderGateOptions(global_limit=2, provider_limits={"codex": 3})
