from __future__ import annotations

from dataclasses import replace

import pytest

from arc_llm import (
    CapabilityPolicy,
    ExecutionLimits,
    InvalidRequestError,
    JsonOutput,
    LLMExecutionOptions,
    LLMRequest,
    ModelSelection,
    ResumeAction,
    ResumeInput,
    decode_request,
    decode_resume_input,
    request_to_document,
    resume_input_to_document,
)
from arc_llm.identity import semantic_document, semantic_key


def test_request_and_resume_codecs_are_closed_round_trips() -> None:
    request = LLMRequest(
        "review-1",
        "Review this.",
        JsonOutput({"type": "object", "required": ["ok"]}),
        ModelSelection("codex"),
        capabilities=CapabilityPolicy(allowed_tools=("z", "a", "z")),
    )
    assert decode_request(request_to_document(request)) == request
    document = request_to_document(request)
    document["unknown"] = True
    with pytest.raises(InvalidRequestError):
        decode_request(document)

    resume = ResumeInput("resume-3", ResumeAction.REPLACE, reason="new evidence")
    assert decode_resume_input(resume_input_to_document(resume)) == resume


def test_model_constraints_and_json_booleans_are_strict() -> None:
    with pytest.raises(InvalidRequestError):
        ModelSelection(model="exact")
    with pytest.raises(InvalidRequestError):
        ModelSelection("codex", "exact", "high")
    with pytest.raises(InvalidRequestError):
        CapabilityPolicy(internet="false")  # type: ignore[arg-type]


def test_operational_limits_do_not_change_semantic_key() -> None:
    request = LLMRequest("task", "prompt", JsonOutput({"type": "object"}))
    first = LLMExecutionOptions(ExecutionLimits(idle_timeout_seconds=1))
    second = replace(first, limits=ExecutionLimits(idle_timeout_seconds=900))
    assert semantic_key(request) == semantic_key(request)
    assert first != second


def test_semantic_capability_and_prompt_changes_are_detected() -> None:
    request = LLMRequest("task", "prompt", JsonOutput({"type": "object"}))
    assert semantic_key(request) != semantic_key(replace(request, prompt="other"))
    assert semantic_key(request) != semantic_key(
        replace(request, capabilities=CapabilityPolicy(internet=True))
    )


def test_semantic_identity_uses_explicit_task_vocabulary() -> None:
    document = semantic_document(
        LLMRequest("task", "prompt", JsonOutput({"type": "object"}))
    )
    assert document["task_id"] == "task"
    assert "logical_key" not in document
