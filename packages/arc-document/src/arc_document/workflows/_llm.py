from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from arc_jobs import JsonValue, RunContext
from arc_llm import (
    LLMCompleted,
    LLMRequest,
    LLMTaskExecutor,
    ModelSelection,
    ProviderUsage,
    ResumeInput,
    awaiting_from_pause,
    decode_resume_input,
    execute_or_resume_matching,
    run_error_from_failure,
    semantic_retry_request as _shared_semantic_retry_request,
)

TaskService = LLMTaskExecutor


class DocumentWorkflowError(RuntimeError):
    """A stable domain error suitable for a run or group-unit result."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def semantic_retry_request(
    request: LLMRequest,
    *,
    validator_contract: str,
    feedback: str,
) -> LLMRequest:
    """Create one deterministic fresh task after package semantic validation."""

    return _shared_semantic_retry_request(
        request,
        identity_schema_version="arc.document.semantic_output_retry.v1",
        validator_contract=validator_contract,
        feedback=feedback,
    )


@dataclass(frozen=True)
class LLMCallProvenance:
    task_id: str
    provider: str | None
    model: str | None
    usage: Mapping[str, JsonValue] | None

    def to_document(self) -> dict[str, JsonValue]:
        return {
            "task_id": self.task_id,
            "provider": self.provider,
            "model": self.model,
            "usage": None if self.usage is None else dict(self.usage),
        }


execute_routed = execute_or_resume_matching


def outer_resume_input(
    context: RunContext,
    *,
    error_code: str,
) -> ResumeInput | None:
    if context.resume_input is None:
        return None
    try:
        return decode_resume_input(context.resume_input)
    except Exception as exc:
        raise DocumentWorkflowError(
            error_code, f"Invalid LLM resume input: {exc}"
        ) from exc


def model_document(value: ModelSelection) -> dict[str, JsonValue]:
    return {
        "provider": value.provider,
        "model": value.model,
        "tier": value.tier,
    }


def provenance(task_id: str, outcome: LLMCompleted) -> LLMCallProvenance:
    return LLMCallProvenance(
        task_id,
        outcome.provider,
        outcome.model,
        usage_document(outcome.usage),
    )


def usage_document(
    value: ProviderUsage | None,
) -> Mapping[str, JsonValue] | None:
    if value is None:
        return None
    return {
        "input_tokens": value.input_tokens,
        "output_tokens": value.output_tokens,
        "cached_input_tokens": value.cached_input_tokens,
    }


__all__ = [
    "LLMCallProvenance",
    "DocumentWorkflowError",
    "TaskService",
    "awaiting_from_pause",
    "execute_routed",
    "model_document",
    "outer_resume_input",
    "provenance",
    "run_error_from_failure",
    "semantic_retry_request",
    "usage_document",
]
