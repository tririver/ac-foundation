from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any, Mapping, Protocol

from arc_jobs import (
    Awaiting,
    JsonValue,
    RunContext,
    RunError,
    canonical_json_bytes,
)
from arc_llm import (
    LLMCompleted,
    LLMExecutionOptions,
    LLMFailed,
    LLMRequest,
    LLMTaskOutcome,
    ModelSelection,
    ProviderUsage,
    ResumeInput,
    decode_resume_input,
    resume_input_matches,
)


class TaskService(Protocol):
    def execute_or_resume(
        self,
        context: RunContext,
        request: LLMRequest,
        *,
        input: ResumeInput | None = None,
        options: Any = ...,
    ) -> LLMTaskOutcome: ...


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

    bounded_feedback = feedback.strip()[:4000]
    identity = {
        "schema_version": "arc.document.semantic_output_retry.v1",
        "source_task_id": request.task_id,
        "validator_contract": validator_contract,
        "feedback": bounded_feedback,
    }
    digest = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    task_id = (
        f"{request.task_id[:72]}-semantic-retry-{digest[:24]}"
    )
    prompt = "\n\n".join(
        (
            request.prompt,
            (
                "ARC package validation found that the previous JSON response "
                "was structurally valid but unusable. Produce a complete fresh "
                "response for the original task; do not merely describe or patch "
                "the prior response."
            ),
            f"Validator contract: {validator_contract}",
            f"Validation feedback:\n{bounded_feedback}",
        )
    )
    return replace(request, task_id=task_id, prompt=prompt)


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


def execute_routed(
    service: TaskService,
    context: RunContext,
    request: LLMRequest,
    *,
    resume_input: ResumeInput | None,
    options: LLMExecutionOptions,
) -> LLMTaskOutcome:
    if resume_input is not None and resume_input_matches(request, resume_input):
        return service.execute_or_resume(
            context, request, input=resume_input, options=options
        )
    return service.execute_or_resume(context, request, options=options)


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


def awaiting_from_pause(outcome: Any) -> Awaiting:
    return Awaiting(
        outcome.reason,
        outcome.resume_key,
        outcome.input_required,
        outcome.request_ref,
        outcome.response_contract,
        outcome.details,
    )


def run_error_from_failure(outcome: LLMFailed) -> RunError:
    return RunError(
        outcome.error.code.value,
        str(outcome.error),
        outcome.error.details,
    )


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
