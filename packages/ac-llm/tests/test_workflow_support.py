from __future__ import annotations

from ac_llm import (
    JsonOutput,
    LLMExecutionOptions,
    LLMRequest,
    execute_or_resume_matching,
    semantic_retry_request,
)
from ac_llm import workflow_support


class _TaskService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def execute_or_resume(self, context, request, **kwargs):
        self.calls.append({"context": context, "request": request, **kwargs})
        return "outcome"


def test_execute_or_resume_only_passes_matching_input(monkeypatch) -> None:
    service = _TaskService()
    context = object()
    request = object()
    resume_input = object()
    options = LLMExecutionOptions()
    monkeypatch.setattr(
        workflow_support,
        "resume_input_matches",
        lambda _request, _resume: True,
    )

    assert execute_or_resume_matching(
        service,
        context,
        request,
        resume_input=resume_input,
        options=options,
    ) == "outcome"
    assert service.calls == [
        {
            "context": context,
            "request": request,
            "input": resume_input,
            "options": options,
        }
    ]


def test_semantic_retry_is_deterministic_and_namespace_owned() -> None:
    request = LLMRequest("source-task", "Original", JsonOutput({"type": "object"}))

    first = semantic_retry_request(
        request,
        identity_schema_version="ac.example.semantic_retry.v1",
        validator_contract="ac.example.output.v1",
        feedback="missing field",
    )
    second = semantic_retry_request(
        request,
        identity_schema_version="ac.example.semantic_retry.v1",
        validator_contract="ac.example.output.v1",
        feedback="missing field",
    )
    changed_namespace = semantic_retry_request(
        request,
        identity_schema_version="ac.other.semantic_retry.v1",
        validator_contract="ac.example.output.v1",
        feedback="missing field",
    )

    assert first == second
    assert first.task_id != request.task_id
    assert changed_namespace.task_id != first.task_id
    assert "Validator contract: ac.example.output.v1" in first.prompt
