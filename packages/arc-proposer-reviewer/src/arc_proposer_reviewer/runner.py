"""Reusable durable facade for proposer-reviewer batches."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from arc_jobs import (
    JsonValue,
    RunEngine,
    RunRepository,
    RunSnapshot,
    RunSpec,
    RunView,
)
from arc_llm import LLMTaskService

from .handler import ProposerReviewerHandler
from .identity import derive_batch_run_id
from .models import BatchRequest, ExecutionOptions
from .projection import BatchProjection
from .protocol import decode_batch_request, encode_batch_request
from .service import ProposerReviewerService
from .validation import validate_execution_options


_RunRoot = RunRepository | str | Path


class BatchRunner:
    """Prepare, execute, resume, and inspect one durable batch lineage."""

    def __init__(self, task_service: LLMTaskService | None = None) -> None:
        self._task_service = task_service

    def prepare(
        self,
        request: BatchRequest,
        run_root: _RunRoot,
        run_id: str | None = None,
    ) -> RunSnapshot:
        repository = _repository(run_root)
        return repository.create(_run_spec(request, run_id))

    def run(
        self,
        request: BatchRequest,
        run_root: _RunRoot,
        run_id: str | None = None,
        options: ExecutionOptions = ExecutionOptions(),
    ) -> RunSnapshot:
        validate_execution_options(options)
        repository = _repository(run_root)
        spec = _run_spec(request, run_id)
        repository.create(spec)
        return RunEngine(repository).execute(spec, self._handler(options))

    def resume(
        self,
        run_root: _RunRoot,
        run_id: str,
        input: Mapping[str, JsonValue] | None = None,
        options: ExecutionOptions = ExecutionOptions(),
    ) -> RunSnapshot:
        validate_execution_options(options)
        repository = _repository(run_root)
        _read_request(repository, run_id)
        return RunEngine(repository).resume(
            run_id,
            self._handler(options),
            input=input,
        )

    def stop(
        self,
        run_root: _RunRoot,
        run_id: str,
        reason: str | None = None,
    ) -> RunView:
        repository = _repository(run_root)
        repository.inspect(run_id)
        _read_request(repository, run_id)
        return repository.request_stop(run_id, reason=reason)

    def inspect(self, run_root: _RunRoot, run_id: str) -> RunView:
        repository = _repository(run_root)
        view = repository.inspect(run_id)
        _read_request(repository, run_id)
        return view

    def projection(self, run_root: _RunRoot, run_id: str) -> BatchProjection:
        return BatchProjection(_repository(run_root), run_id)

    def read_request(self, run_root: _RunRoot, run_id: str) -> BatchRequest:
        return _read_request(_repository(run_root), run_id)

    def _handler(self, options: ExecutionOptions) -> ProposerReviewerHandler:
        task_service = (
            self._task_service
            if self._task_service is not None
            else LLMTaskService()
        )
        return ProposerReviewerHandler(
            ProposerReviewerService(task_service),
            options=options,
        )


def _repository(run_root: _RunRoot) -> RunRepository:
    if isinstance(run_root, RunRepository):
        return run_root
    return RunRepository(run_root)


def _run_spec(request: BatchRequest, run_id: str | None) -> RunSpec:
    resolved_run_id = run_id or derive_batch_run_id(request.batch_id)
    return RunSpec(
        resolved_run_id,
        ProposerReviewerHandler.name,
        encode_batch_request(request),
    )


def _read_request(repository: RunRepository, run_id: str) -> BatchRequest:
    spec = repository.read_spec(run_id)
    if spec.handler != ProposerReviewerHandler.name:
        raise ValueError("run is not a proposer-reviewer batch")
    return decode_batch_request(spec.semantic_input)


__all__ = ["BatchRunner"]
