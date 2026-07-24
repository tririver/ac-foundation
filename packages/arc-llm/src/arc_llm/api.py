"""Public facades over the single task executor and arc-jobs engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arc_jobs import (
    ArtifactSourceRef,
    Awaiting,
    CancelledError,
    Failed,
    IdempotencyConflictError as JobsIdempotencyConflictError,
    Paused,
    RunContext,
    RunEngine,
    RunError,
    RunRepository,
    RunSnapshot,
    RunSpec,
    RunView,
    ResumeInputConflictError as JobsResumeInputConflictError,
    ResumeMismatchError as JobsResumeMismatchError,
    Succeeded,
)

from .executor import HANDLER_NAME, LLMTaskExecutor
from .identity import AdoptionAuthorization, derive_run_id
from .errors import (
    IdempotencyConflictError,
    ResumeInputConflictError,
    ResumeKeyMismatchError,
)
from .outcome import (
    LLMCancelled,
    LLMCompleted,
    LLMFailed,
    LLMPaused,
    LLMTaskOutcome,
)
from .providers import ProviderRegistry
from .request import (
    LLMExecutionOptions,
    LLMRequest,
    ResumeInput,
    decode_request,
    decode_resume_input,
    request_to_document,
    resume_input_to_document,
)


@dataclass(frozen=True)
class LLMRunResult:
    snapshot: RunSnapshot
    outcome: LLMTaskOutcome | None


@dataclass(frozen=True)
class LLMRunView:
    run: RunView


class LLMTaskService:
    """Reusable in-run LLM task service."""

    def __init__(self, *, registry: ProviderRegistry | None = None) -> None:
        self._executor = LLMTaskExecutor(registry)

    def execute(
        self,
        context: RunContext,
        request: LLMRequest,
        *,
        options: LLMExecutionOptions = LLMExecutionOptions(),
    ) -> LLMTaskOutcome:
        return self._executor.execute(context, request, options=options)

    def resume(
        self,
        context: RunContext,
        task_id: str,
        *,
        input: ResumeInput | None = None,
        options: LLMExecutionOptions = LLMExecutionOptions(),
    ) -> LLMTaskOutcome:
        return self._executor.resume(
            context,
            task_id,
            input=input,
            options=options,
        )

    def adopt_and_revalidate(
        self,
        context: RunContext,
        request: LLMRequest,
        source: ArtifactSourceRef,
        *,
        authorization: AdoptionAuthorization | None = None,
    ) -> LLMTaskOutcome:
        return self._executor.adopt_and_revalidate(
            context,
            request,
            source,
            authorization=authorization,
        )


class _LLMHandler:
    name = HANDLER_NAME

    def __init__(
        self,
        service: LLMTaskService,
        *,
        options: LLMExecutionOptions,
    ) -> None:
        self.service = service
        self.options = options
        self.last_outcome: LLMTaskOutcome | None = None

    def execute(self, context: RunContext) -> Any:
        request = decode_request(context.semantic_input)
        state = self.service._executor._task_store(context, request.task_id).read()
        if state is not None and state.pause is not None:
            resume_input = (
                None
                if context.resume_input is None
                else decode_resume_input(context.resume_input)
            )
            outcome = self.service.resume(
                context,
                request.task_id,
                input=resume_input,
                options=self.options,
            )
        else:
            outcome = self.service.execute(context, request, options=self.options)
        self.last_outcome = outcome
        return _run_outcome(self.service, context, request.task_id, outcome)


class _AdoptHandler:
    name = HANDLER_NAME

    def __init__(
        self,
        service: LLMTaskService,
        request: LLMRequest,
        source: ArtifactSourceRef,
        authorization: AdoptionAuthorization | None,
    ) -> None:
        self.service = service
        self.request = request
        self.source = source
        self.authorization = authorization
        self.last_outcome: LLMTaskOutcome | None = None

    def execute(self, context: RunContext) -> Any:
        outcome = self.service.adopt_and_revalidate(
            context,
            self.request,
            self.source,
            authorization=self.authorization,
        )
        self.last_outcome = outcome
        return _run_outcome(self.service, context, self.request.task_id, outcome)


def _run_outcome(
    service: LLMTaskService,
    context: RunContext,
    task_id: str,
    outcome: LLMTaskOutcome,
) -> Any:
    if isinstance(outcome, LLMCompleted):
        state = service._executor._task_store(context, task_id).read()
        assert state is not None and state.accepted is not None
        return Succeeded(state.accepted.artifact_ref)
    if isinstance(outcome, LLMPaused):
        return Paused(
            Awaiting(
                outcome.reason,
                outcome.resume_key,
                outcome.input_required,
                outcome.request_ref,
                outcome.response_contract,
                outcome.details,
            )
        )
    if isinstance(outcome, LLMFailed):
        return Failed(
            RunError(
                outcome.error.code.value,
                str(outcome.error),
                outcome.error.details,
            )
        )
    if isinstance(outcome, LLMCancelled):
        raise CancelledError("LLM task cancelled")
    raise RuntimeError("Unknown LLM task outcome.")


class LLMClient:
    """Blocking standalone facade backed by a durable arc-jobs run."""

    def __init__(
        self,
        *,
        service: LLMTaskService | None = None,
        registry: ProviderRegistry | None = None,
    ) -> None:
        self.service = service or LLMTaskService(registry=registry)

    def generate(
        self,
        request: LLMRequest,
        *,
        run_root: Path,
        run_id: str | None = None,
        options: LLMExecutionOptions = LLMExecutionOptions(),
    ) -> LLMRunResult:
        resolved_run_id = run_id or derive_run_id(HANDLER_NAME, request.task_id)
        repository = RunRepository(run_root)
        handler = _LLMHandler(self.service, options=options)
        try:
            snapshot = RunEngine(repository).execute(
                RunSpec(
                    resolved_run_id,
                    HANDLER_NAME,
                    request_to_document(request),
                ),
                handler,
            )
        except JobsIdempotencyConflictError:
            snapshot = repository.inspect(resolved_run_id).snapshot
            return LLMRunResult(snapshot, LLMFailed(IdempotencyConflictError()))
        outcome = handler.last_outcome or self._replay_succeeded(
            repository, snapshot, request
        )
        return LLMRunResult(snapshot, outcome)

    def resume(
        self,
        *,
        run_root: Path,
        run_id: str,
        input: ResumeInput | None = None,
        options: LLMExecutionOptions = LLMExecutionOptions(),
    ) -> LLMRunResult:
        repository = RunRepository(run_root)
        handler = _LLMHandler(self.service, options=options)
        try:
            snapshot = RunEngine(repository).resume(
                run_id,
                handler,
                input=None if input is None else resume_input_to_document(input),
            )
        except JobsResumeInputConflictError:
            return LLMRunResult(
                repository.inspect(run_id).snapshot,
                LLMFailed(ResumeInputConflictError()),
            )
        except JobsResumeMismatchError:
            return LLMRunResult(
                repository.inspect(run_id).snapshot,
                LLMFailed(ResumeKeyMismatchError()),
            )
        outcome = handler.last_outcome
        if outcome is None and snapshot.result_ref is not None:
            request = decode_request(repository.read_spec(run_id).semantic_input)
            outcome = self._replay_succeeded(repository, snapshot, request)
        return LLMRunResult(snapshot, outcome)

    def adopt(
        self,
        request: LLMRequest,
        source: ArtifactSourceRef,
        *,
        run_root: Path,
        run_id: str | None = None,
        authorization: AdoptionAuthorization | None = None,
    ) -> LLMRunResult:
        resolved_run_id = run_id or derive_run_id(HANDLER_NAME, request.task_id)
        repository = RunRepository(run_root)
        spec = RunSpec(resolved_run_id, HANDLER_NAME, request_to_document(request))
        handler = _AdoptHandler(
            self.service,
            request,
            source,
            authorization,
        )
        snapshot = RunEngine(repository).execute(spec, handler)
        return LLMRunResult(
            snapshot,
            handler.last_outcome
            or self._replay_succeeded(repository, snapshot, request),
        )

    def inspect(self, *, run_root: Path, run_id: str) -> LLMRunView:
        return LLMRunView(RunRepository(run_root).inspect(run_id))

    def _replay_succeeded(
        self,
        repository: RunRepository,
        snapshot: RunSnapshot,
        request: LLMRequest,
    ) -> LLMTaskOutcome | None:
        if snapshot.result_ref is None:
            return None
        context = RunContext(
            repository,
            snapshot,
            resume_input=None,
            execution_slice=None,
        )
        return self.service.execute(context, request)
