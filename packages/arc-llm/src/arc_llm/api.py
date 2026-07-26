"""Public facades over the single task executor and arc-jobs engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from arc_jobs import (
    AtomicStateStore,
    ArtifactSourceRef,
    Awaiting,
    StoppedError,
    CorruptStateError,
    decode_artifact_digest,
    encode_artifact_digest,
    Failed,
    IdempotencyConflictError as JobsIdempotencyConflictError,
    JsonValue,
    EventSink,
    Paused,
    RunContext,
    RunEngine,
    RunError,
    RunRepository,
    RunSnapshot,
    RunSpec,
    RunView,
    SemanticKeyDigest,
    ResumeInputConflictError as JobsResumeInputConflictError,
    ResumeMismatchError as JobsResumeMismatchError,
    StateConflictError,
    Succeeded,
    validate_artifact_id,
    validate_simple_id,
)

from .executor import HANDLER_NAME, LLMTaskExecutor
from .identity import (
    AdoptionAuthorization,
    derive_run_id,
    semantic_document,
    semantic_key,
)
from .errors import (
    CorruptTaskStateError,
    IdempotencyConflictError,
    ResumeInputConflictError,
    ResumeKeyMismatchError,
)
from .outcome import (
    LLMStopped,
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


@dataclass(frozen=True)
class _StandaloneAdoption:
    source: ArtifactSourceRef
    authorization: AdoptionAuthorization | None


@dataclass(frozen=True)
class _StandaloneInvocation:
    revision: int
    mode: str
    request: LLMRequest
    adoption: _StandaloneAdoption | None


class _StandaloneInvocationContract:
    schema_version = "arc.llm.standalone_invocation.v2"

    def encode(self, value: _StandaloneInvocation) -> Mapping[str, JsonValue]:
        self._validate(value)
        adoption: JsonValue = None
        if value.adoption is not None:
            source = value.adoption.source
            authorization = value.adoption.authorization
            adoption = {
                "source_run_id": source.source_run_id,
                "source_artifact_id": source.source_artifact_id,
                "expected_digest": encode_artifact_digest(source.expected_digest),
                "authorization": (
                    None
                    if authorization is None
                    else {
                        "source_semantic_key_sha256": (
                            authorization.source_semantic_key.sha256
                        ),
                        "target_semantic_key_sha256": (
                            authorization.target_semantic_key.sha256
                        ),
                        "reason": authorization.reason,
                    }
                ),
            }
        return {
            "revision": value.revision,
            "mode": value.mode,
            "request": request_to_document(value.request),
            "adoption": adoption,
        }

    def decode(self, document: Mapping[str, JsonValue]) -> _StandaloneInvocation:
        if set(document) != {"revision", "mode", "request", "adoption"}:
            raise CorruptStateError("invalid standalone invocation fields")
        revision = document["revision"]
        mode = document["mode"]
        request_document = document["request"]
        if (
            not isinstance(revision, int)
            or isinstance(revision, bool)
            or revision != 0
            or not isinstance(mode, str)
            or not isinstance(request_document, Mapping)
        ):
            raise CorruptStateError("invalid standalone invocation")
        try:
            request = decode_request(request_document)
        except Exception as exc:
            raise CorruptStateError("invalid standalone invocation request") from exc
        adoption_document = document["adoption"]
        adoption = None
        if adoption_document is not None:
            if (
                not isinstance(adoption_document, Mapping)
                or set(adoption_document)
                != {
                    "source_run_id",
                    "source_artifact_id",
                    "expected_digest",
                    "authorization",
                }
            ):
                raise CorruptStateError("invalid standalone adoption fields")
            try:
                source = ArtifactSourceRef(
                    validate_simple_id(
                        adoption_document["source_run_id"],
                        label="source run id",
                    ),
                    validate_artifact_id(
                        adoption_document["source_artifact_id"]
                    ),
                    decode_artifact_digest(
                        adoption_document["expected_digest"]
                    ),
                )
            except Exception as exc:
                raise CorruptStateError("invalid standalone adoption source") from exc
            authorization_document = adoption_document["authorization"]
            authorization = None
            if authorization_document is not None:
                if (
                    not isinstance(authorization_document, Mapping)
                    or set(authorization_document)
                    != {
                        "source_semantic_key_sha256",
                        "target_semantic_key_sha256",
                        "reason",
                    }
                ):
                    raise CorruptStateError(
                        "invalid standalone adoption authorization fields"
                    )
                try:
                    authorization = AdoptionAuthorization(
                        SemanticKeyDigest(
                            authorization_document[
                                "source_semantic_key_sha256"
                            ]
                        ),
                        SemanticKeyDigest(
                            authorization_document[
                                "target_semantic_key_sha256"
                            ]
                        ),
                        authorization_document["reason"],
                    )
                except Exception as exc:
                    raise CorruptStateError(
                        "invalid standalone adoption authorization"
                    ) from exc
            adoption = _StandaloneAdoption(source, authorization)
        invocation = _StandaloneInvocation(revision, mode, request, adoption)
        self._validate(invocation, reading=True)
        return invocation

    def validate_transition(
        self,
        previous: _StandaloneInvocation | None,
        next: _StandaloneInvocation,
    ) -> None:
        self._validate(next)
        if previous is not None or next.revision != 0:
            raise ValueError("Standalone LLM invocation is immutable.")

    @staticmethod
    def _validate(
        value: _StandaloneInvocation,
        *,
        reading: bool = False,
    ) -> None:
        valid = (
            isinstance(value, _StandaloneInvocation)
            and type(value.revision) is int
            and value.revision == 0
            and isinstance(value.mode, str)
            and value.mode in {"generate", "adopt"}
            and isinstance(value.request, LLMRequest)
            and (
                (value.mode == "generate" and value.adoption is None)
                or (
                    value.mode == "adopt"
                    and isinstance(value.adoption, _StandaloneAdoption)
                )
            )
        )
        if valid and value.adoption is not None:
            valid = (
                isinstance(value.adoption.source, ArtifactSourceRef)
                and (
                    value.adoption.authorization is None
                    or isinstance(
                        value.adoption.authorization,
                        AdoptionAuthorization,
                    )
                )
            )
        if valid and value.adoption is not None:
            try:
                validate_simple_id(
                    value.adoption.source.source_run_id,
                    label="source run id",
                )
                validate_artifact_id(
                    value.adoption.source.source_artifact_id
                )
                encode_artifact_digest(
                    value.adoption.source.expected_digest
                )
                if value.adoption.authorization is not None:
                    AdoptionAuthorization(
                        value.adoption.authorization.source_semantic_key,
                        value.adoption.authorization.target_semantic_key,
                        value.adoption.authorization.reason,
                    )
            except Exception:
                valid = False
        if not valid:
            error_type = CorruptStateError if reading else ValueError
            raise error_type("invalid standalone invocation")


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

    def execute_or_resume(
        self,
        context: RunContext,
        request: LLMRequest,
        *,
        input: ResumeInput | None = None,
        options: LLMExecutionOptions = LLMExecutionOptions(),
    ) -> LLMTaskOutcome:
        """Drive one child task without creating a nested arc-jobs run."""

        state = self._executor._task_store(context, request.task_id).read()
        if state is not None and state.pause is not None:
            if state.semantic_key.sha256 != semantic_key(request).sha256:
                return LLMFailed(IdempotencyConflictError())
            if state.pause.input_required and input is None:
                return self._executor._paused_outcome(state.pause)
            return self._executor.resume(
                context,
                request.task_id,
                input=input,
                options=options,
            )
        if input is not None:
            return LLMFailed(ResumeKeyMismatchError())
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
        options: LLMExecutionOptions = LLMExecutionOptions(),
    ) -> LLMTaskOutcome:
        return self._executor.adopt_and_revalidate(
            context,
            request,
            source,
            authorization=authorization,
            options=options,
        )


class _StandaloneInvocationMissing(CorruptStateError):
    pass


class _StandaloneHandler:
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
        try:
            invocation = self._load_durable_invocation(context)
            request = invocation.request
            if invocation.mode == "adopt":
                assert invocation.adoption is not None
                outcome = self.service.adopt_and_revalidate(
                    context,
                    request,
                    invocation.adoption.source,
                    authorization=invocation.adoption.authorization,
                    options=self.options,
                )
            else:
                state = self.service._executor._task_store(
                    context, request.task_id
                ).read()
                if state is not None and state.pause is not None:
                    resume_input = (
                        None
                        if context.resume_input is None
                        else decode_resume_input(context.resume_input)
                    )
                    outcome = self.service.execute_or_resume(
                        context,
                        request,
                        input=resume_input,
                        options=self.options,
                    )
                else:
                    outcome = self.service.execute_or_resume(
                        context,
                        request,
                        options=self.options,
                    )
        except (
            _StandaloneInvocationMissing,
            CorruptTaskStateError,
            CorruptStateError,
        ) as exc:
            outcome = _standalone_invocation_failure(exc)
            request = None
        self.last_outcome = outcome
        return _run_outcome(
            self.service,
            context,
            "" if request is None else request.task_id,
            outcome,
        )

    def _load_durable_invocation(
        self,
        context: RunContext,
    ) -> _StandaloneInvocation:
        invocation = _standalone_invocation_store(
            context.repository, context.run_id
        ).read()
        if invocation is not None:
            if semantic_document(invocation.request) != dict(
                context.semantic_input
            ):
                raise CorruptStateError(
                    "standalone invocation does not match the run spec"
                )
            return invocation
        raise _StandaloneInvocationMissing(
            "Standalone LLM run has no recoverable invocation."
        )


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
    if isinstance(outcome, LLMStopped):
        raise StoppedError("LLM task stopped")
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
        event_sink: EventSink | None = None,
    ) -> LLMRunResult:
        resolved_run_id = run_id or derive_run_id(HANDLER_NAME, request.task_id)
        return self._invoke(
            RunRepository(run_root),
            resolved_run_id,
            _StandaloneInvocation(0, "generate", request, None),
            options=options,
            event_sink=event_sink,
        )

    def resume(
        self,
        *,
        run_root: Path,
        run_id: str,
        input: ResumeInput | None = None,
        options: LLMExecutionOptions = LLMExecutionOptions(),
        event_sink: EventSink | None = None,
    ) -> LLMRunResult:
        repository = RunRepository(run_root)
        handler = _StandaloneHandler(self.service, options=options)
        try:
            snapshot = RunEngine(repository).resume(
                run_id,
                handler,
                input=None if input is None else resume_input_to_document(input),
                event_sink=event_sink,
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
        if outcome is None:
            context = RunContext(
                repository,
                snapshot,
                resume_input=None,
            )
            try:
                invocation = handler._load_durable_invocation(context)
            except (
                _StandaloneInvocationMissing,
                CorruptTaskStateError,
                CorruptStateError,
            ) as exc:
                outcome = _standalone_invocation_failure(exc)
            else:
                if snapshot.result_ref is not None:
                    outcome = self._replay_succeeded(
                        repository,
                        snapshot,
                        invocation.request,
                        options=options,
                    )
        return LLMRunResult(snapshot, outcome)

    def adopt(
        self,
        request: LLMRequest,
        source: ArtifactSourceRef,
        *,
        run_root: Path,
        run_id: str | None = None,
        authorization: AdoptionAuthorization | None = None,
        options: LLMExecutionOptions = LLMExecutionOptions(),
        event_sink: EventSink | None = None,
    ) -> LLMRunResult:
        resolved_run_id = run_id or derive_run_id(HANDLER_NAME, request.task_id)
        return self._invoke(
            RunRepository(run_root),
            resolved_run_id,
            _StandaloneInvocation(
                0,
                "adopt",
                request,
                _StandaloneAdoption(source, authorization),
            ),
            options=options,
            event_sink=event_sink,
        )

    def _invoke(
        self,
        repository: RunRepository,
        run_id: str,
        invocation: _StandaloneInvocation,
        *,
        options: LLMExecutionOptions,
        event_sink: EventSink | None,
    ) -> LLMRunResult:
        try:
            durable = _reserve_standalone_invocation(
                repository,
                run_id,
                invocation,
            )
        except CorruptStateError as exc:
            snapshot = _snapshot_for_standalone_result(
                repository,
                run_id,
                fallback_request=invocation.request,
            )
            return LLMRunResult(
                snapshot,
                LLMFailed(
                    CorruptTaskStateError(
                        f"Standalone LLM invocation is corrupt: {exc}",
                        details={"code": "standalone_invocation_corrupt"},
                    )
                ),
            )
        if durable is not None and durable != invocation:
            snapshot = _snapshot_for_standalone_result(
                repository,
                run_id,
                fallback_request=durable.request,
            )
            return LLMRunResult(
                snapshot,
                LLMFailed(IdempotencyConflictError()),
            )
        handler = _StandaloneHandler(
            self.service,
            options=options,
        )
        try:
            snapshot = RunEngine(repository).execute(
                RunSpec(
                    run_id,
                    HANDLER_NAME,
                    semantic_document(invocation.request),
                ),
                handler,
                event_sink=event_sink,
            )
        except JobsIdempotencyConflictError:
            snapshot = repository.inspect(run_id).snapshot
            return LLMRunResult(
                snapshot,
                LLMFailed(IdempotencyConflictError()),
            )
        outcome = handler.last_outcome or self._replay_succeeded(
            repository,
            snapshot,
            invocation.request,
            options=options,
        )
        return LLMRunResult(snapshot, outcome)

    def inspect(self, *, run_root: Path, run_id: str) -> LLMRunView:
        return LLMRunView(RunRepository(run_root).inspect(run_id))

    def _replay_succeeded(
        self,
        repository: RunRepository,
        snapshot: RunSnapshot,
        request: LLMRequest,
        *,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome | None:
        if snapshot.result_ref is None:
            return None
        context = RunContext(
            repository,
            snapshot,
            resume_input=None,
        )
        return self.service.execute(context, request, options=options)

def _standalone_invocation_store(
    repository: RunRepository,
    run_id: str,
) -> AtomicStateStore[_StandaloneInvocation]:
    return AtomicStateStore(
        repository.run_directory(run_id)
        / "state"
        / "llm-standalone-invocation.json",
        _StandaloneInvocationContract(),
    )


def _standalone_invocation_failure(
    error: _StandaloneInvocationMissing
    | CorruptTaskStateError
    | CorruptStateError,
) -> LLMFailed:
    if isinstance(error, _StandaloneInvocationMissing):
        return LLMFailed(
            CorruptTaskStateError(
                str(error),
                details={"code": "standalone_invocation_missing"},
            )
        )
    if isinstance(error, CorruptTaskStateError):
        return LLMFailed(error)
    return LLMFailed(
        CorruptTaskStateError(
            f"Standalone LLM invocation is corrupt: {error}",
            details={"code": "standalone_invocation_corrupt"},
        )
    )


def _reserve_standalone_invocation(
    repository: RunRepository,
    run_id: str,
    invocation: _StandaloneInvocation,
) -> _StandaloneInvocation | None:
    run_directory = repository.run_directory(run_id)
    spec_path = run_directory / "spec.json"
    store = _standalone_invocation_store(repository, run_id)
    if spec_path.exists():
        spec = repository.read_spec(run_id)
        existing = store.read()
        if (
            existing is not None
            and semantic_document(existing.request)
            != dict(spec.semantic_input)
        ):
            raise CorruptStateError(
                "standalone invocation does not match the immutable run spec"
            )
        return existing
    existing = store.read()
    if existing is None:
        try:
            store.create(invocation)
            return invocation
        except StateConflictError:
            existing = store.read()
    if existing is None:
        raise CorruptStateError(
            "standalone invocation disappeared after create conflict"
        )
    return existing


def _snapshot_for_standalone_result(
    repository: RunRepository,
    run_id: str,
    *,
    fallback_request: LLMRequest,
) -> RunSnapshot:
    run_directory = repository.run_directory(run_id)
    if (run_directory / "snapshot.json").exists():
        return repository.inspect(run_id).snapshot
    if (run_directory / "spec.json").exists():
        spec = repository.read_spec(run_id)
    else:
        spec = RunSpec(
            run_id,
            HANDLER_NAME,
            semantic_document(fallback_request),
        )
    return repository.create(spec)
