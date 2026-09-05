"""The single durable LLM execution and recovery loop."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ac_jobs import (
    ArtifactRef,
    ArtifactSourceRef,
    BoundedLeasePool,
    ResumeReason,
    RevisionConflictError,
    SemanticKeyDigest,
    StateConflictError,
    StoppedError,
    atomic_write_bytes,
    decode_artifact_ref,
    encode_artifact_ref,
)

from .config import resolve_model_selection
from .errors import (
    AcLLMError,
    AdoptionAuthorizationError,
    AdoptionConflictError,
    CandidateConflictError,
    CorruptTaskStateError,
    DuplicateHostRequestError,
    ExecutionMismatchError,
    FailureCategory,
    HostRequestIdentityConflictError,
    IdempotencyConflictError,
    InvalidRequestError,
    OutputInvalidError,
    ProviderFailure,
    ResumeKeyMismatchError,
)
from .gate import ProviderCallGate
from .host import (
    HostRequest,
    HostResponse,
    HostResponseStatus,
    broker_execution_document,
    decode_host_continuation,
    decode_host_turn,
    effective_host_mode,
    encode_host_turn,
    host_continuation_document,
    host_turn_schema,
)
from .identity import (
    AdoptionAuthorization,
    _make_resume_key,
    canonical_json_bytes,
    document_sha256,
    execution_document,
    execution_fingerprint,
    semantic_key,
)
from .outcome import (
    LLMCompleted,
    LLMFailed,
    LLMPaused,
    LLMStopped,
    LLMTaskOutcome,
)
from .output import (
    CandidateMaterial,
    enumerate_valid_candidates,
    provider_schema,
    select_output,
    validate_value,
)
from .progress import DurableProviderObserver, message_preview
from .providers import (
    NativeResumeHandle,
    ProviderExecution,
    ProviderInputFile,
    ProviderRegistry,
    ProviderRequest,
    ProviderResumeRequest,
    ProviderTerminalKind,
    ProviderUsage,
    default_registry,
)
from .recovery import (
    AcceptedOrigin,
    AcceptedRecord,
    AcceptedSessionTurn,
    GenerationRecord,
    LLMSessionState,
    LLMTaskState,
    SessionStateContract,
    TaskPause,
    TaskStateContract,
    fresh_generation,
)
from .request import (
    RESUME_SCHEMA_VERSION,
    JsonOutput,
    LLMExecutionOptions,
    LLMExecutionProfile,
    LLMRequest,
    ModelSelection,
    ResumeAction,
    ResumeInput,
    SessionRef,
    TextOutput,
    decode_request,
    encode_output_contract,
    request_to_document,
)
from .schema_formatter import (
    FORMATTER_DECISION_SCHEMA,
    FormattingDecision,
    SchemaFormatterError,
    decode_formatting_decision,
    formatter_prompt,
    formatter_task_id,
    select_formatting_source,
)

PROVIDER_INSTRUCTION_POLICY = (
    "Use ac.llm.host_turn.v1. Request host interaction only when its expected "
    "result has a concrete contribution to completing the original task. State "
    "that desired contribution in the request. Do not continue merely because "
    "more turns remain. After a refused request, retry only when the declared "
    "retry_condition has changed. When no host operation has a concrete "
    "expected contribution, return the best supported final result. A host "
    "request_id is unique for the whole task: never reuse one, including after "
    "a refusal or retry; make a new request_id for every new host request."
)

HANDLER_NAME = "ac.llm.task.v4"

_OUTPUT_RETRY_ARTIFACT = "recovery/output-retry.json"


@dataclass(frozen=True)
class _FormattingFailure:
    reason: str
    record_ref: ArtifactRef | None = None


@dataclass(frozen=True)
class _HostRequestIdentity:
    request: HostRequest
    generation: int
    round: int


class LLMTaskExecutor:
    def __init__(
        self,
        registry: ProviderRegistry | None = None,
        *,
        automatic_output_retry: bool = True,
    ) -> None:
        self.registry = registry or default_registry()
        self._automatic_output_retry = automatic_output_retry

    def execute(
        self,
        context: Any,
        request: LLMRequest,
        *,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        key = semantic_key(request)
        store = self._task_store(context, request.task_id)
        state = store.read()
        if state is not None:
            if state.semantic_key.sha256 != key.sha256:
                return LLMFailed(IdempotencyConflictError())
            if state.accepted is not None:
                return self._replay(context, request, state, options)
        session_key = (
            state.session_key
            if state is not None and state.session_key is not None
            else request.session.session_key
            if request.session is not None
            else request.task_id
        )
        try:
            lease = self._session_lineage_pool(context, session_key).acquire(
                checkpoint=context.checkpoint
            )
        except StoppedError:
            return LLMStopped()
        except Exception as exc:
            return LLMFailed(
                ProviderFailure(
                    f"LLM session lineage lock failed: {exc}",
                    category=FailureCategory.LOCAL_IO,
                )
            )
        try:
            return self._execute_locked(context, request, options=options)
        finally:
            lease.release()

    def _execute_locked(
        self,
        context: Any,
        request: LLMRequest,
        *,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        key = semantic_key(request)
        store = self._task_store(context, request.task_id)
        state = store.read()
        if state is not None:
            if state.semantic_key.sha256 != key.sha256:
                return LLMFailed(IdempotencyConflictError())
            if state.accepted is not None:
                return self._replay(context, request, state, options)
            try:
                durable_request = self._load_request(context, state)
                self._validate_session_lineage(
                    context, durable_request, state, options
                )
            except AcLLMError as exc:
                return LLMFailed(exc)
            if state.pause is not None:
                return self._paused_outcome(state.pause)
            if state.pending_host_turn is not None:
                try:
                    return self._resolve_pending_host_turn(
                        context,
                        durable_request,
                        state,
                        store,
                        options,
                    )
                except StoppedError:
                    return LLMStopped()
                except AcLLMError as exc:
                    return LLMFailed(exc)
                except Exception as exc:
                    return LLMFailed(
                        ProviderFailure(
                            f"Host-turn resolution failed locally: {exc}",
                            category=FailureCategory.LOCAL_IO,
                        )
                    )
            return self._drive(context, durable_request, state, store, options)
        try:
            resolved = self._resolve_model(request)
            adapter = self.registry.create(resolved.provider)
            scoped = self._artifacts(context, key)
            durable_request = self._canonicalize_inputs(
                context,
                request,
                scoped,
            )
            execution_doc = self._execution_document(
                adapter,
                resolved.model,
                durable_request,
                options,
            )
            execution = execution_fingerprint(execution_doc)
            request_ref = scoped.publish_json(
                "requests/semantic.json",
                request_to_document(durable_request),
            )
            scoped.publish_json("execution/1/recipe.json", execution_doc)
            self._publish_policy(scoped, 1, options)
            initial_handle = self._existing_session_handle(
                context,
                durable_request,
                resolved.provider,
                resolved.model,
                execution,
            )
            generation = GenerationRecord(
                1,
                execution,
                native_handle=initial_handle,
            )
            state = LLMTaskState(
                revision=0,
                task_id=request.task_id,
                semantic_key=SemanticKeyDigest(key.sha256),
                resolved_provider=resolved.provider,
                resolved_model=resolved.model,
                generation=generation,
                request_ref=request_ref,
                session_key=(
                    durable_request.session.session_key
                    if durable_request.session is not None
                    else durable_request.task_id
                ),
            )
            store.create(state)
            return self._drive(
                context,
                durable_request,
                state,
                store,
                options,
            )
        except AcLLMError as exc:
            return LLMFailed(exc)
        except StoppedError:
            return LLMStopped()
        except Exception as exc:
            return LLMFailed(
                ProviderFailure(
                    f"Local LLM setup failed: {exc}",
                    category=FailureCategory.LOCAL_IO,
                )
            )

    def resume(
        self,
        context: Any,
        task_id: str,
        *,
        input: ResumeInput | None,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        store = self._task_store(context, task_id)
        state = store.read()
        if state is None:
            return LLMFailed(InvalidRequestError("Unknown LLM task_id."))
        if state.pause is None:
            return LLMFailed(InvalidRequestError("The selected LLM task is not paused."))
        session_key = state.session_key or state.task_id
        try:
            lease = self._session_lineage_pool(context, session_key).acquire(
                checkpoint=context.checkpoint
            )
        except StoppedError:
            return LLMStopped()
        except Exception as exc:
            return LLMFailed(
                ProviderFailure(
                    f"LLM session lineage lock failed: {exc}",
                    category=FailureCategory.LOCAL_IO,
                )
            )
        try:
            return self._resume_locked(
                context,
                task_id,
                input=input,
                options=options,
            )
        except StoppedError:
            return LLMStopped()
        finally:
            lease.release()

    def _resume_locked(
        self,
        context: Any,
        task_id: str,
        *,
        input: ResumeInput | None,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        store = self._task_store(context, task_id)
        state = store.read()
        if state is None:
            return LLMFailed(InvalidRequestError("Unknown LLM task_id."))
        if state.pause is None:
            return LLMFailed(InvalidRequestError("The selected LLM task is not paused."))
        if state.pause.input_required and input is None:
            return LLMFailed(InvalidRequestError("This pause requires resume input."))
        if input is not None and input.resume_key != state.pause.resume_key:
            return LLMFailed(ResumeKeyMismatchError())
        try:
            request = self._load_request(context, state)
            self._validate_session_lineage(
                context,
                request,
                state,
                options,
                allow_execution_replace=(
                    input is not None
                    and input.action is ResumeAction.REPLACE
                ),
            )
        except StoppedError:
            return LLMStopped()
        except AcLLMError as exc:
            return LLMFailed(exc)
        if input is not None and input.action is ResumeAction.ACCEPT_CANDIDATE:
            return self._accept_candidate(
                context, request, state, store, input, options
            )
        if input is not None and input.action is ResumeAction.REPLACE:
            adapter = self.registry.create(state.resolved_provider or "")
            execution_doc = self._execution_document(
                adapter, state.resolved_model or "", request, options
            )
            session_handle = (
                state.current.native_handle
                if request.session is not None
                else None
            )
            state = fresh_generation(
                state,
                execution=execution_fingerprint(execution_doc),
            )
            if session_handle is not None:
                state = replace(
                    state,
                    generation=replace(
                        state.current,
                        native_handle=session_handle,
                    ),
                )
            state = replace(state, pause=None)
            store.compare_and_swap(state.revision - 1, state)
            self._artifacts(context, state.semantic_key).publish_json(
                f"execution/{state.current.generation}/recipe.json",
                execution_doc,
            )
            self._publish_policy(
                self._artifacts(context, state.semantic_key),
                state.current.generation,
                options,
            )
            return self._drive(context, request, state, store, options)
        if state.pending_host_turn is not None:
            if input is None or input.action is not ResumeAction.CONTINUE:
                return LLMFailed(
                    InvalidRequestError("Pending host turn requires a continue input.")
                )
            return self._resume_host_turn(context, request, state, store, input, options)
        state = replace(state, revision=state.revision + 1, pause=None)
        store.compare_and_swap(state.revision - 1, state)
        return self._drive(context, request, state, store, options)

    def adopt_and_revalidate(
        self,
        context: Any,
        request: LLMRequest,
        source: ArtifactSourceRef,
        *,
        authorization: AdoptionAuthorization | None,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        key = semantic_key(request)
        store = self._task_store(context, request.task_id)
        existing = store.read()
        if existing is not None:
            if existing.semantic_key.sha256 != key.sha256:
                return LLMFailed(IdempotencyConflictError())
            if existing.accepted is not None:
                return self._replay(context, request, existing, options)
            return LLMFailed(
                AdoptionConflictError("Adoption requires a task with no provider generation.")
            )
        try:
            verified = context.artifacts.read_source(source)
            value = self._decode_artifact_value(verified.content, request)
            validate_value(value, request.output)
            source_key = self._source_semantic_key(source)
            if authorization is not None and (
                authorization.source_semantic_key != source_key
                or authorization.target_semantic_key != key
            ):
                raise AdoptionAuthorizationError()
            if source_key != key and authorization is None:
                raise AdoptionAuthorizationError()
            scoped = self._artifacts(context, key)
            request_ref = scoped.publish_json(
                "requests/semantic.json", request_to_document(request)
            )
            target_id = self._accepted_artifact_id(request)
            target = scoped.adopt(
                source,
                artifact_id=target_id,
                expected_verified_digest=verified.digest,
            )
            state = LLMTaskState(
                revision=0,
                task_id=request.task_id,
                semantic_key=SemanticKeyDigest(key.sha256),
                resolved_provider=None,
                resolved_model=None,
                generation=None,
                request_ref=request_ref,
                accepted=AcceptedRecord(
                    target,
                    AcceptedOrigin.ADOPTED,
                    None,
                    None,
                    None,
                    reused_from={
                        "source_run_id": source.source_run_id,
                        "source_artifact_id": source.source_artifact_id,
                    },
                ),
            )
            store.create(state)
            return LLMCompleted(
                value, None, None, None, None, self._runtime_warnings(options)
            )
        except AcLLMError as exc:
            return LLMFailed(exc)
        except Exception as exc:
            return LLMFailed(AdoptionConflictError(str(exc)))

    def _drive(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        options: LLMExecutionOptions,
        *,
        crash_retry_available: bool = True,
    ) -> LLMTaskOutcome:
        """Run or locally recover one task with one fresh-generation crash retry.

        Raw provider material is durable before any local validation. A later
        public invocation first replays that material locally. Provider crashes
        clear a native handle and may start one new generation in this call.
        """
        while True:
            try:
                current = store.read()
                if current is None:
                    raise InvalidRequestError("LLM task state disappeared.")
                state = current
                if state.accepted is not None:
                    return self._replay(context, request, state, options)
                state = self._recover_published_raw(context, state, store)
                if state.current.raw_response is not None:
                    return self._recover_saved(
                        context,
                        request,
                        state,
                        store,
                        state.current.raw_response,
                        options,
                    )
                if state.current.attempt_started:
                    if not crash_retry_available:
                        return self._pause(
                            store,
                            state,
                            ResumeReason.EXECUTION_INTERRUPTED,
                            "provider_crash_retry_exhausted",
                            input_required=False,
                        )
                    state = self._fresh_after_crash(
                        context, request, state, store, options
                    )
                    crash_retry_available = False
                    continue
                duplicate_recovery = self._read_duplicate_recovery(context, state)
                if duplicate_recovery is not None:
                    return self._resume_duplicate_host_turn(
                        context,
                        request,
                        state,
                        store,
                        canonical_json_bytes(
                            duplicate_recovery["continuation"]
                        ).decode("utf-8"),
                        options,
                    )
                adapter = self.registry.create(state.resolved_provider or "")
                diagnostic = adapter.doctor()
                if not diagnostic.available:
                    failure = ProviderFailure(
                        "The selected provider is unavailable.",
                        category=FailureCategory.UNAVAILABLE,
                        retryable=True,
                        details={"code": "provider_unavailable"},
                    )
                    provider_failure = (
                        self._publish_provider_failure_diagnostic(
                            context,
                            state,
                            failure,
                            execution=None,
                            fresh_retry_available=crash_retry_available,
                        )
                    )
                    return self._pause(
                        store,
                        state,
                        ResumeReason.EXTERNAL_CONDITION,
                        "provider_unavailable",
                        input_required=False,
                        provider_failure=provider_failure,
                    )
                continuation_prompt = self._prepared_host_turn_prompt(context, state)
                if continuation_prompt is not None or (
                    state.current.native_handle is not None
                    and request.session is not None
                ):
                    execution_result = self._call_resume(
                        context,
                        request,
                        state,
                        store,
                        adapter,
                        options,
                        prompt=continuation_prompt,
                    )
                else:
                    execution_result = self._call_start(
                        context, request, state, store, adapter, options
                    )
                if execution_result.terminal_kind is ProviderTerminalKind.FAILED:
                    assert execution_result.failure is not None
                    failure = execution_result.failure
                    active_state = store.read() or state
                    provider_failure = self._publish_provider_failure_diagnostic(
                        context,
                        active_state,
                        failure,
                        execution=execution_result,
                        fresh_retry_available=crash_retry_available,
                    )
                    if self._is_crash_failure(failure):
                        if not crash_retry_available:
                            return self._pause(
                                store,
                                active_state,
                                ResumeReason.EXECUTION_INTERRUPTED,
                                "provider_crash_retry_exhausted",
                                input_required=False,
                                provider_failure=provider_failure,
                            )
                        state = self._fresh_after_crash(
                            context, request, active_state, store, options
                        )
                        crash_retry_available = False
                        continue
                    return self._provider_failure(
                        store,
                        active_state,
                        failure,
                        provider_failure=provider_failure,
                    )
                if execution_result.terminal_kind is ProviderTerminalKind.STOPPED:
                    self._clear_attempt_started(store.read() or state, store)
                    return LLMStopped()
                outcome = self._consume_execution(
                    context,
                    request,
                    store.read() or state,
                    store,
                    execution_result,
                    options,
                )
                if outcome is not None:
                    return outcome
            except ProviderFailure as failure:
                active_state = store.read() or state
                provider_failure = self._publish_provider_failure_diagnostic(
                    context,
                    active_state,
                    failure,
                    execution=None,
                    fresh_retry_available=crash_retry_available,
                )
                if self._is_crash_failure(failure):
                    if not crash_retry_available:
                        return self._pause(
                            store,
                            active_state,
                            ResumeReason.EXECUTION_INTERRUPTED,
                            "provider_crash_retry_exhausted",
                            input_required=False,
                            provider_failure=provider_failure,
                        )
                    state = self._fresh_after_crash(
                        context, request, active_state, store, options
                    )
                    crash_retry_available = False
                    continue
                return self._provider_failure(
                    store,
                    active_state,
                    failure,
                    provider_failure=provider_failure,
                )
            except StoppedError:
                return LLMStopped()
            except AcLLMError as exc:
                return LLMFailed(exc)
            except Exception as exc:
                return LLMFailed(
                    ProviderFailure(
                        f"LLM execution failed locally: {exc}",
                        category=FailureCategory.LOCAL_IO,
                    )
                )

    def _call_start(
        self, context: Any, request: LLMRequest, state: LLMTaskState, store: Any, adapter: Any, options: LLMExecutionOptions
    ) -> ProviderExecution:
        observer = self._observer(context, state, store)
        observer.progress(
            "llm_message",
            {
                "direction": "request",
                "message_kind": "task_prompt",
                **message_preview(request.prompt),
            },
        )
        workspace = self._prepare_workspace(context, request, state, options=options)
        gate = self._provider_gate(context, options)
        with gate.acquire(
            adapter.name,
            checkpoint=context.checkpoint,
            observe=lambda event, data: self._observe_gate(context, event, data),
        ) as permit:
            state = self._mark_attempt_started(state, store)
            try:
                execution = adapter.start(
                    ProviderRequest(
                        prompt=self._workspace_prompt(),
                        model=state.resolved_model or "",
                        output_schema=self._provider_output_schema(request, options),
                        capabilities=self._capability_document(options),
                        idle_timeout_seconds=options.limits.idle_timeout_seconds,
                        workspace=workspace,
                        environment=options.runtime_environment.apply_to(),
                        inputs=self._provider_input_files(request),
                        reasoning_effort=request.model.reasoning_effort,
                    ),
                    observer,
                    context.stop,
                )
            except ProviderFailure as exc:
                permit.record_failure(exc)
                self._emit_gate_record_warning(context, permit)
                raise
            self._record_gate_execution(context, permit, execution)
            return self._with_gate_warnings(execution, permit)

    def _mark_attempt_started(self, state: LLMTaskState, store: Any) -> LLMTaskState:
        if state.current.attempt_started:
            return state
        next_state = self._update_current(state, attempt_started=True)
        store.compare_and_swap(state.revision, next_state)
        return next_state

    def _clear_attempt_started(self, state: LLMTaskState, store: Any) -> LLMTaskState:
        if not state.current.attempt_started:
            return state
        next_state = self._update_current(state, attempt_started=False)
        store.compare_and_swap(state.revision, next_state)
        return next_state

    def _call_resume(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        adapter: Any,
        options: LLMExecutionOptions,
        *,
        prompt: str | None = None,
    ) -> ProviderExecution:
        handle = state.current.native_handle
        if handle is None:
            raise InvalidRequestError("Native continuation has no saved handle.")
        observer = self._observer(context, state, store)
        logical_prompt = request.prompt if prompt is None else prompt
        observer.progress(
            "llm_message",
            {
                "direction": "request",
                "message_kind": (
                    "task_prompt" if prompt is None else "continuation"
                ),
                **message_preview(logical_prompt),
            },
        )
        workspace = self._prepare_workspace(
            context,
            request,
            state,
            options=options,
            continuation_prompt=prompt,
        )
        gate = self._provider_gate(context, options)
        with gate.acquire(
            adapter.name,
            checkpoint=context.checkpoint,
            observe=lambda event, data: self._observe_gate(context, event, data),
        ) as permit:
            state = self._mark_attempt_started(state, store)
            try:
                execution = adapter.resume(
                    NativeResumeHandle(adapter.name, handle),
                    ProviderResumeRequest(
                        prompt=self._workspace_prompt(),
                        output_schema=self._provider_output_schema(request, options),
                        capabilities=self._capability_document(options),
                        idle_timeout_seconds=options.limits.idle_timeout_seconds,
                        workspace=workspace,
                        environment=options.runtime_environment.apply_to(),
                        inputs=self._provider_input_files(request),
                        model=state.resolved_model,
                        reasoning_effort=request.model.reasoning_effort,
                    ),
                    observer,
                    context.stop,
                )
            except ProviderFailure as exc:
                permit.record_failure(exc)
                self._emit_gate_record_warning(context, permit)
                raise
            self._record_gate_execution(context, permit, execution)
            return self._with_gate_warnings(execution, permit)

    def _observer(self, context: Any, state: LLMTaskState, store: Any) -> DurableProviderObserver:
        def save_handle(handle: NativeResumeHandle) -> None:
            current = store.read()
            if current is None:
                raise InvalidRequestError("Task state disappeared while saving a handle.")
            next_state = self._update_current(current, native_handle=handle.value)
            store.compare_and_swap(current.revision, next_state)

        return DurableProviderObserver(
            context=context,
            on_handle=save_handle,
            task_id=state.task_id,
            provider=state.resolved_provider or "",
            generation=state.current.generation,
            host_turn_round=state.host_turn_round,
        )

    @staticmethod
    def _provider_gate(context: Any, options: LLMExecutionOptions) -> ProviderCallGate:
        return ProviderCallGate(
            context.repository.root / "operational" / "llm",
            options.gate,
        )

    def _record_gate_execution(
        self, context: Any, permit: Any, execution: ProviderExecution
    ) -> None:
        if execution.terminal_kind is ProviderTerminalKind.FAILED:
            assert execution.failure is not None
            permit.record_failure(execution.failure)
        else:
            permit.record_success()
        self._emit_gate_record_warning(context, permit)

    @staticmethod
    def _observe_gate(
        context: Any,
        event: str,
        data: Mapping[str, Any],
    ) -> None:
        try:
            details = dict(data)
            if event == "llm_memory_guard_warning":
                details["attempt"] = context.attempt
                if any(
                    item["event"] == event
                    and item["data"].get("attempt") == context.attempt
                    for item in context.events.tail()
                ):
                    return
            context.events.emit(event, details)
        except Exception:
            pass

    @staticmethod
    def _with_gate_warnings(
        execution: ProviderExecution,
        permit: Any,
    ) -> ProviderExecution:
        if not permit.warnings:
            return execution
        diagnostics = dict(execution.diagnostics)
        existing = diagnostics.get("warnings")
        warnings = list(existing) if isinstance(existing, (list, tuple)) else []
        warnings.extend(permit.warnings)
        diagnostics["warnings"] = warnings
        return replace(execution, diagnostics=diagnostics)

    @staticmethod
    def _emit_gate_record_warning(context: Any, permit: Any) -> None:
        if permit.record_error is None:
            return
        try:
            context.events.emit(
                "llm_gate_state_warning",
                {"code": "provider_gate_state_write_failed"},
            )
        except Exception:
            # Provider results remain authoritative even if both the
            # best-effort circuit write and its bounded warning fail.
            pass

    def _consume_execution(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        execution: ProviderExecution,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome | None:
        scoped = self._artifacts(context, state.semantic_key)
        raw_doc = self._execution_document_value(execution)
        raw_ref = scoped.publish_json(
            self._raw_response_artifact_id(context, state), raw_doc
        )
        current = store.read() or state
        changes: dict[str, Any] = {"raw_response": raw_ref}
        if execution.native_handle is not None:
            changes["native_handle"] = execution.native_handle.value
        current = self._update_current(current, **changes)
        store.compare_and_swap(current.revision - 1, current)
        return self._consume_candidates(
            context,
            request,
            current,
            store,
            execution,
            options,
        )

    def _recover_published_raw(
        self,
        context: Any,
        state: LLMTaskState,
        store: Any,
    ) -> LLMTaskState:
        """Attach raw material that was published before its state CAS.

        Provider material is immutable and its location is deterministic for a
        generation/host-turn pair.  This closes the only publication window in
        which an interrupted process could otherwise issue the provider call a
        second time despite already having a locally recoverable response. A
        dedicated duplicate-recovery raw response supersedes a stale original
        duplicate raw reference after a composed crash.
        """
        recovery = self._read_duplicate_recovery(context, state)
        if state.current.raw_response is not None and recovery is None:
            return state
        scoped = self._artifacts(context, state.semantic_key)
        ref = scoped.find(self._raw_response_artifact_id(context, state))
        if ref is None:
            return state
        if state.current.raw_response == ref:
            return state
        # Verify the persisted document before it affects durable state.
        self._execution_from_raw(context, ref, state.resolved_provider or "")
        next_state = self._update_current(state, raw_response=ref)
        try:
            store.compare_and_swap(state.revision, next_state)
            return next_state
        except RevisionConflictError:
            current = store.read()
            if current is None:
                raise InvalidRequestError("LLM task state disappeared.")
            return current

    def _consume_candidates(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        execution: ProviderExecution,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome | None:
        current = store.read() or state
        try:
            value = select_output(
                execution.candidates,
                self._provider_output_contract(request, options),
            )
        except CandidateConflictError as exc:
            assert not isinstance(request.output, TextOutput)
            candidates = enumerate_valid_candidates(
                execution.candidates,
                self._provider_output_contract(request, options),
            )
            candidate_digests = [item.digest for item in candidates]
            assert tuple(candidate_digests) == exc.candidate_digests
            candidate_ref = self._artifacts(context, current.semantic_key).publish_json(
                f"generations/{current.current.generation}/candidates.json",
                {
                    "candidate_digests": candidate_digests,
                    "candidates": [
                        {
                            "digest": item.digest,
                            "value": self._candidate_display_value(
                                item.value, request, options
                            ),
                            "terminal": item.terminal,
                        }
                        for item in candidates
                    ],
                },
            )
            return self._pause(
                store,
                store.read() or current,
                ResumeReason.SUPERVISION_REQUIRED,
                "candidate_selection_required",
                input_required=True,
                request_ref=candidate_ref,
            )
        except OutputInvalidError as exc:
            retry_record = self._output_retry_record(context, current)
            if retry_record is not None:
                if current.current.generation == retry_record["from_generation"]:
                    return self._start_output_retry(
                        context,
                        request,
                        current,
                        store,
                        options,
                        code=str(retry_record["code"]),
                        message=str(retry_record["message"]),
                        formatting_record_ref=None,
                        retry_record=retry_record,
                    )
                return self._pause_after_output_retry(
                    context,
                    current,
                    store,
                    code="output_invalid",
                    message=str(exc),
                )
            formatting_failure: _FormattingFailure | None = None
            if (
                isinstance(request.output, JsonOutput)
                and request.output.repair == "format"
            ):
                formatted = self._format_invalid_output(
                    context,
                    request,
                    current,
                    store,
                    execution,
                    options,
                )
                if isinstance(formatted, FormattingDecision):
                    if formatted.action == "format":
                        return self._accept(
                            context,
                            request,
                            store.read() or current,
                            store,
                            formatted.value,
                            execution,
                            options,
                        )
                    # The formatter found that required content is absent.
                    formatting_failure = _FormattingFailure(formatted.reason)
                elif isinstance(formatted, _FormattingFailure):
                    formatting_failure = formatted
                elif formatted is not None:
                    return formatted
            if (
                self._automatic_output_retry
                and isinstance(request.output, JsonOutput)
            ):
                return self._start_output_retry(
                    context,
                    request,
                    store.read() or current,
                    store,
                    options,
                    code="output_invalid",
                    message=(
                        formatting_failure.reason
                        if formatting_failure is not None
                        else str(exc)
                    ),
                    formatting_record_ref=(
                        None
                        if formatting_failure is None
                        else formatting_failure.record_ref
                    ),
                )
            return self._pause_after_output_retry(
                context,
                store.read() or current,
                store,
                code="output_invalid",
                message=str(exc),
            )
        if self._uses_host_turn(request, options):
            try:
                turn = decode_host_turn(
                    value,
                    seen_host_request_ids=set(current.seen_host_request_ids),
                )
            except DuplicateHostRequestError as exc:
                return self._recover_duplicate_host_turn(
                    context, request, current, store, exc, options
                )
            if turn.state == "request_host":
                return self._handle_host_turn(
                    context, request, current, store, turn, execution, options
                )
            value = turn.result
        return self._accept(
            context, request, store.read() or current, store, value, execution, options
        )

    def _format_invalid_output(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        execution: ProviderExecution,
        options: LLMExecutionOptions,
    ) -> FormattingDecision | _FormattingFailure | LLMTaskOutcome | None:
        assert isinstance(request.output, JsonOutput)
        source = select_formatting_source(execution.candidates)
        if source is None:
            return None
        current = store.read() or state
        task_id = formatter_task_id(
            outer_semantic_key=current.semantic_key.sha256,
            generation=current.current.generation,
            source_sha256=source.sha256,
        )
        formatter_request = LLMRequest(
            task_id,
            formatter_prompt(source, schema=request.output.schema),
            JsonOutput(FORMATTER_DECISION_SCHEMA, repair="strict"),
            ModelSelection(
                provider=current.resolved_provider or "",
                model=current.resolved_model,
                reasoning_effort=request.model.reasoning_effort,
            ),
        )
        # The formatter is its own durable task and therefore owns its single
        # automatic crash retry independently from the parent generation.
        formatter_options = options
        formatter_executor = LLMTaskExecutor(
            self.registry,
            automatic_output_retry=False,
        )
        formatter_store = formatter_executor._task_store(context, task_id)
        child_state = formatter_store.read()
        if (
            child_state is not None
            and child_state.pause is not None
            and not child_state.pause.input_required
        ):
            outcome = formatter_executor.resume(
                context,
                task_id,
                input=None,
                options=formatter_options,
            )
        else:
            outcome = formatter_executor.execute(
                context,
                formatter_request,
                options=formatter_options,
            )
        child_state = formatter_store.read()
        accepted_ref = (
            None
            if child_state is None or child_state.accepted is None
            else child_state.accepted.artifact_ref
        )
        raw_ref = (
            None
            if child_state is None or child_state.generation is None
            else child_state.current.raw_response
        )
        formatter_usage = outcome.usage if isinstance(outcome, LLMCompleted) else None
        if formatter_usage is None and raw_ref is not None:
            formatter_usage = self._execution_from_raw(
                context,
                raw_ref,
                current.resolved_provider or "",
            ).usage
        child_revision = 0 if child_state is None else child_state.revision
        if isinstance(outcome, LLMStopped):
            return outcome
        if isinstance(outcome, LLMCompleted):
            try:
                decision = decode_formatting_decision(
                    outcome.value,
                    source=source,
                    target_schema=request.output.schema,
                )
            except SchemaFormatterError as exc:
                record_ref = self._publish_formatting_record(
                    context,
                    current,
                    formatter_task_id=task_id,
                    source_sha256=source.sha256,
                    status="invalid",
                    reason=str(exc),
                    child_ref=accepted_ref,
                    child_request_ref=None,
                    child_raw_ref=raw_ref,
                    child_revision=child_revision,
                    formatter_usage=formatter_usage,
                )
                return _FormattingFailure(
                    str(exc),
                    record_ref,
                )
            status = "formatted" if decision.action == "format" else "insufficient"
            record_ref = self._publish_formatting_record(
                context,
                current,
                formatter_task_id=task_id,
                source_sha256=source.sha256,
                status=status,
                reason=decision.reason,
                child_ref=accepted_ref,
                child_request_ref=None,
                child_raw_ref=raw_ref,
                child_revision=child_revision,
                formatter_usage=formatter_usage,
            )
            self._emit_formatting_event(
                context,
                status=status,
                generation=current.current.generation,
                formatter_task_id=task_id,
                record_ref=record_ref,
            )
            return decision
        if isinstance(outcome, LLMPaused):
            status = "paused"
            reason = str(outcome.details.get("code", "formatter_paused"))
            child_request_ref = outcome.request_ref
        else:
            assert isinstance(outcome, LLMFailed)
            status = "failed"
            reason = outcome.error.code.value
            child_request_ref = None
        record_ref = self._publish_formatting_record(
            context,
            current,
            formatter_task_id=task_id,
            source_sha256=source.sha256,
            status=status,
            reason=reason,
            child_ref=accepted_ref,
            child_request_ref=child_request_ref,
            child_raw_ref=raw_ref,
            child_revision=child_revision,
            formatter_usage=formatter_usage,
        )
        if (
            isinstance(outcome, LLMPaused)
            and str(outcome.details.get("code", ""))
            in {"output_invalid", "output_formatting_failed"}
        ):
            return _FormattingFailure(reason, record_ref)
        return self._pause(
            store,
            store.read() or current,
            (
                outcome.reason
                if isinstance(outcome, LLMPaused) and not outcome.input_required
                else ResumeReason.SUPERVISION_REQUIRED
            ),
            "output_formatting_failed",
            input_required=not (
                isinstance(outcome, LLMPaused) and not outcome.input_required
            ),
            request_ref=record_ref,
        )

    def _publish_formatting_record(
        self,
        context: Any,
        state: LLMTaskState,
        *,
        formatter_task_id: str,
        source_sha256: str,
        status: str,
        reason: str,
        child_ref: ArtifactRef | None,
        child_request_ref: ArtifactRef | None,
        child_raw_ref: ArtifactRef | None,
        child_revision: int,
        formatter_usage: ProviderUsage | None,
    ) -> ArtifactRef:
        return self._artifacts(context, state.semantic_key).publish_json(
            (
                f"generations/{state.current.generation}/formatting/"
                f"{child_revision:08d}-{status}.json"
            ),
            {
                "schema_version": "ac.llm.output_formatting.v1",
                "formatter_task_id": formatter_task_id,
                "source_sha256": source_sha256,
                "status": status,
                "reason": reason,
                "child_accepted_ref": (
                    None if child_ref is None else encode_artifact_ref(child_ref)
                ),
                "child_request_ref": (
                    None
                    if child_request_ref is None
                    else encode_artifact_ref(child_request_ref)
                ),
                "child_raw_response_ref": (
                    None if child_raw_ref is None else encode_artifact_ref(child_raw_ref)
                ),
                "formatter_usage": (
                    None
                    if formatter_usage is None
                    else {
                        "input_tokens": formatter_usage.input_tokens,
                        "output_tokens": formatter_usage.output_tokens,
                        "cached_input_tokens": formatter_usage.cached_input_tokens,
                    }
                ),
            },
        )

    @staticmethod
    def _emit_formatting_event(
        context: Any,
        *,
        status: str,
        generation: int,
        formatter_task_id: str,
        record_ref: ArtifactRef,
    ) -> None:
        try:
            context.events.emit(
                "llm_output_formatted",
                {
                    "code": (
                        "schema_formatter_used"
                        if status == "formatted"
                        else "schema_formatter_insufficient"
                    ),
                    "generation": generation,
                    "formatter_task_id": formatter_task_id,
                    "record_artifact_id": record_ref.artifact_id,
                },
            )
        except Exception:
            pass

    def _output_retry_record(
        self,
        context: Any,
        state: LLMTaskState,
    ) -> Mapping[str, Any] | None:
        scoped = self._artifacts(context, state.semantic_key)
        ref = scoped.find(_OUTPUT_RETRY_ARTIFACT)
        if ref is None:
            return None
        try:
            value = json.loads(scoped.read_bytes(ref).decode("utf-8"))
        except Exception as exc:
            raise CorruptTaskStateError(
                "Output-retry record is unreadable."
            ) from exc
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "schema_version",
                "from_generation",
                "to_generation",
                "code",
                "message",
                "formatting_record_ref",
            }
            or value.get("schema_version") != "ac.llm.output_retry.v1"
            or type(value.get("from_generation")) is not int
            or type(value.get("to_generation")) is not int
            or value["to_generation"] != value["from_generation"] + 1
            or not isinstance(value.get("code"), str)
            or not isinstance(value.get("message"), str)
        ):
            raise CorruptTaskStateError("Output-retry record is invalid.")
        return value

    def _start_output_retry(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        options: LLMExecutionOptions,
        *,
        code: str,
        message: str,
        formatting_record_ref: ArtifactRef | None,
        retry_record: Mapping[str, Any] | None = None,
    ) -> LLMTaskOutcome:
        current = store.read() or state
        if retry_record is None:
            retry_record = {
                "schema_version": "ac.llm.output_retry.v1",
                "from_generation": current.current.generation,
                "to_generation": current.current.generation + 1,
                "code": code,
                "message": message[:4000],
                "formatting_record_ref": (
                    None
                    if formatting_record_ref is None
                    else encode_artifact_ref(formatting_record_ref)
                ),
            }
            self._artifacts(context, current.semantic_key).publish_json(
                _OUTPUT_RETRY_ARTIFACT,
                retry_record,
            )
        from_generation = int(retry_record["from_generation"])
        to_generation = int(retry_record["to_generation"])
        if current.current.generation >= to_generation:
            return self._pause_after_output_retry(
                context,
                current,
                store,
                code=code,
                message=message,
            )
        if current.current.generation != from_generation:
            raise CorruptTaskStateError(
                "Output-retry generation does not match durable task state."
            )
        adapter = self.registry.create(current.resolved_provider or "")
        execution_doc = self._execution_document(
            adapter,
            current.resolved_model or "",
            request,
            options,
        )
        next_state = fresh_generation(
            current,
            execution=execution_fingerprint(execution_doc),
        )
        if request.session is not None and current.current.native_handle is not None:
            next_state = replace(
                next_state,
                generation=replace(
                    next_state.current,
                    native_handle=current.current.native_handle,
                ),
            )
        store.compare_and_swap(current.revision, next_state)
        scoped = self._artifacts(context, next_state.semantic_key)
        scoped.publish_json(
            f"execution/{next_state.current.generation}/recipe.json",
            execution_doc,
        )
        self._publish_policy(
            scoped,
            next_state.current.generation,
            options,
        )
        try:
            context.events.emit(
                "llm_output_retry_started",
                {
                    "code": code,
                    "from_generation": from_generation,
                    "to_generation": to_generation,
                },
            )
        except Exception:
            pass
        return self._drive(
            context,
            request,
            next_state,
            store,
            options,
            crash_retry_available=True,
        )

    def _pause_after_output_retry(
        self,
        context: Any,
        state: LLMTaskState,
        store: Any,
        *,
        code: str,
        message: str,
    ) -> LLMPaused:
        current = store.read() or state
        retry_record_ref = self._artifacts(
            context, current.semantic_key
        ).find(_OUTPUT_RETRY_ARTIFACT)
        request_ref = self._artifacts(
            context, current.semantic_key
        ).publish_json(
            (
                f"generations/{current.current.generation}/supervision/"
                "output-retry-exhausted.json"
            ),
            {
                "schema_version": "ac.llm.supervision_request.v1",
                "code": code,
                "message": message[:4000],
                "automatic_retry_exhausted": True,
                "output_attempts": 2,
                "retry_record_ref": (
                    None
                    if retry_record_ref is None
                    else encode_artifact_ref(retry_record_ref)
                ),
            },
        )
        return self._pause(
            store,
            current,
            ResumeReason.SUPERVISION_REQUIRED,
            code,
            input_required=True,
            request_ref=request_ref,
            details={
                "automatic_retry_exhausted": True,
                "output_attempts": 2,
            },
        )

    def _recover_saved(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        raw_ref: ArtifactRef | None,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        if raw_ref is None:
            return LLMFailed(OutputInvalidError("Saved output has no artifact reference."))
        execution = self._execution_from_raw(context, raw_ref, state.resolved_provider or "")
        outcome = self._consume_candidates(
            context,
            request,
            state,
            store,
            execution,
            options,
        )
        assert outcome is not None
        return outcome

    @staticmethod
    def _host_turn_artifact_id(generation: int, round: int, name: str) -> str:
        return f"host-turns/{generation}/{round}/{name}"

    @staticmethod
    def _host_request_identity_artifact_id(request_id: str) -> str:
        digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        return f"host-request-identities/{digest}.json"

    def _duplicate_recovery_artifact_id(self, state: LLMTaskState) -> str:
        return self._host_turn_artifact_id(
            state.current.generation,
            state.host_turn_round,
            "duplicate-recovery.json",
        )

    def _raw_response_artifact_id(self, context: Any, state: LLMTaskState) -> str:
        recovery = self._read_duplicate_recovery(context, state)
        suffix = (
            self._duplicate_recovery_raw_response_name(state)
            if recovery is not None
            else f"{state.host_turn_round}.json"
        )
        return f"generations/{state.current.generation}/raw-responses/{suffix}"

    @staticmethod
    def _duplicate_recovery_raw_response_name(state: LLMTaskState) -> str:
        return f"{state.host_turn_round}-duplicate-recovery-1.json"

    def _duplicate_recovery_has_raw_response(self, state: LLMTaskState) -> bool:
        raw_response = state.current.raw_response
        relative_artifact_id = (
            f"generations/{state.current.generation}/raw-responses/"
            f"{self._duplicate_recovery_raw_response_name(state)}"
        )
        return (
            raw_response is not None
            and (
                raw_response.artifact_id == relative_artifact_id
                or raw_response.artifact_id.endswith(f"/{relative_artifact_id}")
            )
        )

    def _publish_host_request_identity(
        self,
        context: Any,
        state: LLMTaskState,
        host_request: HostRequest,
        *,
        generation: int,
        round: int,
    ) -> ArtifactRef:
        return self._artifacts(context, state.semantic_key).publish_json(
            self._host_request_identity_artifact_id(host_request.request_id),
            {
                "schema_version": "ac.llm.host_request_identity.v1",
                "request_id": host_request.request_id,
                "instruction": host_request.instruction,
                "purpose": host_request.purpose,
                "generation": generation,
                "round": round,
            },
        )

    def _read_host_request_identity(
        self,
        context: Any,
        state: LLMTaskState,
        request_id: str,
    ) -> _HostRequestIdentity | None:
        scoped = self._artifacts(context, state.semantic_key)
        ref = scoped.find(self._host_request_identity_artifact_id(request_id))
        if ref is None:
            return None
        try:
            document = json.loads(scoped.read_bytes(ref).decode("utf-8"))
            if not isinstance(document, dict) or set(document) != {
                "schema_version",
                "request_id",
                "instruction",
                "purpose",
                "generation",
                "round",
            }:
                raise ValueError("invalid closed shape")
            if document["schema_version"] != "ac.llm.host_request_identity.v1":
                raise ValueError("unsupported schema")
            if document["request_id"] != request_id:
                raise ValueError("request ID does not match identity artifact")
            generation = document["generation"]
            round = document["round"]
            if (
                not isinstance(generation, int)
                or isinstance(generation, bool)
                or generation < 1
                or not isinstance(round, int)
                or isinstance(round, bool)
                or round < 1
            ):
                raise ValueError("invalid generation or round")
            return _HostRequestIdentity(
                HostRequest(
                    document["request_id"],
                    document["instruction"],
                    document["purpose"],
                ),
                generation,
                round,
            )
        except Exception as exc:
            raise CorruptTaskStateError("Host request identity artifact is corrupt.") from exc

    def _ensure_host_request_identity(
        self,
        context: Any,
        state: LLMTaskState,
        host_request: HostRequest,
    ) -> None:
        """Attach the identity only after the pending turn is durable."""

        identity = self._read_host_request_identity(
            context, state, host_request.request_id
        )
        if identity is None:
            self._publish_host_request_identity(
                context,
                state,
                host_request,
                generation=state.current.generation,
                round=state.host_turn_round,
            )
            return
        if (
            identity.request != host_request
            or identity.generation != state.current.generation
            or identity.round != state.host_turn_round
        ):
            raise CorruptTaskStateError(
                "Host request identity conflicts with the pending host turn."
            )

    def _read_duplicate_recovery(
        self,
        context: Any,
        state: LLMTaskState,
    ) -> Mapping[str, Any] | None:
        scoped = self._artifacts(context, state.semantic_key)
        ref = scoped.find(self._duplicate_recovery_artifact_id(state))
        if ref is None:
            return None
        try:
            document = json.loads(scoped.read_bytes(ref).decode("utf-8"))
            if not isinstance(document, dict) or set(document) != {
                "schema_version",
                "request_id",
                "instruction",
                "action",
                "continuation",
            }:
                raise ValueError("invalid closed shape")
            if (
                document["schema_version"] != "ac.llm.host_duplicate_recovery.v1"
                or not isinstance(document["request_id"], str)
                or not document["request_id"]
                or not isinstance(document["instruction"], str)
                or not document["instruction"]
                or document["action"] not in {"replay", "synthetic_refusal"}
            ):
                raise ValueError("invalid duplicate recovery")
            continuation = decode_host_continuation(document["continuation"])
            canonical = host_continuation_document(
                continuation.request_id, continuation.response
            )
            if document["continuation"] != canonical:
                raise ValueError("non-canonical continuation")
            if continuation.request_id != document["request_id"]:
                raise ValueError("continuation request ID mismatch")
            return document
        except Exception as exc:
            raise CorruptTaskStateError("Duplicate host-request recovery artifact is corrupt.") from exc

    def _persisted_continuation_prompt(
        self,
        context: Any,
        state: LLMTaskState,
        identity: _HostRequestIdentity,
    ) -> str | None:
        scoped = self._artifacts(context, state.semantic_key)
        ref = scoped.find(
            self._host_turn_artifact_id(
                identity.generation, identity.round, "continuation.json"
            )
        )
        if ref is None:
            return None
        try:
            document = json.loads(scoped.read_bytes(ref).decode("utf-8"))
            continuation = decode_host_continuation(document)
            canonical = host_continuation_document(
                continuation.request_id, continuation.response
            )
            if (
                document != canonical
                or continuation.request_id != identity.request.request_id
            ):
                raise ValueError("continuation does not match request identity")
            return canonical_json_bytes(canonical).decode("utf-8")
        except Exception as exc:
            raise CorruptTaskStateError("Host-turn continuation artifact is corrupt.") from exc

    def _recover_duplicate_host_turn(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        duplicate: DuplicateHostRequestError,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        current = store.read() or state
        recovery = self._read_duplicate_recovery(context, current)
        identity = self._read_host_request_identity(
            context, current, duplicate.request_id
        )
        if identity is not None:
            if identity.request.instruction != duplicate.instruction:
                return LLMFailed(
                    HostRequestIdentityConflictError(
                        duplicate.request_id,
                        expected_instruction=identity.request.instruction,
                        received_instruction=duplicate.instruction,
                    )
                )
            if recovery is not None:
                if self._duplicate_recovery_has_raw_response(current):
                    return LLMFailed(
                        DuplicateHostRequestError(
                            duplicate.request_id,
                            duplicate.instruction,
                            recovery_exhausted=True,
                        )
                    )
                return self._resume_duplicate_host_turn(
                    context,
                    request,
                    current,
                    store,
                    canonical_json_bytes(recovery["continuation"]).decode("utf-8"),
                    options,
                )
            continuation_prompt = self._persisted_continuation_prompt(
                context, current, identity
            )
        else:
            if recovery is not None:
                if self._duplicate_recovery_has_raw_response(current):
                    return LLMFailed(
                        DuplicateHostRequestError(
                            duplicate.request_id,
                            duplicate.instruction,
                            recovery_exhausted=True,
                        )
                    )
                return self._resume_duplicate_host_turn(
                    context,
                    request,
                    current,
                    store,
                    canonical_json_bytes(recovery["continuation"]).decode("utf-8"),
                    options,
                )
            continuation_prompt = None
        action = "replay"
        continuation_document: Mapping[str, Any]
        if continuation_prompt is None:
            action = "synthetic_refusal"
            synthetic = HostResponse(
                HostResponseStatus.REFUSED,
                reason_code="duplicate_host_request_id",
                reason=(
                    "This host request ID was already processed. Use a new unique "
                    "request_id before making another host request."
                ),
                retryable=True,
                retry_condition="a new unique request_id",
            )
            continuation_document = host_continuation_document(
                duplicate.request_id, synthetic
            )
            continuation_prompt = canonical_json_bytes(continuation_document).decode(
                "utf-8"
            )
        else:
            continuation_document = json.loads(continuation_prompt)
        self._artifacts(context, current.semantic_key).publish_json(
            self._duplicate_recovery_artifact_id(current),
            {
                "schema_version": "ac.llm.host_duplicate_recovery.v1",
                "request_id": duplicate.request_id,
                "instruction": duplicate.instruction,
                "action": action,
                "continuation": continuation_document,
            },
        )
        next_state = self._update_current(
            current, raw_response=None, attempt_started=False
        )
        store.compare_and_swap(current.revision, next_state)
        return self._resume_duplicate_host_turn(
            context, request, next_state, store, continuation_prompt, options
        )

    def _resume_duplicate_host_turn(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        continuation_prompt: str,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        if state.current.native_handle is None:
            return LLMFailed(
                DuplicateHostRequestError(
                    self._read_duplicate_recovery(context, state)["request_id"],
                    self._read_duplicate_recovery(context, state)["instruction"],
                    recovery_exhausted=True,
                )
            )
        adapter = self.registry.create(state.resolved_provider or "")
        try:
            execution = self._call_resume(
                context,
                request,
                state,
                store,
                adapter,
                options,
                prompt=continuation_prompt,
            )
        except ProviderFailure as failure:
            return self._provider_failure(store, state, failure)
        active_state = store.read() or state
        if execution.terminal_kind is ProviderTerminalKind.FAILED:
            assert execution.failure is not None
            return self._provider_failure(store, active_state, execution.failure)
        if execution.terminal_kind is ProviderTerminalKind.STOPPED:
            self._clear_attempt_started(active_state, store)
            return LLMStopped()
        outcome = self._consume_execution(
            context, request, active_state, store, execution, options
        )
        assert outcome is not None
        return outcome

    def _handle_host_turn(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        turn: Any,
        execution: ProviderExecution,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        del execution
        assert turn.request is not None
        scoped = self._artifacts(context, state.semantic_key)
        existing_identity = self._read_host_request_identity(
            context, state, turn.request.request_id
        )
        if (
            existing_identity is not None
            and turn.request.request_id in state.seen_host_request_ids
        ):
            return self._recover_duplicate_host_turn(
                context,
                request,
                state,
                store,
                DuplicateHostRequestError(
                    turn.request.request_id, turn.request.instruction
                ),
                options,
            )
        next_round = state.host_turn_round + 1
        turn_ref = scoped.publish_json(
            self._host_turn_artifact_id(
                state.current.generation, next_round, "request.json"
            ),
            encode_host_turn(turn),
        )
        current = store.read() or state
        next_state = replace(
            current,
            revision=current.revision + 1,
            generation=replace(
                current.current,
                raw_response=None,
                attempt_started=False,
            ),
            host_turn_round=next_round,
            pending_host_turn=turn_ref,
            seen_host_request_ids=current.seen_host_request_ids
            + (turn.request.request_id,),
        )
        store.compare_and_swap(next_state.revision - 1, next_state)
        self._ensure_host_request_identity(context, next_state, turn.request)
        if options.host_broker is None:
            return self._pause(
                store,
                next_state,
                ResumeReason.SUPERVISION_REQUIRED,
                "host_broker_required",
                input_required=True,
                request_ref=turn_ref,
            )
        return self._resolve_pending_host_turn(
            context, request, next_state, store, options
        )

    def _broker_invocation_artifact_id(self, state: LLMTaskState) -> str:
        return self._host_turn_artifact_id(
            state.current.generation,
            state.host_turn_round,
            "broker-invocation.json",
        )

    def _broker_completion_artifact_id(self, state: LLMTaskState) -> str:
        return self._host_turn_artifact_id(
            state.current.generation,
            state.host_turn_round,
            "broker-completion.json",
        )

    def _record_host_broker_invocation(
        self,
        context: Any,
        state: LLMTaskState,
        host_request: HostRequest,
    ) -> ArtifactRef:
        """Persist intent before a broker can cause an external side effect."""

        return self._artifacts(context, state.semantic_key).publish_json(
            self._broker_invocation_artifact_id(state),
            {
                "schema_version": "ac.llm.host_broker_invocation.v1",
                "request_id": host_request.request_id,
                "instruction": host_request.instruction,
                "purpose": host_request.purpose,
            },
        )

    def _broker_invocation_started(
        self,
        context: Any,
        state: LLMTaskState,
        host_request: HostRequest,
    ) -> bool:
        scoped = self._artifacts(context, state.semantic_key)
        ref = scoped.find(self._broker_invocation_artifact_id(state))
        if ref is None:
            return False
        try:
            document = json.loads(scoped.read_bytes(ref).decode("utf-8"))
            if document != {
                "schema_version": "ac.llm.host_broker_invocation.v1",
                "request_id": host_request.request_id,
                "instruction": host_request.instruction,
                "purpose": host_request.purpose,
            }:
                raise ValueError("invocation does not match pending host request")
        except Exception as exc:
            raise CorruptTaskStateError("Host broker invocation artifact is corrupt.") from exc
        return True

    def _record_host_broker_completion(
        self,
        context: Any,
        state: LLMTaskState,
        host_request: HostRequest,
        host_response: HostResponse,
        workspace: Path,
    ) -> ArtifactRef:
        """Make a completed broker turn replayable without repeating the broker."""

        scoped = self._artifacts(context, state.semantic_key)
        files: list[dict[str, Any]] = []
        for index, relative in enumerate(host_response.files):
            ref = scoped.publish_bytes(
                self._host_turn_artifact_id(
                    state.current.generation,
                    state.host_turn_round,
                    f"broker-files/{index:03d}",
                ),
                (workspace / relative).read_bytes(),
                media_type="application/octet-stream",
            )
            files.append(
                {
                    "path": relative,
                    "artifact_ref": encode_artifact_ref(ref),
                }
            )
        return scoped.publish_json(
            self._broker_completion_artifact_id(state),
            {
                "schema_version": "ac.llm.host_broker_completion.v1",
                "continuation": host_continuation_document(
                    host_request.request_id, host_response
                ),
                "files": files,
            },
        )

    def _read_host_broker_completion(
        self,
        context: Any,
        state: LLMTaskState,
        host_request: HostRequest,
    ) -> tuple[HostResponse, tuple[tuple[str, ArtifactRef], ...]] | None:
        scoped = self._artifacts(context, state.semantic_key)
        ref = scoped.find(self._broker_completion_artifact_id(state))
        if ref is None:
            return None
        try:
            document = json.loads(scoped.read_bytes(ref).decode("utf-8"))
            if not isinstance(document, dict) or set(document) != {
                "schema_version",
                "continuation",
                "files",
            }:
                raise ValueError("invalid closed shape")
            if document["schema_version"] != "ac.llm.host_broker_completion.v1":
                raise ValueError("unsupported schema")
            continuation = decode_host_continuation(document["continuation"])
            canonical = host_continuation_document(
                continuation.request_id, continuation.response
            )
            if (
                document["continuation"] != canonical
                or continuation.request_id != host_request.request_id
                or not isinstance(document["files"], list)
            ):
                raise ValueError("completion does not match pending host request")
            restored: list[tuple[str, ArtifactRef]] = []
            for index, item in enumerate(document["files"]):
                if not isinstance(item, dict) or set(item) != {"path", "artifact_ref"}:
                    raise ValueError("invalid broker file record")
                relative = item["path"]
                if (
                    not isinstance(relative, str)
                    or index >= len(continuation.response.files)
                    or relative != continuation.response.files[index]
                ):
                    raise ValueError("broker file record does not match response")
                restored.append((relative, decode_artifact_ref(item["artifact_ref"])))
            if len(restored) != len(continuation.response.files):
                raise ValueError("broker completion omits response files")
            return continuation.response, tuple(restored)
        except Exception as exc:
            raise CorruptTaskStateError("Host broker completion artifact is corrupt.") from exc

    def _restore_host_broker_files(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        response: HostResponse,
        files: tuple[tuple[str, ArtifactRef], ...],
        options: LLMExecutionOptions,
    ) -> None:
        if not files:
            return
        workspace = self._prepare_workspace(context, request, state, options=options)
        for relative, ref in files:
            self._publish_workspace_file(
                workspace / relative,
                context.artifacts.read_bytes(ref),
            )
        self._validate_host_files(workspace, response)

    def _resolve_pending_host_turn(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        assert state.pending_host_turn is not None
        raw = context.artifacts.read_bytes(state.pending_host_turn)
        turn = decode_host_turn(json.loads(raw.decode("utf-8")))
        assert turn.request is not None
        self._ensure_host_request_identity(context, state, turn.request)
        completed = self._read_host_broker_completion(context, state, turn.request)
        if completed is not None:
            response, files = completed
            self._restore_host_broker_files(
                context, request, state, response, files, options
            )
            return self._continue_host_turn(
                context, request, state, store, turn.request, response, options
            )
        if self._broker_invocation_started(context, state, turn.request):
            return self._pause(
                store,
                state,
                ResumeReason.SUPERVISION_REQUIRED,
                "host_broker_reconciliation_required",
                input_required=True,
                request_ref=state.pending_host_turn,
                details={"broker_invocation_started": True},
            )
        if options.host_broker is None:
            return self._pause(
                store,
                state,
                ResumeReason.SUPERVISION_REQUIRED,
                "host_broker_required",
                input_required=True,
                request_ref=state.pending_host_turn,
            )
        workspace = self._prepare_workspace(context, request, state, options=options)
        self._record_host_broker_invocation(context, state, turn.request)
        response = options.host_broker.execute(turn.request, workspace=workspace)
        if not isinstance(response, HostResponse):
            raise InvalidRequestError("host broker must return a HostResponse.")
        self._validate_host_files(workspace, response)
        self._record_host_broker_completion(
            context, state, turn.request, response, workspace
        )
        return self._continue_host_turn(
            context, request, state, store, turn.request, response, options
        )

    @staticmethod
    def _validate_host_files(workspace: Path, response: HostResponse) -> None:
        for relative in response.files:
            path = workspace / relative
            if not path.is_file():
                raise InvalidRequestError(
                    f"host broker did not deliver declared workspace file: {relative}"
                )

    def _resume_host_turn(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        input: ResumeInput,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        assert state.pending_host_turn is not None
        raw = context.artifacts.read_bytes(state.pending_host_turn)
        turn = decode_host_turn(json.loads(raw.decode("utf-8")))
        assert turn.request is not None
        response = input.host_response
        if response is None:
            return LLMFailed(
                InvalidRequestError("Host-turn resume requires a host_response.")
            )
        if input.action is not ResumeAction.CONTINUE:
            return LLMFailed(
                InvalidRequestError("Host-turn resume requires continue.")
            )
        workspace = self._prepare_workspace(context, request, state, options=options)
        try:
            self._validate_host_files(workspace, response)
        except InvalidRequestError as exc:
            return LLMFailed(exc)
        state = replace(state, revision=state.revision + 1, pause=None)
        store.compare_and_swap(state.revision - 1, state)
        return self._continue_host_turn(
            context, request, state, store, turn.request, response, options
        )

    def _continue_host_turn(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        host_request: HostRequest,
        host_response: HostResponse,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        scoped = self._artifacts(context, state.semantic_key)
        document = host_continuation_document(host_request.request_id, host_response)
        scoped.publish_json(
            self._host_turn_artifact_id(
                state.current.generation,
                state.host_turn_round,
                "continuation.json",
            ),
            document,
        )
        next_state = replace(
            state,
            revision=state.revision + 1,
            pending_host_turn=None,
            pause=None,
        )
        store.compare_and_swap(state.revision, next_state)
        return self._drive(context, request, next_state, store, options)

    @staticmethod
    def _is_crash_failure(failure: ProviderFailure) -> bool:
        return failure.category in {
            FailureCategory.TRANSPORT,
            FailureCategory.TIMEOUT,
            FailureCategory.INTERNAL,
        }

    def _provider_failure(
        self,
        store: Any,
        state: LLMTaskState,
        failure: ProviderFailure,
        *,
        provider_failure: Mapping[str, Any] | None = None,
    ) -> LLMTaskOutcome:
        state = self._clear_attempt_started(store.read() or state, store)
        if failure.category in {
            FailureCategory.AUTHENTICATION,
            FailureCategory.QUOTA,
            FailureCategory.RATE_LIMIT,
            FailureCategory.UNAVAILABLE,
        }:
            return self._pause(
                store,
                state,
                ResumeReason.EXTERNAL_CONDITION,
                failure.category.value,
                input_required=False,
                provider_failure=provider_failure,
            )
        if failure.category is FailureCategory.STOPPED:
            return LLMStopped()
        if provider_failure is not None:
            failure.details["provider_failure"] = dict(provider_failure)
        return LLMFailed(failure)

    def _publish_provider_failure_diagnostic(
        self,
        context: Any,
        state: LLMTaskState,
        failure: ProviderFailure,
        *,
        execution: ProviderExecution | None,
        fresh_retry_available: bool,
    ) -> Mapping[str, Any]:
        diagnostics = {} if execution is None else execution.diagnostics
        returncode = diagnostics.get("returncode")
        if type(returncode) is not int:
            returncode = None
        terminal_types = diagnostics.get("terminal_event_types")
        safe_terminal_types = (
            [
                value
                for value in terminal_types
                if isinstance(value, str)
                and value in {"error", "turn.completed", "turn.failed"}
            ][:16]
            if isinstance(terminal_types, (list, tuple))
            else []
        )
        detail_code = failure.details.get("code")
        provider_code = failure.details.get("provider_code")

        def safe_code(value: Any) -> str | None:
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 256
                or any(
                    not (character.isalnum() or character in "._:-")
                    for character in value
                )
            ):
                return None
            return value

        retry_after = failure.retry_after_seconds
        if (
            isinstance(retry_after, bool)
            or not isinstance(retry_after, (int, float))
            or not math.isfinite(retry_after)
        ):
            retry_after = None
        summary = {
            "category": failure.category.value,
            "ac_error_code": failure.code.value,
            "provider_code": safe_code(provider_code),
            "detail_code": safe_code(detail_code),
            "returncode": returncode,
            "retryable": failure.retryable,
            "retry_after_seconds": retry_after,
            "fresh_retry_available": fresh_retry_available,
            "terminal_event_types": safe_terminal_types,
        }
        try:
            document: dict[str, Any] = {
                "schema_version": "ac.llm.provider_failure.v2",
                "provider": state.resolved_provider,
                "generation": state.current.generation,
                "host_turn_round": state.host_turn_round,
                **summary,
                "last_terminal_evidence": diagnostics.get(
                    "last_terminal_evidence"
                ),
                "event_count": diagnostics.get("event_count"),
                "raw_events": diagnostics.get("raw_events", []),
                "raw_events_truncated": bool(
                    diagnostics.get("raw_events_truncated")
                ),
                "stdout_bytes": diagnostics.get("stdout_bytes"),
                "stderr_bytes": diagnostics.get("stderr_bytes"),
                "stdout_truncated": bool(
                    diagnostics.get("stdout_truncated")
                ),
                "stderr_truncated": bool(
                    diagnostics.get("stderr_truncated")
                ),
                "stderr_tail": diagnostics.get("stderr_tail"),
                "last_activity_at": diagnostics.get("last_activity_at"),
                "termination_reason": diagnostics.get(
                    "termination_reason"
                ),
                "observation_errors": diagnostics.get(
                    "observation_errors", []
                ),
            }
            digest = document_sha256(document)
            artifact_id = (
                f"generations/{state.current.generation}/provider-failures/"
                f"{state.host_turn_round}-{digest}.json"
            )
            ref = self._artifacts(context, state.semantic_key).publish_json(
                artifact_id,
                document,
            )
            encoded_ref = encode_artifact_ref(ref)
        except Exception:
            return {**summary, "diagnostic_persistence_failed": True}
        return {
            **summary,
            "diagnostic_artifact_ref": encoded_ref,
        }

    def _fresh_after_crash(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        options: LLMExecutionOptions,
    ) -> LLMTaskState:
        """Discard an interrupted generation and start one fresh attempt."""
        adapter = self.registry.create(state.resolved_provider or "")
        execution_doc = self._execution_document(
            adapter, state.resolved_model or "", request, options
        )
        next_state = fresh_generation(
            state,
            execution=execution_fingerprint(execution_doc),
        )
        store.compare_and_swap(state.revision, next_state)
        scoped = self._artifacts(context, next_state.semantic_key)
        scoped.publish_json(
            f"execution/{next_state.current.generation}/recipe.json", execution_doc
        )
        self._publish_policy(scoped, next_state.current.generation, options)
        return next_state

    def _accept(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        value: Any,
        execution: ProviderExecution,
        options: LLMExecutionOptions,
    ) -> LLMCompleted:
        validate_value(value, request.output)
        scoped = self._artifacts(context, state.semantic_key)
        if isinstance(request.output, TextOutput):
            ref = scoped.publish_bytes(
                self._accepted_artifact_id(request),
                value.encode("utf-8"),
                media_type="text/plain; charset=utf-8",
            )
        else:
            ref = scoped.publish_json(self._accepted_artifact_id(request), value)
        current = store.read() or state
        accepted = AcceptedRecord(
            ref,
            AcceptedOrigin.PROVIDER,
            current.current.generation,
            current.resolved_provider,
            current.resolved_model,
        )
        # The validated immutable artifact plus the accepted-turn record is
        # the durable acceptance commit. Recording the lineage first prevents
        # an old-prefix sibling from reaching the provider if this process
        # stops before the task-local accepted field is repaired.
        session = self._advance_session(context, request, current, ref)
        next_state = replace(
            current,
            revision=current.revision + 1,
            accepted=accepted,
            pause=None,
            pending_host_turn=None,
        )
        store.compare_and_swap(current.revision, next_state)
        return LLMCompleted(
            value,
            next_state.resolved_provider,
            next_state.resolved_model,
            session,
            execution.usage,
            self._runtime_warnings(options) + self._provider_warnings(execution),
        )

    def _accept_candidate(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        input: ResumeInput,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        assert input.candidate_digest is not None
        raw_ref = state.current.raw_response
        if raw_ref is None:
            return LLMFailed(OutputInvalidError("Candidate selection has no raw response."))
        execution = self._execution_from_raw(
            context, raw_ref, state.resolved_provider or ""
        )
        try:
            value = select_output(
                execution.candidates,
                self._provider_output_contract(request, options),
                selected_digest=input.candidate_digest,
            )
        except AcLLMError as exc:
            return LLMFailed(exc)
        if self._uses_host_turn(request, options):
            try:
                turn = decode_host_turn(
                    value,
                    seen_host_request_ids=set(state.seen_host_request_ids),
                )
            except DuplicateHostRequestError as exc:
                return self._recover_duplicate_host_turn(
                    context, request, state, store, exc, options
                )
            except AcLLMError as exc:
                return LLMFailed(exc)
            if turn.state == "request_host":
                return self._handle_host_turn(
                    context, request, state, store, turn, execution, options
                )
            value = turn.result
        state = replace(state, revision=state.revision + 1, pause=None)
        store.compare_and_swap(state.revision - 1, state)
        return self._accept(context, request, state, store, value, execution, options)

    @staticmethod
    def _candidate_display_value(
        value: Any,
        request: LLMRequest,
        options: LLMExecutionOptions,
    ) -> Any:
        if not LLMTaskExecutor._uses_host_turn(request, options):
            return value
        turn = decode_host_turn(value)
        return turn.result if turn.state == "complete" else value

    @staticmethod
    def _runtime_warnings(
        options: LLMExecutionOptions,
    ) -> tuple[Mapping[str, Any], ...]:
        if not options.internet:
            return ()
        return (
            {
                "code": "internet_best_effort",
                "message": (
                    "Internet availability is provider/host best effort and cannot "
                    "be guaranteed by AC Foundation."
                ),
            },
        )

    @staticmethod
    def _provider_warnings(
        execution: ProviderExecution,
    ) -> tuple[Mapping[str, Any], ...]:
        raw_warnings = execution.diagnostics.get("warnings")
        if not isinstance(raw_warnings, (list, tuple)):
            return ()
        warnings: list[Mapping[str, Any]] = []
        for raw_warning in raw_warnings:
            if not isinstance(raw_warning, Mapping):
                continue
            code = raw_warning.get("code")
            message = raw_warning.get("message")
            if code == "memory_guard_unavailable":
                error_type = raw_warning.get("error_type")
                if not isinstance(message, str) or not isinstance(error_type, str):
                    continue
                warnings.append(
                    {
                        "code": code,
                        "message": message[:512],
                        "error_type": error_type[:128],
                    }
                )
                continue
            if code != "provider_nonzero_exit_with_valid_output":
                continue
            provider = raw_warning.get("provider")
            returncode = raw_warning.get("returncode")
            if (
                not isinstance(message, str)
                or not isinstance(provider, str)
                or type(returncode) is not int
            ):
                continue
            warnings.append(
                {
                    "code": "provider_nonzero_exit_with_valid_output",
                    "message": message[:512],
                    "provider": provider[:64],
                    "returncode": returncode,
                }
            )
        return tuple(warnings)

    def _replay(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        assert state.accepted is not None
        try:
            content = context.artifacts.read_bytes(state.accepted.artifact_ref)
            value = self._decode_artifact_value(content, request)
            validate_value(value, request.output)
            session = None
            if state.session_key is not None:
                session = self._advance_session(
                    context,
                    request,
                    state,
                    state.accepted.artifact_ref,
                )
            provider_warnings: tuple[Mapping[str, Any], ...] = ()
            if (
                state.accepted.origin is AcceptedOrigin.PROVIDER
                and state.current.raw_response is not None
            ):
                try:
                    execution = self._execution_from_raw(
                        context,
                        state.current.raw_response,
                        state.resolved_provider or "",
                    )
                except AcLLMError:
                    pass
                else:
                    provider_warnings = self._provider_warnings(execution)
            return LLMCompleted(
                value,
                state.accepted.provider,
                state.accepted.model,
                session,
                None,
                self._runtime_warnings(options) + provider_warnings,
            )
        except StoppedError:
            return LLMStopped()
        except Exception as exc:
            if isinstance(exc, AcLLMError):
                return LLMFailed(exc)
            return LLMFailed(OutputInvalidError(f"Accepted artifact is invalid: {exc}"))

    def _pause(
        self,
        store: Any,
        state: LLMTaskState,
        reason: ResumeReason,
        code: str,
        *,
        input_required: bool,
        request_ref: ArtifactRef | None = None,
        provider_failure: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> LLMPaused:
        response_contract = RESUME_SCHEMA_VERSION if input_required else None
        resume_key = _make_resume_key(state.semantic_key, state.revision + 1)
        pause_details: dict[str, Any] = {"code": code}
        if provider_failure is not None:
            pause_details["provider_failure"] = dict(provider_failure)
        if details is not None:
            pause_details.update(details)
        pause = TaskPause(
            reason,
            resume_key,
            input_required,
            request_ref,
            response_contract,
            pause_details,
        )
        next_state = replace(
            state,
            revision=state.revision + 1,
            pause=pause,
        )
        store.compare_and_swap(state.revision, next_state)
        return self._paused_outcome(pause)

    @staticmethod
    def _paused_outcome(pause: TaskPause) -> LLMPaused:
        return LLMPaused(
            pause.reason,
            pause.resume_key,
            pause.details,
            pause.request_ref,
            pause.input_required,
            pause.response_contract,
        )

    def _prepared_host_turn_prompt(
        self,
        context: Any,
        state: LLMTaskState,
    ) -> str | None:
        if state.host_turn_round == 0:
            return None
        if state.pending_host_turn is not None:
            raise CorruptTaskStateError(
                "Host-turn continuation has no persisted response artifact."
            )
        scoped = self._artifacts(context, state.semantic_key)
        try:
            ref = scoped.find(
                self._host_turn_artifact_id(
                    state.current.generation,
                    state.host_turn_round,
                    "continuation.json",
                )
            )
            if ref is None:
                raise ValueError("continuation artifact is missing")
            document = json.loads(scoped.read_bytes(ref).decode("utf-8"))
            continuation = decode_host_continuation(document)
            canonical_document = host_continuation_document(
                continuation.request_id,
                continuation.response,
            )
            if document != canonical_document:
                raise ValueError("continuation artifact is not canonical")
            return canonical_json_bytes(canonical_document).decode("utf-8")
        except AcLLMError as exc:
            raise CorruptTaskStateError(
                "Host-turn continuation artifact is corrupt."
            ) from exc
        except Exception as exc:
            raise CorruptTaskStateError(
                "Host-turn continuation artifact is corrupt."
            ) from exc

    def _execution_document(
        self,
        adapter: Any,
        model: str,
        request: LLMRequest,
        options: LLMExecutionOptions,
    ) -> dict[str, Any]:
        capabilities = adapter.capabilities()
        return execution_document(
            provider=adapter.name,
            model=model,
            capabilities={
                "runtime": self._capability_document(options),
                "structured_output": capabilities.structured_output.value,
                "config_isolation": capabilities.config_isolation.value,
                "tool_isolation": capabilities.tool_isolation.value,
                "input_transport": "workspace_control.v1",
                "instruction_policy_sha256": document_sha256(
                    PROVIDER_INSTRUCTION_POLICY
                ),
            },
            adapter_compatibility_version=adapter.compatibility_version,
            session_compatibility={},
        )

    @staticmethod
    def _capability_document(options: LLMExecutionOptions) -> dict[str, Any]:
        mode = effective_host_mode(options.host_authority)
        return {
            "internet": options.internet,
            "host_authority": options.host_authority.value,
            "effective_host_mode": mode.value,
            "ac_environment": options.runtime_environment.execution_document(),
            "host_broker": broker_execution_document(options.host_broker),
            "execution_profile": options.profile.value,
        }

    @classmethod
    def _provider_input_files(
        cls, request: LLMRequest
    ) -> tuple[ProviderInputFile, ...]:
        return tuple(
            ProviderInputFile(
                item.input_id,
                item.media_type,
                Path(
                    f"inputs/{index:04d}-{item.input_id}"
                    f"{cls._input_suffix(item.media_type)}"
                ),
            )
            for index, item in enumerate(request.inputs)
        )

    @staticmethod
    def _uses_host_turn(
        request: LLMRequest,
        options: LLMExecutionOptions,
    ) -> bool:
        return (
            isinstance(request.output, (TextOutput, JsonOutput))
            and options.profile is not LLMExecutionProfile.BOUNDED
            and effective_host_mode(options.host_authority).value == "brokered"
        )

    def _provider_output_contract(
        self,
        request: LLMRequest,
        options: LLMExecutionOptions,
    ) -> Any:
        if not self._uses_host_turn(request, options):
            return request.output
        result_schema = (
            {"type": "string"}
            if isinstance(request.output, TextOutput)
            else dict(request.output.schema)
        )
        return JsonOutput(host_turn_schema(result_schema), repair="strict")

    def _provider_output_schema(
        self,
        request: LLMRequest,
        options: LLMExecutionOptions,
    ) -> Mapping[str, Any] | None:
        return provider_schema(self._provider_output_contract(request, options))

    def _provider_instructions(
        self,
        request: LLMRequest,
        options: LLMExecutionOptions,
    ) -> str | None:
        if self._uses_host_turn(request, options):
            internet = (
                "Internet access is requested on a best-effort basis; ask the host "
                "when it is needed. "
                if options.internet
                else "Internet access is not requested for this task. "
            )
            return f"{PROVIDER_INSTRUCTION_POLICY} {internet}"
        return None

    def _resolve_model(self, request: LLMRequest) -> Any:
        available = self.registry.names()
        if request.model.provider == "auto":
            healthy: list[str] = []
            for name in available:
                try:
                    if self.registry.create(name).doctor().available:
                        healthy.append(name)
                except Exception:
                    continue
            if healthy:
                available = tuple(healthy)
        return resolve_model_selection(request.model, available=available)

    def _canonicalize_inputs(
        self,
        context: Any,
        request: LLMRequest,
        scoped: Any,
    ) -> LLMRequest:
        if not request.inputs:
            return request
        canonical = []
        key = semantic_key(request)
        for index, item in enumerate(request.inputs):
            target_artifact_id = (
                f"llm/tasks/{key.sha256}/inputs/source/"
                f"{index:04d}-{item.input_id}"
            )
            canonical_source = ArtifactSourceRef(
                context.run_id,
                target_artifact_id,
                item.source.expected_digest,
            )
            try:
                verified = context.artifacts.read_source(canonical_source)
            except Exception:
                try:
                    verified = context.artifacts.read_source(item.source)
                except Exception as exc:
                    raise InvalidRequestError(
                        f"Input artifact could not be verified: {item.input_id}",
                        details={
                            "code": "invalid_input_artifact",
                            "input_id": item.input_id,
                        },
                    ) from exc
            if verified.media_type != item.media_type:
                raise InvalidRequestError(
                    f"Input artifact media type differs from the request: {item.input_id}",
                    details={
                        "code": "input_media_type_mismatch",
                        "input_id": item.input_id,
                        "expected": item.media_type,
                        "actual": verified.media_type,
                    },
                )
            target = scoped.publish_bytes(
                f"inputs/source/{index:04d}-{item.input_id}",
                verified.content,
                media_type=item.media_type,
            )
            canonical.append(
                replace(
                    item,
                    source=ArtifactSourceRef(
                        context.run_id,
                        target.artifact_id,
                        target.digest,
                    ),
                )
            )
        durable = replace(request, inputs=tuple(canonical))
        if semantic_key(durable) != semantic_key(request):
            raise InvalidRequestError(
                "Canonical input materialization changed semantic identity."
            )
        return durable

    def _prepare_workspace(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        *,
        options: LLMExecutionOptions,
        continuation_prompt: str | None = None,
    ) -> Path:
        """Publish one self-contained, relative-path-only provider workspace."""

        workspace_root = context.run_directory / "llm-workspaces"
        if context.recovery_epoch:
            workspace_root = (
                workspace_root / f"recovery-{context.recovery_epoch:04d}"
            )
        workspace = (
            workspace_root
            / state.semantic_key.sha256
            / f"generation-{state.current.generation:04d}"
        )
        inputs_root = workspace / "inputs"
        work_root = workspace / "work"
        host_root = workspace / "host"
        for directory in (inputs_root, work_root, host_root):
            directory.mkdir(parents=True, exist_ok=True)

        input_documents: list[dict[str, Any]] = []
        for index, item in enumerate(request.inputs):
            try:
                verified = context.artifacts.read_source(item.source)
            except Exception as exc:
                raise InvalidRequestError(
                    f"Input artifact could not be verified: {item.input_id}",
                    details={
                        "code": "invalid_input_artifact",
                        "input_id": item.input_id,
                    },
                ) from exc
            if verified.media_type != item.media_type:
                raise InvalidRequestError(
                    f"Input artifact media type differs from the request: {item.input_id}",
                    details={
                        "code": "input_media_type_mismatch",
                        "input_id": item.input_id,
                        "expected": item.media_type,
                        "actual": verified.media_type,
                    },
                )
            digest = hashlib.sha256(verified.content).hexdigest()
            if (
                digest != item.source.expected_digest.value
                or len(verified.content) != item.source.expected_digest.size_bytes
            ):
                raise InvalidRequestError(
                    f"Input artifact digest differs from the request: {item.input_id}",
                    details={
                        "code": "input_digest_mismatch",
                        "input_id": item.input_id,
                    },
                )
            relative_path = (
                f"inputs/{index:04d}-{item.input_id}{self._input_suffix(item.media_type)}"
            )
            self._publish_workspace_file(workspace / relative_path, verified.content)
            input_documents.append(
                {
                    "input_id": item.input_id,
                    "media_type": item.media_type,
                    "sha256": digest,
                    "size_bytes": len(verified.content),
                    "path": relative_path,
                }
            )

        continuation_path = None
        if continuation_prompt is not None:
            continuation_path = "host/continuation.json"
            self._publish_workspace_file(
                workspace / continuation_path,
                continuation_prompt.encode("utf-8"),
            )
        control = {
            "schema_version": "ac.llm.workspace_control.v1",
            "task_id": request.task_id,
            "prompt": self._workspace_task_prompt(
                context,
                request,
                state,
            ),
            "output_contract": encode_output_contract(
                self._provider_output_contract(request, options)
            ),
            "runtime": self._capability_document(options),
            "inputs": input_documents,
            "work_directory": "work",
            "continuation_response": continuation_path,
            "provider_instructions": self._provider_instructions(request, options),
        }
        self._publish_workspace_file(
            host_root / "control.json",
            canonical_json_bytes(control),
        )
        return workspace

    def _workspace_task_prompt(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
    ) -> str:
        retry_record = self._output_retry_record(context, state)
        if (
            retry_record is None
            or state.current.generation < int(retry_record["to_generation"])
        ):
            return request.prompt
        return (
            f"{request.prompt}\n\n"
            "The previous response could not satisfy the requested output "
            "contract after format recovery. Produce one complete fresh answer "
            "to the original task, not a patch or commentary about the error. "
            f"Validation feedback: {retry_record['message']}"
        )

    @staticmethod
    def _workspace_prompt() -> str:
        return (
            "Read host/control.json from the current working directory. It contains "
            "the task, output contract, verified relative input paths, and any "
            "continuation response. Use work/ for scratch work when useful. Return "
            "only the requested final response."
        )

    @staticmethod
    def _input_suffix(media_type: str) -> str:
        base_media_type = media_type.split(";", 1)[0]
        return {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "text/markdown": ".md",
            "application/json": ".json",
            "application/pdf": ".pdf",
            "text/html": ".html",
            "text/plain": ".txt",
            "text/x-tex": ".tex",
            "application/x-tex": ".tex",
            "application/tex": ".tex",
            "application/x-latex": ".tex",
        }.get(base_media_type, ".bin")

    @staticmethod
    def _publish_workspace_file(path: Path, content: bytes) -> None:
        if path.exists() and path.read_bytes() == content:
            return
        atomic_write_bytes(path, content)

    @staticmethod
    def _task_namespace(task_id: str) -> str:
        digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:32]
        return f"llm-task-{digest}"

    def _task_store(self, context: Any, task_id: str) -> Any:
        return context.state(self._task_namespace(task_id), TaskStateContract())

    @staticmethod
    def _session_namespace(session_key: str) -> str:
        digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()[:32]
        return f"llm-session-{digest}"

    def _session_store(self, context: Any, session_key: str) -> Any:
        return context.state(self._session_namespace(session_key), SessionStateContract())

    @staticmethod
    def _session_lineage_pool(context: Any, session_key: str) -> BoundedLeasePool:
        namespace = hashlib.sha256(session_key.encode("utf-8")).hexdigest()
        return BoundedLeasePool(
            context.repository.root
            / "operational"
            / "llm"
            / "sessions"
            / namespace,
            1,
        )

    def _validate_session_lineage(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        options: LLMExecutionOptions,
        *,
        allow_execution_replace: bool = False,
    ) -> None:
        adapter = self.registry.create(state.resolved_provider or "")
        execution = execution_fingerprint(
            self._execution_document(
                adapter,
                state.resolved_model or "",
                request,
                options,
            )
        )
        if (
            state.current.execution != execution
            and not allow_execution_replace
        ):
            raise ExecutionMismatchError()
        if request.session is None:
            return
        session = self._session_store(
            context, request.session.session_key
        ).read()
        if session is None:
            raise InvalidRequestError("The requested LLM session does not exist.")
        if (
            session.provider != (state.resolved_provider or "")
            or session.model != (state.resolved_model or "")
        ):
            raise InvalidRequestError("The requested session is execution-incompatible.")
        if (
            session.session_compatibility != execution
            and not allow_execution_replace
        ):
            raise InvalidRequestError("The requested session is execution-incompatible.")
        if session.accepted_prefix_sha256 == request.session.accepted_prefix_sha256:
            return
        if any(
            turn.task_semantic_key_sha256 == state.semantic_key.sha256
            for turn in session.accepted_turn_records
        ):
            # The session record is the durable acceptance commit. A crash may
            # leave this task-local state awaiting recovery of the already
            # saved raw response and immutable accepted artifact.
            return
        raise InvalidRequestError("The session accepted prefix changed.")

    @staticmethod
    def _artifacts(context: Any, key: Any) -> Any:
        digest = key.sha256
        return context.artifacts.scoped(f"llm/tasks/{digest}")

    def _load_request(self, context: Any, state: LLMTaskState) -> LLMRequest:
        content = context.artifacts.read_bytes(state.request_ref)
        try:
            document = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidRequestError("The durable LLM request is corrupt.") from exc
        return decode_request(document)

    @staticmethod
    def _update_current(state: LLMTaskState, **changes: Any) -> LLMTaskState:
        generation = replace(state.current, **changes)
        return replace(
            state,
            revision=state.revision + 1,
            generation=generation,
        )

    def _existing_session_handle(
        self,
        context: Any,
        request: LLMRequest,
        provider: str,
        model: str,
        execution: Any,
    ) -> str | None:
        if request.session is None:
            return None
        session = self._session_store(context, request.session.session_key).read()
        if session is None:
            raise InvalidRequestError("The requested LLM session does not exist.")
        if session.accepted_prefix_sha256 != request.session.accepted_prefix_sha256:
            raise InvalidRequestError("The session accepted prefix changed.")
        if (
            session.provider != provider
            or session.model != model
            or session.session_compatibility != execution
        ):
            raise InvalidRequestError("The requested session is execution-incompatible.")
        return session.native_handle

    def _advance_session(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        accepted_ref: ArtifactRef,
    ) -> SessionRef | None:
        if state.session_key is None:
            return None
        store = self._session_store(context, state.session_key)
        prefix = (
            request.session.accepted_prefix_sha256
            if request.session is not None
            else hashlib.sha256(b"").hexdigest()
        )
        next_prefix = hashlib.sha256(
            f"{prefix}\0{accepted_ref.digest.value}".encode()
        ).hexdigest()
        while True:
            current = store.read()
            if current is None:
                initial = LLMSessionState(
                    0,
                    state.session_key,
                    state.current.generation,
                    state.resolved_provider or "",
                    state.resolved_model or "",
                    state.current.execution,
                    state.current.native_handle,
                    0,
                    prefix,
                    (),
                )
                try:
                    store.create(initial)
                    current = initial
                except StateConflictError:
                    continue
            for turn in current.accepted_turn_records:
                if turn.task_semantic_key_sha256 != state.semantic_key.sha256:
                    continue
                if turn.artifact_sha256 != accepted_ref.digest.value:
                    raise InvalidRequestError(
                        "The accepted task is bound to a different session artifact."
                    )
                return SessionRef(current.session_key, turn.result_prefix_sha256)
            if current.accepted_prefix_sha256 != prefix:
                raise InvalidRequestError("The session accepted prefix changed.")
            next_state = replace(
                current,
                revision=current.revision + 1,
                generation=state.current.generation,
                session_compatibility=state.current.execution,
                native_handle=state.current.native_handle,
                accepted_turns=current.accepted_turns + 1,
                accepted_prefix_sha256=next_prefix,
                accepted_turn_records=current.accepted_turn_records
                + (
                    AcceptedSessionTurn(
                        state.semantic_key.sha256,
                        accepted_ref.digest.value,
                        next_prefix,
                    ),
                ),
            )
            try:
                store.compare_and_swap(current.revision, next_state)
                return SessionRef(
                    next_state.session_key,
                    next_state.accepted_prefix_sha256,
                )
            except RevisionConflictError:
                continue

    @staticmethod
    def _publish_policy(scoped: Any, generation: int, options: LLMExecutionOptions) -> None:
        document = {
            "schema_version": "ac.llm.operational_policy.v2",
            "limits": {
                "idle_timeout_seconds": options.limits.idle_timeout_seconds,
            },
            "gate": {
                "enabled": options.gate.enabled,
                "global_limit": options.gate.global_limit,
                "provider_limits": dict(sorted(options.gate.provider_limits.items())),
                "circuit_failure_threshold": options.gate.circuit_failure_threshold,
                "circuit_cooldown_seconds": options.gate.circuit_cooldown_seconds,
                "minimum_available_memory_fraction": (
                    options.gate.minimum_available_memory_fraction
                ),
                "memory_poll_interval_seconds": (
                    options.gate.memory_poll_interval_seconds
                ),
                "memory_launch_interval_seconds": (
                    options.gate.memory_launch_interval_seconds
                ),
            },
        }
        digest = document_sha256(document)
        scoped.publish_json(f"execution/{generation}/policy/{digest}.json", document)

    @staticmethod
    def _execution_document_value(execution: ProviderExecution) -> dict[str, Any]:
        return {
            "schema_version": "ac.llm.provider_material.v1",
            "terminal_kind": execution.terminal_kind.value,
            "candidates": [
                LLMTaskExecutor._candidate_doc(item) for item in execution.candidates
            ],
            "native_handle": (
                None
                if execution.native_handle is None
                else {
                    "provider": execution.native_handle.provider,
                    "value": execution.native_handle.value,
                }
            ),
            "usage": (
                None
                if execution.usage is None
                else {
                    "input_tokens": execution.usage.input_tokens,
                    "output_tokens": execution.usage.output_tokens,
                    "cached_input_tokens": execution.usage.cached_input_tokens,
                }
            ),
            "diagnostics": dict(execution.diagnostics),
        }

    @staticmethod
    def _candidate_doc(candidate: CandidateMaterial) -> dict[str, Any]:
        return {
            "kind": "value" if candidate.has_value else "text",
            "value": candidate.value if candidate.has_value else None,
            "text": candidate.text,
            "terminal": candidate.terminal,
        }

    def _execution_from_raw(
        self, context: Any, ref: ArtifactRef, provider: str
    ) -> ProviderExecution:
        try:
            document = json.loads(
                context.artifacts.read_bytes(ref).decode("utf-8")
            )
            return self._decode_execution_document(document, provider)
        except CorruptTaskStateError:
            raise
        except Exception as exc:
            raise _provider_material_error(
                "Saved provider material is corrupt."
            ) from exc

    @staticmethod
    def _decode_execution_document(
        value: Any,
        provider: str,
    ) -> ProviderExecution:
        document = _closed_provider_material_object(
            value,
            {
                "schema_version",
                "terminal_kind",
                "candidates",
                "native_handle",
                "usage",
                "diagnostics",
            },
            "provider material",
        )
        if document["schema_version"] != "ac.llm.provider_material.v1":
            raise _provider_material_error(
                "Saved provider material has an unsupported schema."
            )
        if document["terminal_kind"] != ProviderTerminalKind.COMPLETED.value:
            raise _provider_material_error(
                "Saved provider material is not a completed execution."
            )

        raw_candidates = document["candidates"]
        if not isinstance(raw_candidates, list):
            raise _provider_material_error(
                "Saved provider candidates must be an array."
            )
        candidates: list[CandidateMaterial] = []
        for index, raw_candidate in enumerate(raw_candidates):
            candidate = _closed_provider_material_object(
                raw_candidate,
                {"kind", "value", "text", "terminal"},
                f"provider candidate {index}",
            )
            terminal = candidate["terminal"]
            if type(terminal) is not bool:
                raise _provider_material_error(
                    "Saved provider candidate terminal must be a boolean."
                )
            kind = candidate["kind"]
            if kind == "value":
                if candidate["text"] is not None:
                    raise _provider_material_error(
                        "Saved value candidate must have null text."
                    )
                candidates.append(
                    CandidateMaterial(
                        value=candidate["value"],
                        terminal=terminal,
                    )
                )
            elif kind == "text":
                text = candidate["text"]
                if candidate["value"] is not None or not isinstance(text, str):
                    raise _provider_material_error(
                        "Saved text candidate must have null value and string text."
                    )
                candidates.append(CandidateMaterial(text=text, terminal=terminal))
            else:
                raise _provider_material_error(
                    "Saved provider candidate has an unknown kind."
                )

        raw_handle = document["native_handle"]
        native_handle = None
        if raw_handle is not None:
            handle = _closed_provider_material_object(
                raw_handle,
                {"provider", "value"},
                "provider native handle",
            )
            handle_provider = handle["provider"]
            handle_value = handle["value"]
            if (
                not isinstance(handle_provider, str)
                or not handle_provider
                or handle_provider != provider
                or not isinstance(handle_value, str)
                or not handle_value
            ):
                raise _provider_material_error(
                    "Saved provider native handle is invalid."
                )
            native_handle = NativeResumeHandle(handle_provider, handle_value)

        raw_usage = document["usage"]
        usage = None
        if raw_usage is not None:
            usage_document = _closed_provider_material_object(
                raw_usage,
                {
                    "input_tokens",
                    "output_tokens",
                    "cached_input_tokens",
                },
                "provider usage",
            )
            usage = ProviderUsage(
                _provider_usage_value(
                    usage_document["input_tokens"], "input_tokens"
                ),
                _provider_usage_value(
                    usage_document["output_tokens"], "output_tokens"
                ),
                _provider_usage_value(
                    usage_document["cached_input_tokens"],
                    "cached_input_tokens",
                ),
            )

        diagnostics = document["diagnostics"]
        if not isinstance(diagnostics, Mapping):
            raise _provider_material_error(
                "Saved provider diagnostics must be an object."
            )
        return ProviderExecution(
            ProviderTerminalKind.COMPLETED,
            tuple(candidates),
            native_handle=native_handle,
            usage=usage,
            diagnostics=dict(diagnostics),
        )

    def _supervision_artifact(
        self, context: Any, state: LLMTaskState, code: str
    ) -> ArtifactRef:
        return self._artifacts(context, state.semantic_key).publish_json(
            f"generations/{state.current.generation}/supervision/{code}.json",
            {"schema_version": "ac.llm.supervision_request.v1", "code": code},
        )

    @staticmethod
    def _accepted_artifact_id(request: LLMRequest) -> str:
        return "accepted/result.txt" if isinstance(request.output, TextOutput) else "accepted/result.json"

    @staticmethod
    def _source_semantic_key(source: ArtifactSourceRef) -> SemanticKeyDigest:
        parts = source.source_artifact_id.split("/")
        if (
            len(parts) != 5
            or parts[:2] != ["llm", "tasks"]
            or parts[3] != "accepted"
            or parts[4] not in {"result.json", "result.txt"}
            or len(parts[2]) != 64
        ):
            raise AdoptionAuthorizationError()
        try:
            int(parts[2], 16)
        except ValueError as exc:
            raise AdoptionAuthorizationError() from exc
        return SemanticKeyDigest(parts[2])

    @staticmethod
    def _decode_artifact_value(content: bytes, request: LLMRequest) -> Any:
        if isinstance(request.output, TextOutput):
            return content.decode("utf-8")
        return json.loads(content.decode("utf-8"))


def _closed_provider_material_object(
    value: Any,
    fields: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _provider_material_error(
            f"Saved {label} does not use the current closed shape."
        )
    return value


def _provider_usage_value(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise _provider_material_error(
            f"Saved provider usage {field} must be a non-negative integer or null."
        )
    return value


def _provider_material_error(message: str) -> CorruptTaskStateError:
    return CorruptTaskStateError(
        message,
        details={"code": "provider_material_corrupt"},
    )
