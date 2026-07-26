"""The single durable LLM execution and recovery loop."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from arc_jobs import (
    ArtifactRef,
    ArtifactSourceRef,
    BoundedLeasePool,
    ResumeReason,
    RevisionConflictError,
    SemanticKeyDigest,
    StateConflictError,
    StoppedError,
    encode_artifact_ref,
)

from .config import resolve_model_selection
from .errors import (
    AdoptionAuthorizationError,
    AdoptionConflictError,
    ArcLLMError,
    CandidateConflictError,
    CorruptTaskStateError,
    FailureCategory,
    IdempotencyConflictError,
    InvalidRequestError,
    OutputInvalidError,
    ProviderFailure,
    ResumeKeyMismatchError,
)
from .identity import (
    AdoptionAuthorization,
    _make_resume_key,
    canonical_json_bytes,
    document_sha256,
    execution_document,
    execution_fingerprint,
    semantic_document,
    semantic_key,
)
from .gate import ProviderCallGate
from .host import (
    HostRequest,
    HostResponse,
    broker_execution_document,
    decode_host_continuation,
    decode_host_turn,
    effective_host_mode,
    encode_host_turn,
    host_continuation_document,
    host_turn_schema,
)
from .outcome import (
    LLMStopped,
    LLMCompleted,
    LLMFailed,
    LLMPaused,
    LLMTaskOutcome,
)
from .output import (
    CandidateMaterial,
    candidate_digest,
    enumerate_valid_candidates,
    provider_schema,
    select_output,
    validate_value,
)
from .progress import DurableProviderObserver
from .providers import (
    NativeResumeHandle,
    ProviderExecution,
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
    LLMTaskState,
    LLMSessionState,
    SessionStateContract,
    TaskPause,
    TaskStateContract,
    fresh_generation,
)
from .request import (
    RESUME_SCHEMA_VERSION,
    LLMExecutionOptions,
    LLMRequest,
    JsonOutput,
    ModelSelection,
    ResumeAction,
    ResumeInput,
    SessionRef,
    TextOutput,
    encode_output_contract,
    decode_request,
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

HANDLER_NAME = "arc.llm.task.v4"
class LLMTaskExecutor:
    def __init__(self, registry: ProviderRegistry | None = None) -> None:
        self.registry = registry or default_registry()

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
            if state.pause is not None:
                return self._paused_outcome(state.pause)
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
            if state.pause is not None:
                return self._paused_outcome(state.pause)
            try:
                durable_request = self._load_request(context, state)
                self._validate_session_lineage(
                    context, durable_request, state, options
                )
            except ArcLLMError as exc:
                return LLMFailed(exc)
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
                except ArcLLMError as exc:
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
        except ArcLLMError as exc:
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
            self._validate_session_lineage(context, request, state, options)
        except StoppedError:
            return LLMStopped()
        except ArcLLMError as exc:
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
            state = fresh_generation(
                state,
                execution=execution_fingerprint(execution_doc),
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
        except ArcLLMError as exc:
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
                adapter = self.registry.create(state.resolved_provider or "")
                diagnostic = adapter.doctor()
                if not diagnostic.available:
                    return self._pause(
                        store,
                        state,
                        ResumeReason.EXTERNAL_CONDITION,
                        "provider_unavailable",
                        input_required=False,
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
                    if self._is_crash_failure(failure):
                        if not crash_retry_available:
                            return self._pause(
                                store,
                                active_state,
                                ResumeReason.EXECUTION_INTERRUPTED,
                                "provider_crash_retry_exhausted",
                                input_required=False,
                            )
                        state = self._fresh_after_crash(
                            context, request, active_state, store, options
                        )
                        crash_retry_available = False
                        continue
                    return self._provider_failure(store, active_state, failure)
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
                if self._is_crash_failure(failure):
                    active_state = store.read() or state
                    if not crash_retry_available:
                        return self._pause(
                            store,
                            active_state,
                            ResumeReason.EXECUTION_INTERRUPTED,
                            "provider_crash_retry_exhausted",
                            input_required=False,
                        )
                    state = self._fresh_after_crash(
                        context, request, active_state, store, options
                    )
                    crash_retry_available = False
                    continue
                return self._provider_failure(store, store.read() or state, failure)
            except StoppedError:
                return LLMStopped()
            except ArcLLMError as exc:
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
        workspace = self._prepare_workspace(context, request, state, options=options)
        gate = self._provider_gate(context, options)
        with gate.acquire(adapter.name, checkpoint=context.checkpoint) as permit:
            state = self._mark_attempt_started(state, store)
            try:
                execution = adapter.start(
                    ProviderRequest(
                        self._workspace_prompt(),
                        state.resolved_model or "",
                        self._provider_output_schema(request, options),
                        self._capability_document(options),
                        options.limits.idle_timeout_seconds,
                        workspace,
                        options.runtime_environment.apply_to(),
                    ),
                    observer,
                    context.stop,
                )
            except ProviderFailure as exc:
                permit.record_failure(exc)
                self._emit_gate_record_warning(context, permit)
                raise
            self._record_gate_execution(context, permit, execution)
            return execution

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
        workspace = self._prepare_workspace(
            context,
            request,
            state,
            options=options,
            continuation_prompt=prompt,
        )
        gate = self._provider_gate(context, options)
        with gate.acquire(adapter.name, checkpoint=context.checkpoint) as permit:
            state = self._mark_attempt_started(state, store)
            try:
                execution = adapter.resume(
                    NativeResumeHandle(adapter.name, handle),
                    ProviderResumeRequest(
                        self._workspace_prompt(),
                        self._provider_output_schema(request, options),
                        self._capability_document(options),
                        options.limits.idle_timeout_seconds,
                        workspace,
                        options.runtime_environment.apply_to(),
                    ),
                    observer,
                    context.stop,
                )
            except ProviderFailure as exc:
                permit.record_failure(exc)
                self._emit_gate_record_warning(context, permit)
                raise
            self._record_gate_execution(context, permit, execution)
            return execution

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
            (
                f"generations/{state.current.generation}/raw-responses/"
                f"{state.host_turn_round}.json"
            ),
            raw_doc,
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
        second time despite already having a locally recoverable response.
        """
        if state.current.raw_response is not None:
            return state
        scoped = self._artifacts(context, state.semantic_key)
        ref = scoped.find(
            f"generations/{state.current.generation}/raw-responses/"
            f"{state.host_turn_round}.json"
        )
        if ref is None:
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
        except OutputInvalidError:
            if isinstance(request.output, JsonOutput) and request.output.repair == "format":
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
                elif formatted is not None:
                    return formatted
            return self._pause(
                store,
                store.read() or current,
                ResumeReason.SUPERVISION_REQUIRED,
                "output_invalid",
                input_required=True,
                request_ref=self._supervision_artifact(context, current, "output_invalid"),
            )
        if self._uses_host_turn(request, options):
            turn = decode_host_turn(
                value,
                seen_host_request_ids=set(current.seen_host_request_ids),
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
    ) -> FormattingDecision | LLMTaskOutcome | None:
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
            ),
        )
        # The formatter is its own durable task and therefore owns its single
        # automatic crash retry independently from the parent generation.
        formatter_options = options
        formatter_executor = LLMTaskExecutor(self.registry)
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
                return self._pause(
                    store,
                    store.read() or current,
                    ResumeReason.SUPERVISION_REQUIRED,
                    "output_formatting_failed",
                    input_required=True,
                    request_ref=record_ref,
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
                "schema_version": "arc.llm.output_formatting.v1",
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
        next_round = state.host_turn_round + 1
        turn_ref = scoped.publish_json(
            f"host-turns/{next_round}/request.json",
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

    def _resolve_pending_host_turn(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        assert state.pending_host_turn is not None
        if options.host_broker is None:
            return self._pause(
                store,
                state,
                ResumeReason.SUPERVISION_REQUIRED,
                "host_broker_required",
                input_required=True,
                request_ref=state.pending_host_turn,
        )
        raw = context.artifacts.read_bytes(state.pending_host_turn)
        turn = decode_host_turn(json.loads(raw.decode("utf-8")))
        assert turn.request is not None
        workspace = self._prepare_workspace(context, request, state, options=options)
        response = options.host_broker.execute(turn.request, workspace=workspace)
        if not isinstance(response, HostResponse):
            raise InvalidRequestError("host broker must return a HostResponse.")
        self._validate_host_files(workspace, response)
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
            f"host-turns/{state.host_turn_round}/continuation.json",
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
            )
        if failure.category is FailureCategory.STOPPED:
            return LLMStopped()
        return LLMFailed(failure)

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
            self._runtime_warnings(options),
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
        except ArcLLMError as exc:
            return LLMFailed(exc)
        if self._uses_host_turn(request, options):
            try:
                turn = decode_host_turn(
                    value,
                    seen_host_request_ids=set(state.seen_host_request_ids),
                )
            except ArcLLMError as exc:
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
                    "be guaranteed by ARC."
                ),
            },
        )

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
            return LLMCompleted(
                value,
                state.accepted.provider,
                state.accepted.model,
                session,
                None,
                self._runtime_warnings(options),
            )
        except StoppedError:
            return LLMStopped()
        except Exception as exc:
            if isinstance(exc, ArcLLMError):
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
    ) -> LLMPaused:
        response_contract = RESUME_SCHEMA_VERSION if input_required else None
        resume_key = _make_resume_key(state.semantic_key, state.revision + 1)
        pause = TaskPause(
            reason,
            resume_key,
            input_required,
            request_ref,
            response_contract,
            {"code": code},
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

    def _initial_provider_prompt(
        self, request: LLMRequest, options: LLMExecutionOptions
    ) -> str:
        if self._uses_host_turn(request, options):
            internet = (
                "Internet access is requested on a best-effort basis; ask the host "
                "when it is needed. "
                if options.internet
                else "Internet access is not requested for this task. "
            )
            return (
                f"{request.prompt}\n\nUse the arc.llm.host_turn.v1 envelope. "
                f"{internet}Request the host only when needed. After a refused request, do not "
                "repeat the same request until its retry_condition has changed."
            )
        return request.prompt

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
                f"host-turns/{state.host_turn_round}/continuation.json"
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
        except ArcLLMError as exc:
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
            "arc_environment": options.runtime_environment.execution_document(),
            "host_broker": broker_execution_document(options.host_broker),
        }

    @staticmethod
    def _uses_host_turn(
        request: LLMRequest,
        options: LLMExecutionOptions,
    ) -> bool:
        return (
            isinstance(request.output, (TextOutput, JsonOutput))
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
            return (
                "Use arc.llm.host_turn.v1. Request the host only when needed. "
                f"{internet}"
                "After a refused host request, do not repeat the same request until "
                "its retry_condition has changed."
            )
        return None

    def _resolve_model(self, request: LLMRequest) -> Any:
        return resolve_model_selection(request.model, available=self.registry.names())

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

        workspace = (
            context.run_directory
            / "llm-workspaces"
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
            "schema_version": "arc.llm.workspace_control.v1",
            "task_id": request.task_id,
            "prompt": request.prompt,
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
        return {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "text/markdown": ".md",
            "application/json": ".json",
        }.get(media_type, ".bin")

    @staticmethod
    def _publish_workspace_file(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() == content:
            return
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary_path = Path(temporary)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            if os.name != "nt":
                directory_descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        finally:
            temporary_path.unlink(missing_ok=True)

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
    ) -> None:
        if request.session is None:
            return
        adapter = self.registry.create(state.resolved_provider or "")
        execution = execution_fingerprint(
            self._execution_document(
                adapter,
                state.resolved_model or "",
                request,
                options,
            )
        )
        session = self._session_store(
            context, request.session.session_key
        ).read()
        if session is None:
            raise InvalidRequestError("The requested LLM session does not exist.")
        if (
            session.provider != (state.resolved_provider or "")
            or session.model != (state.resolved_model or "")
            or session.session_compatibility != execution
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
            "schema_version": "arc.llm.operational_policy.v1",
            "limits": {
                "idle_timeout_seconds": options.limits.idle_timeout_seconds,
            },
            "gate": {
                "enabled": options.gate.enabled,
                "global_limit": options.gate.global_limit,
                "provider_limits": dict(sorted(options.gate.provider_limits.items())),
                "circuit_failure_threshold": options.gate.circuit_failure_threshold,
                "circuit_cooldown_seconds": options.gate.circuit_cooldown_seconds,
            },
        }
        digest = document_sha256(document)
        scoped.publish_json(f"execution/{generation}/policy/{digest}.json", document)

    @staticmethod
    def _execution_document_value(execution: ProviderExecution) -> dict[str, Any]:
        return {
            "schema_version": "arc.llm.provider_material.v1",
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
        if document["schema_version"] != "arc.llm.provider_material.v1":
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
            {"schema_version": "arc.llm.supervision_request.v1", "code": code},
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
