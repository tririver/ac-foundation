"""The single durable LLM execution and recovery loop."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Mapping

from arc_jobs import (
    ArtifactRef,
    ArtifactSourceRef,
    EffectRequestDigest,
    EffectStage,
    ResumeReason,
    SemanticKeyDigest,
)

from .config import resolve_model_selection
from .errors import (
    AdoptionAuthorizationError,
    AdoptionConflictError,
    ArcLLMError,
    CandidateConflictError,
    DeliveryState,
    FailureCategory,
    IdempotencyConflictError,
    InvalidRequestError,
    OutputInvalidError,
    ProviderFailure,
    ResumeKeyMismatchError,
)
from .identity import (
    AdoptionAuthorization,
    canonical_json_bytes,
    document_sha256,
    execution_document,
    execution_fingerprint,
    semantic_document,
    semantic_key,
)
from .interaction import (
    decode_interactive_turn,
    encode_interactive_turn,
    response_document,
    validate_responses,
)
from .outcome import (
    LLMCancelled,
    LLMCompleted,
    LLMFailed,
    LLMPaused,
    LLMTaskOutcome,
)
from .output import (
    CandidateMaterial,
    candidate_digest,
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
    default_registry,
)
from .recovery import (
    AcceptedOrigin,
    AcceptedRecord,
    GenerationRecord,
    LLMTaskState,
    LLMSessionState,
    RecoveryAction,
    SessionStateContract,
    TaskPause,
    TaskStateContract,
    decide_recovery,
    effect_id_for,
    replace_current,
)
from .request import (
    RESUME_SCHEMA_VERSION,
    InteractiveJsonOutput,
    LLMExecutionOptions,
    LLMRequest,
    ResumeAction,
    ResumeInput,
    SessionRef,
    TextOutput,
    decode_request,
    request_to_document,
)

HANDLER_NAME = "arc.llm.task.v1"


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
                return self._replay(context, request, state)
            if state.pause is not None:
                return self._paused_outcome(state.pause)
            return self._drive(context, request, state, store, options)
        try:
            resolved = resolve_model_selection(
                request.model,
                available=self.registry.names(),
            )
            adapter = self.registry.create(resolved.provider)
            execution_doc = self._execution_document(adapter, resolved.model, request)
            execution = execution_fingerprint(execution_doc)
            scoped = self._artifacts(context, key)
            request_ref = scoped.publish_json(
                "requests/semantic.json",
                request_to_document(request),
            )
            scoped.publish_json("execution/1/recipe.json", execution_doc)
            self._publish_policy(scoped, 1, options)
            initial_handle = self._existing_session_handle(
                context, request, resolved.provider, resolved.model, execution
            )
            generation = GenerationRecord(
                1,
                effect_id_for(request.task_id, 1),
                execution,
                native_handle=initial_handle,
            )
            state = LLMTaskState(
                revision=0,
                task_id=request.task_id,
                semantic_key=SemanticKeyDigest(key.sha256),
                resolved_provider=resolved.provider,
                resolved_model=resolved.model,
                current_generation=1,
                generations=(generation,),
                request_ref=request_ref,
                session_key=(
                    request.session.session_key
                    if request.session is not None
                    else request.task_id
                ),
            )
            store.create(state)
            self._prepare_effect(context, request, state)
            return self._drive(context, request, state, store, options)
        except ArcLLMError as exc:
            return LLMFailed(exc)
        except Exception as exc:
            return LLMFailed(
                ProviderFailure(
                    f"Local LLM setup failed: {exc}",
                    category=FailureCategory.LOCAL_IO,
                    delivery=DeliveryState.NOT_DELIVERED,
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
        if state.pause.input_required and input is None:
            return LLMFailed(InvalidRequestError("This pause requires resume input."))
        if input is not None and input.resume_key != state.pause.resume_key:
            return LLMFailed(ResumeKeyMismatchError())
        try:
            request = self._load_request(context, state)
        except ArcLLMError as exc:
            return LLMFailed(exc)
        if input is not None and input.action is ResumeAction.CANCEL:
            return LLMCancelled()
        if input is not None and input.action is ResumeAction.ACCEPT_CANDIDATE:
            return self._accept_candidate(context, request, state, store, input)
        if input is not None and input.action is ResumeAction.REPLACE:
            adapter = self.registry.create(state.resolved_provider or "")
            execution_doc = self._execution_document(adapter, state.resolved_model or "", request)
            state = replace_current(
                state,
                execution=execution_fingerprint(execution_doc),
                reason=input.reason or "user_replacement",
                possible_duplicate=True,
            )
            state = replace(state, pause=None)
            store.compare_and_swap(state.revision - 1, state)
            self._artifacts(context, state.semantic_key).publish_json(
                f"execution/{state.current_generation}/recipe.json",
                execution_doc,
            )
            self._publish_policy(
                self._artifacts(context, state.semantic_key),
                state.current_generation,
                options,
            )
            self._prepare_effect(context, request, state)
            return self._drive(context, request, state, store, options)
        if state.pending_interaction is not None:
            if input is None or input.action is not ResumeAction.CONTINUE:
                return LLMFailed(
                    InvalidRequestError("Pending interaction requires continue responses.")
                )
            return self._resume_interaction(context, request, state, store, input, options)
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
    ) -> LLMTaskOutcome:
        key = semantic_key(request)
        store = self._task_store(context, request.task_id)
        existing = store.read()
        if existing is not None:
            if existing.semantic_key.sha256 != key.sha256:
                return LLMFailed(IdempotencyConflictError())
            if existing.accepted is not None:
                return self._replay(context, request, existing)
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
                current_generation=0,
                generations=(),
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
            return LLMCompleted(value, None, None, None, None)
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
    ) -> LLMTaskOutcome:
        while True:
            try:
                current = store.read()
                if current is None:
                    raise InvalidRequestError("LLM task state disappeared.")
                state = current
                if state.accepted is not None:
                    return self._replay(context, request, state)
                adapter = self.registry.create(state.resolved_provider or "")
                execution_doc = self._execution_document(
                    adapter, state.resolved_model or "", request
                )
                execution = execution_fingerprint(execution_doc)
                effect = context.effects.read(state.current.effect_id)
                if effect is None:
                    self._prepare_effect(context, request, state)
                    effect = context.effects.read(state.current.effect_id)
                assert effect is not None
                action = decide_recovery(
                    state,
                    effect.stage,
                    execution=execution,
                    supports_native_resume=adapter.capabilities().native_resume,
                    safe_retry_limit=options.limits.safe_retry_limit,
                    native_resume_limit=options.limits.native_resume_limit,
                    automatic_replacement_limit=options.limits.automatic_replacement_limit,
                )
                if action is RecoveryAction.REPLAY_ACCEPTED:
                    return self._replay(context, request, state)
                if action is RecoveryAction.RECOVER_SAVED_OUTPUT:
                    return self._recover_saved(context, request, state, store, effect.output_ref, options)
                if action is RecoveryAction.PAUSE_UNCERTAIN:
                    return self._pause(
                        store,
                        state,
                        ResumeReason.SUPERVISION_REQUIRED,
                        "recovery_limit_reached",
                        input_required=True,
                        request_ref=self._supervision_artifact(
                            context, state, "recovery_limit_reached"
                        ),
                    )
                if action is RecoveryAction.REPLACE:
                    state = replace_current(
                        state,
                        execution=execution,
                        reason="uncertain_delivery",
                        possible_duplicate=True,
                    )
                    store.compare_and_swap(state.revision - 1, state)
                    scoped = self._artifacts(context, state.semantic_key)
                    scoped.publish_json(
                        f"execution/{state.current_generation}/recipe.json",
                        execution_doc,
                    )
                    self._publish_policy(scoped, state.current_generation, options)
                    self._prepare_effect(context, request, state)
                    continue
                diagnostic = adapter.doctor()
                if not diagnostic.available:
                    return self._pause(
                        store,
                        state,
                        ResumeReason.EXTERNAL_CONDITION,
                        "provider_unavailable",
                        input_required=False,
                    )
                planned_continuation = (
                    effect.stage is EffectStage.PREPARED
                    and state.current.native_handle is not None
                    and request.session is not None
                )
                if action is RecoveryAction.NATIVE_RESUME or planned_continuation:
                    execution_result = self._call_resume(
                        context,
                        request,
                        state,
                        store,
                        adapter,
                        options,
                        count_recovery=action is RecoveryAction.NATIVE_RESUME,
                    )
                else:
                    execution_result = self._call_start(
                        context, request, state, store, adapter, options
                    )
                if execution_result.terminal_kind is ProviderTerminalKind.FAILED:
                    assert execution_result.failure is not None
                    outcome = self._provider_failure(
                        context,
                        request,
                        store.read() or state,
                        store,
                        execution_result.failure,
                        options,
                    )
                    if outcome is None:
                        continue
                    return outcome
                if execution_result.terminal_kind is ProviderTerminalKind.CANCELLED:
                    return LLMCancelled()
                outcome = self._consume_execution(
                    context,
                    request,
                    store.read() or state,
                    store,
                    execution_result,
                    options,
                )
                if outcome is None:
                    continue
                return outcome
            except ProviderFailure as exc:
                outcome = self._provider_failure(
                    context, request, store.read() or state, store, exc, options
                )
                if outcome is None:
                    continue
                return outcome
            except ArcLLMError as exc:
                return LLMFailed(exc)
            except Exception as exc:
                return LLMFailed(
                    ProviderFailure(
                        f"LLM execution failed locally: {exc}",
                        category=FailureCategory.LOCAL_IO,
                        delivery=DeliveryState.NOT_DELIVERED,
                    )
                )

    def _call_start(
        self, context: Any, request: LLMRequest, state: LLMTaskState, store: Any, adapter: Any, options: LLMExecutionOptions
    ) -> ProviderExecution:
        observer = self._observer(context, state, store)
        return adapter.start(
            ProviderRequest(
                request.prompt,
                state.resolved_model or "",
                provider_schema(request.output),
                self._capability_document(request),
                options.limits.idle_timeout_seconds,
            ),
            observer,
            context.cancel,
        )

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
        count_recovery: bool = True,
    ) -> ProviderExecution:
        handle = state.current.native_handle
        if handle is None:
            raise InvalidRequestError("Native continuation has no saved handle.")
        observer = self._observer(context, state, store)
        if count_recovery:
            next_state = self._update_current(
                state, native_resumes=state.current.native_resumes + 1
            )
            store.compare_and_swap(state.revision, next_state)
        return adapter.resume(
            NativeResumeHandle(adapter.name, handle),
            ProviderResumeRequest(
                request.prompt if prompt is None else prompt,
                provider_schema(request.output),
                self._capability_document(request),
                options.limits.idle_timeout_seconds,
            ),
            observer,
            context.cancel,
        )

    def _observer(self, context: Any, state: LLMTaskState, store: Any) -> DurableProviderObserver:
        def save_handle(handle: NativeResumeHandle) -> None:
            current = store.read()
            if current is None:
                raise InvalidRequestError("Task state disappeared while saving a handle.")
            next_state = self._update_current(current, native_handle=handle.value)
            store.compare_and_swap(current.revision, next_state)

        return DurableProviderObserver(
            context=context,
            effect_id=state.current.effect_id,
            on_handle=save_handle,
        )

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
                f"generations/{state.current_generation}/raw-responses/"
                f"{state.interaction_round}.json"
            ),
            raw_doc,
        )
        context.effects.save_output(state.current.effect_id, raw_ref)
        current = store.read() or state
        changes: dict[str, Any] = {"raw_response": raw_ref}
        if execution.native_handle is not None:
            changes["native_handle"] = execution.native_handle.value
        current = self._update_current(current, **changes)
        store.compare_and_swap(current.revision - 1, current)
        try:
            value = select_output(execution.candidates, request.output)
        except CandidateConflictError as exc:
            candidate_ref = scoped.publish_json(
                f"generations/{current.current_generation}/candidates.json",
                {
                    "candidate_digests": list(exc.candidate_digests),
                    "candidates": [
                        self._candidate_doc(item) for item in execution.candidates
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
            replacements = sum(
                item.replacement_of is not None for item in current.generations
            )
            if replacements < options.limits.automatic_replacement_limit:
                adapter = self.registry.create(current.resolved_provider or "")
                execution_doc = self._execution_document(
                    adapter, current.resolved_model or "", request
                )
                next_state = replace_current(
                    store.read() or current,
                    execution=execution_fingerprint(execution_doc),
                    reason="output_invalid",
                    possible_duplicate=False,
                )
                store.compare_and_swap(next_state.revision - 1, next_state)
                self._prepare_effect(context, request, next_state)
                return None
            return self._pause(
                store,
                store.read() or current,
                ResumeReason.SUPERVISION_REQUIRED,
                "output_invalid",
                input_required=True,
                request_ref=self._supervision_artifact(context, current, "output_invalid"),
            )
        if isinstance(request.output, InteractiveJsonOutput):
            turn = decode_interactive_turn(
                value,
                request.output,
                seen_request_ids=set(current.seen_request_ids),
            )
            if turn.state == "interact":
                return self._handle_interaction(
                    context, request, current, store, turn, execution, options
                )
            value = turn.result
        return self._accept(
            context, request, store.read() or current, store, value, execution
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
        outcome = self._consume_saved_candidates(
            context, request, state, store, execution, options
        )
        return outcome

    def _consume_saved_candidates(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        execution: ProviderExecution,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        try:
            value = select_output(execution.candidates, request.output)
        except CandidateConflictError as exc:
            ref = self._artifacts(context, state.semantic_key).publish_json(
                f"generations/{state.current_generation}/candidates.json",
                {
                    "candidate_digests": list(exc.candidate_digests),
                    "candidates": [self._candidate_doc(item) for item in execution.candidates],
                },
            )
            return self._pause(
                store,
                state,
                ResumeReason.SUPERVISION_REQUIRED,
                "candidate_selection_required",
                input_required=True,
                request_ref=ref,
            )
        except OutputInvalidError:
            return self._pause(
                store,
                state,
                ResumeReason.SUPERVISION_REQUIRED,
                "output_invalid",
                input_required=True,
                request_ref=self._supervision_artifact(context, state, "output_invalid"),
            )
        if isinstance(request.output, InteractiveJsonOutput):
            turn = decode_interactive_turn(
                value, request.output, seen_request_ids=set(state.seen_request_ids)
            )
            if turn.state == "interact":
                return self._handle_interaction(
                    context, request, state, store, turn, execution, options
                )
            value = turn.result
        return self._accept(context, request, state, store, value, execution)

    def _handle_interaction(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        turn: Any,
        execution: ProviderExecution,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        assert isinstance(request.output, InteractiveJsonOutput)
        scoped = self._artifacts(context, state.semantic_key)
        next_round = state.interaction_round + 1
        turn_ref = scoped.publish_json(
            f"interactions/{next_round}/request.json",
            encode_interactive_turn(turn),
        )
        current = store.read() or state
        context.effects.commit(current.current.effect_id)
        current_generation = replace(
            current.current,
            effect_id=(
                f"{effect_id_for(current.task_id, current.current_generation)}"
                f"-i{next_round}"
            ),
        )
        next_state = replace(
            current,
            revision=current.revision + 1,
            generations=current.generations[:-1] + (current_generation,),
            interaction_round=next_round,
            pending_interaction=turn_ref,
            seen_request_ids=current.seen_request_ids
            + tuple(item.request_id for item in turn.requests),
        )
        store.compare_and_swap(next_state.revision - 1, next_state)
        if next_round >= request.output.max_interaction_turns:
            return self._pause(
                store,
                next_state,
                ResumeReason.EXECUTION_BUDGET_EXHAUSTED,
                "interaction_limit_reached",
                input_required=False,
            )
        resolver = options.interaction_resolver
        if resolver is None:
            return self._pause(
                store,
                next_state,
                ResumeReason.INTERACTION_REQUIRED,
                "operation_requests_pending",
                input_required=True,
                request_ref=turn_ref,
            )
        responses = tuple(resolver.resolve(item) for item in turn.requests)
        resume_input = ResumeInput(
            resume_key=f"internal-{next_state.revision}",
            action=ResumeAction.CONTINUE,
            responses=responses,
        )
        return self._continue_interaction(
            context, request, next_state, store, resume_input, options
        )

    def _resume_interaction(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        input: ResumeInput,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        state = replace(state, revision=state.revision + 1, pause=None)
        store.compare_and_swap(state.revision - 1, state)
        return self._continue_interaction(context, request, state, store, input, options)

    def _continue_interaction(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        input: ResumeInput,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome:
        assert isinstance(request.output, InteractiveJsonOutput)
        assert state.pending_interaction is not None
        raw = context.artifacts.read_bytes(state.pending_interaction)
        turn = decode_interactive_turn(
            json.loads(raw.decode("utf-8")),
            request.output,
            seen_request_ids=set(state.seen_request_ids)
            - set(item.request_id for item in input.responses),
        )
        responses = validate_responses(turn, input.responses, request.output)
        scoped = self._artifacts(context, state.semantic_key)
        scoped.publish_json(
            f"interactions/{state.interaction_round}/response.json",
            response_document(responses),
        )
        next_state = replace(
            state,
            revision=state.revision + 1,
            pending_interaction=None,
            pause=None,
        )
        store.compare_and_swap(state.revision, next_state)
        adapter = self.registry.create(next_state.resolved_provider or "")
        prompt = canonical_json_bytes(response_document(responses)).decode("utf-8")
        self._prepare_effect(context, request, next_state, prompt=prompt)
        try:
            execution = self._call_resume(
                context,
                request,
                next_state,
                store,
                adapter,
                options,
                prompt=prompt,
                count_recovery=False,
            )
        except ProviderFailure as exc:
            outcome = self._provider_failure(
                context, request, store.read() or next_state, store, exc, options
            )
            return outcome or self._drive(
                context, request, store.read() or next_state, store, options
            )
        if execution.terminal_kind is ProviderTerminalKind.FAILED:
            assert execution.failure is not None
            outcome = self._provider_failure(
                context,
                request,
                store.read() or next_state,
                store,
                execution.failure,
                options,
            )
            return outcome or self._drive(
                context, request, store.read() or next_state, store, options
            )
        outcome = self._consume_execution(
            context,
            request,
            store.read() or next_state,
            store,
            execution,
            options,
        )
        return outcome or self._drive(
            context, request, store.read() or next_state, store, options
        )

    def _provider_failure(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        failure: ProviderFailure,
        options: LLMExecutionOptions,
    ) -> LLMTaskOutcome | None:
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
        if failure.category is FailureCategory.CANCELLED:
            return LLMCancelled()
        if failure.category in {
            FailureCategory.INVALID_REQUEST,
            FailureCategory.SCHEMA,
            FailureCategory.LOCAL_IO,
            FailureCategory.INTERNAL,
        }:
            return LLMFailed(failure)
        if failure.delivery is DeliveryState.NOT_DELIVERED:
            if state.current.safe_retries < options.limits.safe_retry_limit:
                next_state = self._update_current(
                    state, safe_retries=state.current.safe_retries + 1
                )
                store.compare_and_swap(state.revision, next_state)
                return None
            return LLMFailed(failure)
        if failure.delivery is DeliveryState.MAY_HAVE_RUN:
            return None
        return LLMFailed(failure)

    def _accept(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        value: Any,
        execution: ProviderExecution,
    ) -> LLMCompleted:
        validate_value(value, request.output if not isinstance(request.output, InteractiveJsonOutput) else _result_contract(request.output))
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
            current.current_generation,
            current.resolved_provider,
            current.resolved_model,
        )
        next_state = replace(
            current,
            revision=current.revision + 1,
            accepted=accepted,
            pause=None,
            pending_interaction=None,
        )
        store.compare_and_swap(current.revision, next_state)
        context.effects.commit(current.current.effect_id)
        session = self._advance_session(context, request, next_state, ref)
        return LLMCompleted(
            value,
            next_state.resolved_provider,
            next_state.resolved_model,
            session,
            execution.usage,
        )

    def _accept_candidate(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        store: Any,
        input: ResumeInput,
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
                request.output,
                selected_digest=input.candidate_digest,
            )
        except ArcLLMError as exc:
            return LLMFailed(exc)
        state = replace(state, revision=state.revision + 1, pause=None)
        store.compare_and_swap(state.revision - 1, state)
        return self._accept(context, request, state, store, value, execution)

    def _replay(
        self, context: Any, request: LLMRequest, state: LLMTaskState
    ) -> LLMTaskOutcome:
        assert state.accepted is not None
        try:
            content = context.artifacts.read_bytes(state.accepted.artifact_ref)
            value = self._decode_artifact_value(content, request)
            validate_value(
                value,
                request.output
                if not isinstance(request.output, InteractiveJsonOutput)
                else _result_contract(request.output),
            )
            session = None
            if state.session_key is not None:
                session_state = self._session_store(context, state.session_key).read()
                if session_state is not None:
                    session = SessionRef(
                        session_state.session_key,
                        session_state.accepted_prefix_sha256,
                    )
            return LLMCompleted(
                value,
                state.accepted.provider,
                state.accepted.model,
                session,
                None,
            )
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
        resume_key = f"resume-{state.revision + 1}"
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

    def _prepare_effect(
        self,
        context: Any,
        request: LLMRequest,
        state: LLMTaskState,
        *,
        prompt: str | None = None,
    ) -> None:
        exact_request = {
            "provider": state.resolved_provider,
            "model": state.resolved_model,
            "prompt": request.prompt if prompt is None else prompt,
            "output_schema": provider_schema(request.output),
            "capabilities": self._capability_document(request),
            "generation": state.current_generation,
        }
        context.effects.prepare(
            state.current.effect_id,
            effect_request_digest=EffectRequestDigest(document_sha256(exact_request)),
            details={"task_id": request.task_id, "generation": state.current_generation},
        )

    def _execution_document(self, adapter: Any, model: str, request: LLMRequest) -> dict[str, Any]:
        capabilities = adapter.capabilities()
        return execution_document(
            provider=adapter.name,
            model=model,
            capabilities={
                "requested": self._capability_document(request),
                "structured_output": capabilities.structured_output.value,
                "config_isolation": capabilities.config_isolation.value,
                "tool_isolation": capabilities.tool_isolation.value,
            },
            adapter_compatibility_version=adapter.compatibility_version,
            session_compatibility={},
        )

    @staticmethod
    def _capability_document(request: LLMRequest) -> dict[str, Any]:
        return {
            "internet": request.capabilities.internet,
            "inherit_host_config": request.capabilities.inherit_host_config,
            "allowed_tools": list(request.capabilities.allowed_tools),
        }

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
            generations=state.generations[:-1] + (generation,),
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
        current = store.read()
        prefix = (
            request.session.accepted_prefix_sha256
            if request.session is not None
            else hashlib.sha256(b"").hexdigest()
        )
        next_prefix = hashlib.sha256(
            f"{prefix}\0{accepted_ref.digest.value}".encode()
        ).hexdigest()
        if current is None:
            initial = LLMSessionState(
                0,
                state.session_key,
                state.current_generation,
                state.resolved_provider or "",
                state.resolved_model or "",
                state.current.execution,
                state.current.native_handle,
                0,
                prefix,
            )
            store.create(initial)
            current = initial
        next_state = replace(
            current,
            revision=current.revision + 1,
            generation=state.current_generation,
            native_handle=state.current.native_handle,
            accepted_turns=current.accepted_turns + 1,
            accepted_prefix_sha256=next_prefix,
        )
        store.compare_and_swap(current.revision, next_state)
        return SessionRef(next_state.session_key, next_state.accepted_prefix_sha256)

    @staticmethod
    def _publish_policy(scoped: Any, generation: int, options: LLMExecutionOptions) -> None:
        document = {
            "schema_version": "arc.llm.operational_policy.v1",
            "limits": {
                "idle_timeout_seconds": options.limits.idle_timeout_seconds,
                "safe_retry_limit": options.limits.safe_retry_limit,
                "native_resume_limit": options.limits.native_resume_limit,
                "automatic_replacement_limit": options.limits.automatic_replacement_limit,
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
            "digest": (
                candidate_digest(candidate.value)
                if candidate.has_value
                else hashlib.sha256((candidate.text or "").encode()).hexdigest()
            ),
        }

    def _execution_from_raw(
        self, context: Any, ref: ArtifactRef, provider: str
    ) -> ProviderExecution:
        document = json.loads(context.artifacts.read_bytes(ref).decode("utf-8"))
        candidates = tuple(
            (
                CandidateMaterial(value=item["value"], terminal=item["terminal"])
                if item["kind"] == "value"
                else CandidateMaterial(text=item["text"], terminal=item["terminal"])
            )
            for item in document["candidates"]
        )
        from .providers import ProviderUsage

        usage_doc = document["usage"]
        usage = (
            None
            if usage_doc is None
            else ProviderUsage(
                usage_doc["input_tokens"],
                usage_doc["output_tokens"],
                usage_doc["cached_input_tokens"],
            )
        )
        return ProviderExecution(
            ProviderTerminalKind.COMPLETED,
            candidates,
            usage=usage,
        )

    def _supervision_artifact(
        self, context: Any, state: LLMTaskState, code: str
    ) -> ArtifactRef:
        return self._artifacts(context, state.semantic_key).publish_json(
            f"generations/{state.current_generation}/supervision/{code}.json",
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


def _result_contract(contract: InteractiveJsonOutput) -> Any:
    from .request import JsonOutput

    return JsonOutput(contract.result_schema, repair="strict")
