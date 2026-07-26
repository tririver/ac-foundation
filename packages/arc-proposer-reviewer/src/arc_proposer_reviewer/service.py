from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Mapping, cast

from arc_jobs import (
    ArtifactRef,
    Awaiting,
    FailureMode,
    GroupResult,
    JsonValue,
    Paused,
    RevisionConflictError,
    RunContext,
    RunError,
    RunOutcome,
    Succeeded,
    StoppedError,
    UnitResult,
    WorkUnit,
)
from arc_llm import (
    JsonOutput,
    LLMStopped,
    LLMCompleted,
    LLMFailed,
    LLMInputArtifact,
    LLMPaused,
    LLMRequest,
    LLMTaskService,
    SessionRef,
    decode_resume_input,
)

from .artifacts import (
    artifact_ref_from_document,
    artifact_ref_to_document,
    batch_result_artifact_id,
    loop_result_artifact_id,
    proposal_artifact_id,
    read_json_artifact,
    request_artifact_id,
    review_artifact_id,
    transcript_artifact_id,
)
from .dialogue import TranscriptTurn, encode_transcript_turn
from .identity import (
    loop_semantic_projection,
    worker_semantic_projection,
    worker_task_id,
)
from .models import (
    RESULT_SCHEMA_VERSION,
    BatchFailurePolicy,
    BatchRequest,
    BatchResult,
    ExecutionOptions,
    LoopResult,
    LoopSpec,
    LoopTermination,
    ProposerFailurePolicy,
    WorkerSpec,
)
from .prompts import (
    render_delta_proposer_prompt,
    render_initial_proposer_prompt,
    render_reviewer_prompt,
    reviewer_envelope_schema,
)
from .protocol import encode_batch_request, encode_batch_result
from .state import (
    _LoopState,
    _LoopStateContract,
    _PauseRecord,
    _session_document,
    _session_from_document,
    batch_group_id,
    proposer_group_id,
    state_namespace,
)
from .validation import (
    decode_review,
    validate_batch_request,
    validate_execution_options,
)


@dataclass(frozen=True)
class _ExecutionProgress:
    event: Literal[
        "loop_started",
        "loop_finished",
        "round_started",
        "round_finished",
        "worker_started",
        "worker_finished",
    ]
    loop_id: str
    round_number: int | None = None
    role: Literal["proposer", "reviewer"] | None = None
    worker_id: str | None = None
    task_id: str | None = None
    status: str | None = None


class ProposerReviewerService:
    def __init__(self, llm: LLMTaskService) -> None:
        self.llm = llm

    def execute(
        self,
        context: RunContext,
        request: BatchRequest,
        *,
        options: ExecutionOptions,
    ) -> RunOutcome:
        validate_batch_request(request)
        validate_execution_options(options)
        artifacts = context.artifacts.scoped("proposer-reviewer")
        artifacts.publish_json(request_artifact_id(), encode_batch_request(request))

        units = tuple(
            WorkUnit(
                loop.loop_id,
                loop_semantic_projection(loop, inputs=request.inputs),
            )
            for loop in request.loops
        )
        loop_by_id = {loop.loop_id: loop for loop in request.loops}

        def run_loop(unit: WorkUnit) -> UnitResult | Paused:
            loop = loop_by_id[unit.unit_id]
            _emit_progress(
                context,
                _ExecutionProgress("loop_started", loop.loop_id),
            )
            try:
                outcome = self._execute_loop(
                    context,
                    artifacts,
                    loop,
                    options=options,
                    inputs=request.inputs,
                )
            except StoppedError:
                _emit_progress(
                    context,
                    _ExecutionProgress(
                        "loop_finished", loop.loop_id, status="stopped"
                    ),
                )
                raise
            except Exception:
                _emit_progress(
                    context,
                    _ExecutionProgress(
                        "loop_finished", loop.loop_id, status="failed"
                    ),
                )
                raise
            if isinstance(outcome, Paused):
                _emit_progress(
                    context,
                    _ExecutionProgress(
                        "loop_finished", loop.loop_id, status="paused"
                    ),
                )
                return outcome
            document = _loop_result_document(outcome)
            artifacts.publish_json(loop_result_artifact_id(loop.loop_id), document)
            if outcome.termination is LoopTermination.FAILED:
                _emit_progress(
                    context,
                    _ExecutionProgress(
                        "loop_finished", loop.loop_id, status="failed"
                    ),
                )
                return UnitResult(
                    unit.unit_id,
                    "failed",
                    document,
                    outcome.error
                    or RunError("loop_failed", f"loop {loop.loop_id} failed"),
                )
            _emit_progress(
                context,
                _ExecutionProgress(
                    "loop_finished", loop.loop_id, status="succeeded"
                ),
            )
            return UnitResult(unit.unit_id, "succeeded", document)

        grouped = context.run_group(
            batch_group_id(),
            units,
            run_loop,
            max_workers=options.max_concurrent_loops,
            failure_mode=(
                FailureMode.FAIL_FAST
                if request.failure_policy is BatchFailurePolicy.FAIL_FAST
                else FailureMode.COLLECT
            ),
        )
        if isinstance(grouped, Paused):
            return grouped
        assert isinstance(grouped, GroupResult)
        result_by_id: dict[str, LoopResult] = {}
        for unit in grouped.units:
            if isinstance(unit.value, Mapping):
                result_by_id[unit.unit_id] = _loop_result_from_document(unit.value)
            else:
                result_by_id[unit.unit_id] = LoopResult(
                    loop_id=unit.unit_id,
                    termination=LoopTermination.FAILED,
                    rounds_completed=0,
                    final_proposals={},
                    final_review=None,
                    error=unit.error
                    or RunError("loop_failed", f"loop {unit.unit_id} failed"),
                )
        for loop in request.loops:
            if loop.loop_id not in result_by_id:
                result_by_id[loop.loop_id] = LoopResult(
                    loop_id=loop.loop_id,
                    termination=LoopTermination.FAILED,
                    rounds_completed=0,
                    final_proposals={},
                    final_review=None,
                    error=RunError(
                        "fail_fast_skipped",
                        "loop was not started after an earlier loop failed",
                    ),
                )
        result = BatchResult(
            schema_version=RESULT_SCHEMA_VERSION,
            loops=tuple(result_by_id[loop.loop_id] for loop in request.loops),
        )
        for loop_result in result.loops:
            artifacts.publish_json(
                loop_result_artifact_id(loop_result.loop_id),
                _loop_result_document(loop_result),
            )
        result_ref = artifacts.publish_json(
            batch_result_artifact_id(), encode_batch_result(result)
        )
        return Succeeded(result_ref)

    def _execute_loop(
        self,
        context: RunContext,
        artifacts: object,
        loop: LoopSpec,
        *,
        options: ExecutionOptions,
        inputs: tuple[LLMInputArtifact, ...] = (),
    ) -> LoopResult | Paused:
        store = context.state(state_namespace(loop.loop_id), _LoopStateContract())
        state = store.read()
        if state is None:
            state = store.create(
                _LoopState(
                    revision=0,
                    loop_id=loop.loop_id,
                    rounds_completed=0,
                    proposal_refs={},
                    current_proposer_ids=(),
                    review_ref=None,
                    proposer_sessions={},
                    reviewer_session=None,
                    transcript_refs=(),
                    pauses={},
                    termination=None,
                )
            )

        if state.termination is not None:
            return _successful_loop(loop, state, state.termination, artifacts)

        while state.rounds_completed < loop.max_rounds:
            context.checkpoint()
            round_number = state.rounds_completed + 1
            _emit_progress(
                context,
                _ExecutionProgress(
                    "round_started",
                    loop.loop_id,
                    round_number=round_number,
                ),
            )
            prior_review = (
                None
                if state.review_ref is None
                else read_json_artifact(artifacts, state.review_ref)
            )
            previous_proposals = {
                worker_id: read_json_artifact(artifacts, ref)
                for worker_id, ref in state.proposal_refs.items()
            }
            previous_feedback = (
                prior_review.get("feedback", {})
                if isinstance(prior_review, Mapping)
                else {}
            )
            transcript_refs = tuple(
                {"content_sha256": ref.digest.value} for ref in state.transcript_refs
            )

            proposer_units: list[WorkUnit] = []
            worker_by_id = {worker.worker_id: worker for worker in loop.proposers}
            for worker in loop.proposers:
                upstream: dict[str, str] = {}
                if worker.worker_id in state.proposal_refs:
                    upstream["previous_proposal"] = state.proposal_refs[
                        worker.worker_id
                    ].digest.value
                    if state.review_ref is not None:
                        upstream["previous_review"] = state.review_ref.digest.value
                    for index, ref in enumerate(state.transcript_refs):
                        upstream[f"transcript:{index:03d}"] = ref.digest.value
                if worker.worker_id in state.proposer_sessions:
                    upstream["session_accepted_prefix"] = state.proposer_sessions[
                        worker.worker_id
                    ].accepted_prefix_sha256
                proposer_units.append(
                    WorkUnit(
                        worker.worker_id,
                        worker_semantic_projection(
                            role="proposer",
                            loop=loop,
                            round_number=round_number,
                            worker=worker,
                            upstream_digests=upstream,
                            inputs=inputs,
                        ),
                    )
                )

            def run_proposer(unit: WorkUnit) -> UnitResult | Paused:
                worker = worker_by_id[unit.unit_id]
                upstream = {
                    key: value
                    for key, value in (
                        (
                            "previous_proposal",
                            state.proposal_refs.get(worker.worker_id).digest.value
                            if worker.worker_id in state.proposal_refs
                            else None,
                        ),
                        (
                            "previous_review",
                            state.review_ref.digest.value
                            if (
                                state.review_ref is not None
                                and worker.worker_id in state.proposal_refs
                            )
                            else None,
                        ),
                    )
                    if value is not None
                }
                if worker.worker_id in state.proposal_refs:
                    upstream.update(
                        {
                            f"transcript:{index:03d}": ref.digest.value
                            for index, ref in enumerate(state.transcript_refs)
                        }
                    )
                if worker.worker_id in state.proposer_sessions:
                    upstream["session_accepted_prefix"] = state.proposer_sessions[
                        worker.worker_id
                    ].accepted_prefix_sha256
                task_id = worker_task_id(
                    role="proposer",
                    loop=loop,
                    round_number=round_number,
                    worker=worker,
                    upstream_digests=upstream,
                    inputs=inputs,
                )
                if round_number == 1 or worker.worker_id not in previous_proposals:
                    prompt = render_initial_proposer_prompt(
                        loop=loop, worker=worker, round_number=round_number
                    )
                else:
                    prompt = render_delta_proposer_prompt(
                        loop=loop,
                        worker=worker,
                        round_number=round_number,
                        previous_proposal=previous_proposals[worker.worker_id],
                        targeted_feedback=str(
                            previous_feedback.get(
                                worker.worker_id,
                                (
                                    "No targeted feedback was produced for your prior "
                                    "turn. Recompute the complete proposal from the "
                                    "current context."
                                ),
                            )
                        ),
                        transcript_refs=transcript_refs,
                    )
                request = LLMRequest(
                    task_id=task_id,
                    prompt=prompt,
                    output=_worker_output(worker.output_schema),
                    model=worker.model,
                    session=state.proposer_sessions.get(worker.worker_id),
                    inputs=inputs,
                )
                pause_key = _pause_key("proposer", worker.worker_id)
                outcome = self._observed_worker_call(
                    context,
                    request,
                    loop_id=loop.loop_id,
                    task_id=task_id,
                    pause=state.pauses.get(pause_key),
                    options=options,
                    round_number=round_number,
                    role="proposer",
                    worker_id=worker.worker_id,
                )
                if isinstance(outcome, LLMPaused):
                    paused = _outer_pause(
                        outcome,
                        loop=loop,
                        round_number=round_number,
                        worker=worker,
                        role="proposer",
                        task_id=task_id,
                    )
                    _put_pause(
                        context,
                        store,
                        pause_key,
                        _PauseRecord(
                            "proposer",
                            worker.worker_id,
                            round_number,
                            task_id,
                            paused.awaiting,
                        ),
                    )
                    return paused
                if isinstance(outcome, LLMStopped):
                    raise StoppedError("proposer LLM task stopped")
                if isinstance(outcome, LLMFailed):
                    return UnitResult(
                        unit.unit_id,
                        "failed",
                        error=_llm_error(outcome),
                    )
                assert isinstance(outcome, LLMCompleted)
                value = cast(JsonValue, outcome.value)
                ref = artifacts.publish_json(
                    proposal_artifact_id(loop.loop_id, round_number, worker.worker_id),
                    value,
                )
                _remove_pause(context, store, pause_key)
                return UnitResult(
                    unit.unit_id,
                    "succeeded",
                    {
                        "content_ref": artifact_ref_to_document(ref),
                        "session": (
                            None
                            if outcome.session is None
                            else _session_document(outcome.session)
                        ),
                    },
                )

            current_state = store.read()
            assert current_state is not None
            grouped = context.run_group(
                proposer_group_id(loop.loop_id, round_number),
                tuple(proposer_units),
                run_proposer,
                max_workers=(
                    1 if current_state.pauses else options.max_concurrent_workers
                ),
                failure_mode=FailureMode.COLLECT,
            )
            if isinstance(grouped, Paused):
                return grouped
            assert isinstance(grouped, GroupResult)

            successful_refs: dict[str, ArtifactRef] = {}
            successful_sessions: dict[str, SessionRef] = {}
            failed: list[UnitResult] = []
            for unit in grouped.units:
                if unit.status != "succeeded":
                    failed.append(unit)
                    continue
                value = _mapping(unit.value, "proposer unit value")
                successful_refs[unit.unit_id] = artifact_ref_from_document(
                    value["content_ref"]
                )
                if value["session"] is not None:
                    successful_sessions[unit.unit_id] = _session_from_document(
                        value["session"]
                    )

            if failed and loop.on_proposer_failure is ProposerFailurePolicy.FAIL_LOOP:
                return _failed_loop(
                    loop,
                    state,
                    "proposer_failed",
                    "one or more proposers failed",
                    failed,
                )
            if not successful_refs:
                return _failed_loop(
                    loop,
                    state,
                    "all_proposers_failed",
                    "all proposers failed",
                    failed,
                )
            failed_ids = tuple(unit.unit_id for unit in failed)
            if failed_ids:
                context.events.emit(
                    "proposer_partial_failure",
                    {
                        "loop_id": loop.loop_id,
                        "round": round_number,
                        "failed_worker_ids": list(failed_ids),
                    },
                )
            proposals = {
                worker_id: read_json_artifact(artifacts, ref)
                for worker_id, ref in successful_refs.items()
            }
            review_upstream = {
                f"proposal:{worker_id}": ref.digest.value
                for worker_id, ref in successful_refs.items()
            }
            if state.review_ref is not None:
                review_upstream["previous_review"] = state.review_ref.digest.value
            review_upstream.update(
                {
                    f"transcript:{index:03d}": ref.digest.value
                    for index, ref in enumerate(state.transcript_refs)
                }
            )
            if state.reviewer_session is not None:
                review_upstream[
                    "session_accepted_prefix"
                ] = state.reviewer_session.accepted_prefix_sha256
            reviewer_task_id = worker_task_id(
                role="reviewer",
                loop=loop,
                round_number=round_number,
                worker=loop.reviewer,
                upstream_digests=review_upstream,
                inputs=inputs,
            )
            reviewer_request = LLMRequest(
                task_id=reviewer_task_id,
                prompt=render_reviewer_prompt(
                    loop=loop,
                    round_number=round_number,
                    proposals=proposals,
                    previous_review=prior_review,
                    failed_proposer_ids=failed_ids,
                    transcript_refs=transcript_refs,
                ),
                output=_worker_output(
                    reviewer_envelope_schema(
                        payload_schema=loop.reviewer.output_schema,
                        active_proposer_ids=tuple(proposals),
                    ),
                ),
                model=loop.reviewer.model,
                session=state.reviewer_session,
                inputs=inputs,
            )
            reviewer_pause_key = _pause_key("reviewer", loop.reviewer.worker_id)
            latest_state = store.read()
            assert latest_state is not None
            reviewer_outcome = self._observed_worker_call(
                context,
                reviewer_request,
                loop_id=loop.loop_id,
                task_id=reviewer_task_id,
                pause=latest_state.pauses.get(reviewer_pause_key),
                options=options,
                round_number=round_number,
                role="reviewer",
                worker_id=loop.reviewer.worker_id,
            )
            if isinstance(reviewer_outcome, LLMPaused):
                paused = _outer_pause(
                    reviewer_outcome,
                    loop=loop,
                    round_number=round_number,
                    worker=loop.reviewer,
                    role="reviewer",
                    task_id=reviewer_task_id,
                )
                _put_pause(
                    context,
                    store,
                    reviewer_pause_key,
                    _PauseRecord(
                        "reviewer",
                        loop.reviewer.worker_id,
                        round_number,
                        reviewer_task_id,
                        paused.awaiting,
                    ),
                )
                return paused
            if isinstance(reviewer_outcome, LLMStopped):
                raise StoppedError("reviewer LLM task stopped")
            if isinstance(reviewer_outcome, LLMFailed):
                return _failed_loop(
                    loop,
                    state,
                    "reviewer_failed",
                    "reviewer failed",
                    (),
                    error=_llm_error(reviewer_outcome),
                    error_worker_id=loop.reviewer.worker_id,
                )
            assert isinstance(reviewer_outcome, LLMCompleted)
            review_value = cast(JsonValue, reviewer_outcome.value)
            review = decode_review(
                review_value,
                active_proposer_ids=tuple(proposals),
                validate_payload=lambda value: None,
            )
            review_ref = artifacts.publish_json(
                review_artifact_id(
                    loop.loop_id, round_number, loop.reviewer.worker_id
                ),
                review_value,
            )
            _remove_pause(context, store, reviewer_pause_key)

            new_transcript_refs = list(state.transcript_refs)
            turn_number = 0
            for worker in loop.proposers:
                if worker.worker_id not in successful_refs:
                    continue
                turn_number += 1
                turn = TranscriptTurn(
                    role="proposer",
                    worker_id=worker.worker_id,
                    round_number=round_number,
                    content_ref=successful_refs[worker.worker_id],
                    addressed_worker_ids=(loop.reviewer.worker_id,),
                )
                new_transcript_refs.append(
                    artifacts.publish_json(
                        transcript_artifact_id(
                            loop.loop_id,
                            round_number,
                            f"{turn_number:03d}",
                        ),
                        encode_transcript_turn(turn),
                    )
                )
            turn_number += 1
            review_turn = TranscriptTurn(
                role="reviewer",
                worker_id=loop.reviewer.worker_id,
                round_number=round_number,
                content_ref=review_ref,
                addressed_worker_ids=tuple(proposals),
            )
            new_transcript_refs.append(
                artifacts.publish_json(
                    transcript_artifact_id(
                        loop.loop_id,
                        round_number,
                        f"{turn_number:03d}",
                    ),
                    encode_transcript_turn(review_turn),
                )
            )
            before_commit = store.read()
            assert before_commit is not None
            termination = (
                LoopTermination.REVIEWER_STOP
                if review.action == "stop" and loop.allow_early_stop
                else (
                    LoopTermination.ROUND_LIMIT
                    if round_number == loop.max_rounds
                    else None
                )
            )
            state = store.compare_and_swap(
                before_commit.revision,
                _LoopState(
                    revision=before_commit.revision + 1,
                    loop_id=loop.loop_id,
                    rounds_completed=round_number,
                    proposal_refs={
                        **state.proposal_refs,
                        **successful_refs,
                    },
                    current_proposer_ids=tuple(successful_refs),
                    review_ref=review_ref,
                    proposer_sessions={
                        **state.proposer_sessions,
                        **successful_sessions,
                    },
                    reviewer_session=reviewer_outcome.session,
                    transcript_refs=tuple(new_transcript_refs),
                    pauses={},
                    termination=termination,
                ),
            )
            _emit_progress(
                context,
                _ExecutionProgress(
                    "round_finished",
                    loop.loop_id,
                    round_number=round_number,
                    status="succeeded",
                ),
            )
            if termination is not None:
                return _successful_loop(
                    loop, state, termination, artifacts
                )

        return _successful_loop(
            loop,
            state,
            state.termination or LoopTermination.ROUND_LIMIT,
            artifacts,
        )

    def _observed_worker_call(
        self,
        context: RunContext,
        request: LLMRequest,
        *,
        loop_id: str,
        task_id: str,
        pause: _PauseRecord | None,
        options: ExecutionOptions,
        round_number: int,
        role: Literal["proposer", "reviewer"],
        worker_id: str,
    ):
        _emit_progress(
            context,
            _ExecutionProgress(
                "worker_started",
                loop_id,
                round_number=round_number,
                role=role,
                worker_id=worker_id,
                task_id=task_id,
            ),
        )
        status = "failed"
        try:
            outcome = self._call_worker(
                context,
                request,
                task_id=task_id,
                pause=pause,
                options=options,
            )
            status = _worker_outcome_status(outcome)
            return outcome
        except StoppedError:
            status = "stopped"
            raise
        finally:
            _emit_progress(
                context,
                _ExecutionProgress(
                    "worker_finished",
                    loop_id,
                    round_number=round_number,
                    role=role,
                    worker_id=worker_id,
                    task_id=task_id,
                    status=status,
                ),
            )

    def _call_worker(
        self,
        context: RunContext,
        request: LLMRequest,
        *,
        task_id: str,
        pause: _PauseRecord | None,
        options: ExecutionOptions,
    ):
        if pause is None:
            return self.llm.execute(context, request, options=options.llm)
        resume_input = None
        if context.resume_input is not None:
            if context.resume_input.get("resume_key") != pause.awaiting.resume_key:
                return _llm_pause_from_record(pause)
            resume_input = decode_resume_input(context.resume_input)
        elif pause.awaiting.input_required:
            return _llm_pause_from_record(pause)
        return self.llm.resume(
            context,
            task_id,
            input=resume_input,
            options=options.llm,
        )


def _successful_loop(
    loop: LoopSpec,
    state: _LoopState,
    termination: LoopTermination,
    artifacts: object,
) -> LoopResult:
    return LoopResult(
        loop_id=loop.loop_id,
        termination=termination,
        rounds_completed=state.rounds_completed,
        final_proposals={
            worker_id: read_json_artifact(artifacts, ref)
            for worker_id, ref in state.proposal_refs.items()
            if worker_id in state.current_proposer_ids
        },
        final_review=(
            None
            if state.review_ref is None
            else read_json_artifact(artifacts, state.review_ref)
        ),
        error=None,
    )


def _worker_output(result_schema: Mapping[str, JsonValue]) -> JsonOutput:
    """Build the worker's single structured result contract."""
    return JsonOutput(result_schema)


def _worker_outcome_status(outcome: object) -> str:
    if isinstance(outcome, LLMCompleted):
        return "succeeded"
    if isinstance(outcome, LLMPaused):
        return "paused"
    if isinstance(outcome, LLMStopped):
        return "stopped"
    return "failed"


def _failed_loop(
    loop: LoopSpec,
    state: _LoopState,
    code: str,
    message: str,
    failures: tuple[UnitResult, ...] | list[UnitResult],
    *,
    error: RunError | None = None,
    error_worker_id: str | None = None,
) -> LoopResult:
    failed_worker_ids = [unit.unit_id for unit in failures]
    if error_worker_id is not None and error_worker_id not in failed_worker_ids:
        failed_worker_ids.append(error_worker_id)
    causes: list[JsonValue] = []
    for unit in failures:
        if unit.error is not None:
            causes.append(
                {
                    "worker_id": unit.unit_id,
                    "code": unit.error.code,
                    "message": unit.error.message,
                    "details": dict(unit.error.details),
                }
            )
    if error is not None:
        causes.append(
            {
                "worker_id": error_worker_id or "reviewer",
                "code": error.code,
                "message": error.message,
                "details": dict(error.details),
            }
        )
    details: dict[str, JsonValue] = {
        "failed_worker_ids": failed_worker_ids,
        "causes": causes,
    }
    return LoopResult(
        loop_id=loop.loop_id,
        termination=LoopTermination.FAILED,
        rounds_completed=state.rounds_completed,
        final_proposals={},
        final_review=None,
        error=RunError(code, message, details),
    )


def _emit_progress(
    context: RunContext,
    progress: _ExecutionProgress,
) -> None:
    data: dict[str, JsonValue] = {
        "loop_id": progress.loop_id,
    }
    for name, value in (
        ("round", progress.round_number),
        ("role", progress.role),
        ("worker_id", progress.worker_id),
        ("task_id", progress.task_id),
        ("status", progress.status),
    ):
        if value is not None:
            data[name] = value
    try:
        event = {
            "round_finished": "proposer_reviewer_round_committed",
        }.get(progress.event, f"proposer_reviewer_{progress.event}")
        context.events.emit(event, data)
    except Exception:
        pass


def _outer_pause(
    outcome: LLMPaused,
    *,
    loop: LoopSpec,
    round_number: int,
    worker: WorkerSpec,
    role: str,
    task_id: str,
) -> Paused:
    details = dict(outcome.details)
    inner_code = details.pop("code", None)
    details.update(
        {
            "code": "proposer_reviewer_worker_paused",
            "llm_code": inner_code,
            "loop_id": loop.loop_id,
            "round": round_number,
            "role": role,
            "worker_id": worker.worker_id,
            "task_id": task_id,
        }
    )
    return Paused(
        Awaiting(
            reason=outcome.reason,
            resume_key=outcome.resume_key,
            input_required=outcome.input_required,
            request_ref=outcome.request_ref,
            response_contract=outcome.response_contract,
            details=details,
        )
    )


def _llm_pause_from_record(record: _PauseRecord) -> LLMPaused:
    awaiting = record.awaiting
    return LLMPaused(
        reason=awaiting.reason,
        resume_key=awaiting.resume_key,
        details=awaiting.details,
        request_ref=awaiting.request_ref,
        input_required=awaiting.input_required,
        response_contract=awaiting.response_contract,
    )


def _llm_error(outcome: LLMFailed) -> RunError:
    code = outcome.error.code
    return RunError(
        code=getattr(code, "value", str(code)),
        message=str(outcome.error),
        details=cast(Mapping[str, JsonValue], outcome.error.details),
    )


def _put_pause(
    context: RunContext,
    store: object,
    key: str,
    record: _PauseRecord,
) -> None:
    for _ in range(64):
        context.checkpoint()
        current = store.read()  # type: ignore[attr-defined]
        assert current is not None
        existing = current.pauses.get(key)
        if existing == record:
            return
        next_pauses = dict(current.pauses)
        next_pauses[key] = record
        try:
            store.compare_and_swap(  # type: ignore[attr-defined]
                current.revision,
                replace(
                    current,
                    revision=current.revision + 1,
                    pauses=next_pauses,
                ),
            )
            return
        except RevisionConflictError:
            continue
    raise RevisionConflictError("could not record paused worker after concurrent updates")


def _remove_pause(context: RunContext, store: object, key: str) -> None:
    for _ in range(64):
        context.checkpoint()
        current = store.read()  # type: ignore[attr-defined]
        assert current is not None
        if key not in current.pauses:
            return
        next_pauses = dict(current.pauses)
        del next_pauses[key]
        try:
            store.compare_and_swap(  # type: ignore[attr-defined]
                current.revision,
                replace(
                    current,
                    revision=current.revision + 1,
                    pauses=next_pauses,
                ),
            )
            return
        except RevisionConflictError:
            continue
    raise RevisionConflictError("could not clear paused worker after concurrent updates")


def _pause_key(role: str, worker_id: str) -> str:
    return f"{role}.{worker_id}"


def _loop_result_document(value: LoopResult) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], encode_batch_result(
        BatchResult(RESULT_SCHEMA_VERSION, (value,))
    )["loops"][0])


def _loop_result_from_document(value: Mapping[str, JsonValue]) -> LoopResult:
    from .protocol import decode_batch_result

    return decode_batch_result(
        {"schema_version": RESULT_SCHEMA_VERSION, "loops": [dict(value)]}
    ).loops[0]


def _mapping(value: object, name: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, JsonValue], value)
