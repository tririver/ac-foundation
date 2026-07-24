from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, cast

from arc_jobs import (
    ArtifactRef,
    Awaiting,
    CancelledError,
    FailureMode,
    GroupResult,
    JsonValue,
    Paused,
    RevisionConflictError,
    RunContext,
    RunError,
    RunOutcome,
    semantic_key,
    StateContract,
    Succeeded,
    UnitResult,
    WorkUnit,
)
from arc_llm import (
    JsonOutput,
    LLMCancelled,
    LLMCompleted,
    LLMExecutionOptions,
    LLMFailed,
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
from .validation import (
    decode_review,
    validate_batch_request,
    validate_execution_options,
)


_STATE_SCHEMA = "arc.proposer_reviewer.loop_state.v1"


@dataclass(frozen=True)
class _PauseRecord:
    role: str
    worker_id: str
    round_number: int
    task_id: str
    awaiting: Awaiting


@dataclass(frozen=True)
class _LoopState:
    revision: int
    loop_id: str
    rounds_completed: int
    proposal_refs: Mapping[str, ArtifactRef]
    current_proposer_ids: tuple[str, ...]
    review_ref: ArtifactRef | None
    proposer_sessions: Mapping[str, SessionRef]
    reviewer_session: SessionRef | None
    transcript_refs: tuple[ArtifactRef, ...]
    pauses: Mapping[str, _PauseRecord]
    termination: LoopTermination | None


class _LoopStateContract(StateContract[_LoopState]):
    schema_version = _STATE_SCHEMA

    def encode(self, value: _LoopState) -> Mapping[str, JsonValue]:
        return {
            "revision": value.revision,
            "loop_id": value.loop_id,
            "rounds_completed": value.rounds_completed,
            "proposal_refs": {
                worker_id: artifact_ref_to_document(ref)
                for worker_id, ref in value.proposal_refs.items()
            },
            "current_proposer_ids": list(value.current_proposer_ids),
            "review_ref": (
                None
                if value.review_ref is None
                else artifact_ref_to_document(value.review_ref)
            ),
            "proposer_sessions": {
                worker_id: _session_document(session)
                for worker_id, session in value.proposer_sessions.items()
            },
            "reviewer_session": (
                None
                if value.reviewer_session is None
                else _session_document(value.reviewer_session)
            ),
            "transcript_refs": [
                artifact_ref_to_document(ref) for ref in value.transcript_refs
            ],
            "pauses": {
                key: _pause_document(record) for key, record in value.pauses.items()
            },
            "termination": (
                None if value.termination is None else value.termination.value
            ),
        }

    def decode(self, document: Mapping[str, JsonValue]) -> _LoopState:
        expected = {
            "revision",
            "loop_id",
            "rounds_completed",
            "proposal_refs",
            "current_proposer_ids",
            "review_ref",
            "proposer_sessions",
            "reviewer_session",
            "transcript_refs",
            "pauses",
            "termination",
        }
        if set(document) != expected:
            raise ValueError("loop state uses an invalid closed shape")
        revision = document["revision"]
        rounds_completed = document["rounds_completed"]
        loop_id = document["loop_id"]
        if (
            type(revision) is not int
            or revision < 0
            or type(rounds_completed) is not int
            or rounds_completed < 0
            or not isinstance(loop_id, str)
        ):
            raise ValueError("loop state has invalid scalar fields")
        raw_proposals = _mapping(document["proposal_refs"], "proposal_refs")
        raw_current_ids = document["current_proposer_ids"]
        if not isinstance(raw_current_ids, list) or not all(
            isinstance(item, str) for item in raw_current_ids
        ):
            raise ValueError("current_proposer_ids must be an array of strings")
        raw_sessions = _mapping(document["proposer_sessions"], "proposer_sessions")
        raw_pauses = _mapping(document["pauses"], "pauses")
        raw_transcript = document["transcript_refs"]
        if not isinstance(raw_transcript, list):
            raise ValueError("transcript_refs must be an array")
        raw_review = document["review_ref"]
        raw_reviewer_session = document["reviewer_session"]
        raw_termination = document["termination"]
        if raw_termination is not None and not isinstance(raw_termination, str):
            raise ValueError("termination must be a string or null")
        return _LoopState(
            revision=revision,
            loop_id=loop_id,
            rounds_completed=rounds_completed,
            proposal_refs={
                key: artifact_ref_from_document(value)
                for key, value in raw_proposals.items()
            },
            current_proposer_ids=tuple(raw_current_ids),
            review_ref=(
                None
                if raw_review is None
                else artifact_ref_from_document(raw_review)
            ),
            proposer_sessions={
                key: _session_from_document(value)
                for key, value in raw_sessions.items()
            },
            reviewer_session=(
                None
                if raw_reviewer_session is None
                else _session_from_document(raw_reviewer_session)
            ),
            transcript_refs=tuple(
                artifact_ref_from_document(value) for value in raw_transcript
            ),
            pauses={
                key: _pause_from_document(value) for key, value in raw_pauses.items()
            },
            termination=(
                None
                if raw_termination is None
                else LoopTermination(raw_termination)
            ),
        )

    def validate_transition(
        self, previous: _LoopState | None, next: _LoopState
    ) -> None:
        if previous is None:
            if next.revision != 0 or next.rounds_completed != 0:
                raise ValueError("loop state must start at revision zero")
            return
        if next.loop_id != previous.loop_id:
            raise ValueError("loop_id cannot change")
        if next.revision != previous.revision + 1:
            raise ValueError("loop state revision must advance by one")
        if next.rounds_completed not in {
            previous.rounds_completed,
            previous.rounds_completed + 1,
        }:
            raise ValueError("rounds_completed must stay fixed or advance by one")
        if (
            next.rounds_completed == previous.rounds_completed + 1
            and next.pauses
        ):
            raise ValueError("a committed round cannot retain paused workers")
        if previous.termination is not None and next != previous:
            raise ValueError("terminated loop state is immutable")


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
            WorkUnit(loop.loop_id, loop_semantic_projection(loop))
            for loop in request.loops
        )
        loop_by_id = {loop.loop_id: loop for loop in request.loops}

        def run_loop(unit: WorkUnit) -> UnitResult | Paused:
            loop = loop_by_id[unit.unit_id]
            outcome = self._execute_loop(
                context,
                artifacts,
                loop,
                options=options,
            )
            if isinstance(outcome, Paused):
                return outcome
            document = _loop_result_document(outcome)
            artifacts.publish_json(loop_result_artifact_id(loop.loop_id), document)
            if outcome.termination is LoopTermination.FAILED:
                return UnitResult(
                    unit.unit_id,
                    "failed",
                    document,
                    outcome.error
                    or RunError("loop_failed", f"loop {loop.loop_id} failed"),
                )
            return UnitResult(unit.unit_id, "succeeded", document)

        grouped = context.run_group(
            "batch.loops",
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
            if unit.status == "cancelled":
                raise CancelledError(unit.error.message if unit.error else "cancelled")
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
    ) -> LoopResult | Paused:
        runtime_loop_id = _runtime_loop_id(loop.loop_id)
        store = context.state(f"pr-loop-{runtime_loop_id}", _LoopStateContract())
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
                    output=JsonOutput(worker.output_schema),
                    model=worker.model,
                    session=state.proposer_sessions.get(worker.worker_id),
                    capabilities=worker.capabilities,
                )
                pause_key = _pause_key("proposer", worker.worker_id)
                outcome = self._call_worker(
                    context,
                    request,
                    task_id=task_id,
                    pause=state.pauses.get(pause_key),
                    options=options,
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
                if isinstance(outcome, LLMCancelled):
                    return UnitResult(
                        unit.unit_id,
                        "cancelled",
                        error=RunError("cancelled", "LLM task was cancelled"),
                    )
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
                f"pr.{runtime_loop_id}.r{round_number:03d}.proposers",
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
                if unit.status == "cancelled":
                    raise CancelledError(
                        unit.error.message if unit.error else "LLM task cancelled"
                    )
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
                output=JsonOutput(
                    reviewer_envelope_schema(
                        payload_schema=loop.reviewer.output_schema,
                        active_proposer_ids=tuple(proposals),
                    )
                ),
                model=loop.reviewer.model,
                session=state.reviewer_session,
                capabilities=loop.reviewer.capabilities,
            )
            reviewer_pause_key = _pause_key("reviewer", loop.reviewer.worker_id)
            latest_state = store.read()
            assert latest_state is not None
            reviewer_outcome = self._call_worker(
                context,
                reviewer_request,
                task_id=reviewer_task_id,
                pause=latest_state.pauses.get(reviewer_pause_key),
                options=options,
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
            if isinstance(reviewer_outcome, LLMCancelled):
                raise CancelledError("reviewer LLM task was cancelled")
            if isinstance(reviewer_outcome, LLMFailed):
                return _failed_loop(
                    loop,
                    state,
                    "reviewer_failed",
                    "reviewer failed",
                    (),
                    error=_llm_error(reviewer_outcome),
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

    def _call_worker(
        self,
        context: RunContext,
        request: LLMRequest,
        *,
        task_id: str,
        pause: _PauseRecord | None,
        options: ExecutionOptions,
    ):
        llm_options = LLMExecutionOptions(
            limits=options.llm_limits,
            interaction_resolver=options.interaction_resolver,
        )
        if pause is None:
            return self.llm.execute(context, request, options=llm_options)
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
            options=llm_options,
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


def _failed_loop(
    loop: LoopSpec,
    state: _LoopState,
    code: str,
    message: str,
    failures: tuple[UnitResult, ...] | list[UnitResult],
    *,
    error: RunError | None = None,
) -> LoopResult:
    details: dict[str, JsonValue] = {
        "failed_worker_ids": [unit.unit_id for unit in failures],
    }
    return LoopResult(
        loop_id=loop.loop_id,
        termination=LoopTermination.FAILED,
        rounds_completed=state.rounds_completed,
        final_proposals={},
        final_review=None,
        error=error or RunError(code, message, details),
    )


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


def _runtime_loop_id(loop_id: str) -> str:
    return semantic_key(
        {
            "semantic_key_schema": "arc.proposer_reviewer.runtime_loop_id.v1",
            "loop_id": loop_id,
        }
    ).sha256[:32]


def _session_document(value: SessionRef) -> dict[str, JsonValue]:
    return {
        "session_key": value.session_key,
        "accepted_prefix_sha256": value.accepted_prefix_sha256,
    }


def _session_from_document(value: JsonValue) -> SessionRef:
    document = _mapping(value, "session")
    if set(document) != {"session_key", "accepted_prefix_sha256"}:
        raise ValueError("session uses an invalid closed shape")
    session_key = document["session_key"]
    digest = document["accepted_prefix_sha256"]
    if not isinstance(session_key, str) or not isinstance(digest, str):
        raise ValueError("session has invalid fields")
    return SessionRef(session_key, digest)


def _pause_document(value: _PauseRecord) -> dict[str, JsonValue]:
    awaiting = value.awaiting
    return {
        "role": value.role,
        "worker_id": value.worker_id,
        "round": value.round_number,
        "task_id": value.task_id,
        "awaiting": {
            "reason": awaiting.reason.value,
            "resume_key": awaiting.resume_key,
            "input_required": awaiting.input_required,
            "request_ref": (
                None
                if awaiting.request_ref is None
                else artifact_ref_to_document(awaiting.request_ref)
            ),
            "response_contract": awaiting.response_contract,
            "details": dict(awaiting.details),
        },
    }


def _pause_from_document(value: JsonValue) -> _PauseRecord:
    from arc_jobs import ResumeReason

    document = _mapping(value, "pause")
    if set(document) != {"role", "worker_id", "round", "task_id", "awaiting"}:
        raise ValueError("pause uses an invalid closed shape")
    awaiting_doc = _mapping(document["awaiting"], "awaiting")
    if set(awaiting_doc) != {
        "reason",
        "resume_key",
        "input_required",
        "request_ref",
        "response_contract",
        "details",
    }:
        raise ValueError("awaiting uses an invalid closed shape")
    request_ref = awaiting_doc["request_ref"]
    details = _mapping(awaiting_doc["details"], "awaiting.details")
    return _PauseRecord(
        role=cast(str, document["role"]),
        worker_id=cast(str, document["worker_id"]),
        round_number=cast(int, document["round"]),
        task_id=cast(str, document["task_id"]),
        awaiting=Awaiting(
            reason=ResumeReason(cast(str, awaiting_doc["reason"])),
            resume_key=cast(str, awaiting_doc["resume_key"]),
            input_required=cast(bool, awaiting_doc["input_required"]),
            request_ref=(
                None
                if request_ref is None
                else artifact_ref_from_document(request_ref)
            ),
            response_contract=cast(str | None, awaiting_doc["response_contract"]),
            details=details,
        ),
    )


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
