"""Private durable-state codec and locators for proposer-reviewer batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, cast

from ac_jobs import (
    ArtifactRef,
    Awaiting,
    InvalidRunIdError,
    JsonValue,
    StateContract,
    semantic_key,
    validate_simple_id,
)
from ac_llm import SessionRef

from .artifacts import artifact_ref_from_document, artifact_ref_to_document
from .identity import execution_scope_token
from .models import LoopTermination


_STATE_SCHEMA = "ac.proposer_reviewer.loop_state.v1"
_PAUSE_ROLES = frozenset({"proposer", "reviewer"})


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


def runtime_loop_id(loop_id: str) -> str:
    return semantic_key(
        {
            "semantic_key_schema": "ac.proposer_reviewer.runtime_loop_id.v1",
            "loop_id": loop_id,
        }
    ).sha256[:32]


def state_namespace(
    loop_id: str,
    *,
    execution_scope: str | None = None,
) -> str:
    scope_token = execution_scope_token(execution_scope)
    if scope_token is None:
        return f"pr-loop-{runtime_loop_id(loop_id)}"
    return f"pr-scope-{scope_token}-loop-{runtime_loop_id(loop_id)}"


def batch_group_id(*, execution_scope: str | None = None) -> str:
    scope_token = execution_scope_token(execution_scope)
    if scope_token is None:
        return "batch.loops"
    return f"pr.scope-{scope_token}.batch.loops"


def proposer_group_id(
    loop_id: str,
    round_number: int,
    *,
    execution_scope: str | None = None,
) -> str:
    scope_token = execution_scope_token(execution_scope)
    prefix = (
        "pr"
        if scope_token is None
        else f"pr.scope-{scope_token}"
    )
    return f"{prefix}.{runtime_loop_id(loop_id)}.r{round_number:03d}.proposers"


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
    from ac_jobs import ResumeReason

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
    role = document["role"]
    if not isinstance(role, str) or role not in _PAUSE_ROLES:
        raise ValueError("pause role must be proposer or reviewer")
    worker_id = _pause_id(document["worker_id"], "worker_id")
    round_number = document["round"]
    if type(round_number) is not int or round_number < 1:
        raise ValueError("pause round must be a positive integer")
    task_id = _pause_id(document["task_id"], "task_id")
    raw_reason = awaiting_doc["reason"]
    if not isinstance(raw_reason, str):
        raise ValueError("awaiting reason must be a string")
    try:
        reason = ResumeReason(raw_reason)
    except ValueError as exc:
        raise ValueError("awaiting reason is unknown") from exc
    resume_key = _pause_id(awaiting_doc["resume_key"], "resume_key")
    input_required = awaiting_doc["input_required"]
    if type(input_required) is not bool:
        raise ValueError("awaiting input_required must be a boolean")
    response_contract = awaiting_doc["response_contract"]
    if response_contract is not None and not isinstance(response_contract, str):
        raise ValueError("awaiting response_contract must be a string or null")
    request_ref = awaiting_doc["request_ref"]
    details = _mapping(awaiting_doc["details"], "awaiting.details")
    return _PauseRecord(
        role=role,
        worker_id=worker_id,
        round_number=round_number,
        task_id=task_id,
        awaiting=Awaiting(
            reason=reason,
            resume_key=resume_key,
            input_required=input_required,
            request_ref=(
                None
                if request_ref is None
                else artifact_ref_from_document(request_ref)
            ),
            response_contract=response_contract,
            details=details,
        ),
    )


def _pause_id(value: JsonValue, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"pause {name} is invalid")
    try:
        return validate_simple_id(value, label=f"pause {name}")
    except InvalidRunIdError as exc:
        raise ValueError(f"pause {name} is invalid") from exc


def _mapping(value: object, name: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, JsonValue], value)
