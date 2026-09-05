"""Durable LLM task/session records and pure recovery decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from ac_jobs import (
    ArtifactRef,
    CorruptStateError,
    ExecutionFingerprint,
    JsonValue,
    ResumeReason,
    SemanticKeyDigest,
    decode_artifact_ref,
    encode_artifact_ref,
)

TASK_SCHEMA_VERSION = "ac.llm.task.v6"
SESSION_SCHEMA_VERSION = "ac.llm.session.v2"


class AcceptedOrigin(StrEnum):
    PROVIDER = "provider"
    ADOPTED = "adopted"


@dataclass(frozen=True)
class AcceptedRecord:
    artifact_ref: ArtifactRef
    origin: AcceptedOrigin
    generation: int | None
    provider: str | None
    model: str | None
    reused_from: Mapping[str, JsonValue] | None = None


@dataclass(frozen=True)
class TaskPause:
    reason: ResumeReason
    resume_key: str
    input_required: bool
    request_ref: ArtifactRef | None
    response_contract: str | None
    details: Mapping[str, JsonValue]


@dataclass(frozen=True)
class GenerationRecord:
    """The sole currently active provider generation for a task."""

    generation: int
    execution: ExecutionFingerprint
    native_handle: str | None = None
    raw_response: ArtifactRef | None = None
    attempt_started: bool = False


@dataclass(frozen=True)
class LLMTaskState:
    revision: int
    task_id: str
    semantic_key: SemanticKeyDigest
    resolved_provider: str | None
    resolved_model: str | None
    generation: GenerationRecord | None
    request_ref: ArtifactRef
    accepted: AcceptedRecord | None = None
    session_key: str | None = None
    host_turn_round: int = 0
    pending_host_turn: ArtifactRef | None = None
    seen_host_request_ids: tuple[str, ...] = ()
    pause: TaskPause | None = None

    @property
    def current(self) -> GenerationRecord:
        if self.generation is None:
            raise CorruptStateError("current provider generation is missing")
        return self.generation


@dataclass(frozen=True)
class AcceptedSessionTurn:
    task_semantic_key_sha256: str
    artifact_sha256: str
    result_prefix_sha256: str


@dataclass(frozen=True)
class LLMSessionState:
    revision: int
    session_key: str
    generation: int
    provider: str
    model: str
    session_compatibility: ExecutionFingerprint
    native_handle: str | None
    accepted_turns: int
    accepted_prefix_sha256: str
    accepted_turn_records: tuple[AcceptedSessionTurn, ...]


def fresh_generation(
    state: LLMTaskState,
    *,
    execution: ExecutionFingerprint,
) -> LLMTaskState:
    generation = 1 if state.generation is None else state.current.generation + 1
    return replace(
        state,
        revision=state.revision + 1,
        generation=GenerationRecord(generation, execution),
        host_turn_round=0,
        pending_host_turn=None,
    )


class TaskStateContract:
    schema_version = TASK_SCHEMA_VERSION

    def encode(self, value: LLMTaskState) -> Mapping[str, JsonValue]:
        return {
            "revision": value.revision,
            "task_id": value.task_id,
            "semantic_key_schema": "ac.llm.semantic_key.v3",
            "semantic_key_sha256": value.semantic_key.sha256,
            "resolved_provider": value.resolved_provider,
            "resolved_model": value.resolved_model,
            "generation": (
                None if value.generation is None else _generation_doc(value.generation)
            ),
            "request_ref": _ref_doc(value.request_ref),
            "accepted": _accepted_doc(value.accepted),
            "session_key": value.session_key,
            "host_turn_round": value.host_turn_round,
            "pending_host_turn": _ref_doc(value.pending_host_turn),
            "seen_host_request_ids": list(value.seen_host_request_ids),
            "pause": _pause_doc(value.pause),
        }

    def decode(self, document: Mapping[str, JsonValue]) -> LLMTaskState:
        required = {
            "revision",
            "task_id",
            "semantic_key_schema",
            "semantic_key_sha256",
            "resolved_provider",
            "resolved_model",
            "generation",
            "request_ref",
            "accepted",
            "session_key",
            "host_turn_round",
            "pending_host_turn",
            "seen_host_request_ids",
            "pause",
        }
        _exact(document, required)
        if document["semantic_key_schema"] != "ac.llm.semantic_key.v3":
            raise CorruptStateError("unsupported LLM semantic key schema")
        seen_doc = document["seen_host_request_ids"]
        if not isinstance(seen_doc, list):
            raise CorruptStateError("host request IDs must be an array")
        raw_generation = document["generation"]
        state = LLMTaskState(
            revision=_int(document["revision"], "revision"),
            task_id=_str(document["task_id"], "task_id"),
            semantic_key=SemanticKeyDigest(_str(document["semantic_key_sha256"], "semantic key")),
            resolved_provider=_nullable_str(document["resolved_provider"], "provider"),
            resolved_model=_nullable_str(document["resolved_model"], "model"),
            generation=None if raw_generation is None else _generation(raw_generation),
            request_ref=_required_ref(document["request_ref"]),
            accepted=_accepted(document["accepted"]),
            session_key=_nullable_str(document["session_key"], "session key"),
            host_turn_round=_int(document["host_turn_round"], "host turn round"),
            pending_host_turn=_ref(document["pending_host_turn"]),
            seen_host_request_ids=tuple(_str(item, "host request id") for item in seen_doc),
            pause=_pause(document["pause"]),
        )
        _validate_task(state)
        return state

    def validate_transition(
        self, previous: LLMTaskState | None, next: LLMTaskState
    ) -> None:
        _validate_task(next)
        if previous is None:
            if next.revision != 0:
                raise ValueError("Initial LLM task state has revision 0.")
            return
        if next.revision != previous.revision + 1:
            raise ValueError("LLM task state revision must increase by one.")
        if next.task_id != previous.task_id or next.semantic_key != previous.semantic_key:
            raise ValueError("Task identity is immutable.")
        if previous.accepted is not None and next.accepted != previous.accepted:
            raise ValueError("Accepted result is immutable.")
        if previous.generation is not None and next.generation is not None:
            if next.generation.generation < previous.generation.generation:
                raise ValueError("Generation cannot decrease.")

class SessionStateContract:
    schema_version = SESSION_SCHEMA_VERSION

    def encode(self, value: LLMSessionState) -> Mapping[str, JsonValue]:
        return {
            "revision": value.revision,
            "session_key": value.session_key,
            "generation": value.generation,
            "provider": value.provider,
            "model": value.model,
            "session_compatibility_schema": value.session_compatibility.schema_version,
            "session_compatibility_sha256": value.session_compatibility.sha256,
            "native_handle": value.native_handle,
            "accepted_turns": value.accepted_turns,
            "accepted_prefix_sha256": value.accepted_prefix_sha256,
            "accepted_turn_records": [
                {
                    "task_semantic_key_sha256": item.task_semantic_key_sha256,
                    "artifact_sha256": item.artifact_sha256,
                    "result_prefix_sha256": item.result_prefix_sha256,
                }
                for item in value.accepted_turn_records
            ],
        }

    def decode(self, document: Mapping[str, JsonValue]) -> LLMSessionState:
        required = {
            "revision",
            "session_key",
            "generation",
            "provider",
            "model",
            "session_compatibility_schema",
            "session_compatibility_sha256",
            "native_handle",
            "accepted_turns",
            "accepted_prefix_sha256",
            "accepted_turn_records",
        }
        _exact(document, required)
        records = document["accepted_turn_records"]
        if not isinstance(records, list):
            raise CorruptStateError("accepted turn records must be an array")
        state = LLMSessionState(
            _int(document["revision"], "revision"),
            _str(document["session_key"], "session key"),
            _int(document["generation"], "generation"),
            _str(document["provider"], "provider"),
            _str(document["model"], "model"),
            ExecutionFingerprint(
                _str(document["session_compatibility_schema"], "session schema"),
                _str(document["session_compatibility_sha256"], "session digest"),
            ),
            _nullable_str(document["native_handle"], "native handle"),
            _int(document["accepted_turns"], "accepted turns"),
            _str(document["accepted_prefix_sha256"], "accepted prefix"),
            tuple(_accepted_session_turn(item) for item in records),
        )
        _validate_session(state)
        return state

    def validate_transition(
        self, previous: LLMSessionState | None, next: LLMSessionState
    ) -> None:
        _validate_session(next)
        if previous is None:
            if next.revision != 0 or next.accepted_turns != 0:
                raise ValueError("Initial session starts at revision and turn zero.")
            return
        if next.revision != previous.revision + 1:
            raise ValueError("Session revision must increase by one.")
        if next.session_key != previous.session_key:
            raise ValueError("Session key is immutable.")
        if next.accepted_turns != previous.accepted_turns + 1:
            raise ValueError("A session transition accepts exactly one turn.")
        if (
            next.accepted_turn_records[:-1] != previous.accepted_turn_records
            or len(next.accepted_turn_records)
            != len(previous.accepted_turn_records) + 1
        ):
            raise ValueError("Accepted session turn history is append-only.")
        if previous.accepted_turns > 0 and (
            next.provider != previous.provider
            or next.model != previous.model
        ):
            raise ValueError(
                "An accepted session cannot change provider or model."
            )


def _accepted_session_turn(value: JsonValue) -> AcceptedSessionTurn:
    if not isinstance(value, dict):
        raise CorruptStateError("accepted session turn must be an object")
    _exact(
        value,
        {
            "task_semantic_key_sha256",
            "artifact_sha256",
            "result_prefix_sha256",
        },
    )
    return AcceptedSessionTurn(
        _str(value["task_semantic_key_sha256"], "task semantic key"),
        _str(value["artifact_sha256"], "accepted artifact digest"),
        _str(value["result_prefix_sha256"], "result prefix"),
    )


def _validate_session(state: LLMSessionState) -> None:
    if state.accepted_turns != len(state.accepted_turn_records):
        raise CorruptStateError("accepted turn count and history differ")
    keys = [item.task_semantic_key_sha256 for item in state.accepted_turn_records]
    if len(keys) != len(set(keys)):
        raise CorruptStateError("duplicate accepted task in session history")
    if state.accepted_turn_records and (
        state.accepted_turn_records[-1].result_prefix_sha256
        != state.accepted_prefix_sha256
    ):
        raise CorruptStateError("session prefix differs from accepted turn history")


def _validate_task(state: LLMTaskState) -> None:
    if state.revision < 0:
        raise CorruptStateError("invalid task revision")
    if state.generation is None:
        if (
            state.accepted is None
            or state.accepted.origin is not AcceptedOrigin.ADOPTED
        ):
            raise CorruptStateError("only an adopted task may have no provider generation")
        return
    if state.generation.generation < 1:
        raise CorruptStateError("invalid task generation")
    if state.accepted is not None and state.accepted.generation is not None:
        if state.accepted.generation > state.generation.generation:
            raise CorruptStateError("accepted generation does not exist")
    if len(set(state.seen_host_request_ids)) != len(state.seen_host_request_ids):
        raise CorruptStateError("duplicate seen host request IDs")


def _generation_doc(value: GenerationRecord) -> dict[str, JsonValue]:
    return {
        "generation": value.generation,
        "execution_fingerprint_schema": value.execution.schema_version,
        "execution_fingerprint_sha256": value.execution.sha256,
        "native_handle": value.native_handle,
        "raw_response": _ref_doc(value.raw_response),
        "attempt_started": value.attempt_started,
    }


def _generation(value: JsonValue) -> GenerationRecord:
    if not isinstance(value, dict):
        raise CorruptStateError("generation must be an object")
    _exact(
        value,
        {
            "generation",
            "execution_fingerprint_schema",
            "execution_fingerprint_sha256",
            "native_handle",
            "raw_response",
            "attempt_started",
        },
    )
    return GenerationRecord(
        _int(value["generation"], "generation"),
        ExecutionFingerprint(
            _str(value["execution_fingerprint_schema"], "execution schema"),
            _str(value["execution_fingerprint_sha256"], "execution digest"),
        ),
        _nullable_str(value["native_handle"], "native handle"),
        _ref(value["raw_response"]),
        _bool(value["attempt_started"], "attempt started"),
    )


def _accepted_doc(value: AcceptedRecord | None) -> JsonValue:
    if value is None:
        return None
    return {
        "artifact_ref": _ref_doc(value.artifact_ref),
        "origin": value.origin.value,
        "generation": value.generation,
        "provider": value.provider,
        "model": value.model,
        "reused_from": None if value.reused_from is None else dict(value.reused_from),
    }


def _accepted(value: JsonValue) -> AcceptedRecord | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CorruptStateError("accepted record must be an object")
    _exact(value, {"artifact_ref", "origin", "generation", "provider", "model", "reused_from"})
    reused = value["reused_from"]
    if reused is not None and not isinstance(reused, dict):
        raise CorruptStateError("reused_from must be an object")
    try:
        origin = AcceptedOrigin(_str(value["origin"], "accepted origin"))
    except ValueError as exc:
        raise CorruptStateError("unknown accepted origin") from exc
    ref = _ref(value["artifact_ref"])
    if ref is None:
        raise CorruptStateError("accepted result requires an artifact")
    return AcceptedRecord(
        ref,
        origin,
        _nullable_int(value["generation"], "accepted generation"),
        _nullable_str(value["provider"], "accepted provider"),
        _nullable_str(value["model"], "accepted model"),
        reused,
    )


def _pause_doc(value: TaskPause | None) -> JsonValue:
    if value is None:
        return None
    return {
        "reason": value.reason.value,
        "resume_key": value.resume_key,
        "input_required": value.input_required,
        "request_ref": _ref_doc(value.request_ref),
        "response_contract": value.response_contract,
        "details": dict(value.details),
    }


def _pause(value: JsonValue) -> TaskPause | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CorruptStateError("pause must be an object")
    _exact(
        value,
        {
            "reason",
            "resume_key",
            "input_required",
            "request_ref",
            "response_contract",
            "details",
        },
    )
    try:
        reason = ResumeReason(_str(value["reason"], "pause reason"))
    except ValueError as exc:
        raise CorruptStateError("unknown pause reason") from exc
    input_required = value["input_required"]
    details = value["details"]
    if not isinstance(input_required, bool) or not isinstance(details, dict):
        raise CorruptStateError("invalid pause fields")
    request_ref = _ref(value["request_ref"])
    response_contract = _nullable_str(value["response_contract"], "response contract")
    if input_required and (request_ref is None or response_contract is None):
        raise CorruptStateError("input-required pause is incomplete")
    return TaskPause(
        reason,
        _str(value["resume_key"], "resume key"),
        input_required,
        request_ref,
        response_contract,
        details,
    )


def _ref_doc(value: ArtifactRef | None) -> JsonValue:
    if value is None:
        return None
    return encode_artifact_ref(value)


def _ref(value: JsonValue) -> ArtifactRef | None:
    if value is None:
        return None
    try:
        return decode_artifact_ref(value)
    except ValueError as exc:
        raise CorruptStateError("invalid artifact ref") from exc


def _required_ref(value: JsonValue) -> ArtifactRef:
    ref = _ref(value)
    if ref is None:
        raise CorruptStateError("artifact ref is required")
    return ref


def _exact(value: Mapping[str, Any], fields: set[str]) -> None:
    if set(value) != fields:
        raise CorruptStateError("durable LLM state uses an invalid closed shape")


def _str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CorruptStateError(f"{name} must be a non-empty string")
    return value


def _nullable_str(value: Any, name: str) -> str | None:
    return None if value is None else _str(value, name)


def _int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CorruptStateError(f"{name} must be a non-negative integer")
    return value


def _nullable_int(value: Any, name: str) -> int | None:
    return None if value is None else _int(value, name)


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise CorruptStateError(f"{name} must be a boolean")
    return value
