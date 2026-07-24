"""Durable LLM task/session records and pure recovery decisions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Mapping

from arc_jobs import (
    ArtifactDigest,
    ArtifactRef,
    CorruptStateError,
    EffectStage,
    ExecutionFingerprint,
    JsonValue,
    SemanticKeyDigest,
    ResumeReason,
)

TASK_SCHEMA_VERSION = "arc.llm.task.v1"
SESSION_SCHEMA_VERSION = "arc.llm.session.v1"


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
    generation: int
    effect_id: str
    execution: ExecutionFingerprint
    native_handle: str | None = None
    raw_response: ArtifactRef | None = None
    replacement_of: int | None = None
    replacement_reason: str | None = None
    possible_duplicate_execution: bool = False
    safe_retries: int = 0
    native_resumes: int = 0


@dataclass(frozen=True)
class LLMTaskState:
    revision: int
    task_id: str
    semantic_key: SemanticKeyDigest
    resolved_provider: str | None
    resolved_model: str | None
    current_generation: int
    generations: tuple[GenerationRecord, ...]
    request_ref: ArtifactRef
    accepted: AcceptedRecord | None = None
    session_key: str | None = None
    interaction_round: int = 0
    pending_interaction: ArtifactRef | None = None
    seen_request_ids: tuple[str, ...] = ()
    pause: TaskPause | None = None

    @property
    def current(self) -> GenerationRecord:
        if not self.generations or self.generations[-1].generation != self.current_generation:
            raise CorruptStateError("current generation is missing")
        return self.generations[-1]


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


class RecoveryAction(StrEnum):
    REPLAY_ACCEPTED = "replay_accepted"
    RECOVER_SAVED_OUTPUT = "recover_saved_output"
    START = "start"
    NATIVE_RESUME = "native_resume"
    REPLACE = "replace"
    PAUSE_UNCERTAIN = "pause_uncertain"


def decide_recovery(
    state: LLMTaskState,
    effect_stage: EffectStage,
    *,
    execution: ExecutionFingerprint,
    supports_native_resume: bool,
    safe_retry_limit: int,
    native_resume_limit: int,
    automatic_replacement_limit: int,
) -> RecoveryAction:
    if state.accepted is not None:
        return RecoveryAction.REPLAY_ACCEPTED
    current = state.current
    if effect_stage in {EffectStage.OUTPUT_SAVED, EffectStage.COMMITTED}:
        return RecoveryAction.RECOVER_SAVED_OUTPUT
    if effect_stage is EffectStage.PREPARED:
        if current.safe_retries <= safe_retry_limit:
            return RecoveryAction.START
        return RecoveryAction.PAUSE_UNCERTAIN
    if current.execution == execution and current.native_handle and supports_native_resume:
        if current.native_resumes < native_resume_limit:
            return RecoveryAction.NATIVE_RESUME
    replacements = sum(item.replacement_of is not None for item in state.generations)
    if replacements < automatic_replacement_limit:
        return RecoveryAction.REPLACE
    return RecoveryAction.PAUSE_UNCERTAIN


def replace_current(
    state: LLMTaskState,
    *,
    execution: ExecutionFingerprint,
    reason: str,
    possible_duplicate: bool,
) -> LLMTaskState:
    generation = state.current_generation + 1
    record = GenerationRecord(
        generation,
        effect_id_for(state.task_id, generation),
        execution,
        replacement_of=state.current_generation,
        replacement_reason=reason,
        possible_duplicate_execution=possible_duplicate,
    )
    return replace(
        state,
        revision=state.revision + 1,
        current_generation=generation,
        generations=state.generations + (record,),
    )


def effect_id_for(task_id: str, generation: int) -> str:
    task_digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:24]
    return f"llm-{task_digest}-g{generation}"


class TaskStateContract:
    schema_version = TASK_SCHEMA_VERSION

    def encode(self, value: LLMTaskState) -> Mapping[str, JsonValue]:
        return {
            "revision": value.revision,
            "task_id": value.task_id,
            "semantic_key_schema": "arc.llm.semantic_key.v1",
            "semantic_key_sha256": value.semantic_key.sha256,
            "resolved_provider": value.resolved_provider,
            "resolved_model": value.resolved_model,
            "current_generation": value.current_generation,
            "generations": [_generation_doc(item) for item in value.generations],
            "request_ref": _ref_doc(value.request_ref),
            "accepted": _accepted_doc(value.accepted),
            "session_key": value.session_key,
            "interaction_round": value.interaction_round,
            "pending_interaction": _ref_doc(value.pending_interaction),
            "seen_request_ids": list(value.seen_request_ids),
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
            "current_generation",
            "generations",
            "request_ref",
            "accepted",
            "session_key",
            "interaction_round",
            "pending_interaction",
            "seen_request_ids",
            "pause",
        }
        _exact(document, required)
        if document["semantic_key_schema"] != "arc.llm.semantic_key.v1":
            raise CorruptStateError("unsupported LLM semantic key schema")
        generations_doc = document["generations"]
        seen_doc = document["seen_request_ids"]
        if not isinstance(generations_doc, list) or not isinstance(seen_doc, list):
            raise CorruptStateError("invalid task state arrays")
        state = LLMTaskState(
            revision=_int(document["revision"], "revision"),
            task_id=_str(document["task_id"], "task_id"),
            semantic_key=SemanticKeyDigest(_str(document["semantic_key_sha256"], "semantic key")),
            resolved_provider=_nullable_str(document["resolved_provider"], "provider"),
            resolved_model=_nullable_str(document["resolved_model"], "model"),
            current_generation=_int(document["current_generation"], "current generation"),
            generations=tuple(_generation(item) for item in generations_doc),
            request_ref=_required_ref(document["request_ref"]),
            accepted=_accepted(document["accepted"]),
            session_key=_nullable_str(document["session_key"], "session key"),
            interaction_round=_int(document["interaction_round"], "interaction round"),
            pending_interaction=_ref(document["pending_interaction"]),
            seen_request_ids=tuple(_str(item, "request id") for item in seen_doc),
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
        if next.current_generation < previous.current_generation:
            raise ValueError("Generation cannot decrease.")
        if next.generations[: len(previous.generations)] != previous.generations:
            current_changed = (
                len(next.generations) == len(previous.generations)
                and next.generations[:-1] == previous.generations[:-1]
            )
            if not current_changed:
                raise ValueError("Past generations are immutable.")


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
        }
        _exact(document, required)
        return LLMSessionState(
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
        )

    def validate_transition(
        self, previous: LLMSessionState | None, next: LLMSessionState
    ) -> None:
        if previous is None:
            if next.revision != 0 or next.accepted_turns != 0:
                raise ValueError("Initial session starts at revision and turn zero.")
            return
        if next.revision != previous.revision + 1:
            raise ValueError("Session revision must increase by one.")
        if next.session_key != previous.session_key:
            raise ValueError("Session key is immutable.")
        if next.accepted_turns not in {previous.accepted_turns, previous.accepted_turns + 1}:
            raise ValueError("Accepted session turns may increase by one.")
        if previous.accepted_turns > 0 and (
            next.provider != previous.provider
            or next.model != previous.model
            or next.session_compatibility != previous.session_compatibility
        ):
            raise ValueError("An accepted session cannot be rebound.")


def _validate_task(state: LLMTaskState) -> None:
    if state.revision < 0:
        raise CorruptStateError("invalid task revision")
    if not state.generations:
        if (
            state.current_generation != 0
            or state.accepted is None
            or state.accepted.origin is not AcceptedOrigin.ADOPTED
        ):
            raise CorruptStateError("only an adopted task may have no provider generation")
        return
    if state.current_generation < 1:
        raise CorruptStateError("invalid task generation")
    expected = tuple(range(1, len(state.generations) + 1))
    actual = tuple(item.generation for item in state.generations)
    if actual != expected or state.current_generation != actual[-1]:
        raise CorruptStateError("task generations are not contiguous")
    if state.accepted is not None and state.accepted.generation is not None:
        if state.accepted.generation > state.current_generation:
            raise CorruptStateError("accepted generation does not exist")
    if len(set(state.seen_request_ids)) != len(state.seen_request_ids):
        raise CorruptStateError("duplicate seen interaction request IDs")


def _generation_doc(value: GenerationRecord) -> dict[str, JsonValue]:
    return {
        "generation": value.generation,
        "effect_id": value.effect_id,
        "execution_fingerprint_schema": value.execution.schema_version,
        "execution_fingerprint_sha256": value.execution.sha256,
        "native_handle": value.native_handle,
        "raw_response": _ref_doc(value.raw_response),
        "replacement_of": value.replacement_of,
        "replacement_reason": value.replacement_reason,
        "possible_duplicate_execution": value.possible_duplicate_execution,
        "safe_retries": value.safe_retries,
        "native_resumes": value.native_resumes,
    }


def _generation(value: JsonValue) -> GenerationRecord:
    if not isinstance(value, dict):
        raise CorruptStateError("generation must be an object")
    _exact(
        value,
        {
            "generation",
            "effect_id",
            "execution_fingerprint_schema",
            "execution_fingerprint_sha256",
            "native_handle",
            "raw_response",
            "replacement_of",
            "replacement_reason",
            "possible_duplicate_execution",
            "safe_retries",
            "native_resumes",
        },
    )
    duplicate = value["possible_duplicate_execution"]
    if not isinstance(duplicate, bool):
        raise CorruptStateError("possible_duplicate_execution must be boolean")
    return GenerationRecord(
        _int(value["generation"], "generation"),
        _str(value["effect_id"], "effect id"),
        ExecutionFingerprint(
            _str(value["execution_fingerprint_schema"], "execution schema"),
            _str(value["execution_fingerprint_sha256"], "execution digest"),
        ),
        _nullable_str(value["native_handle"], "native handle"),
        _ref(value["raw_response"]),
        _nullable_int(value["replacement_of"], "replacement generation"),
        _nullable_str(value["replacement_reason"], "replacement reason"),
        duplicate,
        _int(value["safe_retries"], "safe retries"),
        _int(value["native_resumes"], "native resumes"),
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
    return {
        "artifact_id": value.artifact_id,
        "digest": {
            "algorithm": value.digest.algorithm,
            "value": value.digest.value,
            "size_bytes": value.digest.size_bytes,
        },
        "media_type": value.media_type,
        "relative_path": value.relative_path,
    }


def _ref(value: JsonValue) -> ArtifactRef | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CorruptStateError("artifact ref must be an object")
    _exact(value, {"artifact_id", "digest", "media_type", "relative_path"})
    digest = value["digest"]
    if not isinstance(digest, dict):
        raise CorruptStateError("artifact digest must be an object")
    _exact(digest, {"algorithm", "value", "size_bytes"})
    if digest["algorithm"] != "sha256":
        raise CorruptStateError("unsupported artifact digest")
    return ArtifactRef(
        _str(value["artifact_id"], "artifact id"),
        ArtifactDigest(
            "sha256",
            _str(digest["value"], "artifact digest"),
            _int(digest["size_bytes"], "artifact size"),
        ),
        _str(value["media_type"], "media type"),
        _str(value["relative_path"], "artifact path"),
    )


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
