from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping

from .errors import CorruptStateError
from .models import Awaiting, JsonValue, ResumeReason, RunSnapshot, RunStatus
from .storage import require_fields, utc_now


class CommandStatus(StrEnum):
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class CommandRun:
    id: str
    revision: int


@dataclass(frozen=True)
class CommandArtifact:
    role: str
    id: str
    path: str


@dataclass(frozen=True)
class CommandWarning:
    code: str
    message: str
    details: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandError:
    code: str
    message: str
    details: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ResumeDescriptor:
    reason: ResumeReason
    resume_key: str
    input_required: bool
    input_schema: str | None
    request_artifact: str | None
    details: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandResult:
    status: CommandStatus
    run: CommandRun | None = None
    data: Mapping[str, JsonValue] = field(default_factory=dict)
    artifacts: tuple[CommandArtifact, ...] = ()
    warnings: tuple[CommandWarning, ...] = ()
    error: CommandError | None = None
    resume: ResumeDescriptor | None = None


@dataclass(frozen=True)
class ProgressEvent:
    run_id: str
    sequence: int
    event: str
    data: Mapping[str, JsonValue] = field(default_factory=dict)
    at: str = field(default_factory=utc_now)


def resume_from_awaiting(value: Awaiting) -> ResumeDescriptor:
    return ResumeDescriptor(
        value.reason,
        value.resume_key,
        value.input_required,
        value.response_contract,
        value.request_ref.relative_path if value.request_ref else None,
        dict(value.details),
    )


def _warning_json(value: CommandWarning) -> dict[str, JsonValue]:
    return {
        "code": value.code,
        "message": value.message,
        "details": dict(value.details),
    }


def _error_json(value: CommandError | None) -> JsonValue:
    if value is None:
        return None
    return {
        "code": value.code,
        "message": value.message,
        "details": dict(value.details),
    }


def _resume_json(value: ResumeDescriptor | None) -> JsonValue:
    if value is None:
        return None
    return {
        "reason": value.reason.value,
        "resume_key": value.resume_key,
        "input_required": value.input_required,
        "input_schema": value.input_schema,
        "request_artifact": value.request_artifact,
        "details": dict(value.details),
    }


def validate_command_result(value: CommandResult) -> None:
    if value.status is CommandStatus.FAILED:
        if value.error is None or value.resume is not None:
            raise ValueError("failed result requires error and forbids resume")
    elif value.status is CommandStatus.PAUSED:
        if value.error is not None or value.resume is None:
            raise ValueError("paused result requires resume and forbids error")
    elif value.error is not None or value.resume is not None:
        raise ValueError("completed/cancelled result forbids error and resume")
    if value.resume is not None:
        descriptor = value.resume
        try:
            ResumeReason(descriptor.reason)
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown resume reason") from exc
        if not descriptor.resume_key:
            raise ValueError("resume descriptor requires resume_key")
        if descriptor.input_required and (
            not descriptor.input_schema or not descriptor.request_artifact
        ):
            raise ValueError(
                "input-required resume needs input_schema and request_artifact"
            )
        if not descriptor.input_required and descriptor.input_schema is not None:
            raise ValueError("no-input resume must not contain input_schema")


def encode_command_result(value: CommandResult) -> dict[str, JsonValue]:
    validate_command_result(value)
    return {
        "schema_version": "arc.command_result.v1",
        "status": value.status.value,
        "run": (
            {"id": value.run.id, "revision": value.run.revision}
            if value.run is not None
            else None
        ),
        "data": dict(value.data),
        "artifacts": [
            {"role": item.role, "id": item.id, "path": item.path}
            for item in value.artifacts
        ],
        "warnings": [_warning_json(item) for item in value.warnings],
        "error": _error_json(value.error),
        "resume": _resume_json(value.resume),
    }


def command_result_json(value: CommandResult) -> str:
    return json.dumps(
        encode_command_result(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _decode_error(value: JsonValue) -> CommandError | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CorruptStateError("command error must be an object")
    require_fields(value, required={"code", "message", "details"})
    if not isinstance(value["code"], str) or not isinstance(value["message"], str) or not isinstance(
        value["details"], dict
    ):
        raise CorruptStateError("invalid command error")
    return CommandError(value["code"], value["message"], value["details"])


def _decode_resume(value: JsonValue) -> ResumeDescriptor | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CorruptStateError("resume descriptor must be an object")
    require_fields(
        value,
        required={
            "reason",
            "resume_key",
            "input_required",
            "input_schema",
            "request_artifact",
            "details",
        },
    )
    if (
        not isinstance(value["reason"], str)
        or not isinstance(value["resume_key"], str)
        or not isinstance(value["input_required"], bool)
        or not (
            value["input_schema"] is None or isinstance(value["input_schema"], str)
        )
        or not (
            value["request_artifact"] is None
            or isinstance(value["request_artifact"], str)
        )
        or not isinstance(value["details"], dict)
    ):
        raise CorruptStateError("invalid resume descriptor")
    try:
        reason = ResumeReason(value["reason"])
    except ValueError as exc:
        raise CorruptStateError("unknown resume reason") from exc
    return ResumeDescriptor(
        reason,
        value["resume_key"],
        value["input_required"],
        value["input_schema"],
        value["request_artifact"],
        value["details"],
    )


def decode_command_result(document: Mapping[str, JsonValue]) -> CommandResult:
    require_fields(
        document,
        required={
            "schema_version",
            "status",
            "run",
            "data",
            "artifacts",
            "warnings",
            "error",
            "resume",
        },
    )
    if document["schema_version"] != "arc.command_result.v1":
        raise CorruptStateError("unsupported command result schema")
    try:
        status = CommandStatus(str(document["status"]))
    except ValueError as exc:
        raise CorruptStateError("unknown command status") from exc
    run_json = document["run"]
    run = None
    if run_json is not None:
        if not isinstance(run_json, dict):
            raise CorruptStateError("run locator must be an object")
        require_fields(run_json, required={"id", "revision"})
        if (
            not isinstance(run_json["id"], str)
            or not isinstance(run_json["revision"], int)
            or isinstance(run_json["revision"], bool)
        ):
            raise CorruptStateError("invalid run locator")
        run = CommandRun(run_json["id"], run_json["revision"])
    if not isinstance(document["data"], dict):
        raise CorruptStateError("command data must be an object")
    artifact_json, warning_json = document["artifacts"], document["warnings"]
    if not isinstance(artifact_json, list) or not isinstance(warning_json, list):
        raise CorruptStateError("artifacts and warnings must be arrays")
    artifacts: list[CommandArtifact] = []
    for item in artifact_json:
        if not isinstance(item, dict):
            raise CorruptStateError("command artifact must be an object")
        require_fields(item, required={"role", "id", "path"})
        if not all(isinstance(item[key], str) for key in ("role", "id", "path")):
            raise CorruptStateError("invalid command artifact")
        artifacts.append(CommandArtifact(item["role"], item["id"], item["path"]))
    warnings: list[CommandWarning] = []
    for item in warning_json:
        if not isinstance(item, dict):
            raise CorruptStateError("command warning must be an object")
        require_fields(item, required={"code", "message", "details"})
        if (
            not isinstance(item["code"], str)
            or not isinstance(item["message"], str)
            or not isinstance(item["details"], dict)
        ):
            raise CorruptStateError("invalid command warning")
        warnings.append(CommandWarning(item["code"], item["message"], item["details"]))
    result = CommandResult(
        status,
        run,
        document["data"],
        tuple(artifacts),
        tuple(warnings),
        _decode_error(document["error"]),
        _decode_resume(document["resume"]),
    )
    try:
        validate_command_result(result)
    except ValueError as exc:
        raise CorruptStateError(str(exc)) from exc
    return result


def _safe_progress(value: JsonValue) -> None:
    forbidden = {"text", "token", "content", "output", "delta", "candidate", "result"}
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in forbidden:
                raise ValueError(f"progress contains forbidden field {key!r}")
            _safe_progress(child)
    elif isinstance(value, list):
        for child in value:
            _safe_progress(child)


def encode_progress_event(value: ProgressEvent) -> dict[str, JsonValue]:
    if not value.run_id:
        raise ValueError("run-less commands must not emit progress")
    if value.sequence < 1:
        raise ValueError("progress sequence must be positive")
    _safe_progress(dict(value.data))
    return {
        "schema_version": "arc.progress_event.v1",
        "run_id": value.run_id,
        "sequence": value.sequence,
        "at": value.at,
        "event": value.event,
        "data": dict(value.data),
    }


def decode_progress_event(document: Mapping[str, JsonValue]) -> ProgressEvent:
    require_fields(
        document,
        required={"schema_version", "run_id", "sequence", "at", "event", "data"},
    )
    if document["schema_version"] != "arc.progress_event.v1":
        raise CorruptStateError("unsupported progress event schema")
    run_id, sequence, at, event, data = (
        document["run_id"],
        document["sequence"],
        document["at"],
        document["event"],
        document["data"],
    )
    if (
        not isinstance(run_id, str)
        or not isinstance(sequence, int)
        or isinstance(sequence, bool)
        or not isinstance(at, str)
        or not isinstance(event, str)
        or not isinstance(data, dict)
    ):
        raise CorruptStateError("invalid progress event")
    value = ProgressEvent(run_id, sequence, event, data, at)
    try:
        encode_progress_event(value)
    except ValueError as exc:
        raise CorruptStateError(str(exc)) from exc
    return value


def snapshot_data(snapshot: RunSnapshot) -> dict[str, JsonValue]:
    data: dict[str, JsonValue] = {
        "id": snapshot.run_id,
        "revision": snapshot.revision,
        "status": snapshot.status.value,
        "attempt": snapshot.attempt,
        "created_at": snapshot.created_at,
        "updated_at": snapshot.updated_at,
        "interrupted": snapshot.interrupted,
        "result": (
            {
                "artifact_id": snapshot.result_ref.artifact_id,
                "path": snapshot.result_ref.relative_path,
            }
            if snapshot.result_ref
            else None
        ),
        "error": (
            {
                "code": snapshot.error.code,
                "message": snapshot.error.message,
                "details": dict(snapshot.error.details),
            }
            if snapshot.error
            else None
        ),
        "resume": (
            _resume_json(resume_from_awaiting(snapshot.awaiting))
            if snapshot.awaiting
            else None
        ),
    }
    return data


def command_result_from_snapshot(
    snapshot: RunSnapshot, *, query: bool = False
) -> CommandResult:
    run = CommandRun(snapshot.run_id, snapshot.revision)
    if query:
        return CommandResult(
            CommandStatus.COMPLETED,
            run,
            {"run": snapshot_data(snapshot)},
        )
    mapping = {
        RunStatus.SUCCEEDED: CommandStatus.COMPLETED,
        RunStatus.PAUSED: CommandStatus.PAUSED,
        RunStatus.FAILED: CommandStatus.FAILED,
        RunStatus.CANCELLED: CommandStatus.CANCELLED,
    }
    if snapshot.status not in mapping:
        raise ValueError("blocking result cannot be pending or running")
    error = (
        CommandError(
            snapshot.error.code,
            snapshot.error.message,
            snapshot.error.details,
        )
        if snapshot.error
        else None
    )
    artifacts = (
        (
            CommandArtifact(
                "result",
                snapshot.result_ref.artifact_id,
                snapshot.result_ref.relative_path,
            ),
        )
        if snapshot.result_ref
        else ()
    )
    return CommandResult(
        mapping[snapshot.status],
        run,
        {"run": snapshot_data(snapshot)},
        artifacts,
        error=error,
        resume=resume_from_awaiting(snapshot.awaiting) if snapshot.awaiting else None,
    )
