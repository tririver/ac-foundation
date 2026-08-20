"""Runtime host authority and broker contracts.

These values describe the process in which ARC is running.  They are runtime
execution policy, never part of an LLM request's semantic identity.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from .errors import InvalidRequestError
from .errors import OutputInvalidError


class HostAuthority(StrEnum):
    UNRESTRICTED = "unrestricted"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


class EffectiveHostMode(StrEnum):
    DIRECT = "direct"
    BROKERED = "brokered"


class HostResponseStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    REFUSED = "refused"


_ARC_ENVIRONMENT_KEYS = (
    "ARC_HOME",
    "ARC_RUNTIME_HOME",
    "ARC_DOCUMENT_CACHE",
    "ARC_PAPER_CACHE",
    "PATH",
)


@dataclass(frozen=True)
class ArcRuntimeEnvironment:
    """The small, immutable ARC environment surface inherited by a provider."""

    values: Mapping[str, str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw = dict(self.values)
        if set(raw) != set(_ARC_ENVIRONMENT_KEYS):
            raise InvalidRequestError(
                "ARC runtime environment must contain exactly ARC_HOME, "
                "ARC_RUNTIME_HOME, ARC_DOCUMENT_CACHE, ARC_PAPER_CACHE, and PATH."
            )
        if any(value is not None and not isinstance(value, str) for value in raw.values()):
            raise InvalidRequestError("ARC runtime environment values must be strings or null.")
        object.__setattr__(self, "values", MappingProxyType(raw))

    @classmethod
    def capture(cls, environ: Mapping[str, str] | None = None) -> "ArcRuntimeEnvironment":
        source = os.environ if environ is None else environ
        return cls({key: source.get(key) for key in _ARC_ENVIRONMENT_KEYS})

    def apply_to(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        result = dict(os.environ if base is None else base)
        for key, value in self.values.items():
            if value is None:
                result.pop(key, None)
            else:
                result[key] = value
        return result

    def execution_document(self) -> dict[str, str | None]:
        return dict(self.values)


@dataclass(frozen=True)
class HostRequest:
    request_id: str
    instruction: str
    purpose: str

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise InvalidRequestError("host request_id is required.")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise InvalidRequestError("host instruction is required.")
        if not isinstance(self.purpose, str) or not self.purpose.strip():
            raise InvalidRequestError("host purpose is required.")


@dataclass(frozen=True)
class HostResponse:
    status: HostResponseStatus
    result: Any = None
    files: tuple[str, ...] = ()
    reason_code: str | None = None
    reason: str | None = None
    retryable: bool | None = None
    retry_condition: str | None = None

    def __post_init__(self) -> None:
        _validate_json_value(self.result, field_name="host response result")
        try:
            normalized_result = json.loads(
                json.dumps(
                    self.result,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError) as exc:
            raise InvalidRequestError(
                "host response result must be JSON-compatible."
            ) from exc
        object.__setattr__(self, "result", normalized_result)
        files = tuple(self.files)
        if any(
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            for path in files
        ):
            raise InvalidRequestError("host response files must be workspace-relative paths.")
        object.__setattr__(self, "files", files)
        refused = self.status is HostResponseStatus.REFUSED
        refusal_fields = (
            self.reason_code,
            self.reason,
            self.retryable,
            self.retry_condition,
        )
        if refused and (
            not isinstance(self.reason_code, str)
            or not self.reason_code
            or not isinstance(self.reason, str)
            or not self.reason
            or not isinstance(self.retryable, bool)
            or not isinstance(self.retry_condition, str)
            or not self.retry_condition
        ):
            raise InvalidRequestError(
                "refused host responses require reason_code, reason, retryable, and retry_condition."
            )
        if not refused and any(value is not None for value in refusal_fields):
            raise InvalidRequestError("only refused host responses may contain refusal fields.")


def _validate_json_value(value: Any, *, field_name: str) -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
    elif isinstance(value, list):
        for item in value:
            _validate_json_value(item, field_name=field_name)
        return
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise InvalidRequestError(f"{field_name} object keys must be strings.")
            _validate_json_value(item, field_name=field_name)
        return
    raise InvalidRequestError(f"{field_name} must be JSON-compatible.")


class HostBroker(Protocol):
    """A host-owned executor for one model-requested host turn."""

    @property
    def execution_identity(self) -> Mapping[str, Any]: ...

    def execute(self, request: HostRequest, *, workspace: Path) -> HostResponse: ...


def effective_host_mode(authority: HostAuthority) -> EffectiveHostMode:
    return (
        EffectiveHostMode.DIRECT
        if authority is HostAuthority.UNRESTRICTED
        else EffectiveHostMode.BROKERED
    )


def broker_execution_document(broker: HostBroker | None) -> Mapping[str, Any] | None:
    if broker is None:
        return None
    identity = broker.execution_identity
    if not isinstance(identity, Mapping):
        raise InvalidRequestError("host broker execution_identity must be an object.")
    return dict(identity)


HOST_TURN_SCHEMA_VERSION = "arc.llm.host_turn.v1"
HOST_CONTINUATION_SCHEMA_VERSION = "arc.llm.host_continuation.v1"


@dataclass(frozen=True)
class HostTurn:
    state: str
    result: Any | None
    request: HostRequest | None

    def __post_init__(self) -> None:
        if self.state == "complete" and self.request is None:
            return
        if self.state == "request_host" and self.result is None and self.request is not None:
            return
        raise OutputInvalidError("Invalid host-turn state or payload.")


def host_turn_schema(result_schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "state", "result", "host_request"],
        "properties": {
            "schema_version": {"const": HOST_TURN_SCHEMA_VERSION},
            "state": {"enum": ["complete", "request_host"]},
            "result": {"anyOf": [dict(result_schema), {"type": "null"}]},
            "host_request": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["request_id", "instruction", "purpose"],
                        "properties": {
                            "request_id": {"type": "string", "minLength": 1},
                            "instruction": {"type": "string", "minLength": 1},
                            "purpose": {"type": "string", "minLength": 1},
                        },
                    },
                ]
            },
        },
    }


def decode_host_turn(
    value: Any,
    *,
    seen_host_request_ids: set[str] | None = None,
) -> HostTurn:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "state",
        "result",
        "host_request",
    }:
        raise OutputInvalidError("Host turn uses an invalid closed shape.")
    if value["schema_version"] != HOST_TURN_SCHEMA_VERSION:
        raise OutputInvalidError("Unsupported host-turn schema.")
    raw_request = value["host_request"]
    request = None
    if raw_request is not None:
        if not isinstance(raw_request, Mapping) or set(raw_request) != {
            "request_id",
            "instruction",
            "purpose",
        }:
            raise OutputInvalidError("Host request uses an invalid closed shape.")
        request = HostRequest(
            raw_request["request_id"],
            raw_request["instruction"],
            raw_request["purpose"],
        )
        if (
            seen_host_request_ids is not None
            and request.request_id in seen_host_request_ids
        ):
            raise OutputInvalidError("Duplicate host request ID.")
    return HostTurn(value["state"], value["result"], request)


def encode_host_turn(turn: HostTurn) -> dict[str, Any]:
    return {
        "schema_version": HOST_TURN_SCHEMA_VERSION,
        "state": turn.state,
        "result": turn.result,
        "host_request": (
            None
            if turn.request is None
            else {
                "request_id": turn.request.request_id,
                "instruction": turn.request.instruction,
                "purpose": turn.request.purpose,
            }
        ),
    }


def host_response_document(response: HostResponse) -> dict[str, Any]:
    return {
        "schema_version": "arc.llm.host_response.v1",
        "status": response.status.value,
        "result": response.result,
        "files": list(response.files),
        "reason_code": response.reason_code,
        "reason": response.reason,
        "retryable": response.retryable,
        "retry_condition": response.retry_condition,
    }


@dataclass(frozen=True)
class HostContinuation:
    """The persisted, provider-visible result of one host turn."""

    request_id: str
    response: HostResponse

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise InvalidRequestError("host continuation request_id is required.")
        if not isinstance(self.response, HostResponse):
            raise InvalidRequestError("host continuation response is required.")


def host_continuation_document(
    request_id: str,
    response: HostResponse,
) -> dict[str, Any]:
    if not isinstance(request_id, str) or not request_id:
        raise InvalidRequestError("host continuation request_id is required.")
    return {
        "schema_version": HOST_CONTINUATION_SCHEMA_VERSION,
        "request_id": request_id,
        "response": host_response_document(response),
    }


def decode_host_continuation(value: Any) -> HostContinuation:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "request_id",
        "response",
    }:
        raise InvalidRequestError("host continuation uses an invalid closed shape.")
    if value["schema_version"] != HOST_CONTINUATION_SCHEMA_VERSION:
        raise InvalidRequestError("unsupported host continuation schema.")
    if not isinstance(value["request_id"], str) or not value["request_id"]:
        raise InvalidRequestError("host continuation request_id is required.")
    if not isinstance(value["response"], Mapping):
        raise InvalidRequestError("host continuation response must be an object.")
    return HostContinuation(
        request_id=value["request_id"],
        response=decode_host_response(value["response"]),
    )


def decode_host_response(value: Any) -> HostResponse:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "status",
        "result",
        "files",
        "reason_code",
        "reason",
        "retryable",
        "retry_condition",
    }:
        raise InvalidRequestError("host response uses an invalid closed shape.")
    if value["schema_version"] != "arc.llm.host_response.v1":
        raise InvalidRequestError("unsupported host response schema.")
    try:
        status = HostResponseStatus(value["status"])
    except (TypeError, ValueError) as exc:
        raise InvalidRequestError("unknown host response status.") from exc
    if not isinstance(value["files"], list):
        raise InvalidRequestError("host response files must be an array.")
    return HostResponse(
        status=status,
        result=value["result"],
        files=tuple(value["files"]),
        reason_code=value["reason_code"],
        reason=value["reason"],
        retryable=value["retryable"],
        retry_condition=value["retry_condition"],
    )
