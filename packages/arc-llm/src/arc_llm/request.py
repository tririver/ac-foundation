"""Immutable requests and strict versioned codecs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias

from arc_jobs import (
    ArtifactDigest,
    ArtifactSourceRef,
    InvalidRunIdError,
    decode_artifact_digest,
    encode_artifact_digest,
    validate_artifact_id,
    validate_simple_id,
)
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .errors import InvalidRequestError, InvalidSchemaError

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
ModelTier: TypeAlias = Literal["low", "medium", "high", "xhigh"]

REQUEST_SCHEMA_VERSION = "arc.llm.request.v4"
RESUME_SCHEMA_VERSION = "arc.llm.resume_input.v3"
DEFAULT_MAX_PARALLEL_PROVIDER_CALLS = 100


class LLMExecutionProfile(StrEnum):
    STANDARD = "standard"
    BOUNDED = "bounded"


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidRequestError("Expected a JSON object.")
    return MappingProxyType(dict(value))


def _validate_sha256(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise InvalidRequestError(f"{field_name} must be a SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise InvalidRequestError(f"{field_name} must be a SHA-256 digest.") from exc


def _validate_identifier(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise InvalidRequestError(f"{field_name} must contain 1..128 characters.")
    first, rest = value[0], value[1:]
    allowed = lambda char: char.isascii() and (char.isalnum() or char in "._-")
    if not first.isascii() or not first.isalnum() or any(not allowed(char) for char in rest):
        raise InvalidRequestError(
            f"{field_name} must match [A-Za-z0-9][A-Za-z0-9._-]*."
        )


@dataclass(frozen=True)
class ModelSelection:
    provider: str = "auto"
    model: str | None = None
    tier: ModelTier = "medium"

    def __post_init__(self) -> None:
        if not self.provider or not isinstance(self.provider, str):
            raise InvalidRequestError("model.provider must be a non-empty string.")
        if self.tier not in {"low", "medium", "high", "xhigh"}:
            raise InvalidRequestError("model.tier must be low, medium, high, or xhigh.")
        if self.model is not None and (
            not isinstance(self.model, str) or not self.model.strip()
        ):
            raise InvalidRequestError("model.model must be null or a non-empty string.")
        if self.provider == "auto" and self.model is not None:
            raise InvalidRequestError("An exact model requires an explicit provider.")
        if self.model is not None and self.tier != "medium":
            raise InvalidRequestError("An exact model and a non-default tier are mutually exclusive.")


@dataclass(frozen=True)
class ExecutionLimits:
    idle_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.idle_timeout_seconds is None:
            return
        if (
            isinstance(self.idle_timeout_seconds, bool)
            or not isinstance(self.idle_timeout_seconds, (int, float))
            or self.idle_timeout_seconds <= 0
        ):
            raise InvalidRequestError(
                "idle_timeout_seconds must be null or positive."
            )


@dataclass(frozen=True)
class ProviderGateOptions:
    enabled: bool = True
    global_limit: int = DEFAULT_MAX_PARALLEL_PROVIDER_CALLS
    provider_limits: Mapping[str, int] = field(default_factory=dict)
    circuit_failure_threshold: int = 3
    circuit_cooldown_seconds: float = 900.0
    minimum_available_memory_fraction: float | None = 0.10
    memory_poll_interval_seconds: float = 1.0
    memory_launch_interval_seconds: float = 0.25

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise InvalidRequestError("gate.enabled must be a boolean.")
        for name in ("global_limit", "circuit_failure_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise InvalidRequestError(f"gate.{name} must be positive.")
        if (
            isinstance(self.circuit_cooldown_seconds, bool)
            or not isinstance(self.circuit_cooldown_seconds, (int, float))
            or self.circuit_cooldown_seconds <= 0
        ):
            raise InvalidRequestError(
                "gate.circuit_cooldown_seconds must be positive."
            )
        if self.minimum_available_memory_fraction is not None and (
            isinstance(self.minimum_available_memory_fraction, bool)
            or not isinstance(self.minimum_available_memory_fraction, (int, float))
            or not 0 < self.minimum_available_memory_fraction <= 1
        ):
            raise InvalidRequestError(
                "gate.minimum_available_memory_fraction must be null or in (0, 1]."
            )
        for name in (
            "memory_poll_interval_seconds",
            "memory_launch_interval_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise InvalidRequestError(f"gate.{name} must be positive.")
        limits = dict(self.provider_limits)
        for provider, limit in limits.items():
            if (
                not isinstance(provider, str)
                or not provider
                or isinstance(limit, bool)
                or not isinstance(limit, int)
                or not 1 <= limit <= self.global_limit
            ):
                raise InvalidRequestError(
                    "gate.provider_limits must map provider names to limits "
                    "between 1 and global_limit."
                )
        object.__setattr__(self, "provider_limits", MappingProxyType(limits))


@dataclass(frozen=True)
class LLMExecutionOptions:
    limits: ExecutionLimits = field(default_factory=ExecutionLimits)
    gate: ProviderGateOptions = field(default_factory=ProviderGateOptions)
    internet: bool = True
    host_authority: Any = None
    runtime_environment: Any = None
    host_broker: Any = None
    profile: LLMExecutionProfile = LLMExecutionProfile.STANDARD

    def __post_init__(self) -> None:
        from .host import ArcRuntimeEnvironment, HostAuthority

        if not isinstance(self.internet, bool):
            raise InvalidRequestError("internet must be a boolean.")
        if not isinstance(self.profile, LLMExecutionProfile):
            raise InvalidRequestError("profile must be an LLMExecutionProfile.")
        authority = HostAuthority.UNKNOWN if self.host_authority is None else self.host_authority
        if not isinstance(authority, HostAuthority):
            raise InvalidRequestError("host_authority must be a HostAuthority.")
        environment = (
            ArcRuntimeEnvironment.capture()
            if self.runtime_environment is None
            else self.runtime_environment
        )
        if not isinstance(environment, ArcRuntimeEnvironment):
            raise InvalidRequestError("runtime_environment must be ArcRuntimeEnvironment.")
        object.__setattr__(self, "host_authority", authority)
        object.__setattr__(self, "runtime_environment", environment)


@dataclass(frozen=True)
class SessionRef:
    session_key: str
    accepted_prefix_sha256: str

    def __post_init__(self) -> None:
        _validate_identifier(self.session_key, field_name="session.session_key")
        _validate_sha256(
            self.accepted_prefix_sha256,
            field_name="session.accepted_prefix_sha256",
        )


@dataclass(frozen=True)
class LLMInputArtifact:
    """One verified immutable artifact supplied to a provider."""

    input_id: str
    source: ArtifactSourceRef
    media_type: str

    def __post_init__(self) -> None:
        _validate_identifier(self.input_id, field_name="inputs.input_id")
        if not isinstance(self.source, ArtifactSourceRef):
            raise InvalidRequestError("inputs.source must be an ArtifactSourceRef.")
        try:
            validate_simple_id(
                self.source.source_run_id,
                label="inputs.source.source_run_id",
            )
            validate_artifact_id(self.source.source_artifact_id)
        except InvalidRunIdError as exc:
            raise InvalidRequestError(
                "inputs.source must contain valid run and artifact identifiers."
            ) from exc
        digest = self.source.expected_digest
        if not isinstance(digest, ArtifactDigest):
            raise InvalidRequestError("inputs.source.expected_digest must use SHA-256.")
        try:
            encode_artifact_digest(digest)
        except ValueError as exc:
            raise InvalidRequestError(
                "inputs.source.expected_digest must use a lowercase SHA-256 digest."
            ) from exc
        normalized_media_type = (
            self.media_type.strip().lower()
            if isinstance(self.media_type, str)
            else self.media_type
        )
        if (
            not isinstance(normalized_media_type, str)
            or "/" not in normalized_media_type
            or any(char.isspace() for char in normalized_media_type)
        ):
            raise InvalidRequestError("inputs.media_type must be a MIME type.")
        object.__setattr__(self, "media_type", normalized_media_type)


@dataclass(frozen=True)
class TextOutput:
    kind: Literal["text"] = "text"


@dataclass(frozen=True)
class JsonOutput:
    schema: Mapping[str, Any]
    repair: Literal["strict", "local", "format"] = "format"
    kind: Literal["json"] = "json"

    def __post_init__(self) -> None:
        if self.repair not in {"strict", "local", "format"}:
            raise InvalidRequestError("output.repair must be strict, local, or format.")
        _check_schema(self.schema)
        object.__setattr__(self, "schema", _frozen_mapping(self.schema))


OutputContract: TypeAlias = TextOutput | JsonOutput


@dataclass(frozen=True)
class LLMRequest:
    task_id: str
    prompt: str
    output: OutputContract
    model: ModelSelection = field(default_factory=ModelSelection)
    session: SessionRef | None = None
    inputs: tuple[LLMInputArtifact, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.task_id, field_name="task_id")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise InvalidRequestError("prompt must be a non-empty string.")
        if not isinstance(self.output, (TextOutput, JsonOutput)):
            raise InvalidRequestError("output must be a supported output contract.")
        if isinstance(self.inputs, (str, bytes)):
            raise InvalidRequestError("inputs must be a sequence of LLMInputArtifact values.")
        try:
            inputs = tuple(self.inputs)
        except TypeError as exc:
            raise InvalidRequestError(
                "inputs must be a sequence of LLMInputArtifact values."
            ) from exc
        if any(not isinstance(item, LLMInputArtifact) for item in inputs):
            raise InvalidRequestError("inputs entries must be LLMInputArtifact values.")
        input_ids = [item.input_id for item in inputs]
        if len(input_ids) != len(set(input_ids)):
            raise InvalidRequestError("inputs.input_id values must be unique.")
        object.__setattr__(self, "inputs", inputs)


class ResumeAction(StrEnum):
    CONTINUE = "continue"
    REPLACE = "replace"
    ACCEPT_CANDIDATE = "accept_candidate"


@dataclass(frozen=True)
class ResumeInput:
    resume_key: str
    action: ResumeAction
    host_response: Any = None
    candidate_digest: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.resume_key, field_name="resume_key")
        if not isinstance(self.action, ResumeAction):
            raise InvalidRequestError("action must be a ResumeAction.")
        from .host import HostResponse

        if self.host_response is not None and not isinstance(
            self.host_response, HostResponse
        ):
            raise InvalidRequestError("host_response must be a HostResponse or null.")
        if self.action is ResumeAction.CONTINUE:
            if self.candidate_digest is not None or self.reason is not None:
                raise InvalidRequestError("continue accepts only an optional host response.")
        elif self.action is ResumeAction.REPLACE:
            if (
                not self.reason
                or self.host_response is not None
                or self.candidate_digest is not None
            ):
                raise InvalidRequestError("replace requires only a non-empty reason.")
        elif self.action is ResumeAction.ACCEPT_CANDIDATE:
            if (
                not self.candidate_digest
                or self.host_response is not None
                or self.reason is not None
            ):
                raise InvalidRequestError("accept_candidate requires only candidate_digest.")
            _validate_sha256(
                self.candidate_digest,
                field_name="candidate_digest",
            )


def _check_schema(schema: Mapping[str, Any]) -> None:
    if not isinstance(schema, Mapping):
        raise InvalidSchemaError("JSON Schema must be an object.")
    try:
        Draft202012Validator.check_schema(dict(schema))
    except SchemaError as exc:
        raise InvalidSchemaError(f"Invalid Draft 2020-12 schema: {exc.message}") from exc


def encode_output_contract(contract: OutputContract) -> dict[str, Any]:
    if isinstance(contract, TextOutput):
        return {"kind": "text"}
    return {"kind": "json", "schema": dict(contract.schema), "repair": contract.repair}


def request_to_document(request: LLMRequest) -> dict[str, Any]:
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "task_id": request.task_id,
        "prompt": request.prompt,
        "output": encode_output_contract(request.output),
        "model": {
            "provider": request.model.provider,
            "model": request.model.model,
            "tier": request.model.tier,
        },
        "session": (
            None
            if request.session is None
            else {
                "session_key": request.session.session_key,
                "accepted_prefix_sha256": request.session.accepted_prefix_sha256,
            }
        ),
        "inputs": [
            {
                "input_id": item.input_id,
                "source": {
                    "source_run_id": item.source.source_run_id,
                    "source_artifact_id": item.source.source_artifact_id,
                    "expected_digest": encode_artifact_digest(
                        item.source.expected_digest
                    ),
                },
                "media_type": item.media_type,
            }
            for item in request.inputs
        ],
    }


def decode_request(document: Mapping[str, Any]) -> LLMRequest:
    _require_exact(
        document,
        {
            "schema_version",
            "task_id",
            "prompt",
            "output",
            "model",
            "session",
            "inputs",
        },
        "request",
    )
    if document["schema_version"] != REQUEST_SCHEMA_VERSION:
        raise InvalidRequestError("Unsupported request schema_version.")
    output_doc = _object(document["output"], "output")
    kind = output_doc.get("kind")
    if kind == "text":
        _require_exact(output_doc, {"kind"}, "output")
        output: OutputContract = TextOutput()
    elif kind == "json":
        _require_exact(output_doc, {"kind", "schema", "repair"}, "output")
        output = JsonOutput(_object(output_doc["schema"], "output.schema"), output_doc["repair"])
    else:
        raise InvalidRequestError("Unknown output.kind.")
    model_doc = _object(document["model"], "model")
    _require_exact(model_doc, {"provider", "model", "tier"}, "model")
    model = ModelSelection(model_doc["provider"], model_doc["model"], model_doc["tier"])
    session_doc = document["session"]
    session = None
    if session_doc is not None:
        session_obj = _object(session_doc, "session")
        _require_exact(session_obj, {"session_key", "accepted_prefix_sha256"}, "session")
        session = SessionRef(
            session_obj["session_key"],
            session_obj["accepted_prefix_sha256"],
        )
    inputs_doc = document["inputs"]
    if not isinstance(inputs_doc, list):
        raise InvalidRequestError("inputs must be an array.")
    inputs: list[LLMInputArtifact] = []
    for raw in inputs_doc:
        item = _object(raw, "input")
        _require_exact(item, {"input_id", "source", "media_type"}, "input")
        source_doc = _object(item["source"], "input.source")
        _require_exact(
            source_doc,
            {"source_run_id", "source_artifact_id", "expected_digest"},
            "input.source",
        )
        digest_doc = _object(
            source_doc["expected_digest"],
            "input.source.expected_digest",
        )
        try:
            digest = decode_artifact_digest(digest_doc)
        except ValueError as exc:
            raise InvalidRequestError(
                "input source digest must use a lowercase SHA-256 digest."
            ) from exc
        inputs.append(
            LLMInputArtifact(
                item["input_id"],
                ArtifactSourceRef(
                    source_doc["source_run_id"],
                    source_doc["source_artifact_id"],
                    digest,
                ),
                item["media_type"],
            )
        )
    return LLMRequest(
        task_id=document["task_id"],
        prompt=document["prompt"],
        output=output,
        model=model,
        session=session,
        inputs=tuple(inputs),
    )


def resume_input_to_document(value: ResumeInput) -> dict[str, Any]:
    from .host import host_response_document

    return {
        "schema_version": RESUME_SCHEMA_VERSION,
        "resume_key": value.resume_key,
        "action": value.action.value,
        "host_response": (
            None
            if value.host_response is None
            else host_response_document(value.host_response)
        ),
        "candidate_digest": value.candidate_digest,
        "reason": value.reason,
    }


def decode_resume_input(document: Mapping[str, Any]) -> ResumeInput:
    _require_exact(
        document,
        {
            "schema_version",
            "resume_key",
            "action",
            "host_response",
            "candidate_digest",
            "reason",
        },
        "resume input",
    )
    if document["schema_version"] != RESUME_SCHEMA_VERSION:
        raise InvalidRequestError("Unsupported resume input schema_version.")
    try:
        action = ResumeAction(document["action"])
    except (TypeError, ValueError) as exc:
        raise InvalidRequestError("Unknown resume action.") from exc
    from .host import decode_host_response

    raw_host_response = document["host_response"]
    host_response = (
        None
        if raw_host_response is None
        else decode_host_response(_object(raw_host_response, "host_response"))
    )
    return ResumeInput(
        resume_key=document["resume_key"],
        action=action,
        host_response=host_response,
        candidate_digest=document["candidate_digest"],
        reason=document["reason"],
    )


def _object(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidRequestError(f"{field_name} must be an object.")
    return value


def _require_exact(document: Mapping[str, Any], fields: set[str], name: str) -> None:
    if not isinstance(document, Mapping):
        raise InvalidRequestError(f"{name} must be an object.")
    actual = set(document)
    if actual != fields:
        unknown = sorted(actual - fields)
        missing = sorted(fields - actual)
        raise InvalidRequestError(
            f"{name} fields do not match the closed schema.",
            details={"unknown": unknown, "missing": missing},
        )
