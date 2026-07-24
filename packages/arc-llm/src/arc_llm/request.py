"""Immutable requests and strict versioned codecs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, TypeAlias

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .errors import InvalidRequestError, InvalidSchemaError

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
ModelTier: TypeAlias = Literal["low", "medium", "high", "xhigh"]

REQUEST_SCHEMA_VERSION = "arc.llm.request.v1"
RESUME_SCHEMA_VERSION = "arc.llm.resume_input.v1"


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
class CapabilityPolicy:
    internet: bool = False
    inherit_host_config: bool = False
    allowed_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.internet, bool) or not isinstance(self.inherit_host_config, bool):
            raise InvalidRequestError("Capability booleans must be JSON booleans.")
        if isinstance(self.allowed_tools, (str, bytes)):
            raise InvalidRequestError("allowed_tools must be a sequence of tool names.")
        try:
            values = tuple(self.allowed_tools)
        except TypeError as exc:
            raise InvalidRequestError(
                "allowed_tools must be a sequence of tool names."
            ) from exc
        if any(not isinstance(item, str) or not item for item in values):
            raise InvalidRequestError("allowed_tools entries must be non-empty strings.")
        normalized = tuple(sorted(set(values)))
        object.__setattr__(self, "allowed_tools", normalized)


@dataclass(frozen=True)
class ExecutionLimits:
    idle_timeout_seconds: float = 1800.0
    safe_retry_limit: int = 1
    native_resume_limit: int = 1
    automatic_replacement_limit: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.idle_timeout_seconds, bool)
            or not isinstance(self.idle_timeout_seconds, (int, float))
            or self.idle_timeout_seconds <= 0
        ):
            raise InvalidRequestError("idle_timeout_seconds must be positive.")
        for name in (
            "safe_retry_limit",
            "native_resume_limit",
            "automatic_replacement_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InvalidRequestError(f"{name} must be a non-negative integer.")


class InteractionResolver(Protocol):
    def resolve(self, request: "InteractionRequest") -> "InteractionResponse": ...


@dataclass(frozen=True)
class ProviderGateOptions:
    enabled: bool = True
    global_limit: int = 24
    provider_limits: Mapping[str, int] = field(default_factory=dict)
    circuit_failure_threshold: int = 3
    circuit_cooldown_seconds: float = 900.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise InvalidRequestError("gate.enabled must be a boolean.")
        for name in ("global_limit", "circuit_failure_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 24
            ):
                raise InvalidRequestError(f"gate.{name} must be between 1 and 24.")
        if (
            isinstance(self.circuit_cooldown_seconds, bool)
            or not isinstance(self.circuit_cooldown_seconds, (int, float))
            or self.circuit_cooldown_seconds <= 0
        ):
            raise InvalidRequestError(
                "gate.circuit_cooldown_seconds must be positive."
            )
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
    interaction_resolver: InteractionResolver | None = None
    gate: ProviderGateOptions = field(default_factory=ProviderGateOptions)


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
class TextOutput:
    kind: Literal["text"] = "text"


@dataclass(frozen=True)
class JsonOutput:
    schema: Mapping[str, Any]
    repair: Literal["strict", "local"] = "local"
    kind: Literal["json"] = "json"

    def __post_init__(self) -> None:
        if self.repair not in {"strict", "local"}:
            raise InvalidRequestError("output.repair must be strict or local.")
        _check_schema(self.schema)
        object.__setattr__(self, "schema", _frozen_mapping(self.schema))


@dataclass(frozen=True)
class OperationContract:
    arguments_schema: Mapping[str, Any]
    response_schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        _check_schema(self.arguments_schema)
        _check_schema(self.response_schema)
        object.__setattr__(self, "arguments_schema", _frozen_mapping(self.arguments_schema))
        object.__setattr__(self, "response_schema", _frozen_mapping(self.response_schema))


@dataclass(frozen=True)
class InteractiveJsonOutput:
    result_schema: Mapping[str, Any]
    operations: Mapping[str, OperationContract]
    max_interaction_turns: int = 3
    kind: Literal["interactive_json"] = "interactive_json"

    def __post_init__(self) -> None:
        _check_schema(self.result_schema)
        if (
            isinstance(self.max_interaction_turns, bool)
            or not isinstance(self.max_interaction_turns, int)
            or self.max_interaction_turns < 1
        ):
            raise InvalidRequestError("max_interaction_turns must be a positive integer.")
        if any(not isinstance(name, str) or not name for name in self.operations):
            raise InvalidRequestError("Operation names must be non-empty strings.")
        if any(
            not isinstance(contract, OperationContract)
            for contract in self.operations.values()
        ):
            raise InvalidRequestError(
                "Every operation must contain an OperationContract."
            )
        object.__setattr__(self, "result_schema", _frozen_mapping(self.result_schema))
        object.__setattr__(self, "operations", MappingProxyType(dict(self.operations)))


OutputContract: TypeAlias = TextOutput | JsonOutput | InteractiveJsonOutput


@dataclass(frozen=True)
class LLMRequest:
    task_id: str
    prompt: str
    output: OutputContract
    model: ModelSelection = field(default_factory=ModelSelection)
    session: SessionRef | None = None
    capabilities: CapabilityPolicy = field(default_factory=CapabilityPolicy)

    def __post_init__(self) -> None:
        _validate_identifier(self.task_id, field_name="task_id")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise InvalidRequestError("prompt must be a non-empty string.")
        if not isinstance(self.output, (TextOutput, JsonOutput, InteractiveJsonOutput)):
            raise InvalidRequestError("output must be a supported output contract.")


class ResumeAction(StrEnum):
    CONTINUE = "continue"
    REPLACE = "replace"
    ACCEPT_CANDIDATE = "accept_candidate"
    CANCEL = "cancel"


@dataclass(frozen=True)
class InteractionRequest:
    request_id: str
    operation: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validate_identifier(self.request_id, field_name="request_id")
        if not self.operation:
            raise InvalidRequestError("operation is required.")
        object.__setattr__(self, "arguments", _frozen_mapping(self.arguments))


@dataclass(frozen=True)
class InteractionResponse:
    request_id: str
    result: JsonValue | None = None
    error: Mapping[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.request_id, field_name="request_id")
        if (self.result is None) == (self.error is None):
            raise InvalidRequestError("An interaction response has exactly one of result or error.")
        if self.error is not None:
            object.__setattr__(self, "error", MappingProxyType(dict(self.error)))


@dataclass(frozen=True)
class ResumeInput:
    resume_key: str
    action: ResumeAction
    responses: tuple[InteractionResponse, ...] = ()
    candidate_digest: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.resume_key, field_name="resume_key")
        if not isinstance(self.action, ResumeAction):
            raise InvalidRequestError("action must be a ResumeAction.")
        if self.action is ResumeAction.CONTINUE:
            if self.candidate_digest is not None or self.reason is not None:
                raise InvalidRequestError("continue accepts only interaction responses.")
        elif self.action is ResumeAction.REPLACE:
            if not self.reason or self.responses or self.candidate_digest is not None:
                raise InvalidRequestError("replace requires only a non-empty reason.")
        elif self.action is ResumeAction.ACCEPT_CANDIDATE:
            if not self.candidate_digest or self.responses or self.reason is not None:
                raise InvalidRequestError("accept_candidate requires only candidate_digest.")
            _validate_sha256(
                self.candidate_digest,
                field_name="candidate_digest",
            )
        elif self.action is ResumeAction.CANCEL:
            if self.responses or self.candidate_digest is not None or self.reason is not None:
                raise InvalidRequestError("cancel accepts no action payload.")


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
    if isinstance(contract, JsonOutput):
        return {"kind": "json", "schema": dict(contract.schema), "repair": contract.repair}
    return {
        "kind": "interactive_json",
        "result_schema": dict(contract.result_schema),
        "operations": {
            name: {
                "arguments_schema": dict(item.arguments_schema),
                "response_schema": dict(item.response_schema),
            }
            for name, item in sorted(contract.operations.items())
        },
        "max_interaction_turns": contract.max_interaction_turns,
    }


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
        "capabilities": {
            "internet": request.capabilities.internet,
            "inherit_host_config": request.capabilities.inherit_host_config,
            "allowed_tools": list(request.capabilities.allowed_tools),
        },
    }


def decode_request(document: Mapping[str, Any]) -> LLMRequest:
    _require_exact(
        document,
        {"schema_version", "task_id", "prompt", "output", "model", "session", "capabilities"},
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
    elif kind == "interactive_json":
        _require_exact(
            output_doc,
            {"kind", "result_schema", "operations", "max_interaction_turns"},
            "output",
        )
        operations_doc = _object(output_doc["operations"], "output.operations")
        operations: dict[str, OperationContract] = {}
        for name, raw in operations_doc.items():
            item = _object(raw, f"output.operations.{name}")
            _require_exact(item, {"arguments_schema", "response_schema"}, f"operation {name}")
            operations[name] = OperationContract(
                _object(item["arguments_schema"], "arguments_schema"),
                _object(item["response_schema"], "response_schema"),
            )
        output = InteractiveJsonOutput(
            _object(output_doc["result_schema"], "output.result_schema"),
            operations,
            output_doc["max_interaction_turns"],
        )
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
    capabilities_doc = _object(document["capabilities"], "capabilities")
    _require_exact(
        capabilities_doc,
        {"internet", "inherit_host_config", "allowed_tools"},
        "capabilities",
    )
    if not isinstance(capabilities_doc["allowed_tools"], list):
        raise InvalidRequestError("capabilities.allowed_tools must be an array.")
    return LLMRequest(
        task_id=document["task_id"],
        prompt=document["prompt"],
        output=output,
        model=model,
        session=session,
        capabilities=CapabilityPolicy(
            capabilities_doc["internet"],
            capabilities_doc["inherit_host_config"],
            tuple(capabilities_doc["allowed_tools"]),
        ),
    )


def resume_input_to_document(value: ResumeInput) -> dict[str, Any]:
    return {
        "schema_version": RESUME_SCHEMA_VERSION,
        "resume_key": value.resume_key,
        "action": value.action.value,
        "responses": [
            {
                "request_id": response.request_id,
                "result": response.result,
                "error": None if response.error is None else dict(response.error),
            }
            for response in value.responses
        ],
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
            "responses",
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
    raw_responses = document["responses"]
    if not isinstance(raw_responses, list):
        raise InvalidRequestError("responses must be an array.")
    responses = []
    for raw in raw_responses:
        item = _object(raw, "interaction response")
        _require_exact(item, {"request_id", "result", "error"}, "interaction response")
        responses.append(InteractionResponse(item["request_id"], item["result"], item["error"]))
    return ResumeInput(
        resume_key=document["resume_key"],
        action=action,
        responses=tuple(responses),
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
