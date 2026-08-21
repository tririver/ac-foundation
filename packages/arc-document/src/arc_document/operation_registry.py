"""Host-neutral typed operation contracts shared by ARC packages."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from jsonschema import Draft202012Validator


T = TypeVar("T")


class OperationEffect(str, Enum):
    NETWORK = "network"
    CACHE_WRITE = "cache_write"
    CACHE_ADMIN = "cache_admin"
    DESTRUCTIVE = "destructive"
    ARBITRARY_LOCAL_PATH = "arbitrary_local_path"
    RECURSIVE_LLM = "recursive_llm"


DEFAULT_EXCLUDED_EFFECTS = frozenset(
    {
        OperationEffect.CACHE_ADMIN,
        OperationEffect.DESTRUCTIVE,
        OperationEffect.ARBITRARY_LOCAL_PATH,
        OperationEffect.RECURSIVE_LLM,
    }
)


class OperationRequestError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class JsonCodec(Generic[T]):
    """Strict JSON-object decoder paired with its published schema."""

    schema_id: str
    schema: Mapping[str, Any]
    decoder: Callable[[Mapping[str, Any]], T]
    encoder: Callable[[T], Any]

    def decode(self, value: Mapping[str, Any]) -> T:
        errors = sorted(
            Draft202012Validator(dict(self.schema)).iter_errors(dict(value)),
            key=lambda error: [str(item) for item in error.absolute_path],
        )
        if errors:
            detail = "; ".join(error.message for error in errors[:5])
            raise OperationRequestError("invalid_parameters", detail)
        return self.decoder(value)

    def encode(self, value: T) -> Any:
        return self.encoder(value)


@dataclass(frozen=True)
class JsonOutputCodec(Generic[T]):
    """Typed result encoder paired with its published JSON schema."""

    schema_id: str
    schema: Mapping[str, Any]
    encoder: Callable[[T], Any]

    def encode(self, value: T) -> Any:
        encoded = self.encoder(value)
        errors = sorted(
            Draft202012Validator(dict(self.schema)).iter_errors(encoded),
            key=lambda error: [str(item) for item in error.absolute_path],
        )
        if errors:
            detail = "; ".join(error.message for error in errors[:5])
            raise OperationRequestError("invalid_result", detail)
        return encoded


@dataclass(frozen=True)
class OperationSpec(Generic[T]):
    operation_id: str
    version: int
    name: str
    input_codec: JsonCodec[Mapping[str, Any]]
    output_codec: JsonOutputCodec[T]
    callable: Callable[..., T]
    effect_flags: frozenset[OperationEffect] = frozenset()

    def invoke(self, parameters: Mapping[str, Any]) -> Any:
        decoded = self.input_codec.decode(parameters)
        return self.output_codec.encode(self.callable(**decoded))

    def document(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "version": self.version,
            "name": self.name,
            "input": {
                "schema_id": self.input_codec.schema_id,
                "schema": dict(self.input_codec.schema),
            },
            "output": {
                "schema_id": self.output_codec.schema_id,
                "schema": dict(self.output_codec.schema),
            },
            "effect_flags": sorted(flag.value for flag in self.effect_flags),
        }


def to_json_value(value: Any) -> Any:
    """Convert ARC value objects to JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return to_json_value(value.value)
    if is_dataclass(value):
        return {
            item.name: to_json_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    if hasattr(value, "to_document"):
        return to_json_value(value.to_document())
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def object_schema(
    properties: Mapping[str, Any],
    *,
    required: Sequence[str] = (),
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": dict(properties),
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


def operation_spec(
    *,
    package_name: str,
    schema_namespace: str,
    name: str,
    schema: Mapping[str, Any],
    callable: Callable[..., T],
    output_schema: Mapping[str, Any],
    effects: frozenset[OperationEffect] = frozenset(),
    version: int = 1,
    decoder: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> OperationSpec[T]:
    return OperationSpec(
        operation_id=f"{package_name}.{name}.v{version}",
        version=version,
        name=name,
        input_codec=JsonCodec(
            f"{schema_namespace}.{name}.parameters.v{version}",
            schema,
            decoder or (lambda value: dict(value)),
            to_json_value,
        ),
        output_codec=JsonOutputCodec(
            f"{schema_namespace}.{name}.result.v{version}",
            output_schema,
            to_json_value,
        ),
        callable=callable,
        effect_flags=effects,
    )


def registry_mapping(
    operations: Sequence[OperationSpec[Any]],
) -> Mapping[str, OperationSpec[Any]]:
    return MappingProxyType(
        {
            key: spec
            for spec in operations
            for key in (spec.operation_id, spec.name)
        }
    )


def resolve_operation_specs(
    operations: Sequence[OperationSpec[Any]],
    *,
    excluded_effects: frozenset[OperationEffect] = DEFAULT_EXCLUDED_EFFECTS,
) -> tuple[OperationSpec[Any], ...]:
    return tuple(
        spec
        for spec in operations
        if not spec.effect_flags.intersection(excluded_effects)
    )


def operation_registry_document(
    operations: Sequence[OperationSpec[Any]],
    *,
    schema_version: str,
    excluded_effects: frozenset[OperationEffect] = DEFAULT_EXCLUDED_EFFECTS,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "operations": [
            spec.document()
            for spec in resolve_operation_specs(
                operations, excluded_effects=excluded_effects
            )
        ],
    }


__all__ = [
    "DEFAULT_EXCLUDED_EFFECTS",
    "JsonCodec",
    "JsonOutputCodec",
    "OperationEffect",
    "OperationRequestError",
    "OperationSpec",
    "object_schema",
    "operation_registry_document",
    "operation_spec",
    "registry_mapping",
    "resolve_operation_specs",
    "to_json_value",
]
