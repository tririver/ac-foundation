from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from arc_jobs import JsonValue
from arc_llm import ArcLLMError, ModelSelection, OperationContract

from .identity import worker_contract_document
from .models import (
    BATCH_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    BatchFailurePolicy,
    BatchRequest,
    BatchResult,
    LoopResult,
    LoopSpec,
    LoopTermination,
    ProposerFailurePolicy,
    WorkerSpec,
)
from .validation import RequestValidationError, validate_batch_request


def decode_batch_request(document: Mapping[str, JsonValue]) -> BatchRequest:
    _exact(document, {"schema_version", "batch_id", "loops", "failure_policy"}, ())
    schema_version = _required_text(document, "schema_version", ())
    if schema_version != BATCH_SCHEMA_VERSION:
        raise RequestValidationError(
            f"schema_version must be {BATCH_SCHEMA_VERSION}", ("schema_version",)
        )
    batch_id = _required_text(document, "batch_id", ())
    raw_loops = document["loops"]
    if not isinstance(raw_loops, list):
        raise RequestValidationError("must be an array", ("loops",))
    loops = tuple(_decode_loop(value, ("loops", index)) for index, value in enumerate(raw_loops))
    failure_policy = _enum(
        BatchFailurePolicy,
        document["failure_policy"],
        ("failure_policy",),
    )
    request = BatchRequest(
        schema_version=BATCH_SCHEMA_VERSION,
        batch_id=batch_id,
        loops=loops,
        failure_policy=failure_policy,
    )
    validate_batch_request(request)
    return request


def encode_batch_request(request: BatchRequest) -> dict[str, JsonValue]:
    validate_batch_request(request)
    return {
        "schema_version": request.schema_version,
        "batch_id": request.batch_id,
        "loops": [_encode_loop(loop) for loop in request.loops],
        "failure_policy": request.failure_policy.value,
    }


def encode_batch_result(result: BatchResult) -> dict[str, JsonValue]:
    if result.schema_version != RESULT_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {RESULT_SCHEMA_VERSION}")
    return {
        "schema_version": result.schema_version,
        "loops": [_encode_loop_result(loop) for loop in result.loops],
    }


def decode_batch_result(document: Mapping[str, JsonValue]) -> BatchResult:
    _exact(document, {"schema_version", "loops"}, ())
    if document["schema_version"] != RESULT_SCHEMA_VERSION:
        raise RequestValidationError(
            f"schema_version must be {RESULT_SCHEMA_VERSION}", ("schema_version",)
        )
    raw_loops = document["loops"]
    if not isinstance(raw_loops, list):
        raise RequestValidationError("must be an array", ("loops",))
    return BatchResult(
        schema_version=RESULT_SCHEMA_VERSION,
        loops=tuple(
            _decode_loop_result(value, ("loops", index))
            for index, value in enumerate(raw_loops)
        ),
    )


def _decode_loop(value: JsonValue, path: tuple[str | int, ...]) -> LoopSpec:
    document = _object(value, path)
    _exact(
        document,
        {
            "loop_id",
            "context",
            "proposers",
            "reviewer",
            "max_rounds",
            "allow_early_stop",
            "on_proposer_failure",
        },
        path,
    )
    raw_context = document["context"]
    if not isinstance(raw_context, Mapping):
        raise RequestValidationError("must be an object", path + ("context",))
    raw_proposers = document["proposers"]
    if not isinstance(raw_proposers, list):
        raise RequestValidationError("must be an array", path + ("proposers",))
    allow_early_stop = document["allow_early_stop"]
    if type(allow_early_stop) is not bool:
        raise RequestValidationError("must be a boolean", path + ("allow_early_stop",))
    max_rounds = document["max_rounds"]
    if type(max_rounds) is not int:
        raise RequestValidationError("must be an integer", path + ("max_rounds",))
    return LoopSpec(
        loop_id=_required_text(document, "loop_id", path),
        context=dict(raw_context),
        proposers=tuple(
            _decode_worker(item, path + ("proposers", index))
            for index, item in enumerate(raw_proposers)
        ),
        reviewer=_decode_worker(document["reviewer"], path + ("reviewer",)),
        max_rounds=max_rounds,
        allow_early_stop=allow_early_stop,
        on_proposer_failure=_enum(
            ProposerFailurePolicy,
            document["on_proposer_failure"],
            path + ("on_proposer_failure",),
        ),
    )


def _decode_worker(value: JsonValue, path: tuple[str | int, ...]) -> WorkerSpec:
    document = _object(value, path)
    _exact(
        document,
        {
            "worker_id",
            "instructions",
            "output_schema",
            "model",
            "interaction_operations",
        },
        path,
    )
    raw_schema = document["output_schema"]
    if not isinstance(raw_schema, Mapping):
        raise RequestValidationError("must be an object", path + ("output_schema",))
    raw_model = _object(document["model"], path + ("model",))
    _exact(raw_model, {"provider", "model", "tier"}, path + ("model",))
    raw_exact_model = raw_model["model"]
    if raw_exact_model is not None and not isinstance(raw_exact_model, str):
        raise RequestValidationError("must be a string or null", path + ("model", "model"))
    raw_operations = document["interaction_operations"]
    if not isinstance(raw_operations, Mapping):
        raise RequestValidationError(
            "must be an object", path + ("interaction_operations",)
        )
    interaction_operations: dict[str, OperationContract] = {}
    for name, raw_operation in raw_operations.items():
        if not isinstance(name, str) or not name:
            raise RequestValidationError(
                "operation names must be non-empty strings",
                path + ("interaction_operations",),
            )
        operation_path = path + ("interaction_operations", name)
        operation = _object(raw_operation, operation_path)
        _exact(
            operation,
            {"arguments_schema", "response_schema"},
            operation_path,
        )
        raw_arguments = operation["arguments_schema"]
        raw_response = operation["response_schema"]
        if not isinstance(raw_arguments, Mapping):
            raise RequestValidationError(
                "must be an object", operation_path + ("arguments_schema",)
            )
        if not isinstance(raw_response, Mapping):
            raise RequestValidationError(
                "must be an object", operation_path + ("response_schema",)
            )
        try:
            interaction_operations[name] = OperationContract(
                dict(raw_arguments), dict(raw_response)
            )
        except (ArcLLMError, TypeError, ValueError) as exc:
            raise RequestValidationError(str(exc), operation_path) from exc
    return WorkerSpec(
        worker_id=_required_text(document, "worker_id", path),
        instructions=_required_text(document, "instructions", path),
        output_schema=dict(raw_schema),
        model=ModelSelection(
            provider=_required_text(raw_model, "provider", path + ("model",)),
            model=raw_exact_model,
            tier=cast(Any, _required_text(raw_model, "tier", path + ("model",))),
        ),
        interaction_operations=interaction_operations,
    )


def _encode_loop(loop: LoopSpec) -> dict[str, JsonValue]:
    return {
        "loop_id": loop.loop_id,
        "context": dict(loop.context),
        "proposers": [_encode_worker(worker) for worker in loop.proposers],
        "reviewer": _encode_worker(loop.reviewer),
        "max_rounds": loop.max_rounds,
        "allow_early_stop": loop.allow_early_stop,
        "on_proposer_failure": loop.on_proposer_failure.value,
    }


def _encode_worker(worker: WorkerSpec) -> dict[str, JsonValue]:
    return worker_contract_document(worker)


def _encode_loop_result(result: LoopResult) -> dict[str, JsonValue]:
    error: JsonValue = None
    if result.error is not None:
        error = {
            "code": result.error.code,
            "message": result.error.message,
            "details": dict(result.error.details),
        }
    return {
        "loop_id": result.loop_id,
        "termination": result.termination.value,
        "rounds_completed": result.rounds_completed,
        "final_proposals": dict(result.final_proposals),
        "final_review": result.final_review,
        "error": error,
    }


def _decode_loop_result(
    value: JsonValue, path: tuple[str | int, ...]
) -> LoopResult:
    from arc_jobs import RunError

    document = _object(value, path)
    _exact(
        document,
        {
            "loop_id",
            "termination",
            "rounds_completed",
            "final_proposals",
            "final_review",
            "error",
        },
        path,
    )
    proposals = document["final_proposals"]
    if not isinstance(proposals, Mapping):
        raise RequestValidationError("must be an object", path + ("final_proposals",))
    raw_rounds = document["rounds_completed"]
    if type(raw_rounds) is not int or raw_rounds < 0:
        raise RequestValidationError(
            "must be a non-negative integer", path + ("rounds_completed",)
        )
    raw_error = document["error"]
    error = None
    if raw_error is not None:
        error_document = _object(raw_error, path + ("error",))
        _exact(error_document, {"code", "message", "details"}, path + ("error",))
        details = error_document["details"]
        if not isinstance(details, Mapping):
            raise RequestValidationError("must be an object", path + ("error", "details"))
        error = RunError(
            code=_required_text(error_document, "code", path + ("error",)),
            message=_required_text(error_document, "message", path + ("error",)),
            details=dict(details),
        )
    return LoopResult(
        loop_id=_required_text(document, "loop_id", path),
        termination=_enum(
            LoopTermination, document["termination"], path + ("termination",)
        ),
        rounds_completed=raw_rounds,
        final_proposals=dict(proposals),
        final_review=document["final_review"],
        error=error,
    )


def _object(value: JsonValue, path: tuple[str | int, ...]) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise RequestValidationError("must be an object", path)
    return value


def _exact(
    document: Mapping[str, object],
    fields: set[str],
    path: tuple[str | int, ...],
) -> None:
    missing = fields - set(document)
    if missing:
        field = sorted(missing)[0]
        raise RequestValidationError("required field is missing", path + (field,))
    unknown = set(document) - fields
    if unknown:
        field = sorted(unknown)[0]
        raise RequestValidationError("unknown field", path + (field,))


def _required_text(
    document: Mapping[str, object],
    field: str,
    path: tuple[str | int, ...],
) -> str:
    value = document[field]
    if not isinstance(value, str) or not value:
        raise RequestValidationError("must be a non-empty string", path + (field,))
    return value


def _strict_bool(value: object, path: tuple[str | int, ...]) -> bool:
    if type(value) is not bool:
        raise RequestValidationError("must be a boolean", path)
    return value


def _enum(enum_type: type[Any], value: object, path: tuple[str | int, ...]) -> Any:
    if not isinstance(value, str):
        raise RequestValidationError("must be a string", path)
    try:
        return enum_type(value)
    except ValueError as exc:
        raise RequestValidationError("unknown enum value", path) from exc
