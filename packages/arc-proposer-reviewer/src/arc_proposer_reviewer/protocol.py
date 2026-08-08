from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from arc_jobs import ArtifactSourceRef, JsonValue, decode_artifact_digest, encode_artifact_digest
from arc_llm import LLMInputArtifact, ModelSelection

from .identity import worker_contract_document
from .models import (
    BATCH_SCHEMA_VERSION,
    LEGACY_BATCH_SCHEMA_VERSION_V6,
    LEGACY_BATCH_SCHEMA_VERSION_V4,
    LEGACY_BATCH_SCHEMA_VERSION_V5,
    RESULT_SCHEMA_VERSION,
    BatchFailurePolicy,
    BatchRequest,
    BatchResult,
    LoopResult,
    LoopSpec,
    LoopTermination,
    ProposerFailurePolicy,
    RevisionContextMode,
    WorkerSpec,
)
from .validation import RequestValidationError, validate_batch_request


def decode_batch_request(document: Mapping[str, JsonValue]) -> BatchRequest:
    _exact(document, {"schema_version", "batch_id", "loops", "inputs", "failure_policy"}, ())
    schema_version = _required_text(document, "schema_version", ())
    if schema_version not in {
        BATCH_SCHEMA_VERSION,
        LEGACY_BATCH_SCHEMA_VERSION_V6,
        LEGACY_BATCH_SCHEMA_VERSION_V5,
        LEGACY_BATCH_SCHEMA_VERSION_V4,
    }:
        raise RequestValidationError(
            (
                "schema_version must be "
                f"{BATCH_SCHEMA_VERSION}, {LEGACY_BATCH_SCHEMA_VERSION_V6}, "
                f"{LEGACY_BATCH_SCHEMA_VERSION_V5}, or "
                f"{LEGACY_BATCH_SCHEMA_VERSION_V4}"
            ),
            ("schema_version",),
        )
    batch_id = _required_text(document, "batch_id", ())
    raw_loops = document["loops"]
    if not isinstance(raw_loops, list):
        raise RequestValidationError("must be an array", ("loops",))
    raw_inputs = document["inputs"]
    if not isinstance(raw_inputs, list):
        raise RequestValidationError("must be an array", ("inputs",))
    loops = tuple(
        _decode_loop(
            value,
            ("loops", index),
            schema_version=schema_version,
        )
        for index, value in enumerate(raw_loops)
    )
    failure_policy = _enum(
        BatchFailurePolicy,
        document["failure_policy"],
        ("failure_policy",),
    )
    request = BatchRequest(
        schema_version=BATCH_SCHEMA_VERSION,
        batch_id=batch_id,
        loops=loops,
        inputs=tuple(
            _decode_input(value, ("inputs", index))
            for index, value in enumerate(raw_inputs)
        ),
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
        "inputs": [_encode_input(item) for item in request.inputs],
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


def _decode_loop(
    value: JsonValue,
    path: tuple[str | int, ...],
    *,
    schema_version: str,
) -> LoopSpec:
    document = _object(value, path)
    fields = {
        "loop_id",
        "context",
        "proposers",
        "reviewer",
        "max_rounds",
        "allow_early_stop",
        "on_proposer_failure",
    }
    if schema_version != LEGACY_BATCH_SCHEMA_VERSION_V4:
        fields.add("review_final_round")
    if schema_version in {BATCH_SCHEMA_VERSION, LEGACY_BATCH_SCHEMA_VERSION_V6}:
        fields.add("revision_context_mode")
    if schema_version == BATCH_SCHEMA_VERSION:
        fields.add("input_ids")
    _exact(document, fields, path)
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
    review_final_round = (
        True
        if schema_version == LEGACY_BATCH_SCHEMA_VERSION_V4
        else document["review_final_round"]
    )
    if type(review_final_round) is not bool:
        raise RequestValidationError(
            "must be a boolean", path + ("review_final_round",)
        )
    revision_context_mode = (
        RevisionContextMode.FEEDBACK_ONLY
        if schema_version not in {BATCH_SCHEMA_VERSION, LEGACY_BATCH_SCHEMA_VERSION_V6}
        else _enum(
            RevisionContextMode,
            document["revision_context_mode"],
            path + ("revision_context_mode",),
        )
    )
    raw_input_ids = document.get("input_ids")
    if raw_input_ids is not None and (
        not isinstance(raw_input_ids, list)
        or any(not isinstance(item, str) for item in raw_input_ids)
    ):
        raise RequestValidationError(
            "must be an array of strings or null", path + ("input_ids",)
        )
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
        review_final_round=review_final_round,
        revision_context_mode=revision_context_mode,
        input_ids=(
            None if raw_input_ids is None else tuple(raw_input_ids)
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
    return WorkerSpec(
        worker_id=_required_text(document, "worker_id", path),
        instructions=_required_text(document, "instructions", path),
        output_schema=dict(raw_schema),
        model=ModelSelection(
            provider=_required_text(raw_model, "provider", path + ("model",)),
            model=raw_exact_model,
            tier=cast(Any, _required_text(raw_model, "tier", path + ("model",))),
        ),
    )


def _decode_input(value: JsonValue, path: tuple[str | int, ...]) -> LLMInputArtifact:
    document = _object(value, path)
    _exact(document, {"input_id", "source", "media_type"}, path)
    source_document = _object(document["source"], path + ("source",))
    _exact(
        source_document,
        {"source_run_id", "source_artifact_id", "expected_digest"},
        path + ("source",),
    )
    try:
        digest = decode_artifact_digest(source_document["expected_digest"])
        return LLMInputArtifact(
            _required_text(document, "input_id", path),
            ArtifactSourceRef(
                _required_text(source_document, "source_run_id", path + ("source",)),
                _required_text(source_document, "source_artifact_id", path + ("source",)),
                digest,
            ),
            _required_text(document, "media_type", path),
        )
    except Exception as exc:
        raise RequestValidationError(str(exc), path) from exc


def _encode_loop(loop: LoopSpec) -> dict[str, JsonValue]:
    return {
        "loop_id": loop.loop_id,
        "context": dict(loop.context),
        "proposers": [_encode_worker(worker) for worker in loop.proposers],
        "reviewer": _encode_worker(loop.reviewer),
        "max_rounds": loop.max_rounds,
        "allow_early_stop": loop.allow_early_stop,
        "on_proposer_failure": loop.on_proposer_failure.value,
        "review_final_round": loop.review_final_round,
        "revision_context_mode": loop.revision_context_mode.value,
        "input_ids": None if loop.input_ids is None else list(loop.input_ids),
    }


def _encode_worker(worker: WorkerSpec) -> dict[str, JsonValue]:
    return worker_contract_document(worker)


def _encode_input(item: LLMInputArtifact) -> dict[str, JsonValue]:
    return {
        "input_id": item.input_id,
        "source": {
            "source_run_id": item.source.source_run_id,
            "source_artifact_id": item.source.source_artifact_id,
            "expected_digest": encode_artifact_digest(item.source.expected_digest),
        },
        "media_type": item.media_type,
    }


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
