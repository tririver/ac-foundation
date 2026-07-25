from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from arc_jobs import InvalidRunIdError, JsonValue, validate_simple_id
from arc_llm import ArcLLMError, InteractiveJsonOutput, JsonOutput, OperationContract

from .models import (
    BATCH_SCHEMA_VERSION,
    BatchFailurePolicy,
    BatchRequest,
    ExecutionOptions,
    LoopSpec,
    ProposerFailurePolicy,
    REVIEW_SCHEMA_VERSION,
    Review,
)


@dataclass(frozen=True)
class RequestValidationError(ValueError):
    message: str
    path: tuple[str | int, ...] = ()

    def __str__(self) -> str:
        location = "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}" for part in self.path
        ).lstrip(".")
        return f"{location}: {self.message}" if location else self.message


def validate_batch_request(request: BatchRequest) -> None:
    if request.schema_version != BATCH_SCHEMA_VERSION:
        raise RequestValidationError(
            f"schema_version must be {BATCH_SCHEMA_VERSION}", ("schema_version",)
        )
    _valid_id(request.batch_id, ("batch_id",))
    if not request.loops:
        raise RequestValidationError("must contain at least one loop", ("loops",))
    if not isinstance(request.failure_policy, BatchFailurePolicy):
        raise RequestValidationError("unknown failure policy", ("failure_policy",))

    seen_loops: set[str] = set()
    for loop_index, loop in enumerate(request.loops):
        path = ("loops", loop_index)
        _validate_loop(loop, path)
        if loop.loop_id in seen_loops:
            raise RequestValidationError("duplicate loop_id", path + ("loop_id",))
        seen_loops.add(loop.loop_id)


def validate_execution_options(options: ExecutionOptions) -> None:
    _positive_int(
        options.max_concurrent_loops,
        ("max_concurrent_loops",),
    )
    _positive_int(
        options.max_concurrent_workers,
        ("max_concurrent_workers",),
    )
    if not isinstance(options.loop_interaction_resolvers, Mapping):
        raise RequestValidationError(
            "must be an object",
            ("loop_interaction_resolvers",),
        )
    for loop_id, resolver in options.loop_interaction_resolvers.items():
        _valid_id(loop_id, ("loop_interaction_resolvers", loop_id))
        if not callable(getattr(resolver, "resolve", None)):
            raise RequestValidationError(
                "resolver must provide resolve(request)",
                ("loop_interaction_resolvers", loop_id),
            )


def decode_review(
    value: JsonValue,
    *,
    active_proposer_ids: tuple[str, ...],
    validate_payload: Any,
) -> Review:
    if not isinstance(value, Mapping):
        raise RequestValidationError("review must be an object")
    expected_fields = {"schema_version", "action", "reason", "feedback", "payload"}
    _require_exact_fields(value, expected_fields, ())

    schema_version = value["schema_version"]
    if schema_version != REVIEW_SCHEMA_VERSION:
        raise RequestValidationError(
            f"schema_version must be {REVIEW_SCHEMA_VERSION}", ("schema_version",)
        )
    action = value["action"]
    if action not in {"continue", "stop"}:
        raise RequestValidationError(
            "must be 'continue' or 'stop'", ("action",)
        )
    reason = _nonempty_text(value["reason"], ("reason",))
    raw_feedback = value["feedback"]
    if not isinstance(raw_feedback, Mapping):
        raise RequestValidationError("must be an object", ("feedback",))
    if set(raw_feedback) != set(active_proposer_ids):
        raise RequestValidationError(
            "keys must exactly match successful active proposer IDs",
            ("feedback",),
        )
    feedback: dict[str, str] = {}
    for proposer_id in active_proposer_ids:
        feedback[proposer_id] = _nonempty_text(
            raw_feedback[proposer_id], ("feedback", proposer_id)
        )

    payload = value["payload"]
    validate_payload(payload)
    return Review(
        schema_version=REVIEW_SCHEMA_VERSION,
        action=action,
        reason=reason,
        feedback=feedback,
        payload=payload,
    )


def _validate_loop(loop: LoopSpec, path: tuple[str | int, ...]) -> None:
    _valid_id(loop.loop_id, path + ("loop_id",))
    if not isinstance(loop.context, Mapping):
        raise RequestValidationError("must be an object", path + ("context",))
    if not loop.proposers:
        raise RequestValidationError(
            "must contain at least one proposer", path + ("proposers",)
        )
    _positive_int(loop.max_rounds, path + ("max_rounds",))
    if type(loop.allow_early_stop) is not bool:
        raise RequestValidationError("must be a boolean", path + ("allow_early_stop",))
    if not isinstance(loop.on_proposer_failure, ProposerFailurePolicy):
        raise RequestValidationError(
            "unknown proposer failure policy", path + ("on_proposer_failure",)
        )

    seen_workers: set[str] = set()
    for worker_index, worker in enumerate((*loop.proposers, loop.reviewer)):
        worker_path = (
            path + ("proposers", worker_index)
            if worker_index < len(loop.proposers)
            else path + ("reviewer",)
        )
        _valid_id(worker.worker_id, worker_path + ("worker_id",))
        if worker.worker_id in seen_workers:
            raise RequestValidationError(
                "worker_id must be unique within the loop",
                worker_path + ("worker_id",),
            )
        seen_workers.add(worker.worker_id)
        _nonempty_text(worker.instructions, worker_path + ("instructions",))
        if not isinstance(worker.output_schema, Mapping):
            raise RequestValidationError(
                "must be a JSON Schema object",
                worker_path + ("output_schema",),
            )
        try:
            JsonOutput(worker.output_schema)
        except (ArcLLMError, TypeError, ValueError) as exc:
            raise RequestValidationError(
                str(exc), worker_path + ("output_schema",)
            ) from exc
        _positive_int(
            worker.max_interaction_turns,
            worker_path + ("max_interaction_turns",),
        )
        if not isinstance(worker.interaction_operations, Mapping):
            raise RequestValidationError(
                "must be an object",
                worker_path + ("interaction_operations",),
            )
        if not worker.interaction_operations:
            continue
        for name, contract in worker.interaction_operations.items():
            operation_path = worker_path + ("interaction_operations",)
            if not isinstance(name, str) or not name:
                raise RequestValidationError(
                    "operation names must be non-empty strings", operation_path
                )
            if not isinstance(contract, OperationContract):
                raise RequestValidationError(
                    "operation contracts must be OperationContract values",
                    operation_path + (name,),
                )
        try:
            InteractiveJsonOutput(
                result_schema=worker.output_schema,
                operations=dict(worker.interaction_operations),
                max_interaction_turns=worker.max_interaction_turns,
            )
        except (ArcLLMError, TypeError, ValueError) as exc:
            raise RequestValidationError(
                str(exc), worker_path + ("interaction_operations",)
            ) from exc


def _valid_id(value: object, path: tuple[str | int, ...]) -> None:
    try:
        validate_simple_id(value, label="identifier")  # type: ignore[arg-type]
    except (InvalidRunIdError, TypeError, ValueError) as exc:
        raise RequestValidationError(
            "must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}", path
        ) from exc


def _positive_int(value: object, path: tuple[str | int, ...]) -> int:
    if type(value) is not int or value < 1:
        raise RequestValidationError("must be an integer greater than or equal to 1", path)
    return value


def _nonempty_text(value: object, path: tuple[str | int, ...]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestValidationError("must be a non-empty string", path)
    return value


def _require_exact_fields(
    value: Mapping[str, object],
    expected: set[str],
    path: tuple[str | int, ...],
) -> None:
    missing = expected - set(value)
    if missing:
        field = sorted(missing)[0]
        raise RequestValidationError("required field is missing", path + (field,))
    unknown = set(value) - expected
    if unknown:
        field = sorted(unknown)[0]
        raise RequestValidationError("unknown field", path + (field,))
