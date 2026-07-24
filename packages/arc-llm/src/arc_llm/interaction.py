"""Operation-opaque interactive turn validation and response binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .errors import InvalidRequestError, OutputInvalidError
from .request import (
    InteractionRequest,
    InteractionResponse,
    InteractiveJsonOutput,
)

INTERACTIVE_TURN_SCHEMA_VERSION = "arc.llm.interactive_turn.v1"


@dataclass(frozen=True)
class InteractiveTurn:
    state: str
    result: Any | None
    requests: tuple[InteractionRequest, ...]

    def __post_init__(self) -> None:
        if self.state == "complete":
            if self.result is None or self.requests:
                raise OutputInvalidError("A complete interactive turn has result and no requests.")
        elif self.state == "interact":
            if self.result is not None or not self.requests:
                raise OutputInvalidError("An interaction turn has requests and no result.")
        else:
            raise OutputInvalidError("Unknown interactive turn state.")


def decode_interactive_turn(
    value: Any,
    contract: InteractiveJsonOutput,
    *,
    seen_request_ids: set[str] | None = None,
) -> InteractiveTurn:
    if not isinstance(value, Mapping):
        raise OutputInvalidError("Interactive output must be an object.")
    fields = {"schema_version", "state", "result", "requests"}
    if set(value) != fields:
        raise OutputInvalidError("Interactive output uses an invalid closed shape.")
    if value["schema_version"] != INTERACTIVE_TURN_SCHEMA_VERSION:
        raise OutputInvalidError("Unsupported interactive turn schema.")
    raw_requests = value["requests"]
    if not isinstance(raw_requests, list):
        raise OutputInvalidError("Interactive requests must be an array.")
    seen = set() if seen_request_ids is None else set(seen_request_ids)
    requests: list[InteractionRequest] = []
    for raw in raw_requests:
        if not isinstance(raw, Mapping) or set(raw) != {
            "request_id",
            "operation",
            "arguments",
        }:
            raise OutputInvalidError("Interaction request uses an invalid closed shape.")
        operation = raw["operation"]
        if operation not in contract.operations:
            raise OutputInvalidError("Unknown interaction operation.")
        request = InteractionRequest(
            request_id=raw["request_id"],
            operation=operation,
            arguments=raw["arguments"],
        )
        if request.request_id in seen:
            raise OutputInvalidError("Duplicate interaction request ID.")
        seen.add(request.request_id)
        schema = contract.operations[operation].arguments_schema
        if tuple(Draft202012Validator(schema).iter_errors(dict(request.arguments))):
            raise OutputInvalidError("Interaction arguments do not satisfy their contract.")
        requests.append(request)
    result = value["result"]
    if value["state"] == "complete":
        if tuple(Draft202012Validator(contract.result_schema).iter_errors(result)):
            raise OutputInvalidError("Interactive result does not satisfy its contract.")
    return InteractiveTurn(value["state"], result, tuple(requests))


def encode_interactive_turn(turn: InteractiveTurn) -> dict[str, Any]:
    return {
        "schema_version": INTERACTIVE_TURN_SCHEMA_VERSION,
        "state": turn.state,
        "result": turn.result,
        "requests": [
            {
                "request_id": request.request_id,
                "operation": request.operation,
                "arguments": dict(request.arguments),
            }
            for request in turn.requests
        ],
    }


def validate_responses(
    turn: InteractiveTurn,
    responses: tuple[InteractionResponse, ...],
    contract: InteractiveJsonOutput,
) -> tuple[InteractionResponse, ...]:
    expected = {item.request_id: item for item in turn.requests}
    actual: dict[str, InteractionResponse] = {}
    for response in responses:
        if response.request_id in actual:
            raise InvalidRequestError("Duplicate interaction response ID.")
        actual[response.request_id] = response
    if set(actual) != set(expected):
        raise InvalidRequestError("Responses must cover the pending request IDs exactly.")
    ordered = []
    for request in turn.requests:
        response = actual[request.request_id]
        if response.error is None:
            schema = contract.operations[request.operation].response_schema
            if tuple(Draft202012Validator(schema).iter_errors(response.result)):
                raise InvalidRequestError("Interaction response does not satisfy its contract.")
        ordered.append(response)
    return tuple(ordered)


def response_document(responses: tuple[InteractionResponse, ...]) -> dict[str, Any]:
    return {
        "schema_version": "arc.llm.interaction_response.v1",
        "responses": [
            {
                "request_id": response.request_id,
                "result": response.result,
                "error": None if response.error is None else dict(response.error),
            }
            for response in responses
        ],
    }
