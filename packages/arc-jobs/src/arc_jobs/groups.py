from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Mapping

from .artifacts import encode_artifact_ref
from .errors import CorruptStateError, ResumeMismatchError, StoppedError, UnsupportedSchemaError
from .events import EventWriter
from .identity import semantic_key, validate_simple_id
from .models import (
    ArtifactRef,
    FailureMode,
    GroupExecutionResult,
    GroupResult,
    GroupUnitView,
    GroupView,
    JsonValue,
    RunError,
    Paused,
    UnitResult,
    WorkUnit,
)
from .storage import atomic_write_json, read_json_object, require_fields
from .stopping import StopToken


def _artifact_value(ref: ArtifactRef) -> dict[str, JsonValue]:
    return encode_artifact_ref(ref)


def _unit_result(
    document: Mapping[str, JsonValue],
    *,
    expected_unit_id: str,
    expected_semantic_key: str,
    replayed: bool,
) -> UnitResult:
    require_fields(
        document,
        required={
            "schema_version",
            "unit_id",
            "semantic_key_sha256",
            "status",
            "value",
            "error",
        },
    )
    if document["schema_version"] != "arc.jobs.group_unit.v2":
        raise UnsupportedSchemaError(str(document["schema_version"]))
    unit_id, semantic_key_sha256, status, value, error_json = (
        document["unit_id"],
        document["semantic_key_sha256"],
        document["status"],
        document["value"],
        document["error"],
    )
    if (
        unit_id != expected_unit_id
        or semantic_key_sha256 != expected_semantic_key
        or status not in {"succeeded", "failed"}
    ):
        raise CorruptStateError("invalid group unit document")
    error = None
    if error_json is not None:
        if not isinstance(error_json, dict):
            raise CorruptStateError("invalid group unit error")
        require_fields(error_json, required={"code", "message", "details"})
        code, message, details = (
            error_json["code"],
            error_json["message"],
            error_json["details"],
        )
        if (
            not isinstance(code, str)
            or not isinstance(message, str)
            or not isinstance(details, dict)
        ):
            raise CorruptStateError("invalid group unit error fields")
        error = RunError(code, message, details)
    return UnitResult(unit_id, status, value, error, replayed)


def inspect_group(directory: Path, group_id: str) -> GroupView:
    """Return a read-only projection of one durable work group."""

    validate_simple_id(group_id, label="group id")
    group_directory = directory / group_id
    document = read_json_object(group_directory / "state.json")
    require_fields(
        document,
        required={"schema_version", "group_id", "units"},
    )
    if document["schema_version"] != "arc.jobs.group.v2":
        raise UnsupportedSchemaError(str(document["schema_version"]))
    if document["group_id"] != group_id or not isinstance(document["units"], list):
        raise CorruptStateError("invalid group document")

    views: list[GroupUnitView] = []
    unit_ids: set[str] = set()
    for item in document["units"]:
        if not isinstance(item, dict):
            raise CorruptStateError("invalid group unit descriptor")
        require_fields(item, required={"unit_id", "semantic_key_sha256"})
        unit_id = item["unit_id"]
        semantic_key_sha256 = item["semantic_key_sha256"]
        if (
            not isinstance(unit_id, str)
            or unit_id in unit_ids
            or not isinstance(semantic_key_sha256, str)
            or len(semantic_key_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in semantic_key_sha256
            )
        ):
            raise CorruptStateError("invalid group unit descriptor")
        validate_simple_id(unit_id, label="unit id")
        unit_ids.add(unit_id)
        path = group_directory / "units" / f"{unit_id}.json"
        if not path.exists():
            views.append(GroupUnitView(unit_id, "pending"))
            continue
        result = _unit_result(
            read_json_object(path),
            expected_unit_id=unit_id,
            expected_semantic_key=semantic_key_sha256,
            replayed=False,
        )
        views.append(
            GroupUnitView(
                result.unit_id,
                result.status,
                result.value,
                result.error,
            )
        )
    return GroupView(group_id, tuple(views))


class WorkGroupRunner:
    def __init__(
        self,
        directory: Path,
        *,
        stop: StopToken,
        events: EventWriter,
        checkpoint: Callable[[], None],
    ):
        self.directory = directory
        self.stop = stop
        self.events = events
        self.checkpoint = checkpoint

    def run(
        self,
        group_id: str,
        units: tuple[WorkUnit, ...],
        worker: Callable[
            [WorkUnit], JsonValue | ArtifactRef | UnitResult | Paused
        ],
        *,
        max_workers: int,
        failure_mode: FailureMode,
    ) -> GroupExecutionResult:
        validate_simple_id(group_id, label="group id")
        if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        unit_ids: set[str] = set()
        keys: dict[str, str] = {}
        for unit in units:
            validate_simple_id(unit.unit_id, label="unit id")
            if unit.unit_id in unit_ids:
                raise ValueError(f"duplicate unit id: {unit.unit_id}")
            unit_ids.add(unit.unit_id)
            keys[unit.unit_id] = semantic_key(unit.semantic_input).sha256
        group_directory = self.directory / group_id
        state_path = group_directory / "state.json"
        state: dict[str, JsonValue] = {
            "schema_version": "arc.jobs.group.v2",
            "group_id": group_id,
            "units": [
                {"unit_id": unit.unit_id, "semantic_key_sha256": keys[unit.unit_id]}
                for unit in units
            ],
        }
        if state_path.exists():
            existing_state = read_json_object(state_path)
            if existing_state.get("schema_version") != "arc.jobs.group.v2":
                raise UnsupportedSchemaError(
                    str(existing_state.get("schema_version"))
                )
            if existing_state != state:
                raise ResumeMismatchError(f"group {group_id!r} unit set changed")
        else:
            atomic_write_json(state_path, state, exclusive=True)

        results: dict[str, UnitResult] = {}
        pending: deque[WorkUnit] = deque()
        for unit in units:
            path = group_directory / "units" / f"{unit.unit_id}.json"
            if path.exists():
                document = read_json_object(path)
                if document.get("semantic_key_sha256") != keys[unit.unit_id]:
                    raise ResumeMismatchError(f"unit {unit.unit_id!r} semantic input changed")
                results[unit.unit_id] = _unit_result(
                    document,
                    expected_unit_id=unit.unit_id,
                    expected_semantic_key=keys[unit.unit_id],
                    replayed=True,
                )
            else:
                pending.append(unit)

        def invoke(unit: WorkUnit) -> UnitResult | Paused:
            try:
                self.stop.raise_if_requested()
                value = worker(unit)
                if isinstance(value, Paused):
                    return value
                if isinstance(value, UnitResult):
                    if value.unit_id != unit.unit_id:
                        raise ValueError("worker UnitResult.unit_id does not match its unit")
                    result = UnitResult(
                        value.unit_id,
                        value.status,
                        value.value,
                        value.error,
                        False,
                    )
                elif isinstance(value, ArtifactRef):
                    result = UnitResult(unit.unit_id, "succeeded", _artifact_value(value))
                else:
                    result = UnitResult(unit.unit_id, "succeeded", value)
            except StoppedError:
                # A stop is a run-level pause, not a durable failed unit.
                raise
            except Exception as exc:
                result = UnitResult(
                    unit.unit_id,
                    "failed",
                    error=RunError(
                        "worker_unhandled_exception",
                        f"{type(exc).__name__}: {str(exc)[:300]}",
                    ),
                )
            if result.status not in {"succeeded", "failed"}:
                raise ValueError("worker UnitResult.status is invalid")
            error_json = (
                {
                    "code": result.error.code,
                    "message": result.error.message,
                    "details": dict(result.error.details),
                }
                if result.error
                else None
            )
            document: dict[str, JsonValue] = {
                "schema_version": "arc.jobs.group_unit.v2",
                "unit_id": unit.unit_id,
                "semantic_key_sha256": keys[unit.unit_id],
                "status": result.status,
                "value": result.value,
                "error": error_json,
            }
            atomic_write_json(
                group_directory / "units" / f"{unit.unit_id}.json",
                document,
                exclusive=True,
            )
            self.events.emit(
                "group_unit_finished",
                {"group_id": group_id, "unit_id": unit.unit_id, "status": result.status},
            )
            return result

        stop_submitting = (
            failure_mode is FailureMode.FAIL_FAST
            and any(result.status != "succeeded" for result in results.values())
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            in_flight: dict[Future[UnitResult | Paused], WorkUnit] = {}
            pauses: dict[str, Paused] = {}
            while pending or in_flight:
                self.checkpoint()
                while pending and len(in_flight) < max_workers and not stop_submitting:
                    self.stop.raise_if_requested()
                    unit = pending.popleft()
                    in_flight[executor.submit(invoke, unit)] = unit
                if not in_flight:
                    break
                finished, _ = wait(tuple(in_flight), return_when=FIRST_COMPLETED)
                for future in finished:
                    unit = in_flight.pop(future)
                    result = future.result()
                    if isinstance(result, Paused):
                        pauses[unit.unit_id] = result
                        stop_submitting = True
                        continue
                    results[unit.unit_id] = result
                    if failure_mode is FailureMode.FAIL_FAST and result.status != "succeeded":
                        stop_submitting = True
            # Context manager joins every submitted future before returning.
        self.stop.raise_if_requested()
        if pauses:
            for unit in units:
                if unit.unit_id in pauses:
                    return pauses[unit.unit_id]
        return GroupResult(
            group_id,
            tuple(results[unit.unit_id] for unit in units if unit.unit_id in results),
        )
