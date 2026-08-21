"""Durable runtime concurrency control for work groups."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .contracts import StateContract
from .errors import CorruptStateError, RevisionConflictError, StateConflictError
from .identity import validate_simple_id
from .models import JsonValue
from .storage import AtomicStateStore, require_fields


# Executors create threads lazily.  This capacity therefore reserves no idle
# threads; it only leaves room for an operator to raise a running group's target.
DEFAULT_DYNAMIC_WORKER_CAPACITY = 1024
_SCHEMA_VERSION = "ac.jobs.group_workers.v1"


@dataclass(frozen=True)
class GroupWorkerControl:
    group_id: str
    revision: int
    target_workers: int
    capacity: int


class _GroupWorkerContract(StateContract[GroupWorkerControl]):
    schema_version = _SCHEMA_VERSION

    def encode(self, value: GroupWorkerControl) -> Mapping[str, JsonValue]:
        return {
            "group_id": value.group_id,
            "revision": value.revision,
            "target_workers": value.target_workers,
            "capacity": value.capacity,
        }

    def decode(self, document: Mapping[str, JsonValue]) -> GroupWorkerControl:
        require_fields(
            document,
            required={"group_id", "revision", "target_workers", "capacity"},
        )
        group_id = document["group_id"]
        revision = document["revision"]
        target = document["target_workers"]
        capacity = document["capacity"]
        try:
            if not isinstance(group_id, str):
                raise ValueError("group worker control has invalid group_id")
            validate_simple_id(group_id, label="group id")
            _validate_positive_int(revision, "group worker revision", allow_zero=True)
            _validate_positive_int(target, "target_workers")
            _validate_positive_int(capacity, "worker capacity")
            if target > capacity:
                raise ValueError("target_workers exceeds worker capacity")
        except ValueError as exc:
            raise CorruptStateError(str(exc)) from exc
        return GroupWorkerControl(group_id, revision, target, capacity)

    def validate_transition(
        self,
        previous: GroupWorkerControl | None,
        next: GroupWorkerControl,
    ) -> None:
        if previous is None:
            if next.revision != 0:
                raise ValueError("initial group worker revision must be zero")
            return
        if next.group_id != previous.group_id:
            raise ValueError("group worker control cannot change group_id")
        if next.revision != previous.revision + 1:
            raise ValueError("group worker revision must increase by one")
        if next.capacity < previous.capacity:
            raise ValueError("group worker capacity cannot decrease")


def _validate_positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _store(run_directory: Path, group_id: str) -> AtomicStateStore[GroupWorkerControl]:
    validate_simple_id(group_id, label="group id")
    namespace = hashlib.sha256(group_id.encode("utf-8")).hexdigest()
    return AtomicStateStore(
        run_directory / "state" / "group-workers" / f"{namespace}.json",
        _GroupWorkerContract(),
    )


def initialize_group_workers(
    run_directory: Path,
    group_id: str,
    *,
    target_workers: int,
    capacity: int,
) -> GroupWorkerControl:
    _validate_positive_int(target_workers, "target_workers")
    _validate_positive_int(capacity, "worker capacity")
    if target_workers > capacity:
        raise ValueError("target_workers exceeds worker capacity")
    store = _store(run_directory, group_id)
    while True:
        current = store.read()
        if current is None:
            initial = GroupWorkerControl(group_id, 0, target_workers, capacity)
            try:
                return store.create(initial)
            except StateConflictError:
                continue
        if current.capacity >= capacity:
            return current
        expanded = GroupWorkerControl(
            group_id,
            current.revision + 1,
            current.target_workers,
            capacity,
        )
        try:
            return store.compare_and_swap(current.revision, expanded)
        except RevisionConflictError:
            continue


def read_group_workers(
    run_directory: Path, group_id: str
) -> GroupWorkerControl | None:
    return _store(run_directory, group_id).read()


def set_group_workers(
    run_directory: Path,
    group_id: str,
    target_workers: int,
) -> tuple[GroupWorkerControl, GroupWorkerControl]:
    _validate_positive_int(target_workers, "target_workers")
    store = _store(run_directory, group_id)
    while True:
        current = store.read()
        if current is None:
            raise ValueError(f"work group {group_id!r} has not started")
        if target_workers > current.capacity:
            raise ValueError(
                f"target_workers must be between 1 and {current.capacity}"
            )
        if target_workers == current.target_workers:
            return current, current
        updated = GroupWorkerControl(
            group_id,
            current.revision + 1,
            target_workers,
            current.capacity,
        )
        try:
            return current, store.compare_and_swap(current.revision, updated)
        except RevisionConflictError:
            continue


def validate_group_worker_controls(run_directory: Path) -> None:
    root = run_directory / "state" / "group-workers"
    if not root.exists():
        return
    for path in sorted(root.glob("*.json")):
        control = AtomicStateStore(path, _GroupWorkerContract()).validate()
        assert control is not None
        if _store(run_directory, control.group_id).path != path.resolve():
            raise CorruptStateError(
                f"group worker control namespace does not match group_id: {path}"
            )
