from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from .errors import CorruptStateError
from .identity import canonical_json_bytes, validate_simple_id
from .lease import FileLease
from .models import JsonValue
from .storage import _ensure_directory, _fsync_directory, utc_now

MAX_EVENT_BYTES = 256 * 1024
MAX_TAIL_BYTES = 1024 * 1024
_FORBIDDEN_PROGRESS_KEYS = {"text", "token", "content", "output", "delta"}


def _validate_safe_data(value: JsonValue) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in _FORBIDDEN_PROGRESS_KEYS:
                raise ValueError(
                    f"event data contains forbidden partial-output field {key!r}"
                )
            _validate_safe_data(child)
    elif isinstance(value, list):
        for child in value:
            _validate_safe_data(child)


class EventWriter:
    def __init__(self, path: Path, *, run_id: str):
        self.path = path
        self.run_id = validate_simple_id(run_id, label="run id")

    def _validate_document(
        self, value: dict[str, JsonValue], *, expected_sequence: int
    ) -> None:
        expected_fields = {
            "schema_version",
            "run_id",
            "sequence",
            "event_id",
            "emitted_at",
            "event",
            "data",
        }
        if set(value) != expected_fields:
            raise CorruptStateError("event uses an invalid closed shape")
        if value["schema_version"] != "arc.jobs.event.v1":
            raise CorruptStateError("unsupported event schema")
        if (
            value["run_id"] != self.run_id
            or value["sequence"] != expected_sequence
            or not isinstance(value["event_id"], str)
            or not isinstance(value["emitted_at"], str)
            or not isinstance(value["event"], str)
            or not isinstance(value["data"], dict)
        ):
            raise CorruptStateError("invalid event fields or sequence")
        stable = {
            "run_id": self.run_id,
            "sequence": expected_sequence,
            "event": value["event"],
            "data": value["data"],
        }
        expected_id = hashlib.sha256(canonical_json_bytes(stable)).hexdigest()
        if value["event_id"] != expected_id:
            raise CorruptStateError("event_id does not match event content")
        try:
            _validate_safe_data(value["data"])
        except ValueError as exc:
            raise CorruptStateError(str(exc)) from exc

    def _complete_documents(self) -> list[dict[str, JsonValue]]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        lines = raw.splitlines(keepends=True)
        documents: list[dict[str, JsonValue]] = []
        for index, line in enumerate(lines):
            if not line.endswith((b"\n", b"\r")):
                if index == len(lines) - 1:
                    break
                raise CorruptStateError("incomplete event record in the middle of the log")
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CorruptStateError(f"invalid event at line {index + 1}") from exc
            if not isinstance(value, dict):
                raise CorruptStateError(f"event at line {index + 1} is not an object")
            self._validate_document(value, expected_sequence=len(documents) + 1)
            documents.append(value)
        return documents

    def _truncate_incomplete_tail(self) -> None:
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        if not raw or raw.endswith(b"\n"):
            return
        last_newline = raw.rfind(b"\n")
        with self.path.open("r+b") as handle:
            handle.truncate(last_newline + 1)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self.path.parent)

    def emit(self, event: str, data: Mapping[str, JsonValue]) -> None:
        validate_simple_id(event, label="event")
        _validate_safe_data(dict(data))
        lock = FileLease(self.path.with_suffix(".lock")).acquire(blocking=True)
        try:
            self._truncate_incomplete_tail()
            sequence = len(self._complete_documents()) + 1
            stable = {
                "run_id": self.run_id,
                "sequence": sequence,
                "event": event,
                "data": dict(data),
            }
            event_id = hashlib.sha256(canonical_json_bytes(stable)).hexdigest()
            document: dict[str, JsonValue] = {
                "schema_version": "arc.jobs.event.v1",
                "run_id": self.run_id,
                "sequence": sequence,
                "event_id": event_id,
                "emitted_at": utc_now(),
                "event": event,
                "data": dict(data),
            }
            encoded = canonical_json_bytes(document) + b"\n"
            if len(encoded) > MAX_EVENT_BYTES:
                raise ValueError(f"event exceeds {MAX_EVENT_BYTES} bytes")
            _ensure_directory(self.path.parent)
            with self.path.open("ab") as handle:
                os.chmod(self.path, 0o600)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(self.path.parent)
        finally:
            lock.release()

    def tail(self) -> tuple[dict[str, JsonValue], ...]:
        if not self.path.exists():
            return ()
        size = self.path.stat().st_size
        with self.path.open("rb") as handle:
            if size > MAX_TAIL_BYTES:
                handle.seek(size - MAX_TAIL_BYTES)
                handle.readline()
            raw = handle.read(MAX_TAIL_BYTES)
        if raw and not raw.endswith(b"\n"):
            raw = raw[: raw.rfind(b"\n") + 1]
        documents: list[dict[str, JsonValue]] = []
        for line in raw.splitlines():
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                documents.append(value)
        return tuple(documents)

    def validate(self) -> None:
        self._complete_documents()
