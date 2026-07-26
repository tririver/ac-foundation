from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Callable, Mapping, TypeAlias

from .errors import CorruptStateError
from .identity import canonical_json_bytes, validate_simple_id
from .json_data import validate_json_value
from .lease import FileLease
from .models import JsonValue
from .storage import _ensure_directory, _fsync_directory, utc_now

MAX_EVENT_BYTES = 256 * 1024
MAX_TAIL_BYTES = 1024 * 1024
EventSink: TypeAlias = Callable[[Mapping[str, JsonValue]], None]


class _SinkFailureState:
    """Suppress repeated diagnostics across writers in one runtime execution."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reported = False

    def claim_report(self) -> bool:
        with self._lock:
            if self._reported:
                return False
            self._reported = True
            return True


class EventWriter:
    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        event_sink: EventSink | None = None,
        _sink_failure_state: _SinkFailureState | None = None,
    ):
        self.path = path
        self.run_id = validate_simple_id(run_id, label="run id")
        self._event_sink = event_sink
        self._sink_failure_state = _sink_failure_state or _SinkFailureState()
        self._last_sequence: int | None = None
        self._last_size: int | None = None

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
            validate_json_value(value["data"])
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

    def _truncate_incomplete_tail(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r+b") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size == 0:
                return 0
            handle.seek(size - 1)
            if handle.read(1) in {b"\n", b"\r"}:
                return size
            boundary = size
            while boundary:
                chunk_start = max(0, boundary - 64 * 1024)
                handle.seek(chunk_start)
                chunk = handle.read(boundary - chunk_start)
                newline = max(chunk.rfind(b"\n"), chunk.rfind(b"\r"))
                if newline >= 0:
                    boundary = chunk_start + newline + 1
                    break
                boundary = chunk_start
            handle.truncate(boundary)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self.path.parent)
        self._last_sequence = None
        self._last_size = None
        return boundary

    def _sequence_at_size(self, size: int) -> int:
        if size == 0:
            return 0
        if self._last_sequence is not None and self._last_size == size:
            return self._last_sequence
        with self.path.open("rb") as handle:
            start = max(0, size - MAX_EVENT_BYTES - 1)
            handle.seek(start)
            raw = handle.read(size - start)
        lines = raw.splitlines()
        if not lines or (start > 0 and len(lines) < 2):
            raise CorruptStateError("last event record exceeds the event size limit")
        try:
            value = json.loads(lines[-1].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorruptStateError("invalid last event record") from exc
        if not isinstance(value, dict):
            raise CorruptStateError("last event is not an object")
        sequence = value.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            raise CorruptStateError("invalid last event sequence")
        self._validate_document(value, expected_sequence=sequence)
        self._last_sequence = sequence
        self._last_size = size
        return sequence

    def _append(
        self, event: str, data: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        validate_simple_id(event, label="event")
        validate_json_value(dict(data))
        lock = FileLease(self.path.with_suffix(".lock")).acquire(blocking=True)
        try:
            size = self._truncate_incomplete_tail()
            sequence = self._sequence_at_size(size) + 1
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
            self._last_sequence = sequence
            self._last_size = size + len(encoded)
            return document
        finally:
            lock.release()

    def emit(self, event: str, data: Mapping[str, JsonValue]) -> None:
        document = self._append(event, data)
        if self._event_sink is None:
            return
        try:
            self._event_sink(document)
        except Exception as exc:
            if self._sink_failure_state.claim_report():
                try:
                    self._append(
                        "progress_sink_failed",
                        {
                            "source_event": event,
                            "source_sequence": document["sequence"],
                            "error_type": type(exc).__name__,
                        },
                    )
                except Exception:
                    pass

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
                try:
                    validate_json_value(value)
                except ValueError:
                    continue
                documents.append(value)
        return tuple(documents)

    def validate(self) -> None:
        self._complete_documents()
