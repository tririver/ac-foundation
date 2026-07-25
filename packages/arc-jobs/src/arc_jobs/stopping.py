"""Attempt-scoped cooperative stop requests for durable runs."""

from __future__ import annotations

from pathlib import Path

from .errors import CorruptStateError, StoppedError, UnsupportedSchemaError
from .lease import FileLease
from .models import StopRequest
from .storage import atomic_write_json, read_json_object, require_fields, utc_now


class StopToken:
    """One immutable stop request targeting one execution attempt."""

    def __init__(self, path: Path, *, target_attempt: int):
        if isinstance(target_attempt, bool) or not isinstance(target_attempt, int) or target_attempt < 0:
            raise ValueError("target_attempt must be a non-negative integer")
        self.path = path
        self.target_attempt = target_attempt

    def read(self) -> StopRequest | None:
        if not self.path.exists():
            return None
        document = read_json_object(self.path)
        require_fields(
            document,
            required={"schema_version", "target_attempt", "requested_at", "reason"},
        )
        if document["schema_version"] != "arc.jobs.stop_request.v1":
            raise UnsupportedSchemaError(str(document["schema_version"]))
        target_attempt = document["target_attempt"]
        requested_at = document["requested_at"]
        reason = document["reason"]
        if (
            isinstance(target_attempt, bool)
            or not isinstance(target_attempt, int)
            or target_attempt != self.target_attempt
            or not isinstance(requested_at, str)
            or not (reason is None or isinstance(reason, str))
        ):
            raise CorruptStateError("invalid stop request")
        return StopRequest(target_attempt, requested_at, reason)

    def request(self, *, reason: str | None = None) -> StopRequest:
        if reason is not None and not isinstance(reason, str):
            raise ValueError("stop reason must be a string or None")
        document = {
            "schema_version": "arc.jobs.stop_request.v1",
            "target_attempt": self.target_attempt,
            "requested_at": utc_now(),
            "reason": reason,
        }
        lease = FileLease(self.path.with_suffix(f"{self.path.suffix}.lock")).acquire(
            blocking=True
        )
        try:
            existing = self.read()
            if existing is not None:
                return existing
            atomic_write_json(self.path, document)
            return StopRequest(
                self.target_attempt,
                str(document["requested_at"]),
                reason,
            )
        finally:
            lease.release()

    def raise_if_requested(self) -> None:
        request = self.read()
        if request is not None:
            raise StoppedError(request.reason or "execution stop requested")
