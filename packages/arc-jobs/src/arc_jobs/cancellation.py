from __future__ import annotations

from pathlib import Path

from .errors import CancelledError, CorruptStateError, UnsupportedSchemaError
from .models import CancelRequest
from .lease import FileLease
from .storage import atomic_write_json, read_json_object, require_fields, utc_now


class CancellationToken:
    def __init__(self, path: Path):
        self.path = path

    def read(self) -> CancelRequest | None:
        if not self.path.exists():
            return None
        document = read_json_object(self.path)
        require_fields(document, required={"schema_version", "requested_at", "reason"})
        if document["schema_version"] != "arc.jobs.cancel_request.v1":
            raise UnsupportedSchemaError(str(document["schema_version"]))
        requested_at, reason = document["requested_at"], document["reason"]
        if not isinstance(requested_at, str) or not (
            reason is None or isinstance(reason, str)
        ):
            raise CorruptStateError("invalid cancellation request")
        return CancelRequest(requested_at, reason)

    def request(self, *, reason: str | None = None) -> CancelRequest:
        request = CancelRequest(utc_now(), reason)
        document = {
            "schema_version": "arc.jobs.cancel_request.v1",
            "requested_at": request.requested_at,
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
            return request
        finally:
            lease.release()

    def raise_if_requested(self) -> None:
        request = self.read()
        if request is not None:
            raise CancelledError(request.reason or "run cancellation requested")
