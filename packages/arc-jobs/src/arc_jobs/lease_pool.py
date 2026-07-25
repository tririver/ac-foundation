"""Small cross-process concurrency pools built from durable file leases."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

from .errors import CorruptStateError, RunBusyError
from .lease import FileLease
from .storage import atomic_write_json, read_json_object, require_fields

_SCHEMA_VERSION = "arc.jobs.bounded_lease_pool.v1"
_MAX_CAPACITY = 1024


class BoundedLease:
    """One already-acquired slot from a :class:`BoundedLeasePool`."""

    def __init__(self, slot: int, lease: FileLease):
        self._slot = slot
        self._lease = lease

    @property
    def slot(self) -> int:
        return self._slot

    def release(self) -> None:
        self._lease.release()

    def __enter__(self) -> "BoundedLease":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class BoundedLeasePool:
    """A fixed-capacity, explicit-root lease pool shared by local processes."""

    def __init__(self, root: str | Path, capacity: int):
        if (
            not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or not 1 <= capacity <= _MAX_CAPACITY
        ):
            raise ValueError(f"capacity must be between 1 and {_MAX_CAPACITY}")
        self.root = Path(root).resolve()
        self.capacity = capacity
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass
        self._bind_capacity()

    def _bind_capacity(self) -> None:
        init = FileLease(self.root / "initialize.lock").acquire(blocking=True)
        try:
            path = self.root / "pool.json"
            expected = {
                "schema_version": _SCHEMA_VERSION,
                "capacity": self.capacity,
            }
            if not path.exists():
                atomic_write_json(path, expected)
                return
            document = read_json_object(path)
            require_fields(document, required={"schema_version", "capacity"})
            if document["schema_version"] != _SCHEMA_VERSION:
                raise CorruptStateError("unsupported bounded lease pool schema")
            if document["capacity"] != self.capacity:
                raise ValueError(
                    "bounded lease pool capacity differs from the persisted contract"
                )
        finally:
            init.release()

    def acquire(
        self,
        *,
        limit: int | None = None,
        blocking: bool = True,
        checkpoint: Callable[[], None] | None = None,
        poll_interval_seconds: float = 0.05,
    ) -> BoundedLease:
        eligible_capacity = self.capacity if limit is None else limit
        if (
            not isinstance(eligible_capacity, int)
            or isinstance(eligible_capacity, bool)
            or not 1 <= eligible_capacity <= self.capacity
        ):
            raise ValueError(
                f"limit must be between 1 and the pool capacity ({self.capacity})"
            )
        if (
            not isinstance(poll_interval_seconds, (int, float))
            or isinstance(poll_interval_seconds, bool)
            or poll_interval_seconds <= 0
        ):
            raise ValueError("poll_interval_seconds must be positive")
        while True:
            if checkpoint is not None:
                checkpoint()
            acquisition = FileLease(self.root / "acquire.lock")
            try:
                acquisition.acquire()
            except RunBusyError:
                pass
            else:
                candidate: tuple[int, FileLease] | None = None
                held = 0
                try:
                    for slot in range(self.capacity):
                        probe = FileLease(
                            self.root / "slots" / f"{slot:04d}.lock"
                        )
                        try:
                            probe.acquire()
                        except RunBusyError:
                            held += 1
                            continue
                        if candidate is None:
                            candidate = (slot, probe)
                        else:
                            probe.release()
                    if held < eligible_capacity and candidate is not None:
                        slot, lease = candidate
                        candidate = None
                        return BoundedLease(slot, lease)
                finally:
                    if candidate is not None:
                        candidate[1].release()
                    acquisition.release()
            if not blocking:
                raise RunBusyError("bounded lease pool acquisition limit is reached")
            time.sleep(float(poll_interval_seconds))
