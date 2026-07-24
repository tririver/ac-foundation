from __future__ import annotations

import os
import threading
from pathlib import Path

from .errors import RunBusyError

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None

_LOCAL_LEASES_GUARD = threading.Lock()
_LOCAL_LEASES: dict[str, threading.Lock] = {}


class FileLease:
    def __init__(self, path: Path):
        self.path = path
        self._handle = None
        self._local_lock: threading.Lock | None = None

    def acquire(self, *, blocking: bool = False) -> "FileLease":
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = str(self.path.resolve())
        with _LOCAL_LEASES_GUARD:
            local_lock = _LOCAL_LEASES.setdefault(key, threading.Lock())
        if not local_lock.acquire(blocking=blocking):
            raise RunBusyError(f"lease is held: {self.path}")
        self._local_lock = local_lock
        try:
            handle = self.path.open("a+b")
            if fcntl is not None:
                flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
                fcntl.flock(handle.fileno(), flags)
            elif msvcrt is not None:  # pragma: no cover - Windows
                handle.seek(0)
                handle.write(b"0")
                handle.flush()
                mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
                msvcrt.locking(handle.fileno(), mode, 1)
            else:  # pragma: no cover
                raise RuntimeError("no supported file locking backend")
        except (BlockingIOError, OSError) as exc:
            if "handle" in locals():
                handle.close()
            local_lock.release()
            self._local_lock = None
            raise RunBusyError(f"lease is held: {self.path}") from exc
        except Exception:
            if "handle" in locals():
                handle.close()
            local_lock.release()
            self._local_lock = None
            raise
        os.chmod(self.path, 0o600)
        self._handle = handle
        return self

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self._handle.close()
            self._handle = None
            if self._local_lock is not None:
                self._local_lock.release()
                self._local_lock = None

    def __enter__(self) -> "FileLease":
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        self.release()
