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
        handle = None
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
            self._abort_acquire(handle)
            raise RunBusyError(f"lease is held: {self.path}") from exc
        except BaseException:
            self._abort_acquire(handle)
            raise
        try:
            os.chmod(self.path, 0o600)
        except BaseException:
            self._abort_acquire(handle)
            raise
        self._handle = handle
        return self

    def _abort_acquire(self, handle: object | None) -> None:
        try:
            if handle is not None:
                try:
                    handle.close()
                except BaseException:
                    pass
        finally:
            local_lock = self._local_lock
            self._local_lock = None
            if local_lock is not None:
                try:
                    local_lock.release()
                except BaseException:
                    pass

    def release(self) -> None:
        handle = self._handle
        local_lock = self._local_lock
        self._handle = None
        self._local_lock = None
        try:
            if handle is not None:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                elif msvcrt is not None:  # pragma: no cover - Windows
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            try:
                if handle is not None:
                    handle.close()
            finally:
                if local_lock is not None:
                    local_lock.release()

    def __enter__(self) -> "FileLease":
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        self.release()
