from __future__ import annotations

import errno
import importlib
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator


class _PosixFileLockBackend:
    name = "fcntl"

    def __init__(self, module: Any):
        self._module = module

    def acquire(self, handle: Any) -> None:
        self._module.flock(handle.fileno(), self._module.LOCK_EX)

    def release(self, handle: Any) -> None:
        self._module.flock(handle.fileno(), self._module.LOCK_UN)


class _WindowsFileLockBackend:
    name = "msvcrt"

    def __init__(
        self,
        module: Any,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._module = module
        self._sleep = sleep

    def acquire(self, handle: Any) -> None:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"\0")
            handle.flush()
        while True:
            handle.seek(0)
            try:
                self._module.locking(handle.fileno(), self._module.LK_NBLCK, 1)
                return
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                self._sleep(0.05)

    def release(self, handle: Any) -> None:
        handle.seek(0)
        self._module.locking(handle.fileno(), self._module.LK_UNLCK, 1)


def _load_file_lock_backend(platform_name: str = os.name) -> Any:
    if platform_name == "nt":
        return _WindowsFileLockBackend(importlib.import_module("msvcrt"))
    return _PosixFileLockBackend(importlib.import_module("fcntl"))


_FILE_LOCK_BACKEND = _load_file_lock_backend()


@contextmanager
def exclusive_file_lock(path: str | Path) -> Iterator[None]:
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        _FILE_LOCK_BACKEND.acquire(handle)
        try:
            yield
        finally:
            _FILE_LOCK_BACKEND.release(handle)


__all__ = ["exclusive_file_lock"]
