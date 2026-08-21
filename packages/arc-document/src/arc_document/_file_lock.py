"""Compatibility name for the repository-wide cooperative file lease."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from arc_jobs import file_lease


@contextmanager
def exclusive_file_lock(path: str | Path) -> Iterator[None]:
    with file_lease(path, blocking=True):
        yield


__all__ = ["exclusive_file_lock"]
