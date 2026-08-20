from __future__ import annotations

import hashlib
from pathlib import Path

from arc_jobs import atomic_write_bytes


def payload_matches(path: Path, digest: str, size: int) -> bool:
    try:
        if (
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not path.is_file()
            or path.stat().st_size != size
        ):
            return False
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest() == digest
    except OSError:
        return False

__all__ = ["atomic_write_bytes", "payload_matches"]
