from __future__ import annotations

import hashlib
from pathlib import Path

from ac_jobs import file_matches_sha256


def test_file_matches_sha256_checks_digest_and_size(tmp_path: Path) -> None:
    path = tmp_path / "payload"
    payload = b"foundation"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    assert file_matches_sha256(path, digest, len(payload))
    assert not file_matches_sha256(path, "0" * 64, len(payload))
    assert not file_matches_sha256(path, digest, len(payload) + 1)
    assert not file_matches_sha256(path, "invalid", len(payload))
    assert not file_matches_sha256(tmp_path / "missing", digest, len(payload))
