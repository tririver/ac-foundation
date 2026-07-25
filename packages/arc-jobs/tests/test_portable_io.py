from __future__ import annotations

import inspect
import json

import pytest

import arc_jobs.storage as storage_module
from arc_jobs import (
    FileLease,
    RunBusyError,
    atomic_write_bytes,
    atomic_write_json,
    file_lease,
)


def test_public_atomic_writes_do_not_claim_create_if_absent(tmp_path):
    assert "exclusive" not in inspect.signature(atomic_write_bytes).parameters
    assert "exclusive" not in inspect.signature(atomic_write_json).parameters

    path = tmp_path / "value.bin"
    atomic_write_bytes(path, b"first")
    atomic_write_bytes(path, b"second")

    assert path.read_bytes() == b"second"


def test_public_atomic_json_is_canonical_and_newline_terminated(tmp_path):
    path = tmp_path / "value.json"

    atomic_write_json(path, {"z": 1, "a": ["value"]})

    assert path.read_bytes() == b'{"a":["value"],"z":1}\n'
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "a": ["value"],
        "z": 1,
    }


def test_file_lease_context_helper_acquires_exactly_once(
    tmp_path, monkeypatch
):
    path = tmp_path / "cooperating.lock"
    original_acquire = FileLease.acquire
    calls = 0

    def observed_acquire(self, *, blocking=False):
        nonlocal calls
        calls += 1
        return original_acquire(self, blocking=blocking)

    monkeypatch.setattr(FileLease, "acquire", observed_acquire)
    with file_lease(path, blocking=True) as held:
        assert held.path == path
        assert calls == 1
        with pytest.raises(RunBusyError):
            FileLease(path).acquire()

    assert calls == 2
    FileLease(path).acquire().release()


def test_windows_atomic_write_skips_fchmod_and_directory_fsync(
    tmp_path, monkeypatch
):
    path = tmp_path / "windows.bin"
    monkeypatch.setattr(storage_module, "_WINDOWS", True)

    def unsupported_fchmod(*_args):
        raise AssertionError("Windows fallback must not call fchmod")

    def unsupported_chmod(*_args):
        raise OSError("POSIX mode unsupported")

    monkeypatch.setattr(
        storage_module.os,
        "fchmod",
        unsupported_fchmod,
        raising=False,
    )
    monkeypatch.setattr(storage_module.os, "chmod", unsupported_chmod)

    atomic_write_bytes(path, b"portable")

    assert path.read_bytes() == b"portable"

    def unsupported_directory_open(*_args):
        raise AssertionError("Windows fallback must not open a directory")

    monkeypatch.setattr(storage_module.os, "open", unsupported_directory_open)
    storage_module._fsync_directory(tmp_path)


def test_posix_fchmod_failure_is_not_silently_downgraded(
    tmp_path, monkeypatch
):
    path = tmp_path / "strict-mode.bin"
    monkeypatch.setattr(storage_module, "_WINDOWS", False)

    def fail_fchmod(*_args):
        raise OSError("fchmod failed")

    monkeypatch.setattr(
        storage_module.os,
        "fchmod",
        fail_fchmod,
        raising=False,
    )

    with pytest.raises(OSError, match="fchmod failed"):
        atomic_write_bytes(path, b"value")

    assert not path.exists()
    assert not list(tmp_path.glob(".strict-mode.bin.*"))


def test_posix_directory_fsync_failure_is_not_silently_downgraded(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(storage_module, "_WINDOWS", False)

    def fail_fsync(*_args):
        raise OSError("directory fsync failed")

    monkeypatch.setattr(storage_module.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="directory fsync failed"):
        storage_module._fsync_directory(tmp_path)
