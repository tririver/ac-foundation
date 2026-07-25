from __future__ import annotations

import os

import pytest

import arc_llm.executor as executor_module
from arc_llm.executor import LLMTaskExecutor


def test_windows_readonly_input_tolerates_posix_mode_and_skips_directory_fsync(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "inputs" / "source.txt"
    path.parent.mkdir()
    opened = []
    original_open = os.open

    def observed_open(target, flags, *args, **kwargs):
        opened.append(target)
        return original_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(executor_module, "_WINDOWS", True)
    monkeypatch.setattr(executor_module.os, "open", observed_open)
    monkeypatch.setattr(
        executor_module.os,
        "chmod",
        lambda *_args: (_ for _ in ()).throw(
            OSError("POSIX modes are unsupported")
        ),
    )

    LLMTaskExecutor._publish_readonly_input(path, b"portable")

    assert path.read_bytes() == b"portable"
    assert path.parent not in opened


def test_posix_readonly_mode_failure_is_not_silently_downgraded(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "inputs" / "source.txt"
    path.parent.mkdir()
    monkeypatch.setattr(executor_module, "_WINDOWS", False)
    monkeypatch.setattr(
        executor_module.os,
        "chmod",
        lambda *_args: (_ for _ in ()).throw(OSError("chmod failed")),
    )

    with pytest.raises(OSError, match="chmod failed"):
        LLMTaskExecutor._publish_readonly_input(path, b"strict")

    assert not path.exists()
    assert not tuple(path.parent.glob(".source.txt.*"))
