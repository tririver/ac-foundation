from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ac_runtime_bootstrap", ROOT / "runtime/ac_runtime.py"
)
assert SPEC is not None and SPEC.loader is not None
RUNTIME = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNTIME
SPEC.loader.exec_module(RUNTIME)


def _lock(commit: str = "a" * 40) -> dict[str, object]:
    return {
        "schema_version": "ac.runtime_sources.v1",
        "profile": "test-product",
        "sources": [
            {
                "id": "foundation",
                "repository": "https://github.com/example/foundation.git",
                "commit": commit,
                "version": ">=2,<3",
                "packages": ["ac-jobs"],
                "tools": ["ac-jobs"],
                "local_root_env": "AC_FOUNDATION_REPO_ROOT",
            }
        ],
        "environment_defaults": {},
    }


def test_source_lock_requires_full_sha_and_major_range(tmp_path: Path) -> None:
    path = tmp_path / "runtime-sources.json"
    path.write_text(json.dumps(_lock()), encoding="utf-8")
    lock = RUNTIME.load_lock(path)

    assert lock.profile == "test-product"
    assert lock.tools == ("ac-jobs",)
    assert lock.sources[0].version == ">=2,<3"

    path.write_text(json.dumps(_lock("abc123")), encoding="utf-8")
    with pytest.raises(RUNTIME.RuntimeConfigError, match="full Git SHA"):
        RUNTIME.load_lock(path)


def test_source_lock_rejects_duplicate_package_ownership(tmp_path: Path) -> None:
    document = _lock()
    document["sources"].append(
        {
            "id": "product",
            "repository": "https://github.com/example/product.git",
            "commit": "b" * 40,
            "version": ">=2,<3",
            "packages": ["ac-jobs"],
            "tools": ["product-tool"],
            "local_root_env": "AC_PRODUCT_REPO_ROOT",
        }
    )
    path = tmp_path / "runtime-sources.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RUNTIME.RuntimeConfigError, match="one owning source"):
        RUNTIME.load_lock(path)


def test_runtime_environment_owns_only_generic_ac_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "runtime-sources.json"
    path.write_text(json.dumps(_lock()), encoding="utf-8")
    lock = RUNTIME.load_lock(path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AC_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("AC_RUNTIME_HOME", raising=False)
    monkeypatch.delenv("AC_DOCUMENT_CACHE", raising=False)

    environment = RUNTIME._runtime_environment(lock, None)

    assert set(environment) == {
        "AC_HOME",
        "AC_RUNTIME_HOME",
        "AC_DOCUMENT_CACHE",
    }
    assert environment["AC_DOCUMENT_CACHE"] == str(
        tmp_path / ".ac/cache/ac-document"
    )


def test_install_creates_console_scripts_at_their_final_venv_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "runtime-sources.json"
    lock_path.write_text(json.dumps(_lock()), encoding="utf-8")
    lock = RUNTIME.load_lock(lock_path)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    foundation = tmp_path / "foundation"
    constraints = tmp_path / "missing-constraints.txt"

    def fake_run(command: list[str], _log_path: Path) -> None:
        if command[1:3] == ["-m", "venv"]:
            venv = Path(command[-1])
            (venv / "bin").mkdir(parents=True)
            (venv / "bin/python").write_text("", encoding="utf-8")
            return
        python = Path(command[0])
        (python.parent / "ac-jobs").write_text(
            f"#!{python}\n", encoding="utf-8"
        )

    monkeypatch.setattr(RUNTIME.shutil, "which", lambda _name: None)
    monkeypatch.setattr(RUNTIME, "_run_logged", fake_run)

    RUNTIME._install(
        runtime_dir,
        lock,
        "local",
        {"foundation": foundation},
        constraints,
        "f" * 64,
        {"mode": "local"},
    )

    tool = runtime_dir / "venv/bin/ac-jobs"
    assert tool.read_text(encoding="utf-8") == (
        f"#!{runtime_dir / 'venv/bin/python'}\n"
    )
