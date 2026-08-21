from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "runtime_constraints", ROOT / "scripts/check-runtime-constraints.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_dependency_names_normalize_and_exclude_internal_packages(
    tmp_path: Path,
) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nname="sample"\nversion="1.0.0"\n'
        'dependencies=["ac-jobs>=2,<3", "Pillow>=10", "markdown_it-py>=4"]\n',
        encoding="utf-8",
    )

    assert MODULE.dependency_names([project]) == {"pillow", "markdown-it-py"}


def test_constraints_require_exact_pins(tmp_path: Path) -> None:
    path = tmp_path / "constraints.txt"
    path.write_text("Pillow>=10\n", encoding="utf-8")

    try:
        MODULE.constraint_names(path)
    except ValueError as exc:
        assert "exact pin" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-exact constraint accepted")
