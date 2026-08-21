from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_build_script_discovers_every_arc_package() -> None:
    script = (ROOT / "scripts/build-packages.sh").read_text(encoding="utf-8")
    assert "packages/arc-*/pyproject.toml" in script
    assert "${project%/pyproject.toml}" in script
    assert "packages/arc-document" not in script


def test_release_script_discovers_every_arc_package() -> None:
    script = (ROOT / "scripts/release-arc.sh").read_text(encoding="utf-8")
    assert "packages/arc-*/pyproject.toml" in script
