from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "packages"
EXPECTED = {
    "ac-document": {"ac-jobs", "ac-llm"},
    "ac-jobs": set(),
    "ac-llm": {"ac-jobs"},
    "ac-proposer-reviewer": {"ac-jobs", "ac-llm"},
}
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
MAJOR = int(VERSION.split(".")[0])


def _project(package: str) -> dict[str, object]:
    return tomllib.loads(
        (PACKAGES / package / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]


def test_package_set_metadata_and_dependency_graph() -> None:
    observed = {path.name for path in PACKAGES.iterdir() if path.is_dir()}
    assert observed == set(EXPECTED)
    for package, internal_dependencies in EXPECTED.items():
        project = _project(package)
        assert project["name"] == package
        assert project["version"] == VERSION
        assert project["authors"] == [{"name": "AC Foundation"}]
        assert project["urls"] == {
            "Homepage": "https://github.com/tririver/ac-foundation",
            "Repository": "https://github.com/tririver/ac-foundation",
            "Issues": "https://github.com/tririver/ac-foundation/issues",
        }
        dependencies = {
            dependency.split(">=", 1)[0]
            for dependency in project.get("dependencies", [])
            if dependency.startswith("ac-")
        }
        assert dependencies == internal_dependencies
        for dependency in project.get("dependencies", []):
            if dependency.startswith("ac-"):
                assert dependency.endswith(f">={MAJOR},<{MAJOR + 1}")


def test_foundation_has_no_product_plugin_or_skill() -> None:
    assert not (ROOT / "plugins").exists()
    assert not (ROOT / ".claude-plugin").exists()
    package_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in PACKAGES.rglob("*")
        if path.is_file() and path.suffix in {".py", ".md", ".toml"}
    )
    for stale in ("arc_", "arc-", "arc.", "ARC_", ".arc"):
        assert stale not in package_text


def test_build_and_release_scripts_cover_every_foundation_package() -> None:
    build = (ROOT / "scripts/build-packages.sh").read_text(encoding="utf-8")
    check = (ROOT / "scripts/check-packages.sh").read_text(encoding="utf-8")
    release = (ROOT / "scripts/release-ac-foundation.sh").read_text(
        encoding="utf-8"
    )
    assert "packages/ac-*/pyproject.toml" in build
    assert '"$root/scripts/build-packages.sh"' in check
    for package in EXPECTED:
        assert f'"{package}"' in release
    assert "plugins/" not in release


def test_canonical_runtime_and_dsh_bridge_are_ac_owned() -> None:
    launcher = (ROOT / "runtime/ac-runtime").read_text(encoding="utf-8")
    bootstrap = (ROOT / "runtime/ac_runtime.py").read_text(encoding="utf-8")
    bridge = (ROOT / "dsh/llm-bridge.js").read_text(encoding="utf-8")
    assert "AC_RUNTIME_LAUNCHER_NAME" in launcher
    assert 'LOCK_SCHEMA = "ac.runtime_sources.v2"' in bootstrap
    assert "ac.dsh-llm.request.v1" in bridge
    assert "arc.dsh-llm" not in bridge
