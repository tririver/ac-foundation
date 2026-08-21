#!/usr/bin/env python3
"""Check exact runtime pins against direct external package dependencies."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
INTERNAL_PREFIXES = ("ac-", "arc-", "alc-")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def dependency_names(projects: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in projects:
        project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
        for dependency in project.get("dependencies", []):
            match = NAME_RE.match(dependency)
            if match is None:
                raise ValueError(f"cannot parse dependency in {path}: {dependency}")
            name = normalize(match.group(1))
            if not name.startswith(INTERNAL_PREFIXES):
                names.add(name)
    return names


def constraint_names(path: Path) -> set[str]:
    names: set[str] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"constraint must be an exact pin at {path}:{number}")
        match = NAME_RE.match(line)
        if match is None:
            raise ValueError(f"cannot parse constraint at {path}:{number}")
        names.add(normalize(match.group(1)))
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--constraints", required=True, type=Path)
    parser.add_argument("--package", action="append", required=True, type=Path)
    args = parser.parse_args()
    dependencies = dependency_names(args.package)
    constraints = constraint_names(args.constraints)
    missing = sorted(dependencies - constraints)
    unused = sorted(constraints - dependencies)
    if missing or unused:
        raise SystemExit(
            f"runtime constraint mismatch: missing={missing}, unused={unused}"
        )
    print(f"runtime constraints cover {len(dependencies)} external dependencies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
