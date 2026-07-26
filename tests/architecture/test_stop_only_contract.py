from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SRC_ROOTS = tuple(
    sorted(
        path / "src"
        for path in (ROOT / "packages").iterdir()
        if path.is_dir() and (path / "src").is_dir()
    )
)
SCANNED_ROOTS = (
    *PACKAGE_SRC_ROOTS,
    ROOT / "plugins" / "arc",
    ROOT / "docs",
)
SCANNED_FILES = (
    ROOT / "README.md",
    *tuple(sorted((ROOT / "packages").glob("*/README.md"))),
)
TEXT_SUFFIXES = {".py", ".md", ".json", ".txt", ".toml", ".yaml", ".yml"}
def _text_files() -> tuple[Path, ...]:
    paths = list(SCANNED_FILES)
    for root in SCANNED_ROOTS:
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in TEXT_SUFFIXES
            and ".venv" not in path.parts
            and "vendor" not in path.parts
            and "__pycache__" not in path.parts
        )
    return tuple(sorted(set(paths)))


def test_arc_exposes_stop_without_an_owned_cancel_concept() -> None:
    offenders: list[str] = []
    for path in _text_files():
        relative = path.relative_to(ROOT)
        if "cancel" in path.name.lower():
            offenders.append(f"{relative}: filename")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "cancel" not in line.lower():
                continue
            offenders.append(f"{relative}:{line_number}: {line.strip()}")
    assert offenders == []


def test_public_docs_use_current_control_contract_schemas() -> None:
    offenders: list[str] = []
    for path in _text_files():
        if path.suffix != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for legacy in ("arc.command_result.v1", "arc.llm.resume_input.v1"):
            if legacy in text:
                offenders.append(f"{path.relative_to(ROOT)}: {legacy}")
    assert offenders == []
