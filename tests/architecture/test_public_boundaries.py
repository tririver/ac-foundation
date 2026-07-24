from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _python_files(package: str):
    return (ROOT / "packages" / package / "src").glob("**/*.py")


def _from_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def _all_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            yield from (item.name for item in node.names)


def test_new_core_packages_only_use_dependency_root_facades():
    rules = {
        "arc-llm": {"arc_jobs"},
        "arc-proposer-reviewer": {"arc_jobs", "arc_llm"},
    }
    for package, dependencies in rules.items():
        for path in _python_files(package):
            for module in _from_imports(path):
                root = module.split(".", 1)[0]
                if root in dependencies:
                    assert module == root, (
                        f"{path.relative_to(ROOT)} imports private dependency module {module}"
                    )


def test_arc_paper_only_uses_the_arc_llm_public_facade():
    for path in _python_files("arc-paper"):
        for module in _all_imports(path):
            if module == "arc_llm" or module.startswith("arc_llm."):
                assert module == "arc_llm", (
                    f"{path.relative_to(ROOT)} imports private dependency module {module}"
                )


def test_core_does_not_import_plugin_or_test_code():
    for package in ("arc-jobs", "arc-llm", "arc-proposer-reviewer"):
        for path in _python_files(package):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                assert not any(
                    name == "plugins"
                    or name.startswith("plugins.")
                    or name == "tests"
                    or name.startswith("tests.")
                    for name in names
                ), path


def test_arc_jobs_public_facade_has_no_raw_filesystem_helpers():
    namespace: dict[str, object] = {}
    facade = ROOT / "packages/arc-jobs/src/arc_jobs/__init__.py"
    tree = ast.parse(facade.read_text(encoding="utf-8"))
    exports = None
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
        ):
            exports = ast.literal_eval(node.value)
    assert exports is not None
    forbidden = {"read_json", "write_json", "atomic_write_json", "JobPaths", "_fs"}
    assert not forbidden.intersection(exports)


def test_old_proposer_reviewer_implementation_is_absent_from_arc_llm():
    root = ROOT / "packages/arc-llm/src/arc_llm"
    assert not tuple((root / "proposers_reviewer").glob("**/*.py"))
    assert not tuple((root / "proposers_reviewer_bench").glob("**/*.py"))
