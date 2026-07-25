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


def _public_exports(package: str) -> set[str]:
    facade = ROOT / "packages" / package / "src" / package.replace("-", "_") / "__init__.py"
    tree = ast.parse(facade.read_text(encoding="utf-8"), filename=str(facade))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
        ):
            return set(ast.literal_eval(node.value))
    raise AssertionError(f"{facade.relative_to(ROOT)} does not define a literal __all__")


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


def test_arc_paper_only_uses_public_dependency_facades():
    dependencies = {"arc_jobs", "arc_llm"}
    for path in _python_files("arc-paper"):
        for module in _all_imports(path):
            root = module.split(".", 1)[0]
            if root in dependencies:
                assert module == root, (
                    f"{path.relative_to(ROOT)} imports private dependency module {module}"
                )


def test_arc_domain_only_uses_public_dependency_facades():
    dependencies = {"arc_jobs", "arc_llm", "arc_paper"}
    exports = {package: _public_exports(package.replace("_", "-")) for package in dependencies}
    for path in _python_files("arc-domain"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in dependencies:
                        assert alias.name == root, (
                            f"{path.relative_to(ROOT)} imports private dependency module {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".", 1)[0]
                if root not in dependencies:
                    continue
                assert node.module == root, (
                    f"{path.relative_to(ROOT)} imports private dependency module {node.module}"
                )
                assert all(alias.name in exports[root] for alias in node.names), (
                    f"{path.relative_to(ROOT)} imports non-facade symbols from {root}: "
                    f"{[alias.name for alias in node.names if alias.name not in exports[root]]}"
                )


def test_arc_translate_only_uses_public_dependency_facades():
    dependencies = {"arc_jobs", "arc_llm", "arc_paper"}
    exports = {
        package: _public_exports(package.replace("_", "-"))
        for package in dependencies
    }
    for path in _python_files("arc-translate"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [(alias.name, None) for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [(node.module, node.names)]
            else:
                continue
            for module, names in modules:
                root = module.split(".", 1)[0]
                if root not in dependencies:
                    continue
                assert module == root, (
                    f"{path.relative_to(ROOT)} imports private dependency module {module}"
                )
                if names is not None:
                    assert all(alias.name in exports[root] for alias in names), (
                        f"{path.relative_to(ROOT)} imports non-facade symbols from "
                        f"{root}: "
                        f"{[alias.name for alias in names if alias.name not in exports[root]]}"
                    )


def test_arc_companion_only_uses_public_dependency_facades():
    dependencies = {"arc_jobs", "arc_llm", "arc_paper", "arc_translate"}
    exports = {
        package: _public_exports(package.replace("_", "-"))
        for package in dependencies
    }
    for path in _python_files("arc-companion"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                aliases = node.names
                modules = [(alias.name, None) for alias in aliases]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [(node.module, node.names)]
            else:
                continue
            for module, names in modules:
                root = module.split(".", 1)[0]
                if root not in dependencies:
                    continue
                assert module == root, (
                    f"{path.relative_to(ROOT)} imports private dependency module {module}"
                )
                if names is not None:
                    assert all(alias.name in exports[root] for alias in names), (
                        f"{path.relative_to(ROOT)} imports non-facade symbols from {root}: "
                        f"{[alias.name for alias in names if alias.name not in exports[root]]}"
                    )


def test_arc_domain_owns_no_process_thread_or_file_lock_implementation():
    forbidden = {
        "threading",
        "subprocess",
        "fcntl",
        "filelock",
        "portalocker",
        "fasteners",
        "msvcrt",
    }
    for path in _python_files("arc-domain"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _all_imports(path):
            assert module.split(".", 1)[0] not in forbidden, (
                f"{path.relative_to(ROOT)} imports process, thread, or file-lock machinery: {module}"
            )
        if path.name == "_cache_root.py":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                assert not (
                    node.value.id == "os" and node.attr in {"environ", "getenv"}
                ), f"{path.relative_to(ROOT)} reads environment directly"
            if isinstance(node, ast.ImportFrom) and node.module == "os":
                assert not {alias.name for alias in node.names} & {"environ", "getenv"}, (
                    f"{path.relative_to(ROOT)} reads environment directly"
                )


def test_core_does_not_import_plugin_or_test_code():
    for package in (
        "arc-jobs",
        "arc-llm",
        "arc-proposer-reviewer",
        "arc-paper",
        "arc-domain",
        "arc-translate",
        "arc-companion",
    ):
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


def test_package_source_does_not_depend_on_arc_skill_or_plugin_files():
    """Published packages must be runnable without a source checkout or Skill."""

    forbidden_fragments = (
        "plugins/arc",
        "plugins\\arc",
        "skills/arc",
        "skills\\arc",
        ".arc-install-ref",
    )
    for package in (
        "arc-jobs",
        "arc-llm",
        "arc-proposer-reviewer",
        "arc-paper",
        "arc-domain",
        "arc-translate",
        "arc-companion",
    ):
        for path in _python_files(package):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                normalized = node.value.casefold()
                assert not any(
                    fragment in normalized for fragment in forbidden_fragments
                ), (
                    f"{path.relative_to(ROOT)} contains a runtime reference to "
                    "ARC Skill or plugin files"
                )


def test_arc_jobs_public_facade_has_no_private_storage_helpers():
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
    forbidden = {"read_json", "write_json", "JobPaths", "_fs"}
    assert not forbidden.intersection(exports)
    assert {"atomic_write_bytes", "atomic_write_json", "file_lease"} <= set(exports)


def test_old_proposer_reviewer_implementation_is_absent_from_arc_llm():
    root = ROOT / "packages/arc-llm/src/arc_llm"
    assert not tuple((root / "proposers_reviewer").glob("**/*.py"))
    assert not tuple((root / "proposers_reviewer_bench").glob("**/*.py"))
