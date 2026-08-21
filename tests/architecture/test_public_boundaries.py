from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DEPENDENCIES = {
    "arc-document": {"arc_jobs", "arc_llm"},
    "arc-llm": {"arc_jobs"},
    "arc-proposer-reviewer": {"arc_jobs", "arc_llm"},
    "arc-paper": {"arc_document", "arc_jobs", "arc_llm"},
    "arc-render": {"arc_document"},
    "arc-domain": {"arc_jobs", "arc_llm", "arc_paper"},
    "arc-ocr-proofread": {"arc_document", "arc_jobs", "arc_llm"},
    "arc-translate": {"arc_document", "arc_jobs", "arc_llm", "arc_render"},
    "arc-companion": {
        "arc_document",
        "arc_jobs",
        "arc_llm",
        "arc_proposer_reviewer",
        "arc_render",
        "arc_translate",
    },
}


def _python_files(package: str):
    return (ROOT / "packages" / package / "src").glob("**/*.py")


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _all_imports(path: Path):
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module, node.names
        elif isinstance(node, ast.Import):
            yield from ((item.name, None) for item in node.names)


def _public_exports(module: str) -> set[str]:
    package = module.replace("_", "-")
    facade = ROOT / "packages" / package / "src" / module / "__init__.py"
    for node in _tree(facade).body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, list)
            return set(value)
    raise AssertionError(
        f"{facade.relative_to(ROOT)} does not define a literal __all__"
    )


def _is_compatibility_facade(path: Path) -> bool:
    tree = _tree(path)
    docstring = ast.get_docstring(tree) or ""
    if not docstring.startswith("Compatibility facade"):
        return False
    allowed = (ast.Assign, ast.Expr, ast.ImportFrom)
    return all(
        isinstance(node, allowed)
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
        )
    )


def test_packages_use_only_declared_dependency_root_facades() -> None:
    exports = {
        dependency: _public_exports(dependency)
        for dependencies in PACKAGE_DEPENDENCIES.values()
        for dependency in dependencies
    }
    for package, dependencies in PACKAGE_DEPENDENCIES.items():
        for path in _python_files(package):
            for module, names in _all_imports(path):
                root = module.split(".", 1)[0]
                if root not in dependencies:
                    continue
                if module != root:
                    assert package == "arc-paper" and _is_compatibility_facade(
                        path
                    ), (
                        f"{path.relative_to(ROOT)} imports private dependency "
                        f"module {module}"
                    )
                    continue
                if names is not None:
                    missing = [
                        alias.name
                        for alias in names
                        if alias.name != "*" and alias.name not in exports[root]
                    ]
                    assert not missing, (
                        f"{path.relative_to(ROOT)} imports non-facade symbols "
                        f"from {root}: {missing}"
                    )


def test_arc_domain_owns_no_process_thread_or_file_lock_implementation() -> None:
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
        tree = _tree(path)
        for module, _names in _all_imports(path):
            assert module.split(".", 1)[0] not in forbidden, (
                f"{path.relative_to(ROOT)} owns process/thread/lock machinery"
            )
        if path.name == "_cache_root.py":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(
                node.value, ast.Name
            ):
                assert not (
                    node.value.id == "os"
                    and node.attr in {"environ", "getenv"}
                ), f"{path.relative_to(ROOT)} reads environment directly"


def test_core_does_not_import_plugin_or_test_code() -> None:
    for package in PACKAGE_DEPENDENCIES:
        for path in _python_files(package):
            for module, _names in _all_imports(path):
                assert module.split(".", 1)[0] not in {"plugins", "tests"}, path


def test_package_source_does_not_depend_on_arc_skill_or_plugin_files() -> None:
    forbidden = (
        "plugins/arc",
        "plugins\\arc",
        "skills/arc",
        "skills\\arc",
        ".arc-install-ref",
    )
    for package in PACKAGE_DEPENDENCIES:
        for path in _python_files(package):
            for node in ast.walk(_tree(path)):
                if isinstance(node, ast.Constant) and isinstance(
                    node.value, str
                ):
                    assert not any(
                        item in node.value.casefold() for item in forbidden
                    ), path


def test_arc_jobs_public_facade_has_shared_storage_helpers() -> None:
    exports = _public_exports("arc_jobs")
    assert not {"read_json", "write_json", "JobPaths", "_fs"}.intersection(
        exports
    )
    assert {
        "atomic_write_bytes",
        "atomic_write_json",
        "canonical_json_bytes",
        "file_lease",
    } <= exports


def test_old_proposer_reviewer_implementation_is_absent_from_arc_llm() -> None:
    root = ROOT / "packages/arc-llm/src/arc_llm"
    assert not tuple((root / "proposers_reviewer").glob("**/*.py"))
    assert not tuple((root / "proposers_reviewer_bench").glob("**/*.py"))
