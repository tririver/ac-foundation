from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _trees(package: str):
    for path in (ROOT / "packages" / package / "src").glob("**/*.py"):
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_arc_jobs_has_no_process_or_upper_layer_dependencies():
    forbidden = {
        "subprocess",
        "arc_llm",
        "arc_paper",
        "arc_domain",
        "arc_translate",
        "arc_companion",
        "arc_mcp",
        "arc_proposer_reviewer",
    }
    for path, tree in _trees("arc-jobs"):
        assert not (_import_roots(tree) & forbidden), path.relative_to(ROOT)


def test_arc_llm_has_no_domain_mcp_or_proposer_reviewer_dependencies():
    forbidden = {
        "arc_paper",
        "arc_domain",
        "arc_translate",
        "arc_companion",
        "arc_mcp",
        "arc_proposer_reviewer",
    }
    for path, tree in _trees("arc-llm"):
        assert not (_import_roots(tree) & forbidden), path.relative_to(ROOT)


def test_proposer_reviewer_uses_core_concurrency_and_provider_boundaries():
    forbidden = {
        "subprocess",
        "threading",
        "concurrent",
        "fcntl",
        "arc_paper",
        "arc_domain",
        "arc_translate",
        "arc_mcp",
    }
    for path, tree in _trees("arc-proposer-reviewer"):
        assert not (_import_roots(tree) & forbidden), path.relative_to(ROOT)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
                and node.attr == "environ"
            ):
                raise AssertionError(
                    f"{path.relative_to(ROOT)} reads environment directly"
                )


def test_paper_has_no_translate_domain_or_companion_dependency():
    forbidden = {"arc_translate", "arc_domain", "arc_companion"}
    for path, tree in _trees("arc-paper"):
        assert not (_import_roots(tree) & forbidden), path.relative_to(ROOT)


def test_domain_has_no_translate_or_companion_dependency():
    forbidden = {"arc_translate", "arc_companion"}
    for path, tree in _trees("arc-domain"):
        assert not (_import_roots(tree) & forbidden), path.relative_to(ROOT)


def test_companion_uses_arc_jobs_concurrency_and_has_no_domain_dependency():
    forbidden = {"threading", "concurrent", "fcntl", "arc_domain"}
    for path, tree in _trees("arc-companion"):
        assert not (_import_roots(tree) & forbidden), path.relative_to(ROOT)


def test_translate_uses_core_concurrency_and_has_no_upper_layer_dependency():
    forbidden = {
        "threading",
        "concurrent",
        "fcntl",
        "arc_domain",
        "arc_companion",
    }
    for path, tree in _trees("arc-translate"):
        assert not (_import_roots(tree) & forbidden), path.relative_to(ROOT)


def test_removed_schema_namespace_never_returns():
    for package in ("arc-jobs", "arc-llm", "arc-proposer-reviewer"):
        for path in (ROOT / "packages" / package / "src").glob("**/*.py"):
            assert "arc.llm.review_envelope.v1" not in path.read_text(encoding="utf-8")
