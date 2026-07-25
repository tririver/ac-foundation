from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_run_spec_contains_only_semantic_identity_inputs():
    path = ROOT / "packages/arc-jobs/src/arc_jobs/models.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    run_spec = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RunSpec"
    )
    fields = {
        node.target.id
        for node in run_spec.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert fields == {"run_id", "handler", "semantic_input"}


def test_new_core_durable_models_do_not_use_ambiguous_identity_names():
    forbidden = {
        "identity",
        "identity_sha256",
        "logical_key",
        "request_fingerprint",
        "request_sha256",
        "fingerprint",
    }
    for package in ("arc-jobs", "arc-llm", "arc-proposer-reviewer"):
        for path in (ROOT / "packages" / package / "src").glob("**/*.py"):
            if "providers" in path.parts:
                continue  # Provider-private wire names are not public identity contracts.
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    assert node.target.id not in forbidden, (
                        path.relative_to(ROOT),
                        node.target.id,
                    )
                if isinstance(node, ast.Dict):
                    for key in node.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            assert key.value not in forbidden, (
                                path.relative_to(ROOT),
                                key.value,
                            )
