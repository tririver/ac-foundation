"""Structural guard for the manually reviewed pre-v1 test closure.

This module proves inventory completeness, explicit disposition, and the
existence of replacement tests.  It deliberately does not claim that an AST
check can prove semantic equivalence between an old invariant and a rewritten
test; that remains a code-review responsibility recorded entry by entry in the
closure fixtures.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).with_name("fixtures")
PACKAGES = ("arc_jobs", "arc_llm", "arc_proposer_reviewer")
SOURCE_REVISION = "f7b4424f427a11f40cd3dea61cb56e90067685dd"
EXPECTED_MANIFESTS = {
    "arc_jobs": {
        "package": "arc-jobs",
        "family_count": 62,
        "parameterized_family_count": 10,
        "families_sha256": "906cdb520337d3e87e220272aa37bf3b5e77d6e1b801a85da4d48e69e7a01f20",
        "source_file_count": 2,
        "source_files": {
            "packages/arc-jobs/tests/test_jobs.py",
            "packages/arc-jobs/tests/test_runtime.py",
        },
    },
    "arc_llm": {
        "package": "arc-llm",
        "family_count": 581,
        "parameterized_family_count": 37,
        "families_sha256": "ee98a8e7fa94816ba0bfd98c099d5dee8aebcc3cbc5d2f5505121a6cfeab8e83",
        "source_file_count": 38,
    },
    "arc_proposer_reviewer": {
        "package": "arc-proposer-reviewer",
        "family_count": 126,
        "parameterized_family_count": 1,
        "families_sha256": "c63beeaf14ceb611db4575471c94b03cac0737eea71b1651d870dfc67b80a985",
        "source_file_count": 9,
        "source_files": {
            "packages/arc-llm/tests/test_controlled.py",
            "packages/arc-llm/tests/test_proposers_reviewer_artifacts.py",
            "packages/arc-llm/tests/test_proposers_reviewer_bench.py",
            "packages/arc-llm/tests/test_proposers_reviewer_cli.py",
            "packages/arc-llm/tests/test_proposers_reviewer_config.py",
            "packages/arc-llm/tests/test_proposers_reviewer_evidence.py",
            "packages/arc-llm/tests/test_proposers_reviewer_llm_integration.py",
            "packages/arc-llm/tests/test_proposers_reviewer_runner.py",
            "packages/arc-llm/tests/test_template_materializer.py",
        },
    },
}


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _current_test_functions(path: str) -> set[str]:
    source = ROOT / path
    assert source.is_file(), f"closure target file does not exist: {path}"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    return {
        item.name
        for item in ast.walk(tree)
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name.startswith("test_")
    }


@pytest.mark.parametrize("package", PACKAGES)
def test_pre_v1_manifest_matches_frozen_inventory_contract(package: str) -> None:
    manifest = _load(FIXTURES / f"{package}_pre_v1_tests.json")
    expected = EXPECTED_MANIFESTS[package]
    assert set(manifest) == {
        "schema_version",
        "source_revision",
        "package",
        "source_files",
        "family_count",
        "families",
    }
    assert manifest["schema_version"] == "arc.legacy_test_manifest.v1"
    assert manifest["source_revision"] == SOURCE_REVISION
    assert manifest["package"] == expected["package"]
    source_files = manifest["source_files"]
    assert isinstance(source_files, list)
    assert len(source_files) == expected["source_file_count"]
    if "source_files" in expected:
        assert set(source_files) == expected["source_files"]
    families = manifest["families"]
    assert isinstance(families, list)
    assert manifest["family_count"] == expected["family_count"] == len(families)
    nodeids = [item["old_nodeid"] for item in families]
    assert len(nodeids) == len(set(nodeids))
    parameterized = 0
    for family in families:
        assert set(family) == {"old_nodeid", "parameter_dimensions"}
        assert isinstance(family["old_nodeid"], str)
        dimensions = family["parameter_dimensions"]
        assert isinstance(dimensions, list)
        parameterized += bool(dimensions)
        for dimension in dimensions:
            assert set(dimension) == {"names", "case_count"}
            assert isinstance(dimension["names"], list) and dimension["names"]
            assert all(
                isinstance(name, str) and name.strip() for name in dimension["names"]
            )
            assert dimension["case_count"] is None or (
                type(dimension["case_count"]) is int
                and dimension["case_count"] > 0
            )
    assert parameterized == expected["parameterized_family_count"]
    canonical = json.dumps(
        sorted(families, key=lambda item: str(item["old_nodeid"])),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == expected["families_sha256"]


@pytest.mark.parametrize("package", PACKAGES)
def test_legacy_closure_is_structurally_complete_and_targets_real_tests(
    package: str,
) -> None:
    manifest = _load(FIXTURES / f"{package}_pre_v1_tests.json")
    closure = _load(FIXTURES / f"{package}_legacy_closure.json")
    assert set(closure) == {"schema_version", "package", "manifest", "groups"}
    assert closure["schema_version"] == "arc.legacy_test_closure.v1"
    assert closure["package"] == manifest["package"]
    assert closure["manifest"] == f"{package}_pre_v1_tests.json"
    groups = closure["groups"]
    assert isinstance(groups, list) and groups

    closed_nodeids: list[str] = []
    group_ids: set[str] = set()
    for group in groups:
        assert set(group) == {
            "group_id",
            "disposition",
            "status",
            "target_nodeids",
            "entries",
        }
        group_id = group["group_id"]
        assert isinstance(group_id, str) and group_id not in group_ids
        group_ids.add(group_id)
        disposition = group["disposition"]
        status = group["status"]
        targets = group["target_nodeids"]
        entries = group["entries"]
        assert disposition in {"retain", "rewrite", "move", "delete"}
        assert status == "closed", (
            f"legacy closure group must be resolved before acceptance: {group_id}"
        )
        assert isinstance(targets, list)
        assert isinstance(entries, list) and len(entries) == 1, (
            "each closure group must describe one old invariant so a broad "
            f"matrix cannot close unrelated behavior: {group_id}"
        )
        if disposition == "delete":
            assert targets == []
        else:
            assert targets
        for entry in entries:
            assert set(entry) == {"old_nodeid", "preserved_invariant", "reason"}
            assert isinstance(entry["old_nodeid"], str)
            assert isinstance(entry["preserved_invariant"], str)
            assert isinstance(entry["reason"], str) and entry["reason"].strip()
            normalized_text = (
                entry["preserved_invariant"] + " " + entry["reason"]
            ).lower()
            for placeholder in (
                "unresolved rewrite concern",
                "concern expressed by",
                "narrowed v1 replacement",
                "listed owner tests",
                "listed r-",
                "contract matrix directly asserts",
                "tbd",
                "todo",
            ):
                assert placeholder not in normalized_text
            assert "test_" not in entry["preserved_invariant"]
            assert entry["preserved_invariant"] in entry["reason"], (
                "the closure reason must name the exact audited invariant: "
                f"{entry['old_nodeid']}"
            )
            if disposition == "delete":
                assert entry["reason"].startswith("Retired pre-v1 behavior ")
            else:
                assert entry["reason"].startswith("Retained v1 contract ")
            closed_nodeids.append(entry["old_nodeid"])
        for target in targets:
            assert isinstance(target, str) and target.count("::") == 1
            path, function = target.split("::", 1)
            assert function in _current_test_functions(path), (
                f"closure target is not a collected AST test function: {target}"
            )

    manifest_nodeids = [item["old_nodeid"] for item in manifest["families"]]
    assert len(closed_nodeids) == len(set(closed_nodeids))
    assert sorted(closed_nodeids) == sorted(manifest_nodeids)
