from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ROOT / "packages"
RELEASE = "1.0.1"

ALLOWED = {
    "arc_jobs": set(),
    "arc_llm": {"arc_jobs"},
    "arc_proposer_reviewer": {"arc_jobs", "arc_llm"},
    "arc_paper": {"arc_jobs", "arc_llm"},
    "arc_domain": {"arc_jobs", "arc_llm", "arc_paper"},
    "arc_translate": {"arc_jobs", "arc_llm", "arc_paper"},
    "arc_companion": {
        "arc_jobs",
        "arc_llm",
        "arc_paper",
        "arc_translate",
    },
}

DIST_TO_MODULE = {name.replace("_", "-"): name for name in ALLOWED}
REQUIRED_EXTERNAL_DEPENDENCIES = {
    "arc_companion": {"beautifulsoup4>=4.12"},
}
PROJECT_URLS = {
    "Homepage": "https://github.com/tririver/arc",
    "Repository": "https://github.com/tririver/arc",
    "Issues": "https://github.com/tririver/arc/issues",
}
ARC_DEPENDENCY = re.compile(
    r"^(arc-[a-z0-9-]+)>=1\.0\.1,<1\.1$"
)


def _projects() -> dict[str, tuple[Path, dict]]:
    found: dict[str, tuple[Path, dict]] = {}
    for path in sorted(PACKAGES.glob("arc-*/pyproject.toml")):
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        project = document["project"]
        module = project["name"].replace("-", "_")
        found[module] = (path.parent, project)
    return found


def _literal_dynamic_import(node: ast.Call) -> str | None:
    if (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "importlib"
        and node.func.attr == "import_module"
    ) or (isinstance(node.func, ast.Name) and node.func.id == "__import__"):
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(
            node.args[0].value, str
        ):
            return node.args[0].value
    return None


def _imports(package_dir: Path) -> set[str]:
    edges: set[str] = set()
    for path in package_dir.glob("src/**/*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            elif isinstance(node, ast.Call):
                dynamic = _literal_dynamic_import(node)
                if dynamic:
                    names.append(dynamic)
            for imported in names:
                root = imported.split(".", 1)[0]
                if root.startswith("arc_"):
                    edges.add(root)
    return edges


def _declared(project: dict) -> set[str]:
    edges: set[str] = set()
    for dependency in project.get("dependencies", []):
        if not dependency.startswith("arc-"):
            continue
        match = ARC_DEPENDENCY.fullmatch(dependency)
        assert match, (
            f"ARC dependency {dependency!r} must use the unified "
            f">={RELEASE},<1.1 release train"
        )
        edges.add(match.group(1).replace("-", "_"))
    return edges


def test_all_packages_use_root_release_and_known_dependency_rows():
    projects = _projects()
    assert set(projects) == set(ALLOWED)
    assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() == RELEASE
    for module, (_, project) in projects.items():
        assert project["version"] == RELEASE, module


def test_retired_arc_mcp_surfaces_stay_absent():
    package = PACKAGES / "arc-mcp"
    assert not (package / "pyproject.toml").exists()
    assert not list((package / "src").rglob("*.py"))
    assert not list((package / "tests").rglob("*.py"))
    assert not [
        path
        for path in (ROOT / "plugins/arc-mcp").rglob("*")
        if path.is_file()
    ]


def test_all_packages_publish_complete_distribution_metadata():
    for module, (package_dir, project) in _projects().items():
        assert project["readme"] == "README.md", module
        readme = package_dir / project["readme"]
        assert readme.is_file(), module
        readme_text = readme.read_text(encoding="utf-8")
        command = project["name"]
        assert readme_text.startswith(f"# {command}\n"), module
        assert f"{command} --help" in readme_text, module
        assert f"python -m pytest packages/{command}/tests" in readme_text, module
        assert project["license"] == "MIT", module
        assert project["authors"] == [{"name": "ARC"}], module
        assert "Programming Language :: Python :: 3.11" in project["classifiers"], module
        assert project["urls"] == PROJECT_URLS, module


def test_declared_and_imported_arc_edges_are_direct_and_allowed():
    for module, (package_dir, project) in _projects().items():
        declared = _declared(project)
        imported = _imports(package_dir) - {module}
        assert imported <= declared, (
            f"{module} imports ARC packages without direct dependencies: "
            f"{sorted(imported - declared)}"
        )
        assert declared | imported <= ALLOWED[module], (
            f"{module} has forbidden ARC dependency edges: "
            f"{sorted((declared | imported) - ALLOWED[module])}"
        )


def test_known_direct_external_dependencies_are_declared():
    projects = _projects()
    for module, required in REQUIRED_EXTERNAL_DEPENDENCIES.items():
        dependencies = set(projects[module][1].get("dependencies", ()))
        assert required <= dependencies, (
            f"{module} is missing direct external dependencies: "
            f"{sorted(required - dependencies)}"
        )


def test_arc_dependency_graph_is_acyclic():
    graph = {
        module: (_declared(project) | (_imports(directory) - {module}))
        for module, (directory, project) in _projects().items()
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, trail: tuple[str, ...]) -> None:
        if node in visiting:
            raise AssertionError("ARC package dependency cycle: " + " -> ".join((*trail, node)))
        if node in visited:
            return
        visiting.add(node)
        for child in graph[node]:
            visit(child, (*trail, node))
        visiting.remove(node)
        visited.add(node)

    for module in graph:
        visit(module, ())
