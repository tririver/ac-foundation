#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: release-arc.sh VERSION\n' >&2
  printf 'Example: release-arc.sh 0.2.0\n' >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

pause() {
  printf '\n%s\n' "$1"
  printf 'Press Enter to continue, or Ctrl-C to abort... '
  read -r _
}

print_cmd() {
  prefix="$1"
  shift
  printf '%s:' "$prefix"
  for item in "$@"; do
    printf ' %s' "$item"
  done
  printf '\n'
}

run() {
  print_cmd RUN "$@"
  "$@"
}

run_dry() {
  print_cmd 'DRY RUN' "$@"
  "$@"
}

if [ "$#" -ne 1 ]; then
  usage
  exit 2
fi

version="$1"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  usage
  die "VERSION must be SemVer core format X.Y.Z without a leading v"
fi

tag="v${version}"
major="${version%%.*}"
rest="${version#*.}"
minor="${rest%%.*}"
next_minor=$((minor + 1))
internal_range=">=${major}.${minor},<${major}.${next_minor}"

root="$(git rev-parse --show-toplevel)"
cd "$root"

version_paths=(
  "VERSION"
  "plugins/arc/.codex-plugin/plugin.json"
  "plugins/arc/.claude-plugin/plugin.json"
  "plugins/arc/skills/arc/scripts/runtime-constraints.txt"
  "packages/arc-paper/tests/test_import.py"
  "packages/arc-paper/tests/test_package_metadata.py"
  "packages/arc-llm/tests/test_contract_matrix.py"
  "tests/architecture/test_package_dependencies.py"
)
for pyproject in packages/arc-*/pyproject.toml; do
  package_dir="${pyproject%/pyproject.toml}"
  package_name="${package_dir##*/}"
  module_name="${package_name//-/_}"
  version_paths+=("$pyproject")
  init_path="$package_dir/src/$module_name/__init__.py"
  if [ -e "$init_path" ] && grep -q '^__version__[[:space:]]*=' "$init_path"; then
    version_paths+=("$init_path")
  fi
done
install_ref_path="plugins/arc/skills/arc/.arc-install-ref"

existing_version_paths=()
for path in "${version_paths[@]}"; do
  if [ -e "$path" ]; then
    existing_version_paths+=("$path")
  fi
done

if [ "${#existing_version_paths[@]}" -eq 0 ]; then
  die "No ARC version files found under $root"
fi

path_is_release_resume_path() {
  candidate="$1"
  if [ "$candidate" = "$install_ref_path" ]; then
    return 0
  fi
  for path in "${existing_version_paths[@]}"; do
    if [ "$candidate" = "$path" ]; then
      return 0
    fi
  done
  return 1
}

pause "Step 1/9: preflight checks for clean worktree, upstream freshness, release commits, and tag availability."

dirty="$(git status --short --untracked-files=all)"
if [ -n "$dirty" ]; then
  version_only_dirty=1
  untracked="$(git ls-files --others --exclude-standard)"
  changed_files="$(
    {
      git diff --name-only
      git diff --cached --name-only
    } | sort -u
  )"
  if [ -n "$untracked" ] || [ -z "$changed_files" ]; then
    version_only_dirty=0
  fi
  while IFS= read -r changed_path; do
    [ -z "$changed_path" ] && continue
    if ! path_is_release_resume_path "$changed_path"; then
      version_only_dirty=0
    fi
  done <<< "$changed_files"
  if [ "$version_only_dirty" = "1" ]; then
    printf 'Worktree has only release metadata changes; continuing resume.\n'
  else
    printf '%s\n' "$dirty" >&2
    die "Worktree is dirty; commit or stash changes before release"
  fi
fi

tracked_generated="$(
  git ls-files | grep -E '(^|/)(__pycache__|\.pytest_cache)(/|$)|\.pyc$' || true
)"
if [ -n "$tracked_generated" ]; then
  printf '%s\n' "$tracked_generated" >&2
  die "Generated Python cache files are tracked; remove them before release"
fi

branch="$(git branch --show-current)"
if [ -z "$branch" ]; then
  die "Detached HEAD; checkout a release branch before running this script"
fi

remote_name="$(git config "branch.${branch}.remote" || true)"
if [ -z "$remote_name" ]; then
  die "Branch $branch has no upstream remote"
fi

upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
if [ -z "$upstream" ]; then
  die "Branch $branch has no upstream branch"
fi

run git fetch --tags "$remote_name"

local_rev="$(git rev-parse HEAD)"
target_tag_exists=0
if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
  target_tag_exists=1
  target_tag_rev="$(git rev-list -n 1 "$tag")"
  if [ "$target_tag_rev" != "$local_rev" ]; then
    die "Tag already exists but does not point to HEAD: $tag"
  fi
  printf 'Reusing existing local tag %s at HEAD.\n' "$tag"
fi

upstream_rev="$(git rev-parse "$upstream")"
merge_base="$(git merge-base HEAD "$upstream")"
if [ "$local_rev" = "$upstream_rev" ]; then
  printf 'Branch %s is synchronized with %s.\n' "$branch" "$upstream"
elif [ "$local_rev" = "$merge_base" ]; then
  die "Branch is behind upstream $upstream; pull/rebase before release"
elif [ "$upstream_rev" = "$merge_base" ]; then
  ahead_count="$(git rev-list --count "${upstream}..HEAD")"
  printf 'Branch %s is ahead of %s by %s commit(s); release push will include them.\n' "$branch" "$upstream" "$ahead_count"
else
  die "Branch has diverged from upstream $upstream; reconcile before release"
fi

latest_release_tag="$(git tag --list 'v[0-9]*' --sort=-v:refname | sed -n '1p')"
if [ -n "$latest_release_tag" ]; then
  commit_count="$(git rev-list --count "${latest_release_tag}..HEAD")"
  if [ "$commit_count" = "0" ]; then
    if [ "$latest_release_tag" = "$tag" ] && [ "$target_tag_exists" = "1" ]; then
      printf 'Release tag %s already points at HEAD; resuming release push/stable steps.\n' "$tag"
    else
      die "No committed changes since $latest_release_tag; refusing empty release"
    fi
  else
    printf 'Committed changes since %s: %s\n' "$latest_release_tag" "$commit_count"
  fi
else
  printf 'No existing v* release tag found; treating this as first release.\n'
fi

pause "Step 2/9: bump plugin manifests, Python package versions, internal dependency ranges, and version tests to $version."

python3 - "$version" "$internal_range" "${existing_version_paths[@]}" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

version = sys.argv[1]
internal_range = sys.argv[2]
paths = [Path(item) for item in sys.argv[3:]]

internal_dep_re = re.compile(r"(arc-[a-z0-9-]+)>=\d+\.\d+(?:\.\d+)?,<\d+\.\d+")


def replace_once(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"Could not update version in {path}")
    path.write_text(new_text, encoding="utf-8")


def replace_all(path: Path, pattern: re.Pattern[str], replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    new_text = pattern.sub(replacement, text)
    path.write_text(new_text, encoding="utf-8")


for path in paths:
    if path.name == "VERSION":
        path.write_text(version + "\n", encoding="utf-8")
    elif path.name == "runtime-constraints.txt":
        replace_once(
            path,
            r"^(# Direct external dependencies tested for ARC v)\d+\.\d+\.\d+(\.)$",
            rf"\g<1>{version}\2",
        )
    elif path.suffix == ".json":
        json.loads(path.read_text(encoding="utf-8"))
        replace_once(path, r'^(\s*"version"\s*:\s*")[^"]+(")', rf"\g<1>{version}\2")
    elif path.name == "pyproject.toml":
        replace_once(path, r'^(version\s*=\s*")[^"]+(")', rf"\g<1>{version}\2")
        replace_all(path, internal_dep_re, rf"\1{internal_range}")
    elif path.name == "__init__.py":
        replace_once(path, r'^(__version__\s*=\s*")[^"]+(")', rf"\g<1>{version}\2")
    elif path.name == "test_import.py":
        replace_once(path, r'(__version__\s*==\s*")[^"]+(")', rf"\g<1>{version}\2")
    elif path.name == "test_package_metadata.py":
        replace_once(
            path,
            r'(project\["version"\]\s*==\s*")[^"]+(")',
            rf"\g<1>{version}\2",
        )
        replace_all(path, internal_dep_re, rf"\1{internal_range}")
    elif path.name == "test_contract_matrix.py":
        replace_once(
            path,
            r'(observed\["version"\]\s*==\s*")[^"]+(")',
            rf"\g<1>{version}\2",
        )
    elif path.name == "test_package_dependencies.py":
        replace_once(path, r'^(RELEASE\s*=\s*")[^"]+(")', rf"\g<1>{version}\2")
PY

version_changed=1
if git diff --quiet -- "${existing_version_paths[@]}"; then
  version_changed=0
  printf 'Version files already at %s; continuing without a bump commit.\n' "$version"
else
  if [ "$target_tag_exists" = "1" ]; then
    die "Tag $tag already exists at the pre-bump HEAD; remove it before changing version files"
  fi
  printf '\nVersion diff:\n'
  git diff -- "${existing_version_paths[@]}"
fi

pause "Step 3/9: validate bumped metadata."

python3 - "$version" "$internal_range" "$root" <<'PY'
from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

version = sys.argv[1]
internal_range = sys.argv[2]
root = Path(sys.argv[3])
packages = sorted(
    path.parent.name
    for path in (root / "packages").glob("arc-*/pyproject.toml")
)
if not packages:
    raise SystemExit("no ARC packages discovered")

if (root / "VERSION").read_text(encoding="utf-8").strip() != version:
    raise SystemExit("root VERSION mismatch")

marketplace_path = root / ".agents/plugins/marketplace.json"
marketplace = json.loads(marketplace_path.read_text(encoding="utf-8"))
if marketplace.get("name") != "arc":
    raise SystemExit(f"{marketplace_path} marketplace name mismatch")
plugins = marketplace.get("plugins")
if not isinstance(plugins, list) or len(plugins) != 1:
    raise SystemExit(f"{marketplace_path} must expose exactly the ARC plugin")
entry = plugins[0]
source = entry.get("source")
if entry.get("name") != "arc" or source != {
    "source": "local",
    "path": "./plugins/arc",
}:
    raise SystemExit(f"{marketplace_path} ARC plugin source mismatch")
if entry.get("policy") != {
    "installation": "AVAILABLE",
    "authentication": "ON_INSTALL",
}:
    raise SystemExit(f"{marketplace_path} ARC plugin policy mismatch")
if entry.get("category") != "Productivity":
    raise SystemExit(f"{marketplace_path} ARC plugin category mismatch")

claude_marketplace_path = root / ".claude-plugin/marketplace.json"
claude_marketplace = json.loads(
    claude_marketplace_path.read_text(encoding="utf-8")
)
if claude_marketplace.get("name") != "arc":
    raise SystemExit(f"{claude_marketplace_path} marketplace name mismatch")
if claude_marketplace.get("owner") != {"name": "ARC"}:
    raise SystemExit(f"{claude_marketplace_path} marketplace owner mismatch")
claude_plugins = claude_marketplace.get("plugins")
if not isinstance(claude_plugins, list) or len(claude_plugins) != 1:
    raise SystemExit(
        f"{claude_marketplace_path} must expose exactly the ARC plugin"
    )
claude_entry = claude_plugins[0]
if (
    claude_entry.get("name") != "arc"
    or claude_entry.get("source") != "./plugins/arc"
):
    raise SystemExit(f"{claude_marketplace_path} ARC plugin source mismatch")
if claude_entry.get("category") != "productivity":
    raise SystemExit(f"{claude_marketplace_path} ARC plugin category mismatch")

manifest_paths = [
    root / "plugins/arc/.codex-plugin/plugin.json",
    root / "plugins/arc/.claude-plugin/plugin.json",
]
for manifest in manifest_paths:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("version") != version:
        raise SystemExit(f"{manifest} version mismatch")
claude_manifest = json.loads(manifest_paths[1].read_text(encoding="utf-8"))
if claude_entry.get("description") != claude_manifest.get("description"):
    raise SystemExit(
        f"{claude_marketplace_path} ARC plugin description mismatch"
    )

install_ref_path = root / "plugins/arc/skills/arc/.arc-install-ref"
install_ref = install_ref_path.read_text(encoding="utf-8").strip()
if not re.fullmatch(r"[0-9a-fA-F]{40}", install_ref):
    raise SystemExit(f"{install_ref_path} must contain a full commit SHA")

for constraints in [
    root / "plugins/arc/skills/arc/scripts/runtime-constraints.txt",
]:
    expected = f"# Direct external dependencies tested for ARC v{version}."
    first_line = constraints.read_text(encoding="utf-8").splitlines()[0]
    if first_line != expected:
        raise SystemExit(f"{constraints} release marker mismatch")

for package in packages:
    pyproject = root / "packages" / package / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    if data["project"]["version"] != version:
        raise SystemExit(f"{pyproject} version mismatch")
    for dep in data["project"].get("dependencies", []):
        if dep.startswith("arc-") and internal_range not in dep:
            raise SystemExit(f"{pyproject} dependency mismatch: {dep}")
PY

run git diff --check -- "${existing_version_paths[@]}"
if command -v claude >/dev/null 2>&1; then
  run claude plugin validate plugins/arc
else
  printf 'SKIP: claude not found on PATH; using built-in manifest checks.\n'
fi

pause "Step 4/9: commit version bump if needed."

if [ "$version_changed" = "1" ]; then
  run git add "${existing_version_paths[@]}"
  run git commit -m "chore: release ${tag}"
else
  printf 'SKIP: no version bump commit needed.\n'
fi

pause "Step 5/9: pin the plugin runtime to the complete versioned source commit."

pinned_ref="$(sed -n '1p' "$install_ref_path")"
head_parent="$(git rev-parse HEAD^ 2>/dev/null || true)"
head_paths="$(git diff-tree --no-commit-id --name-only -r HEAD)"
already_pinned=0
if [ -n "$head_parent" ] && [ "$pinned_ref" = "$head_parent" ] && [ "$head_paths" = "$install_ref_path" ]; then
  already_pinned=1
fi

if [ "$target_tag_exists" = "1" ]; then
  if [ "$already_pinned" != "1" ]; then
    die "Tag $tag does not point to a valid plugin source-pin commit"
  fi
  printf 'SKIP: tagged release already pins source commit %s.\n' "$pinned_ref"
elif [ "$already_pinned" = "1" ]; then
  printf 'Plugin runtime already pins the preceding source commit %s.\n' "$pinned_ref"
else
  source_ref="$(git rev-parse HEAD)"
  printf '%s\n' "$source_ref" > "$install_ref_path"
  run git add "$install_ref_path"
  run git commit -m "chore(plugin): pin ARC ${version} source"
fi

pause "Step 6/9: create release tag $tag if needed."

if [ "$target_tag_exists" = "1" ]; then
  printf 'SKIP: release tag %s already exists at HEAD.\n' "$tag"
else
  run git tag -a "$tag" -m "$tag"
fi

pause "Step 7/9: dry-run push release branch and tag."

run_dry git push --dry-run "$remote_name" "HEAD:${branch}" "$tag"

pause "Step 8/9: push release branch and tag."

run git push "$remote_name" "HEAD:${branch}" "$tag"

pause "Step 9/9: dry-run then push stable branch to this release commit."

run_dry git push --dry-run "$remote_name" "HEAD:stable"
pause "Final remote mutation: push stable branch to $tag."
run git push "$remote_name" "HEAD:stable"

printf '\nRelease %s pushed. Create GitHub Release from tag %s when release notes are ready.\n' "$version" "$tag"
