#!/usr/bin/env bash
set -euo pipefail

root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)"
python_bin="${PYTHON:-python3}"
build_dir="${AC_FOUNDATION_BUILD_DIR:-$root/local/dist}"

rm -rf "$build_dir"
mkdir -p "$build_dir"

for project in "$root"/packages/ac-*/pyproject.toml; do
  "$python_bin" -m build --outdir "$build_dir" "${project%/pyproject.toml}"
done
