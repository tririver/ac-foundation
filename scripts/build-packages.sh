#!/usr/bin/env bash
set -euo pipefail

rm -rf dist
mkdir -p dist

for project in packages/ac-*/pyproject.toml; do
  python -m build --outdir dist "${project%/pyproject.toml}"
done
