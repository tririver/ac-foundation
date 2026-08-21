#!/usr/bin/env bash
set -euo pipefail

for pyproject in packages/ac-*/pyproject.toml; do
  python -m build "${pyproject%/pyproject.toml}"
done
