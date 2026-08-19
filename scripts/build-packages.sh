#!/usr/bin/env bash
set -euo pipefail

rm -rf dist
mkdir -p dist

python -m build --outdir dist packages/arc-llm
python -m build --outdir dist packages/arc-jobs
python -m build --outdir dist packages/arc-proposer-reviewer
python -m build --outdir dist packages/arc-paper
python -m build --outdir dist packages/arc-render
python -m build --outdir dist packages/arc-domain
python -m build --outdir dist packages/arc-ocr-proofread
python -m build --outdir dist packages/arc-translate
python -m build --outdir dist packages/arc-companion
