#!/usr/bin/env bash
set -euo pipefail

python -m build packages/arc-llm
python -m build packages/arc-jobs
python -m build packages/arc-proposer-reviewer
python -m build packages/arc-paper
python -m build packages/arc-render
python -m build packages/arc-domain
python -m build packages/arc-ocr-proofread
python -m build packages/arc-translate
python -m build packages/arc-companion
