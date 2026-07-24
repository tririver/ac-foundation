from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for source in (
    ROOT / "arc-proposer-reviewer" / "src",
    ROOT / "arc-jobs" / "src",
    ROOT / "arc-llm" / "src",
):
    sys.path.insert(0, str(source))
