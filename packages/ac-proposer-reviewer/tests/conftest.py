from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for source in (
    ROOT / "ac-proposer-reviewer" / "src",
    ROOT / "ac-jobs" / "src",
    ROOT / "ac-llm" / "src",
):
    sys.path.insert(0, str(source))
