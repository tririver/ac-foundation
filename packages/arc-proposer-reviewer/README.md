# arc-proposer-reviewer

`arc-proposer-reviewer` owns typed proposer-worker and reviewer batches,
rounds, dialogue artifacts, consensus results, and verified read-only
projections. It delegates durable execution to `arc-jobs` and every model call
to `arc-llm`; research-workflow policy belongs to the calling workflow.

## Quick start

Installing the Python distribution exposes the `arc-proposer-reviewer` console
script. The ARC plugin does not install a standalone bin wrapper for this
core-only command: in an active Skill, invoke
`<skill-dir>/scripts/arc-runtime arc-proposer-reviewer`. Inside this source
checkout, `packages/arc-paper/.venv/bin/arc-proposer-reviewer` is a direct
shared development fallback.

Check an installed command before use:

```bash
arc-proposer-reviewer --help
```

Create `local/example/batch.json` from the smallest validated v7 template in
the [ARC Proposer-Reviewer Quick Start](../../plugins/arc/skills/arc/manuals/arc-proposer-reviewer.md).
Validate it locally, then run and query the batch through the public CLI:

```bash
arc-proposer-reviewer validate \
  --request local/example/batch.json

arc-proposer-reviewer run \
  --request local/example/batch.json \
  --run-root local/example/proposer-reviewer \
  --run-id batch-001

arc-proposer-reviewer inspect \
  --run-root local/example/proposer-reviewer --run-id batch-001

arc-proposer-reviewer trace \
  --run-root local/example/proposer-reviewer --run-id batch-001

arc-proposer-reviewer show-round \
  --run-root local/example/proposer-reviewer --run-id batch-001 \
  --loop-id question-1 --round 1
```

Validation is at `data.valid`, lifecycle and pauses at `data.inspection`,
committed references at `data.trace`, and one expanded committed round at
`data.round`. `inspect` reports loop lifecycle counts, current workers and
interactions, actionable pauses, and sanitized failure causes. Help documents
commands and flags; the linked Quick Start owns the complete v7 batch contract,
exact result paths, and v3 resume skeleton.

## Python API

`BatchRunner` is the reusable durable facade:

```python
from arc_proposer_reviewer import BatchRunner

runner = BatchRunner()
inspection = runner.projection(
    "local/example/proposer-reviewer", "batch-001"
).inspect()
```

`BatchRunner.run(..., event_sink=...)` and `resume(..., event_sink=...)`
forward newly persisted `arc.jobs.event.v1` documents for foreground status
rendering. Sink failures do not change the durable batch outcome.

`LoopSpec.review_final_round` defaults to `True`, preserving the usual
proposer-reviewer round. Set it to `False` to make only the final configured
round proposer-only: earlier reviewer `stop` decisions still accept the current
proposal when early stopping is enabled, while a run that reaches its round
limit commits the terminal proposal and retains the latest completed review.

`LoopSpec.revision_context_mode` defaults to `feedback_only`: a delta proposer
receives its previous proposal and its own targeted feedback. Set it to
`RevisionContextMode.FULL_REVIEW_ENVELOPE` when a caller also wants each delta
proposer to receive the complete previous review envelope as broader context.

`LoopSpec.input_ids` optionally selects batch inputs for one loop. Its default,
`None`, preserves the existing behavior of passing every batch input. An empty
tuple passes none; otherwise IDs must be unique references to `BatchRequest.inputs`.

## Tests

The default suite uses fake providers:

```bash
python -m pytest packages/arc-proposer-reviewer/tests
```
