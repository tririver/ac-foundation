# arc-proposer-reviewer

`arc-proposer-reviewer` owns typed proposer-worker and reviewer batches,
rounds, dialogue artifacts, consensus results, and verified read-only
projections. It delegates durable execution to `arc-jobs` and every model call
to `arc-llm`; research-workflow policy belongs to the calling workflow.

## Quick start

Run one validated batch request:

```bash
arc-proposer-reviewer run \
  --request local/example/batch.json \
  --run-root local/example/proposer-reviewer
```

Use `arc-proposer-reviewer --help` and
`arc-proposer-reviewer run --help` for request validation, resume, cooperative
stop, inspection, trace, and committed-round queries. `inspect` reports the
durable lifecycle, loop lifecycle counts, current workers and interactions,
actionable pauses, and sanitized failure causes.

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
