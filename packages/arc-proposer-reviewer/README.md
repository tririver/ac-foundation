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

## Tests

The default suite uses fake providers:

```bash
python -m pytest packages/arc-proposer-reviewer/tests
```
