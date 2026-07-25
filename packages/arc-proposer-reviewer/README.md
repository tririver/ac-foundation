# arc-proposer-reviewer

`arc-proposer-reviewer` coordinates typed proposer workers and one reviewer
over one or more rounds. It delegates durable execution and concurrency to
`arc-jobs`, and delegates every model call, output validation, session, and
provider concern to `arc-llm`.

The package deliberately contains no provider transport, environment handling,
filesystem locking, thread pool, research-workflow policy, or output
fabrication.

## Python API

Construct a `BatchRequest`, an `LLMTaskService`, and a
`ProposerReviewerService`. Execute it inside an `arc-jobs` `RunContext`, either
through `ProposerReviewerHandler` and `RunEngine` or through an embedding
handler.

Completed round history is observed through one read-only projection:

```python
inspection = inspect_batch(repository, run_id)
trace = read_batch_trace(repository, run_id)
round_one = read_batch_round(repository, run_id, loop_id, 1)
```

Inspection is available while a batch is pending, running, paused, or terminal.
Its worker activity counts are explicitly best effort. Trace contains only
rounds atomically committed by each loop state; an artifact published before
that commit is not visible. The run revision and per-loop revision vector
identify the observation without claiming a globally linearized snapshot.
Trace references expose logical IDs and content digests, never sessions, task
IDs, private group IDs, resume records, or physical paths. `read_batch_round`
is the only call that expands proposal and review JSON.

Worker identities bind only the worker's actual semantic inputs; they never
bind a physical run directory or concurrency limit.

## CLI

```text
arc-proposer-reviewer validate --request REQUEST.json
arc-proposer-reviewer run --request REQUEST.json --run-root DIR [--run-id ID]
arc-proposer-reviewer resume --run-root DIR --run-id ID [--input INPUT.json]
arc-proposer-reviewer inspect --run-root DIR --run-id ID [--include-trace]
arc-proposer-reviewer trace --run-root DIR --run-id ID
arc-proposer-reviewer show-round --run-root DIR --run-id ID --loop-id ID --round N
```

`validate`, `inspect`, `trace`, and `show-round` perform no model call. `run`
and `resume` are blocking commands. Every command uses the shared
`arc.command_result.v2` command envelope; `inspect --include-trace` retains its
inspection and emits a warning when a strict trace cannot be verified.
