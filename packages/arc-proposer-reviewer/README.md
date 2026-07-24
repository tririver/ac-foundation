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

The durable identity rules are defined by
[`identity-and-reuse.md`](../../docs/architecture/identity-and-reuse.md).
In particular, worker identities bind only the worker's actual semantic inputs;
they never bind a physical run directory or concurrency limit.

## CLI

```text
arc-proposer-reviewer validate --request REQUEST.json
arc-proposer-reviewer run --request REQUEST.json --run-root DIR [--run-id ID]
arc-proposer-reviewer resume --run-root DIR --run-id ID [--input INPUT.json]
```

`validate` performs no model call. `run` and `resume` are blocking commands and
use the shared `arc.command_result.v1` command envelope.
