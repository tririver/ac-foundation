# arc-jobs

`arc-jobs` is ARC's zero-dependency durable run kernel. It owns atomic state,
immutable artifacts, cooperative cancellation, pause/resume, effect recovery,
joined work groups, and the shared command JSON codec.

It deliberately does **not** own detached processes, provider selection,
arbitrary command execution, daemons, watchdogs, or domain workflows. Agent
hosts that need background execution should launch an ordinary blocking ARC
command in the host's background-job facility.

## Library

```python
from arc_jobs import RunEngine, RunRepository, RunSpec, Succeeded

class Handler:
    name = "example.v1"

    def execute(self, context):
        result = context.artifacts.publish_json("result", {"answer": 42})
        return Succeeded(result)

repository = RunRepository("/explicit/run/root")
snapshot = RunEngine(repository).execute(
    RunSpec("run-001", "example.v1", {"question": "life"}),
    Handler(),
)
```

`RunSpec.semantic_input` is the stable business identity. Runtime controls such
as concurrency, retry, deadlines, credentials, timestamps, and paths do not
belong in it. The repository derives the semantic key; callers never submit a
digest.

The same run ID and semantic input replays. The same run ID with different
semantic input raises `IdempotencyConflictError`. Artifact identity, execution
fingerprints, operational policy, effect-request digests, and resume-input
digests are separate concepts and must not be substituted for one another.
See the canonical
[`identity-and-reuse.md`](../../docs/architecture/identity-and-reuse.md)
policy.

## CLI

The CLI is intentionally limited to read/control operations:

```text
arc-jobs status   --run-root DIR --run-id ID
arc-jobs cancel   --run-root DIR --run-id ID [--reason TEXT]
arc-jobs validate --run-root DIR --run-id ID
```

stdout contains exactly one `arc.command_result.v1` object. A successfully
recorded cancellation is a terminal outcome and uses exit code 0; failure to
request cancellation uses exit code 1.
