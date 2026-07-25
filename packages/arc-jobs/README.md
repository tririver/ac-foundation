# arc-jobs

`arc-jobs` is ARC's zero-dependency durable run kernel. It owns atomic state,
immutable artifacts, cooperative stopping, pause/resume, effect recovery,
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

File leases are transactional: an acquire returns only after the OS lock and
user-only file permissions are in place. Any acquisition failure releases both
the file handle and the in-process lock; `release()` also clears the local lock
even when the OS unlock or close reports an error.

Effects advance from `PREPARED` through `MAY_HAVE_RUN` to saved/committed
output. A `MAY_HAVE_RUN` recovery needs a policy that can prove either
`RETRY_VERIFIED_NOT_RUN` or `RESUME_EXTERNALLY`; absent, uncertain, or
impossible replay decisions pause the run for supervision with the effect ID.
Progress data is recursively body-free: case-insensitive `text`, `token`,
`content`, `output`, `delta`, `prompt`, `candidate`, and `result` keys are
rejected at event, codec, and observer boundaries.

Joined work groups are also immutable within a run. Use
`repository.inspect_group(run_id, group_id)` (or `context.inspect_group(group_id)`
inside a handler) to read every unit's pending or terminal status, value, and
error without claiming or mutating work. Completed units replay in the same run,
including failed units; retrying a failed item requires a new run ID.

For a small cross-process concurrency limit, use an explicit-root lease pool:

```python
from arc_jobs import BoundedLeasePool

pool = BoundedLeasePool("/explicit/run/root/provider-slots", capacity=2)
with pool.acquire(limit=current_limit, checkpoint=context.stop.raise_if_requested):
    call_provider()
```

The first opener durably binds the capacity to the pool root; another process
cannot silently reopen the same root with a different capacity. Acquisition
uses a short cross-process mutex while it counts holders across every capacity
slot. A call may supply a current `limit`, where `1 <= limit <= capacity`,
without changing the persisted pool contract. Lowering that limit does not
revoke existing leases, but no new lease is granted until the total active
holder count is below the new limit. The optional checkpoint keeps blocking
waits cooperatively stoppable. The pool does not create a daemon, detached
worker, or implicit global path.

## CLI

The CLI is intentionally limited to read/control operations:

```text
arc-jobs status   --run-root DIR --run-id ID
arc-jobs stop     --run-root DIR --run-id ID [--reason TEXT]
arc-jobs validate --run-root DIR --run-id ID
```

stdout contains exactly one `arc.command_result.v2` object. A successful stop
acknowledges the current attempt and uses exit code 0. The attempt then pauses
at its next cooperative checkpoint; `resume` continues the same run.
