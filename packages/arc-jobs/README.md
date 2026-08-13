# arc-jobs

`arc-jobs` is ARC's zero-dependency durable-run kernel. It owns atomic run
state, immutable artifacts, cooperative stopping, pause/resume, explicit failed-run
recovery, work groups, and the shared command-result codec. It does not own
provider selection, detached processes, or research-domain workflows.

## Quick start

Installing the Python distribution exposes the `arc-jobs` console script. In
an active ARC Skill, use `<skill-dir>/scripts/arc-runtime arc-jobs`; inside this
source checkout, `packages/arc-paper/.venv/bin/arc-jobs` is the direct shared
development fallback.

Inspect a run whose owning workflow exposed a generic root and ID. For a
direct-root owner such as `arc-llm`, preserve the exact `--run-root` input and
pair it with the returned `run.id`:

```bash
arc-jobs status --run-root local/example/runs --run-id run-001
```

The command envelope is a query: read the durable lifecycle at
`data.run.status`, not only top-level `status`. The result artifact ID and
returned path are at `data.run.result.artifact_id` and
`data.run.result.path`; failures and pauses are at `data.run.error` and
`data.run.resume`; exact editable recovery paths are under
`data.run.working_state`. `validate` returns `data.valid` and `data.issues[]`,
while `stop` returns `data.run.stop_requested` and `data.run.status`.

Work groups expose a durable runtime concurrency target. Owners choose the
initial target through `max_workers`; an operator may inspect or change it
without stopping the run:

```bash
arc-jobs workers get \
  --run-root local/example/runs --run-id run-001 --group-id summaries
arc-jobs workers set \
  --run-root local/example/runs --run-id run-001 --group-id summaries \
  --workers 100
```

Increasing the target admits pending units immediately. Decreasing it never
cancels in-flight work; new units wait until the in-flight count falls below
the new target. The target is operational state, not semantic input, and
survives pause, process replacement, and failed-run recovery. Provider gates,
memory admission, and circuit breakers may keep effective concurrency below
the requested target. Only groups that have started have worker controls.

`arc-jobs` deliberately has no resume command. Resume through the package that
created the run, using the same run root and ID. See the
[ARC Jobs Quick Start](../../plugins/arc/skills/arc/manuals/arc-jobs.md) for
launcher selection, command examples, result paths, and recovery boundaries.
Use `arc-jobs --help` and `arc-jobs <command> --help` for current flags.

## Python API

The public repository API can inspect the same durable run:

```python
from arc_jobs import RunRepository

view = RunRepository("local/example/runs").inspect("run-001")
print(view.snapshot.status.value)

workers = RunRepository("local/example/runs").set_group_workers(
    "run-001", "summaries", 100
)
print(workers.target_workers)
```

Higher-level packages create and resume their own runs; use their public
handlers or facades instead of constructing package-internal run specs.
Project-based packages normally provide project-aware status, validation, and
stop commands; use those rather than deriving their internal job roots. There
is no generic fixture-creation CLI because an ownerless run has neither valid
package semantics nor a package resume path.

`failed` records the latest failed execution attempt; it is not a permanent
project terminal state. Only an explicit `RunEngine.resume` may retry it.
That retry increments `recovery_epoch`, snapshots the editable `working/`
tree, gives LLM tasks and work groups a fresh execution namespace, reuses
matching successful group units, and retries failed units. `execute` never
forms an automatic recovery loop, and a succeeded run remains final.

Each run exposes `working/semantic-input.json`, `working/artifacts/`,
`working/candidates/`, `working/index.json`, and `working/last-error.json`.
An agent may edit a readable file to adopt its current bytes or delete an
artifact/candidate to regenerate it. Recovery rehashes current files and emits
`working_state_modified`; if semantic input changed while downstream files
remain it also emits a broad stale-state warning. Immutable specs, old
fingerprints, object-store content, recovery snapshots, and locks are not
edited.

`RunEngine.execute` and `RunEngine.resume` accept an optional runtime-only
`event_sink`. The sink receives each newly fsynced `arc.jobs.event.v1`
document and never receives historical replay. Sink failures are isolated from
the durable run and recorded best-effort as `progress_sink_failed` events.
Progress event data may contain any valid JSON body; individual durable events
remain limited to 256 KiB.

## Tests

From the repository root:

```bash
python -m pytest packages/arc-jobs/tests
```
