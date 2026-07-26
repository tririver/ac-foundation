# arc-jobs

`arc-jobs` is ARC's zero-dependency durable-run kernel. It owns atomic run
state, immutable artifacts, cooperative stopping, pause/resume, effect
recovery, work groups, and the shared command-result codec. It does not own
provider selection, detached processes, or research-domain workflows.

## Quick start

Inspect a run created by an ARC package:

```bash
arc-jobs status --run-root local/example/runs --run-id run-001
```

Use `arc-jobs --help` and `arc-jobs status --help` for the current control
commands and arguments.

## Python API

The public repository API can inspect the same durable run:

```python
from arc_jobs import RunRepository

view = RunRepository("local/example/runs").inspect("run-001")
print(view.snapshot.status.value)
```

Higher-level packages create and resume their own runs; use their public
handlers or facades instead of constructing package-internal run specs.

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
