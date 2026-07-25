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

## Tests

From the repository root:

```bash
python -m pytest packages/arc-jobs/tests
```
