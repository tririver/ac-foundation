# arc-llm

`arc-llm` owns reusable host-LLM execution for ARC: immutable requests,
provider and model resolution, structured-output validation, sessions, and
durable recovery over `arc-jobs`. Research-specific prompts and orchestration
belong to the packages that call it.

## Quick start

Installing the Python distribution exposes the `arc-llm` console script. In an
active ARC Skill, use `<skill-dir>/scripts/arc-runtime arc-llm`; inside this
source checkout, `packages/arc-paper/.venv/bin/arc-llm` is the direct shared
development fallback.

Create `local/example/request.json` from the validated v4 template in the
[ARC LLM Quick Start](../../plugins/arc/skills/arc/manuals/arc-llm.md), then run
one typed request in an explicit durable root:

```bash
arc-llm generate \
  --request local/example/request.json \
  --run-root local/example/.arc/llm \
  --host-authority <host-authority>
```

Persist `run.id`. Read the lifecycle at `data.run.status`; on success, the
verified result path is `data.run.result.path`, with a matching item in
`artifacts[]` whose `role` is `result`. Resolve only that returned path under
`<run-root>/runs/<run.id>/`; the model result is not inline in the command
envelope. The Quick Start also provides exact status, pause, and result paths
plus the closed v3 resume template. Use `arc-llm --help` and
`arc-llm <command> --help` for current commands and flags, not the JSON request
contract.

Provider admission pauses by default when effective available system or
container memory falls below 10%. On Linux, host availability uses
`MemAvailable`; cgroup availability also counts inactive file cache reported
by `memory.stat`, because the kernel can reclaim it under pressure. Missing
cgroup statistics fall back to conservative raw headroom. Override the
threshold with `--minimum-available-memory-percent PERCENT`, or bypass the
check explicitly with `--disable-memory-guard`. The same flags are available
for `resume`.

ARC's default **max parallel** provider target is 100 concurrent calls. This
is an admission target, not a hard ceiling: callers may set any positive
`ProviderGateOptions.global_limit`, and a caller-owned worker pool must create
the demand. The memory guard, provider-specific limits, and circuit breaker
can reduce effective concurrency below the target.

Set `<host-authority>` once per run: use `unrestricted` only when the host
explicitly reports unrestricted authority; otherwise use `unknown`. Reuse the
identical value when resuming that run.

## Python API

Use `LLMClient` for a standalone durable task:

```python
from pathlib import Path

from arc_llm import LLMClient, LLMRequest, TextOutput

request = LLMRequest("summary-1", "Summarize the argument.", TextOutput())
result = LLMClient().generate(request, run_root=Path("local/example/.arc/llm"))
```

Package workflows that already have an `arc_jobs.RunContext` should use
`LLMTaskService` instead of creating a nested standalone run.

## Tests

The normal suite is offline; real-provider checks are opt-in:

```bash
python -m pytest packages/arc-llm/tests
```
