# ac-llm

`ac-llm` owns reusable host-LLM execution for AC Foundation: immutable requests,
provider and model resolution, structured-output validation, sessions, and
durable recovery over `ac-jobs`. Research-specific prompts and orchestration
belong to the packages that call it.

## Quick start

Installing the Python distribution exposes the `ac-llm` console script. AC
Foundation has no agent-host plugin or Skill; product launchers may install and
invoke this command inside their private runtime.

Run `ac-llm` outside a sandbox when possible; sandbox restrictions can cause
provider subprocesses to fail with permission errors.

Create `local/example/request.json` from the public v4 request contract, then
run one typed request in an explicit durable root:

```bash
ac-llm generate \
  --request local/example/request.json \
  --run-root local/example/.ac/llm \
  --host-authority <host-authority>
```

Persist `run.id`. Read the lifecycle at `data.run.status`; on success, the
verified result path is `data.run.result.path`, with a matching item in
`artifacts[]` whose `role` is `result`. Resolve only that returned path under
`<run-root>/runs/<run.id>/`; the model result is not inline in the command
envelope. Use `ac-llm --help` and
`ac-llm <command> --help` for current commands and flags, not the JSON request
contract.

Provider admission pauses by default when effective available system or
container memory falls below 10%. On Linux, host availability uses
`MemAvailable`; cgroup availability also counts inactive file cache reported
by `memory.stat`, because the kernel can reclaim it under pressure. Missing
cgroup statistics fall back to conservative raw headroom. Override the
threshold with `--minimum-available-memory-percent PERCENT`, or bypass the
check explicitly with `--disable-memory-guard`. The same flags are available
for `resume`.

AC Foundation's default **max parallel** provider target is 100 concurrent calls. This
is an admission target, not a hard ceiling: callers may set any positive
`ProviderGateOptions.global_limit`, and a caller-owned worker pool must create
the demand. The memory guard, provider-specific limits, and circuit breaker
can reduce effective concurrency below the target.

When that caller uses an `ac-jobs` work group, its pending-work demand may be
changed live with `ac-jobs workers set`. This
changes the work-group target only. It does not raise or bypass the independent
provider gate, memory guard, or circuit breaker.

Set `<host-authority>` once per run: use `unrestricted` only when the host
explicitly reports unrestricted authority; otherwise use `unknown`. Reuse the
identical value when resuming that run.

## Python API

Use `LLMClient` for a standalone durable task:

```python
from pathlib import Path

from ac_llm import LLMClient, LLMRequest, TextOutput

request = LLMRequest("summary-1", "Summarize the argument.", TextOutput())
result = LLMClient().generate(request, run_root=Path("local/example/.ac/llm"))
```

Package workflows that already have an `ac_jobs.RunContext` should use
`LLMTaskService` instead of creating a nested standalone run.

## Tests

The normal suite is offline; real-provider checks are opt-in:

```bash
python -m pytest packages/ac-llm/tests
```
