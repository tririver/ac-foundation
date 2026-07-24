# arc-llm

`arc-llm` is ARC's provider-neutral LLM execution package. It owns immutable
LLM requests, provider and model resolution, structured-output validation,
interactive turns, accepted-result sessions, and durable recovery over
`arc-jobs`.

The package intentionally does not own generic jobs or higher-level research
workflow orchestration.

## Python API

Use `LLMTaskService` inside an existing `arc_jobs.RunContext`. A parent run may
execute multiple tasks; `task_id` is the idempotency key and is required again
when resuming one of them.

```python
from arc_llm import JsonOutput, LLMRequest, LLMTaskService

request = LLMRequest(
    task_id="derive-ward-identity",
    prompt="Return a compact derivation.",
    output=JsonOutput({"type": "object"}),
)
outcome = LLMTaskService().execute(context, request)
```

Use `LLMClient` for a standalone durable run:

```python
from pathlib import Path
from arc_llm import LLMClient, LLMRequest, TextOutput

result = LLMClient().generate(
    LLMRequest("summary-1", "Summarize the argument.", TextOutput()),
    run_root=Path("local/my-run/llm"),
)
```

## CLI

```text
arc-llm generate --request REQUEST.json --run-root DIR [--run-id ID]
arc-llm resume --run-root DIR --run-id ID [--input RESUME.json]
arc-llm status --run-root DIR --run-id ID
arc-llm cancel --run-root DIR --run-id ID [--reason TEXT]
arc-llm doctor [--provider auto|codex|claude|kimi]
```

Every command writes exactly one `arc.command_result.v1` JSON object to stdout.
Accepted provider output is stored as an immutable artifact; provider streams
are never copied to stdout or stderr.

## Identity and reuse

Semantic reuse, execution compatibility, explicit adoption, and the distinction
between logical keys and content digests follow the repository-wide
[identity and reuse architecture](../../docs/architecture/identity-and-reuse.md).

## Optional live smoke

The normal test suite skips the real-provider smoke. Run it explicitly with an
ignored output directory below `arc-tests/`:

```bash
ARC_RUN_NET_TESTS=1 \
ARC_RUN_LIVE_PROVIDER_SMOKE=1 \
ARC_LLM_SMOKE_PROVIDER=codex \
ARC_LLM_SMOKE_ROOT="$PWD/arc-tests/live-provider-smoke/manual-$(date -u +%Y%m%d-%H%M%S)" \
packages/arc-paper/.venv/bin/python -m pytest -q -m live_provider_smoke \
  packages/arc-llm/tests/live/test_provider_smoke.py
```

Confirm `ARC_LLM_SMOKE_ROOT` is ignored before running. The smoke is
single-threaded and makes at most two provider calls: strict JSON start and
an explicit session continuation, followed by a zero-call local replay. The
root must be new or empty. It uses a 120-second idle timeout with automatic
retry, recovery resume, and replacement disabled.
