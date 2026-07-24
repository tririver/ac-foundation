# arc-llm

`arc-llm` is ARC's provider-neutral LLM execution package. It owns immutable
LLM requests, provider and model resolution, structured-output validation,
interactive turns, accepted-result sessions, and durable recovery over
`arc-jobs`.

The package intentionally does not own generic jobs or higher-level research
workflow orchestration.

## Python API

Use `LLMTaskService` inside an existing `arc_jobs.RunContext`. A parent run may
execute multiple tasks. `execute_or_resume()` drives a task without creating a
nested run; pause keys are namespaced by the task semantic digest so a parent
can route several paused child tasks safely.

```python
from arc_llm import JsonOutput, LLMRequest, LLMTaskService

request = LLMRequest(
    task_id="derive-ward-identity",
    prompt="Return a compact derivation.",
    output=JsonOutput({"type": "object"}),
)
outcome = LLMTaskService().execute_or_resume(context, request)
```

Resume keys are opaque outside `arc-llm`. A parent workflow that contains
multiple child LLM tasks should call
`resume_input_matches(request, resume_input)` to identify the target task,
rather than parsing or constructing a resume key.

`arc.llm.request.v2` accepts ordered immutable inputs through
`LLMInputArtifact`. Inputs must be `ArtifactSourceRef` values in the same
`arc-jobs` repository; arbitrary paths, URLs, and caller-supplied base64 are
not accepted. The executor verifies digest, size, and media type before a
provider call, then materializes the bytes into the current run.

```python
from arc_llm import LLMInputArtifact, LLMRequest, TextOutput

request = LLMRequest(
    task_id="review-page",
    prompt="Compare the page with the supplied Markdown.",
    output=TextOutput(),
    inputs=(
        LLMInputArtifact("page", page_source_ref, "image/png"),
        LLMInputArtifact("paper", markdown_source_ref, "text/markdown"),
    ),
)
```

The built-in adapters declare delivery per MIME type:

| Provider | PNG/JPEG | Markdown/JSON |
| --- | --- | --- |
| Codex | native `--image` attachment | read-only tool path |
| Claude Code | read-only `Read` path | read-only `Read` path |
| Kimi Code | ACP image content | ACP embedded text resource |

Kimi uses the official `agent-client-protocol` Python SDK for protocol
negotiation, session start/resume, capability checks, and content delivery.
An explicit provider that cannot deliver every input fails with
`invalid_request`; `provider="auto"` considers only adapters supporting all
inputs.

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
Input identity includes input order, ID, normalized MIME type, digest, and
size. Source run IDs, artifact IDs, materialized paths, and auto-resolved
provider/model choices do not affect semantic identity. The execution
fingerprint records the resolved provider/model, adapter version, and actual
per-input delivery modes.

## Optional live smoke

The normal test suite skips the real-provider smoke. Run it explicitly with an
ignored output directory below `local/`:

```bash
ARC_RUN_NET_TESTS=1 \
ARC_RUN_LIVE_PROVIDER_SMOKE=1 \
ARC_LLM_SMOKE_PROVIDER=codex \
ARC_LLM_SMOKE_ROOT="$PWD/local/live-provider-smoke/manual-$(date -u +%Y%m%d-%H%M%S)" \
packages/arc-paper/.venv/bin/python -m pytest -q -m live_provider_smoke \
  packages/arc-llm/tests/live/test_provider_smoke.py
```

Confirm `ARC_LLM_SMOKE_ROOT` is ignored before running. The smoke is
single-threaded and makes at most two provider calls: strict JSON start and
an explicit session continuation, followed by a zero-call local replay. The
root must be new or empty. It uses a 120-second idle timeout with automatic
retry, recovery resume, and replacement disabled.
