# arc-llm

`arc-llm` owns reusable host-LLM execution for ARC: immutable requests,
provider and model resolution, structured-output validation, sessions, and
durable recovery over `arc-jobs`. Research-specific prompts and orchestration
belong to the packages that call it.

## Quick start

Run one typed request in an explicit durable root:

```bash
arc-llm generate \
  --request local/example/request.json \
  --run-root local/example/llm
```

Use `arc-llm --help` and `arc-llm generate --help` for the request contract,
run controls, provider diagnostic, and current options.

## Python API

Use `LLMClient` for a standalone durable task:

```python
from pathlib import Path

from arc_llm import LLMClient, LLMRequest, TextOutput

request = LLMRequest("summary-1", "Summarize the argument.", TextOutput())
result = LLMClient().generate(request, run_root=Path("local/example/llm"))
```

Package workflows that already have an `arc_jobs.RunContext` should use
`LLMTaskService` instead of creating a nested standalone run.

## Tests

The normal suite is offline; real-provider checks are opt-in:

```bash
python -m pytest packages/arc-llm/tests
```
