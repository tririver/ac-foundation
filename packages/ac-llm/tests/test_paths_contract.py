from __future__ import annotations

from pathlib import Path

from ac_jobs import RunContext, RunRepository, RunSpec
from ac_llm import (
    JsonOutput,
    LLMCompleted,
    LLMRequest,
    LLMTaskService,
    ModelSelection,
    NativeResumeHandle,
    ProviderExecution,
    ProviderTerminalKind,
)
from ac_llm.identity import semantic_key
from ac_llm.output import CandidateMaterial


def test_explicit_repository_owns_run_task_and_operational_layout(
    tmp_path: Path, adapter, registry
) -> None:
    adapter.steps.append(
        ProviderExecution(
            ProviderTerminalKind.COMPLETED,
            (CandidateMaterial(value={"ok": True}, terminal=True),),
            NativeResumeHandle("codex", "thread"),
        )
    )
    repository = RunRepository(tmp_path / "explicit-root")
    snapshot = repository.create(RunSpec("parent", "test.parent", {"case": "paths"}))
    context = RunContext(repository, snapshot, resume_input=None)
    request = LLMRequest(
        "paths",
        "Return JSON.",
        JsonOutput({"type": "object", "required": ["ok"]}),
        ModelSelection("codex"),
    )
    service = LLMTaskService(registry=registry)
    outcome = service.execute(context, request)
    assert isinstance(outcome, LLMCompleted)
    run = repository.root / "runs" / "parent"
    digest = semantic_key(request).sha256
    state = service._executor._task_store(context, "paths").read()
    assert state is not None
    assert state.request_ref.artifact_id.startswith(f"llm/tasks/{digest}/")
    assert (run / "artifacts" / "manifests").is_dir()
    assert (repository.root / "operational" / "llm" / "global" / "pool.json").is_file()
    assert not (tmp_path / ".ac").exists()
