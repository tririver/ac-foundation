from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from arc_jobs import RunContext, RunRepository, RunSpec
from arc_llm import (
    JsonOutput,
    ExecutionLimits,
    LLMCompleted,
    LLMExecutionOptions,
    LLMRequest,
    LLMTaskService,
    ModelSelection,
    ProviderGateOptions,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("ARC_RUN_LIVE_PROVIDER_SMOKE") != "1"
    or os.environ.get("ARC_RUN_NET_TESTS") != "1",
    reason=(
        "live provider smoke requires ARC_RUN_LIVE_PROVIDER_SMOKE=1 and "
        "ARC_RUN_NET_TESTS=1"
    ),
)
pytestmark = [pytestmark, pytest.mark.live_provider_smoke]


def _tree_digests(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_live_provider_strict_json_continuation_and_local_replay() -> None:
    root_value = os.environ.get("ARC_LLM_SMOKE_ROOT")
    if not root_value:
        pytest.fail("ARC_LLM_SMOKE_ROOT must name an ignored run directory")
    root = Path(root_value).resolve()
    repository_root = Path(__file__).resolve().parents[4]
    smoke_parent = repository_root / "arc-tests"
    if not root.is_relative_to(smoke_parent):
        pytest.fail("ARC_LLM_SMOKE_ROOT must be located under repository arc-tests/")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        pytest.fail("ARC_LLM_SMOKE_ROOT must not exist or must be an empty directory")
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(root)],
        cwd=repository_root,
        check=False,
    )
    if ignored.returncode != 0:
        pytest.fail("ARC_LLM_SMOKE_ROOT must be covered by repository git-ignore")
    root.mkdir(parents=True, exist_ok=True)
    provider = os.environ.get("ARC_LLM_SMOKE_PROVIDER", "auto")
    repository = RunRepository(root)
    parent = repository.create(
        RunSpec(
            "live-provider-smoke",
            "arc.llm.live_smoke.v1",
            {"provider": provider},
        )
    )
    context = RunContext(
        repository,
        parent,
        resume_input=None,
        execution_slice=None,
    )
    service = LLMTaskService()
    options = LLMExecutionOptions(
        limits=ExecutionLimits(idle_timeout_seconds=120),
        gate=ProviderGateOptions(global_limit=1),
    )
    contract = JsonOutput(
        {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        repair="strict",
    )
    first_request = LLMRequest(
        "live-smoke-first",
        "Return JSON with answer equal to 2. No prose.",
        contract,
        ModelSelection(provider),
    )
    first = service.execute(context, first_request, options=options)
    assert isinstance(first, LLMCompleted)
    assert first.value == {"answer": 2}
    assert first.session is not None

    second = service.execute(
        context,
        LLMRequest(
            "live-smoke-second",
            "Continue the same session. Return JSON with answer equal to 3. No prose.",
            contract,
            ModelSelection(provider),
            session=first.session,
        ),
        options=options,
    )
    assert isinstance(second, LLMCompleted)
    assert second.value == {"answer": 3}

    run_root = root / "runs" / parent.run_id
    before_replay = repository.inspect(parent.run_id).snapshot
    effects_before_replay = _tree_digests(run_root / "effects")
    artifacts_before_replay = _tree_digests(run_root / "artifacts")
    replay = service.execute(context, first_request, options=options)
    assert isinstance(replay, LLMCompleted)
    assert replay.value == {"answer": 2}
    after_replay = repository.inspect(parent.run_id).snapshot
    assert after_replay.attempt == before_replay.attempt
    assert after_replay.revision == before_replay.revision
    assert _tree_digests(run_root / "effects") == effects_before_replay
    assert _tree_digests(run_root / "artifacts") == artifacts_before_replay
