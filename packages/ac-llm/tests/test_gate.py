from __future__ import annotations

import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path

import pytest

from ac_jobs import BoundedLeasePool
from ac_llm import (
    FailureCategory,
    ProviderFailure,
    ProviderGateOptions,
)
from ac_llm.gate import ProviderCallGate
from ac_llm.memory import MemoryAvailability


def _checkpoint() -> None:
    return None


def _crash_while_holding_half_open_probe(root: str, marker: str) -> None:
    gate = ProviderCallGate(
        Path(root),
        ProviderGateOptions(
            global_limit=2,
            circuit_failure_threshold=1,
            circuit_cooldown_seconds=1,
        ),
        clock=lambda: 2.0,
    )
    gate.acquire("codex", checkpoint=_checkpoint)
    Path(marker).write_text("acquired", encoding="utf-8")
    os._exit(0)


def test_rate_limit_opens_durable_circuit_and_allows_one_half_open_probe(
    tmp_path: Path,
) -> None:
    now = [100.0]
    options = ProviderGateOptions(
        global_limit=3,
        circuit_failure_threshold=3,
        circuit_cooldown_seconds=15,
    )
    gate = ProviderCallGate(
        tmp_path / "operational" / "llm",
        options,
        clock=lambda: now[0],
    )
    with gate.acquire("codex", checkpoint=_checkpoint) as permit:
        permit.record_failure(
            ProviderFailure(
                "limited",
                category=FailureCategory.RATE_LIMIT,
            )
        )

    with pytest.raises(ProviderFailure) as opened:
        gate.acquire("codex", checkpoint=_checkpoint)
    assert opened.value.details["code"] == "provider_circuit_open"
    assert opened.value.details["retry_after_seconds"] == 15

    now[0] = 116.0
    probe = gate.acquire("codex", checkpoint=_checkpoint)
    with pytest.raises(ProviderFailure):
        gate.acquire("codex", checkpoint=_checkpoint)
    probe.record_success()
    probe.release()

    with gate.acquire("codex", checkpoint=_checkpoint) as permit:
        permit.record_success()


def test_rate_limit_retry_after_controls_and_clamps_circuit_cooldown(
    tmp_path: Path,
) -> None:
    now = [100.0]
    gate = ProviderCallGate(
        tmp_path / "operational" / "llm",
        ProviderGateOptions(circuit_cooldown_seconds=900),
        clock=lambda: now[0],
    )
    with gate.acquire("codex", checkpoint=_checkpoint) as permit:
        permit.record_failure(
            ProviderFailure(
                "limited",
                category=FailureCategory.RATE_LIMIT,
                retry_after_seconds=2,
            )
        )

    with pytest.raises(ProviderFailure) as opened:
        gate.acquire("codex", checkpoint=_checkpoint)
    assert opened.value.details["retry_after_seconds"] == 2

    now[0] = 102.1
    with gate.acquire("codex", checkpoint=_checkpoint) as permit:
        permit.record_success()


def test_non_circuit_half_open_failure_releases_probe(tmp_path: Path) -> None:
    now = [0.0]
    gate = ProviderCallGate(
        tmp_path / "operational" / "llm",
        ProviderGateOptions(
            global_limit=2,
            circuit_failure_threshold=1,
            circuit_cooldown_seconds=1,
        ),
        clock=lambda: now[0],
    )
    with gate.acquire("codex", checkpoint=_checkpoint) as permit:
        permit.record_failure(
            ProviderFailure(
                "transport",
                category=FailureCategory.TRANSPORT,
            )
        )
    now[0] = 2.0
    with gate.acquire("codex", checkpoint=_checkpoint) as probe:
        probe.record_failure(
            ProviderFailure(
                "bad request",
                category=FailureCategory.INVALID_REQUEST,
            )
        )
    with gate.acquire("codex", checkpoint=_checkpoint) as next_probe:
        next_probe.record_success()


@pytest.mark.parametrize(
    "category",
    [
        FailureCategory.AUTHENTICATION,
        FailureCategory.QUOTA,
        FailureCategory.RATE_LIMIT,
    ],
)
def test_provider_fatal_categories_open_circuit_immediately(
    tmp_path: Path, category: FailureCategory
) -> None:
    gate = ProviderCallGate(
        tmp_path / category.value,
        ProviderGateOptions(global_limit=2, circuit_failure_threshold=9),
    )
    with gate.acquire("codex", checkpoint=_checkpoint) as permit:
        permit.record_failure(
            ProviderFailure(
                "fatal",
                category=category,
            )
        )
    with pytest.raises(ProviderFailure) as opened:
        gate.acquire("codex", checkpoint=_checkpoint)
    assert opened.value.category is category


def test_circuit_record_errors_never_replace_provider_result(
    tmp_path: Path, monkeypatch
) -> None:
    gate = ProviderCallGate(
        tmp_path / "operational" / "llm",
        ProviderGateOptions(global_limit=1),
    )
    success = gate.acquire("codex", checkpoint=_checkpoint)
    monkeypatch.setattr(
        gate,
        "_record_success",
        lambda *args: (_ for _ in ()).throw(OSError("state write failed")),
    )
    success.record_success()
    assert isinstance(success.record_error, OSError)
    success.release()

    failed = gate.acquire("codex", checkpoint=_checkpoint)
    monkeypatch.setattr(
        gate,
        "_record_failure",
        lambda *args: (_ for _ in ()).throw(OSError("state write failed")),
    )
    original = ProviderFailure(
        "provider failure",
        category=FailureCategory.TRANSPORT,
    )
    failed.record_failure(original)
    assert isinstance(failed.record_error, OSError)
    assert original.category is FailureCategory.TRANSPORT
    failed.release()


def test_crashed_half_open_probe_is_released_for_the_next_process(
    tmp_path: Path,
) -> None:
    root = tmp_path / "operational" / "llm"
    now = [0.0]
    gate = ProviderCallGate(
        root,
        ProviderGateOptions(
            global_limit=2,
            circuit_failure_threshold=1,
            circuit_cooldown_seconds=1,
        ),
        clock=lambda: now[0],
    )
    with gate.acquire("codex", checkpoint=_checkpoint) as permit:
        permit.record_failure(
            ProviderFailure(
                "transport",
                category=FailureCategory.TRANSPORT,
            )
        )

    marker = tmp_path / "probe-acquired"
    process = multiprocessing.get_context("fork").Process(
        target=_crash_while_holding_half_open_probe,
        args=(str(root), str(marker)),
    )
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 0
    assert marker.read_text(encoding="utf-8") == "acquired"

    now[0] = 2.0
    with gate.acquire("codex", checkpoint=_checkpoint) as recovered:
        recovered.record_success()


def test_half_open_record_write_failure_still_releases_probe_lease(
    tmp_path: Path, monkeypatch
) -> None:
    now = [0.0]
    gate = ProviderCallGate(
        tmp_path / "operational" / "llm",
        ProviderGateOptions(
            global_limit=2,
            circuit_failure_threshold=1,
            circuit_cooldown_seconds=1,
        ),
        clock=lambda: now[0],
    )
    with gate.acquire("codex", checkpoint=_checkpoint) as permit:
        permit.record_failure(
            ProviderFailure(
                "transport",
                category=FailureCategory.TRANSPORT,
            )
        )
    now[0] = 2.0
    probe = gate.acquire("codex", checkpoint=_checkpoint)
    monkeypatch.setattr(
        gate,
        "_record_success",
        lambda *args: (_ for _ in ()).throw(OSError("state write failed")),
    )
    probe.record_success()
    assert isinstance(probe.record_error, OSError)
    probe.release()

    # The durable circuit write failed, but the crash-safe lease—not a
    # persistent boolean—controls liveness, so another probe can proceed.
    next_probe = gate.acquire("codex", checkpoint=_checkpoint)
    next_probe.release()


def test_changed_operational_limit_reuses_fixed_pool_and_honors_lower_limit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "operational" / "llm"
    wider = ProviderCallGate(root, ProviderGateOptions(global_limit=3))
    held = wider.acquire("codex", checkpoint=_checkpoint)
    calls = 0

    class StopWaiting(Exception):
        pass

    def checkpoint() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise StopWaiting

    narrower = ProviderCallGate(root, ProviderGateOptions(global_limit=1))
    with pytest.raises(StopWaiting):
        narrower.acquire("codex", checkpoint=checkpoint)
    assert calls == 2
    held.record_success()
    held.release()
    with narrower.acquire("codex", checkpoint=_checkpoint) as permit:
        permit.record_success()


def test_provider_limit_does_not_consume_other_provider_capacity(
    tmp_path: Path,
) -> None:
    gate = ProviderCallGate(
        tmp_path / "operational" / "llm",
        ProviderGateOptions(
            global_limit=2,
            provider_limits={"codex": 1, "claude": 1},
        ),
    )
    codex = gate.acquire("codex", checkpoint=_checkpoint)
    claude = gate.acquire("claude", checkpoint=_checkpoint)
    codex.record_success()
    claude.record_success()
    codex.release()
    claude.release()


def test_provider_names_are_hashed_before_use_as_path_names(tmp_path: Path) -> None:
    provider = "../../unexpected"
    root = tmp_path / "operational" / "llm"
    gate = ProviderCallGate(
        root,
        ProviderGateOptions(
            global_limit=1,
            provider_limits={provider: 1},
        ),
    )
    with gate.acquire(provider, checkpoint=_checkpoint) as permit:
        permit.record_success()
    namespace = hashlib.sha256(provider.encode()).hexdigest()
    assert (root / "providers" / namespace / "pool.json").is_file()
    assert not (tmp_path / "unexpected").exists()


def test_inflight_success_cannot_clear_a_newer_circuit_epoch(tmp_path: Path) -> None:
    gate = ProviderCallGate(
        tmp_path / "operational" / "llm",
        ProviderGateOptions(global_limit=3),
    )
    older = gate.acquire("codex", checkpoint=_checkpoint)
    concurrent = gate.acquire("codex", checkpoint=_checkpoint)
    older.record_failure(
        ProviderFailure(
            "limited",
            category=FailureCategory.RATE_LIMIT,
        )
    )
    older.release()
    concurrent.record_success()
    concurrent.release()
    with pytest.raises(ProviderFailure) as opened:
        gate.acquire("codex", checkpoint=_checkpoint)
    assert opened.value.details["code"] == "provider_circuit_open"


def test_gate_accepts_typed_failures_and_has_no_endpoint_status_surface() -> None:
    failure = ProviderFailure(
        "typed",
        category=FailureCategory.AUTHENTICATION,
    )
    assert not failure.retryable
    wrapped = RuntimeError("wrapper")
    wrapped.__cause__ = failure
    assert not isinstance(wrapped, ProviderFailure)
    assert "endpoint" not in inspect.signature(ProviderCallGate.acquire).parameters
    assert "url" not in ProviderGateOptions.__dataclass_fields__


def test_gate_uses_repository_operational_root_at_provider_boundary(
    tmp_path: Path, adapter, registry
) -> None:
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
    from ac_llm.output import CandidateMaterial

    adapter.steps.append(
        ProviderExecution(
            ProviderTerminalKind.COMPLETED,
            (CandidateMaterial(value={"answer": 1}, terminal=True),),
            NativeResumeHandle("codex", "thread"),
        )
    )
    repository = RunRepository(tmp_path)
    snapshot = repository.create(
        RunSpec("parent", "test.parent", {"case": "provider-gate"})
    )
    context = RunContext(repository, snapshot, resume_input=None)
    service = LLMTaskService(registry=registry)
    outcome = service.execute(
        context,
        LLMRequest(
            "gated",
            "Return an object.",
            JsonOutput({"type": "object", "required": ["answer"]}),
            ModelSelection("codex"),
        ),
    )
    assert isinstance(outcome, LLMCompleted)
    assert (tmp_path / "operational" / "llm" / "global" / "pool.json").is_file()


def test_gate_state_write_warning_is_bounded_and_does_not_replace_success(
    tmp_path: Path, adapter, registry, monkeypatch
) -> None:
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
    from ac_llm.output import CandidateMaterial

    adapter.steps.append(
        ProviderExecution(
            ProviderTerminalKind.COMPLETED,
            (CandidateMaterial(value={"answer": 1}, terminal=True),),
            NativeResumeHandle("codex", "thread"),
        )
    )
    monkeypatch.setattr(
        ProviderCallGate,
        "_record_success",
        lambda *args: (_ for _ in ()).throw(OSError("state write failed")),
    )
    repository = RunRepository(tmp_path)
    snapshot = repository.create(
        RunSpec("parent", "test.parent", {"case": "gate-warning"})
    )
    context = RunContext(
        repository,
        snapshot,
        resume_input=None,
    )
    service = LLMTaskService(registry=registry)
    outcome = service.execute(
        context,
        LLMRequest(
            "gated",
            "Return an object.",
            JsonOutput({"type": "object", "required": ["answer"]}),
            ModelSelection("codex"),
        ),
    )
    assert isinstance(outcome, LLMCompleted)
    warnings = [
        event
        for event in context.events.tail()
        if event["event"] == "llm_gate_state_warning"
    ]
    assert [event["data"] for event in warnings] == [
        {"code": "provider_gate_state_write_failed"}
    ]


def test_memory_probe_failure_is_a_completed_outcome_warning(
    tmp_path: Path, adapter, registry, monkeypatch
) -> None:
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
    from ac_llm.output import CandidateMaterial

    adapter.steps.append(
        ProviderExecution(
            ProviderTerminalKind.COMPLETED,
            (CandidateMaterial(value={"answer": 1}, terminal=True),),
            NativeResumeHandle("codex", "thread"),
        )
    )
    monkeypatch.setattr(
        "ac_llm.gate.read_memory_availability",
        lambda: (_ for _ in ()).throw(OSError("unsupported")),
    )
    repository = RunRepository(tmp_path)
    snapshot = repository.create(
        RunSpec("parent", "test.parent", {"case": "memory-warning"})
    )
    context = RunContext(repository, snapshot, resume_input=None)

    service = LLMTaskService(registry=registry)
    outcome = service.execute(
        context,
        LLMRequest(
            "gated",
            "Return an object.",
            JsonOutput({"type": "object", "required": ["answer"]}),
            ModelSelection("codex"),
        ),
    )

    assert isinstance(outcome, LLMCompleted)
    assert [warning["code"] for warning in outcome.warnings] == [
        "internet_best_effort",
        "memory_guard_unavailable",
    ]
    assert any(
        event["event"] == "llm_memory_guard_warning"
        for event in context.events.tail()
    )
    service._executor._observe_gate(
        context,
        "llm_memory_guard_warning",
        {"code": "memory_guard_unavailable"},
    )
    assert sum(
        event["event"] == "llm_memory_guard_warning"
        for event in context.events.tail()
    ) == 1


def test_operational_limit_changes_expand_the_repository_pool(
    tmp_path: Path,
) -> None:
    root = tmp_path / "operational" / "llm"
    first = ProviderCallGate(
        root,
        ProviderGateOptions(global_limit=1, provider_limits={"codex": 1}),
    )
    with first.acquire("codex", checkpoint=_checkpoint) as permit:
        permit.record_success()

    second = ProviderCallGate(
        root,
        ProviderGateOptions(global_limit=200, provider_limits={"codex": 150}),
    )
    with second.acquire("codex", checkpoint=_checkpoint) as permit:
        permit.record_success()

    global_contract = json.loads(
        (root / "global" / "pool.json").read_text(encoding="utf-8")
    )
    provider_namespace = hashlib.sha256(b"codex").hexdigest()
    provider_contract = json.loads(
        (root / "providers" / provider_namespace / "pool.json").read_text(
            encoding="utf-8"
        )
    )
    assert global_contract["capacity"] == 200
    assert provider_contract["capacity"] == 150


def test_configured_gate_can_hold_more_than_twenty_four_permits(
    tmp_path: Path,
) -> None:
    gate = ProviderCallGate(
        tmp_path / "operational" / "llm",
        ProviderGateOptions(
            global_limit=200,
            minimum_available_memory_fraction=None,
        ),
    )
    permits = []
    try:
        for _ in range(25):
            permits.append(gate.acquire("codex", checkpoint=_checkpoint))
        assert len(permits) == 25
    finally:
        for permit in permits:
            permit.release()


def test_low_memory_releases_capacity_then_resumes_with_events(
    tmp_path: Path,
) -> None:
    root = tmp_path / "operational" / "llm"
    measurements = iter(
        (
            MemoryAvailability(5, 100, "test"),
            MemoryAvailability(20, 100, "test"),
            MemoryAvailability(20, 100, "test"),
        )
    )
    now = [100.0]
    sleeps = []
    capacity_was_free = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if seconds == 0.02:
            probe = BoundedLeasePool(root / "global", 1).acquire(blocking=False)
            capacity_was_free.append(True)
            probe.release()
        now[0] += seconds

    events = []
    gate = ProviderCallGate(
        root,
        ProviderGateOptions(
            global_limit=1,
            minimum_available_memory_fraction=0.10,
            memory_poll_interval_seconds=0.02,
            memory_launch_interval_seconds=0.01,
        ),
        clock=lambda: now[0],
        memory_probe=lambda: next(measurements),
        sleeper=sleep,
    )

    permit = gate.acquire(
        "codex",
        checkpoint=_checkpoint,
        observe=lambda event, data: events.append((event, data)),
    )
    permit.release()

    assert capacity_was_free == [True]
    assert sleeps == [0.02]
    assert [event for event, _ in events] == [
        "llm_memory_throttled",
        "llm_memory_resumed",
    ]
    assert events[0][1]["available_fraction"] == 0.05
    assert events[1][1]["wait_seconds"] == pytest.approx(0.02)


def test_memory_guard_paces_repository_launches(tmp_path: Path) -> None:
    now = [100.0]
    sleeps = []
    available = MemoryAvailability(50, 100, "test")

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    gate = ProviderCallGate(
        tmp_path / "operational" / "llm",
        ProviderGateOptions(
            global_limit=2,
            memory_launch_interval_seconds=0.25,
        ),
        clock=lambda: now[0],
        memory_probe=lambda: available,
        sleeper=sleep,
    )
    first = gate.acquire("codex", checkpoint=_checkpoint)
    second = gate.acquire("codex", checkpoint=_checkpoint)
    first.release()
    second.release()

    assert sleeps == [0.25]


def test_unavailable_memory_probe_proceeds_without_pacing_and_warns(
    tmp_path: Path,
) -> None:
    def unavailable() -> MemoryAvailability:
        raise OSError("unsupported")

    events = []
    sleeps = []
    gate = ProviderCallGate(
        tmp_path / "operational" / "llm",
        ProviderGateOptions(global_limit=1),
        memory_probe=unavailable,
        sleeper=sleeps.append,
    )
    permit = gate.acquire(
        "codex",
        checkpoint=_checkpoint,
        observe=lambda event, data: events.append((event, data)),
    )
    permit.release()

    assert sleeps == []
    assert [warning["code"] for warning in permit.warnings] == [
        "memory_guard_unavailable"
    ]
    assert [event for event, _ in events] == ["llm_memory_guard_warning"]


def test_invalid_memory_probe_result_uses_fail_open_warning(tmp_path: Path) -> None:
    gate = ProviderCallGate(
        tmp_path / "operational" / "llm",
        ProviderGateOptions(global_limit=1),
        memory_probe=lambda: None,  # type: ignore[arg-type,return-value]
        sleeper=lambda _: None,
    )

    permit = gate.acquire("codex", checkpoint=_checkpoint)
    permit.release()

    assert permit.warnings[0]["code"] == "memory_guard_unavailable"
    assert permit.warnings[0]["error_type"] == "TypeError"


def test_disabled_memory_guard_skips_probe_and_pacing(tmp_path: Path) -> None:
    gate = ProviderCallGate(
        tmp_path / "operational" / "llm",
        ProviderGateOptions(
            global_limit=1,
            minimum_available_memory_fraction=None,
        ),
        memory_probe=lambda: (_ for _ in ()).throw(
            AssertionError("disabled guard must not probe")
        ),
        sleeper=lambda _: (_ for _ in ()).throw(
            AssertionError("disabled guard must not sleep")
        ),
    )

    gate.acquire("codex", checkpoint=_checkpoint).release()


def test_memory_wait_remains_cooperatively_stoppable(tmp_path: Path) -> None:
    calls = 0

    class StopWaiting(Exception):
        pass

    def checkpoint() -> None:
        nonlocal calls
        calls += 1
        if calls >= 4:
            raise StopWaiting

    gate = ProviderCallGate(
        tmp_path / "operational" / "llm",
        ProviderGateOptions(
            global_limit=1,
            memory_poll_interval_seconds=0.01,
            memory_launch_interval_seconds=0.01,
        ),
        memory_probe=lambda: MemoryAvailability(1, 100, "test"),
        sleeper=lambda _: None,
    )

    with pytest.raises(StopWaiting):
        gate.acquire("codex", checkpoint=checkpoint)
