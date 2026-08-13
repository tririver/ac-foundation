from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from arc_jobs import (
    FailureMode,
    GroupResult,
    RunEngine,
    RunRepository,
    RunSpec,
    RunStatus,
    Succeeded,
    WorkUnit,
    run_control_main,
)


class _Tracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.release = threading.Semaphore(0)
        self.active = 0
        self.peak = 0
        self.started = 0

    def worker(self, unit: WorkUnit) -> None:
        del unit
        with self.lock:
            self.active += 1
            self.started += 1
            self.peak = max(self.peak, self.active)
        self.release.acquire()
        with self.lock:
            self.active -= 1

    def values(self) -> tuple[int, int, int]:
        with self.lock:
            return self.active, self.started, self.peak


class _DynamicGroupHandler:
    name = "dynamic-group-test.v1"

    def __init__(self, tracker: _Tracker, *, initial_workers: int, units: int) -> None:
        self.tracker = tracker
        self.initial_workers = initial_workers
        self.units = units

    def execute(self, context):
        result = context.run_group(
            "adjustable",
            tuple(
                WorkUnit(f"unit-{index:03d}", {"index": index})
                for index in range(self.units)
            ),
            self.tracker.worker,
            max_workers=self.initial_workers,
            failure_mode=FailureMode.COLLECT,
        )
        assert isinstance(result, GroupResult)
        return Succeeded()


def _wait_for(predicate, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def _start(repository: RunRepository, handler: _DynamicGroupHandler, run_id: str):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        RunEngine(repository).execute,
        RunSpec(run_id, handler.name, {"units": handler.units}),
        handler,
    )
    return executor, future


def test_running_group_scales_up_immediately_and_persists_target(tmp_path):
    repository = RunRepository(tmp_path)
    tracker = _Tracker()
    executor, future = _start(
        repository,
        _DynamicGroupHandler(tracker, initial_workers=2, units=6),
        "scale-up",
    )
    try:
        _wait_for(lambda: tracker.values()[0] == 2)
        initial = repository.group_workers("scale-up", "adjustable")
        assert initial.target_workers == 2
        assert initial.capacity >= 5

        updated = repository.set_group_workers("scale-up", "adjustable", 5)
        assert updated.target_workers == 5
        _wait_for(lambda: tracker.values()[0] == 5)

        tracker.release.release(6)
        assert future.result(timeout=5).status is RunStatus.SUCCEEDED
        assert tracker.values()[2] == 5
        assert repository.group_workers("scale-up", "adjustable") == updated

        events = [
            json.loads(line)
            for line in (tmp_path / "runs" / "scale-up" / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        changes = [event for event in events if event["event"] == "group_workers_updated"]
        assert changes[-1]["data"]["previous_target_workers"] == 2
        assert changes[-1]["data"]["target_workers"] == 5
    finally:
        tracker.release.release(20)
        executor.shutdown(wait=True)


def test_running_group_scales_down_without_cancelling_in_flight_units(tmp_path):
    repository = RunRepository(tmp_path)
    tracker = _Tracker()
    executor, future = _start(
        repository,
        _DynamicGroupHandler(tracker, initial_workers=5, units=8),
        "scale-down",
    )
    try:
        _wait_for(lambda: tracker.values()[0] == 5)
        repository.set_group_workers("scale-down", "adjustable", 2)

        # All five already-running units finish normally.  Only two replacements
        # may start after the lower target is observed.
        tracker.release.release(5)
        _wait_for(lambda: tracker.values()[1] >= 7)
        assert tracker.values()[0] == 2
        time.sleep(0.2)
        assert tracker.values()[0] == 2

        tracker.release.release(3)
        assert future.result(timeout=5).status is RunStatus.SUCCEEDED
        assert tracker.values()[1] == 8
        assert repository.group_workers("scale-down", "adjustable").target_workers == 2
    finally:
        tracker.release.release(20)
        executor.shutdown(wait=True)


def test_workers_cli_gets_and_sets_group_target(tmp_path, capsys):
    repository = RunRepository(tmp_path)
    tracker = _Tracker()
    executor, future = _start(
        repository,
        _DynamicGroupHandler(tracker, initial_workers=1, units=2),
        "cli-workers",
    )
    try:
        _wait_for(lambda: tracker.values()[0] == 1)
        assert run_control_main(
            [
                "workers", "set", "--run-root", str(tmp_path),
                "--run-id", "cli-workers", "--group-id", "adjustable",
                "--workers", "2",
            ]
        ) == 0
        set_result = json.loads(capsys.readouterr().out)
        assert set_result["data"]["group_workers"]["target_workers"] == 2
        _wait_for(lambda: tracker.values()[0] == 2)

        assert run_control_main(
            [
                "workers", "get", "--run-root", str(tmp_path),
                "--run-id", "cli-workers", "--group-id", "adjustable",
            ]
        ) == 0
        get_result = json.loads(capsys.readouterr().out)
        assert get_result["data"]["group_workers"]["target_workers"] == 2

        tracker.release.release(2)
        assert future.result(timeout=5).status is RunStatus.SUCCEEDED
    finally:
        tracker.release.release(20)
        executor.shutdown(wait=True)


def test_workers_cli_preserves_missing_run_error_contract(tmp_path, capsys):
    assert run_control_main(
        [
            "workers", "get", "--run-root", str(tmp_path),
            "--run-id", "missing", "--group-id", "adjustable",
        ]
    ) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["error"]["code"] == "run_not_found"
