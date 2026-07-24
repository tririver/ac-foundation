from __future__ import annotations

import pytest

import arc_jobs.events as event_storage
import arc_jobs.storage as durable_storage
from arc_jobs import (
    FailureMode,
    GroupResult,
    RunEngine,
    RunRepository,
    RunSpec,
    RunStatus,
    Succeeded,
    WorkUnit,
)


class NoOpGroupHandler:
    name = "noop-group.v1"

    def __init__(self, size: int):
        self.size = size

    def execute(self, context):
        result = context.run_group(
            "scale",
            tuple(
                WorkUnit(f"unit-{index:05d}", {"index": index})
                for index in range(self.size)
            ),
            lambda unit: None,
            max_workers=32,
            failure_mode=FailureMode.COLLECT,
        )
        assert isinstance(result, GroupResult)
        assert len(result.units) == self.size
        return Succeeded()


@pytest.mark.parametrize("size", (5_000, 10_000))
def test_noop_group_scales_without_quadratic_queue_or_event_scans(
    tmp_path, monkeypatch, size
):
    # Preserve all file and event behavior while avoiding hardware-dependent fsync
    # latency in this algorithmic scale regression.
    monkeypatch.setattr(durable_storage, "_fsync_directory", lambda path: None)
    monkeypatch.setattr(event_storage, "_fsync_directory", lambda path: None)
    monkeypatch.setattr(durable_storage.os, "fsync", lambda fd: None)

    def fail_if_scanned(self):
        raise AssertionError("group execution performed a full event-log scan")

    monkeypatch.setattr(event_storage.EventWriter, "_complete_documents", fail_if_scanned)
    repository = RunRepository(tmp_path)
    snapshot = RunEngine(repository).execute(
        RunSpec(f"scale-{size}", NoOpGroupHandler.name, {"size": size}),
        NoOpGroupHandler(size),
    )

    assert snapshot.status is RunStatus.SUCCEEDED
    view = repository.inspect_group(f"scale-{size}", "scale")
    assert len(view.units) == size
    assert all(unit.status == "succeeded" for unit in view.units)
