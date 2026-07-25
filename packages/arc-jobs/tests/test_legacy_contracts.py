from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
from dataclasses import dataclass, replace

import pytest

from arc_jobs import (
    AtomicStateStore,
    EventWriter,
    Failed,
    FailureMode,
    IdempotencyConflictError,
    ImmutableArtifactStore,
    RunEngine,
    RunError,
    RunRepository,
    RunSpec,
    RunStatus,
    Succeeded,
    WorkUnit,
)
from arc_jobs.cli import main
from arc_jobs import storage as jobs_storage


class _Success:
    name = "legacy-success.v1"

    def execute(self, context):
        result = context.artifacts.publish_json("result", {"ok": True})
        return Succeeded(result)


class _Failure:
    name = "legacy-failure.v1"

    def execute(self, context):
        return Failed(RunError("expected_failure", "expected"))


@dataclass(frozen=True)
class _AtomicValue:
    revision: int
    payload: str


class _AtomicContract:
    schema_version = "legacy.atomic.v1"

    def encode(self, value):
        return {"revision": value.revision, "payload": value.payload}

    def decode(self, value):
        return _AtomicValue(value["revision"], value["payload"])

    def validate_transition(self, previous, next_value):
        return None


def test_state_and_artifact_publication_are_atomic(tmp_path, monkeypatch):
    state = AtomicStateStore(tmp_path / "state.json", _AtomicContract())
    original = state.create(_AtomicValue(0, "old"))
    original_replace = jobs_storage.os.replace
    failed = False

    def fail_state_replace(source, target):
        nonlocal failed
        if not failed and target == state.path:
            failed = True
            raise OSError("simulated replace failure")
        return original_replace(source, target)

    monkeypatch.setattr(jobs_storage.os, "replace", fail_state_replace)
    with pytest.raises(OSError):
        state.compare_and_swap(0, _AtomicValue(1, "new"))
    assert state.read() == original
    monkeypatch.setattr(jobs_storage.os, "replace", original_replace)

    artifacts = ImmutableArtifactStore(tmp_path / "run")
    barrier = threading.Barrier(2)
    refs = []

    def publish() -> None:
        barrier.wait()
        refs.append(
            artifacts.publish_bytes(
                "shared/result",
                b"x" * 200_000,
                media_type="application/octet-stream",
            )
        )

    threads = [threading.Thread(target=publish) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(refs) == 2
    assert refs[0] == refs[1]
    assert artifacts.read_bytes(refs[0]) == b"x" * 200_000


def test_concurrent_create_is_idempotent_and_changed_semantics_conflict(tmp_path):
    repository = RunRepository(tmp_path)
    barrier = threading.Barrier(2)
    results = []

    def create() -> None:
        barrier.wait()
        results.append(
            repository.create(RunSpec("run-1", "legacy-success.v1", {"x": 1}))
        )

    threads = [threading.Thread(target=create) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 2
    assert results[0] == results[1]
    with pytest.raises(IdempotencyConflictError):
        repository.create(RunSpec("run-1", "legacy-success.v1", {"x": 2}))


@pytest.mark.parametrize(
    ("handler", "expected_status"),
    [
        (_Success(), RunStatus.SUCCEEDED),
        (_Failure(), RunStatus.FAILED),
    ],
)
def test_terminal_event_precedes_terminal_snapshot_commit(
    tmp_path, monkeypatch, handler, expected_status
):
    repository = RunRepository(tmp_path)
    store = repository._snapshot_store("run-1")
    original = store.compare_and_swap
    observed = []

    def compare_and_swap(expected_revision, value):
        if value.status is expected_status:
            events = EventWriter(
                repository.run_directory("run-1") / "events.jsonl",
                run_id="run-1",
            ).tail()
            observed.append(events[-1]["data"]["status"])
            if value.result_ref is not None:
                repository.validate("run-1")
        return original(expected_revision, value)

    monkeypatch.setattr(store, "compare_and_swap", compare_and_swap)
    monkeypatch.setattr(repository, "_snapshot_store", lambda _run_id: store)

    snapshot = RunEngine(repository).execute(
        RunSpec("run-1", handler.name, {}), handler
    )

    assert snapshot.status is expected_status
    assert observed == [expected_status.value]


def test_stop_terminal_run_is_idempotent_and_does_not_replace_success(tmp_path):
    repository = RunRepository(tmp_path)
    succeeded = RunEngine(repository).execute(
        RunSpec("run-1", _Success.name, {}), _Success()
    )

    first = repository.request_stop("run-1", reason="too late")
    second = repository.request_stop("run-1", reason="different")

    assert first == second
    assert first.snapshot == succeeded
    assert first.snapshot.status is RunStatus.SUCCEEDED
    assert first.stop_request is None


def test_os_releases_execution_lease_when_owner_exits(tmp_path):
    lease_path = tmp_path / "crashed.lock"
    script = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from arc_jobs.lease import FileLease\n"
        "FileLease(Path(sys.argv[1])).acquire()\n"
        "os._exit(0)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(lease_path)],
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")

    from arc_jobs.lease import FileLease

    FileLease(lease_path).acquire().release()


def test_resume_recovers_interrupted_running_attempt_before_restarting(tmp_path):
    repository = RunRepository(tmp_path)
    pending = repository.create(RunSpec("run-1", _Success.name, {}))
    running = replace(
        pending,
        revision=pending.revision + 1,
        status=RunStatus.RUNNING,
        attempt=1,
    )
    repository._snapshot_store("run-1").compare_and_swap(
        pending.revision, running
    )

    resumed = RunEngine(repository).resume("run-1", _Success())

    assert resumed.status is RunStatus.SUCCEEDED
    assert resumed.attempt == 2
    assert resumed.interrupted
    events = EventWriter(
        repository.run_directory("run-1") / "events.jsonl", run_id="run-1"
    ).tail()
    assert [event["event"] for event in events][:1] == ["run_interrupted"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_run_storage_uses_private_directory_and_file_modes(tmp_path):
    repository = RunRepository(tmp_path)
    RunEngine(repository).execute(
        RunSpec("run-1", _Success.name, {}), _Success()
    )

    assert stat.S_IMODE((tmp_path / "runs").stat().st_mode) == 0o700
    assert stat.S_IMODE(repository.run_directory("run-1").stat().st_mode) == 0o700
    for path in repository.run_directory("run-1").glob("**/*"):
        if path.is_file():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_event_tail_reads_only_the_bounded_suffix(tmp_path):
    writer = EventWriter(tmp_path / "events.jsonl", run_id="run-1")
    for index in range(6):
        writer.emit("progress", {"index": index, "summary": "x" * 220_000})

    events = writer.tail()

    assert 0 < len(events) < 6
    assert events[-1]["sequence"] == 6
    assert events[0]["sequence"] > 1


def test_concurrent_event_appends_receive_monotonic_sequences(tmp_path):
    writer = EventWriter(tmp_path / "events.jsonl", run_id="run-1")
    barrier = threading.Barrier(4)

    def emit(index):
        barrier.wait()
        writer.emit("progress", {"index": index})

    threads = [threading.Thread(target=emit, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    events = writer.tail()
    assert [event["sequence"] for event in events] == [1, 2, 3, 4]
    assert {event["data"]["index"] for event in events} == {0, 1, 2, 3}


def test_event_appends_only_full_scan_during_explicit_validation(
    tmp_path, monkeypatch
):
    writer = EventWriter(tmp_path / "events.jsonl", run_id="run-1")
    writer.emit("progress", {"index": 0})
    full_scan = writer._complete_documents

    def fail_if_scanned():
        raise AssertionError("normal event append performed a full log scan")

    monkeypatch.setattr(writer, "_complete_documents", fail_if_scanned)
    for index in range(1, 100):
        writer.emit("progress", {"index": index})

    # A distinct writer observes the shared append-only log from its last record.
    other = EventWriter(writer.path, run_id="run-1")
    monkeypatch.setattr(other, "_complete_documents", fail_if_scanned)
    other.emit("progress", {"index": 100})
    writer.emit("progress", {"index": 101})

    EventWriter(writer.path, run_id="run-1").validate()
    documents = full_scan()
    assert [document["sequence"] for document in documents] == list(range(1, 103))


def test_cli_commands_queries_and_unexpected_errors_use_one_envelope(
    tmp_path, capsys, monkeypatch
):
    repository = RunRepository(tmp_path)
    RunEngine(repository).execute(
        RunSpec("failed", _Failure.name, {}), _Failure()
    )
    repository.create(RunSpec("pending", _Success.name, {}))

    commands = (
        ["status", "--run-root", str(tmp_path), "--run-id", "failed"],
        ["validate", "--run-root", str(tmp_path), "--run-id", "failed"],
        [
            "stop",
            "--run-root",
            str(tmp_path),
            "--run-id",
            "pending",
            "--reason",
            "stop",
        ],
    )
    for command in commands:
        assert main(command) == 0
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["status"] == "completed"

    def explode(_self, _run_id):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(RunRepository, "inspect", explode)
    assert (
        main(["status", "--run-root", str(tmp_path), "--run-id", "failed"]) == 1
    )
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result["status"] == "failed"
    assert result["error"]["code"] == "internal_error"


def test_group_stop_stops_submissions_and_joins_started_workers(tmp_path):
    repository = RunRepository(tmp_path)
    engine = RunEngine(repository)
    started = []
    workers_started = threading.Event()
    release = threading.Event()

    class GroupHandler:
        name = "legacy-stop-group.v1"

        def execute(self, context):
            def worker(unit):
                started.append(unit.unit_id)
                if len(started) == 2:
                    workers_started.set()
                release.wait(timeout=5)
                return {"unit": unit.unit_id}

            context.run_group(
                "group",
                tuple(WorkUnit(str(index), {"index": index}) for index in range(5)),
                worker,
                max_workers=2,
                failure_mode=FailureMode.COLLECT,
            )
            return Succeeded()

    snapshots = []
    thread = threading.Thread(
        target=lambda: snapshots.append(
            engine.execute(
                RunSpec("run-1", GroupHandler.name, {}),
                GroupHandler(),
            )
        )
    )
    thread.start()
    assert workers_started.wait(timeout=5)

    repository.request_stop("run-1", reason="stop")
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert snapshots[0].status is RunStatus.PAUSED
    assert set(started) == {"0", "1"}
    view = repository.inspect_group("run-1", "group")
    assert [unit.status for unit in view.units] == [
        "succeeded",
        "succeeded",
        "pending",
        "pending",
        "pending",
    ]


def test_context_events_persist_safe_protocol_neutral_progress(tmp_path):
    class EventHandler:
        name = "legacy-event.v1"

        def execute(self, context):
            context.events.emit(
                "phase_changed",
                {"phase": "prepare", "summary": "starting"},
            )
            return Succeeded()

    repository = RunRepository(tmp_path)
    RunEngine(repository).execute(
        RunSpec("run-1", EventHandler.name, {}), EventHandler()
    )

    events = EventWriter(
        repository.run_directory("run-1") / "events.jsonl", run_id="run-1"
    ).tail()
    assert events[0]["event"] == "phase_changed"
    assert events[0]["data"] == {"phase": "prepare", "summary": "starting"}
