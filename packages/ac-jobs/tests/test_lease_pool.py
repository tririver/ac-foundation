from __future__ import annotations

import subprocess
import sys

import pytest

from ac_jobs import BoundedLeasePool, RunBusyError


def test_bounded_lease_pool_enforces_same_process_capacity_and_release(tmp_path):
    pool = BoundedLeasePool(tmp_path / "pool", 3)
    first = pool.acquire(limit=1)
    with pytest.raises(RunBusyError):
        pool.acquire(limit=1, blocking=False)
    second = pool.acquire(limit=2, blocking=False)
    third = pool.acquire(blocking=False)
    assert {first.slot, second.slot, third.slot} == {0, 1, 2}

    released_slot = first.slot
    first.release()
    replacement = pool.acquire(limit=3, blocking=False)
    assert replacement.slot == released_slot
    replacement.release()
    second.release()
    third.release()


def test_bounded_lease_pool_lower_limit_counts_high_slot_holder_across_hole(
    tmp_path,
):
    pool = BoundedLeasePool(tmp_path / "pool", 2)
    low = pool.acquire(limit=1)
    high = pool.acquire(limit=2)
    assert (low.slot, high.slot) == (0, 1)

    low.release()
    with pytest.raises(RunBusyError):
        pool.acquire(limit=1, blocking=False)

    high.release()
    replacement = pool.acquire(limit=1, blocking=False)
    assert replacement.slot == 0
    replacement.release()


def test_bounded_lease_pool_enforces_capacity_across_processes(tmp_path):
    root = tmp_path / "pool"
    script = (
        "import sys\n"
        "from ac_jobs import BoundedLeasePool\n"
        "lease = BoundedLeasePool(sys.argv[1], 2).acquire(limit=1)\n"
        "print('ready', flush=True)\n"
        "sys.stdin.readline()\n"
        "lease.release()\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        pool = BoundedLeasePool(root, 2)
        with pytest.raises(RunBusyError):
            pool.acquire(limit=1, blocking=False)
        expanded = pool.acquire(limit=2, blocking=False)
        assert expanded.slot == 1
        expanded.release()
        assert child.stdin is not None
        child.stdin.write("\n")
        child.stdin.flush()
        assert child.wait(timeout=10) == 0
        narrowed = pool.acquire(limit=1, blocking=False)
        assert narrowed.slot == 0
        narrowed.release()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_bounded_lease_pool_lower_limit_counts_cross_process_high_slot_holder(
    tmp_path,
):
    root = tmp_path / "pool"
    pool = BoundedLeasePool(root, 2)
    low = pool.acquire(limit=1)
    script = (
        "import sys\n"
        "from ac_jobs import BoundedLeasePool\n"
        "lease = BoundedLeasePool(sys.argv[1], 2).acquire(limit=2)\n"
        "print(lease.slot, flush=True)\n"
        "sys.stdin.readline()\n"
        "lease.release()\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "1"
        low.release()

        with pytest.raises(RunBusyError):
            pool.acquire(limit=1, blocking=False)

        assert child.stdin is not None
        child.stdin.write("\n")
        child.stdin.flush()
        assert child.wait(timeout=10) == 0
        replacement = pool.acquire(limit=1, blocking=False)
        assert replacement.slot == 0
        replacement.release()
    finally:
        low.release()
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_bounded_lease_pool_expands_capacity_and_never_shrinks(tmp_path):
    root = tmp_path / "pool"
    initial = BoundedLeasePool(root, 2)
    expanded = BoundedLeasePool(root, 200)
    reopened = BoundedLeasePool(root, 3)

    assert initial.capacity == 2
    assert expanded.capacity == 200
    assert reopened.capacity == 200


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_bounded_lease_pool_rejects_invalid_capacity(tmp_path, capacity):
    with pytest.raises(ValueError, match="capacity must be positive"):
        BoundedLeasePool(tmp_path / "pool", capacity)


@pytest.mark.parametrize("limit", [0, 4, True, 1.5])
def test_bounded_lease_pool_rejects_invalid_acquisition_limit(tmp_path, limit):
    pool = BoundedLeasePool(tmp_path / "pool", 3)
    with pytest.raises(ValueError, match="limit must be between"):
        pool.acquire(limit=limit)


def test_bounded_lease_pool_blocking_wait_calls_checkpoint_and_leaks_no_slot(
    tmp_path,
):
    pool = BoundedLeasePool(tmp_path / "pool", 1)
    held = pool.acquire()
    calls = 0

    class StopWaiting(Exception):
        pass

    def checkpoint() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise StopWaiting

    with pytest.raises(StopWaiting):
        pool.acquire(checkpoint=checkpoint, poll_interval_seconds=0.001)
    assert calls == 2
    with pytest.raises(RunBusyError):
        pool.acquire(blocking=False)

    held.release()
    pool.acquire(blocking=False).release()
