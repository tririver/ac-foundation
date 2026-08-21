from __future__ import annotations

from pathlib import Path

import pytest

from ac_llm.memory import read_memory_availability


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _host_memory(proc: Path, *, total_kib: int, available_kib: int) -> None:
    _write(
        proc / "meminfo",
        f"MemTotal: {total_kib} kB\nMemAvailable: {available_kib} kB\n",
    )


def test_cgroup_v2_headroom_wins_when_more_constrained(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    _host_memory(proc, total_kib=1000, available_kib=600)
    _write(proc / "self" / "cgroup", "0::/ac.slice\n")
    _write(cgroup / "ac.slice" / "memory.max", "1000\n")
    _write(cgroup / "ac.slice" / "memory.current", "950\n")

    measured = read_memory_availability(proc_root=proc, cgroup_root=cgroup)

    assert measured.source == "cgroup_v2"
    assert measured.total_bytes == 1000
    assert measured.available_bytes == 50
    assert measured.fraction == 0.05


def test_cgroup_v2_counts_only_inactive_file_cache_as_available(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    _host_memory(proc, total_kib=1000, available_kib=600)
    _write(proc / "self" / "cgroup", "0::/ac.slice\n")
    _write(cgroup / "ac.slice" / "memory.max", "1000\n")
    _write(cgroup / "ac.slice" / "memory.current", "950\n")
    _write(
        cgroup / "ac.slice" / "memory.stat",
        "inactive_file 400\nactive_file 300\nslab_reclaimable 100\n",
    )

    measured = read_memory_availability(proc_root=proc, cgroup_root=cgroup)

    assert measured.source == "cgroup_v2"
    assert measured.available_bytes == 450
    assert measured.fraction == 0.45


def test_cgroup_v1_headroom_is_supported(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    _host_memory(proc, total_kib=1000, available_kib=900)
    _write(proc / "self" / "cgroup", "7:cpu:/work\n6:memory:/work\n")
    _write(cgroup / "memory" / "work" / "memory.limit_in_bytes", "1000\n")
    _write(cgroup / "memory" / "work" / "memory.usage_in_bytes", "700\n")

    measured = read_memory_availability(proc_root=proc, cgroup_root=cgroup)

    assert measured.source == "cgroup_v1"
    assert measured.available_bytes == 300


def test_cgroup_v1_prefers_hierarchical_inactive_file_cache(
    tmp_path: Path,
) -> None:
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    _host_memory(proc, total_kib=1000, available_kib=900)
    _write(proc / "self" / "cgroup", "6:memory:/work\n")
    root = cgroup / "memory" / "work"
    _write(root / "memory.limit_in_bytes", "1000\n")
    _write(root / "memory.usage_in_bytes", "900\n")
    _write(root / "memory.stat", "inactive_file 50\ntotal_inactive_file 300\n")

    measured = read_memory_availability(proc_root=proc, cgroup_root=cgroup)

    assert measured.source == "cgroup_v1"
    assert measured.available_bytes == 400


def test_unlimited_cgroup_falls_back_to_host_measurement(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    cgroup = tmp_path / "cgroup"
    _host_memory(proc, total_kib=1000, available_kib=400)
    _write(proc / "self" / "cgroup", "0::/\n")
    _write(cgroup / "memory.max", "max\n")
    _write(cgroup / "memory.current", "700\n")

    measured = read_memory_availability(proc_root=proc, cgroup_root=cgroup)

    assert measured.source == "proc_meminfo"
    assert measured.fraction == 0.4


def test_missing_supported_measurement_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="no supported"):
        read_memory_availability(
            proc_root=tmp_path / "missing-proc",
            cgroup_root=tmp_path / "missing-cgroup",
        )
