"""Portable best-effort memory availability for provider admission."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryAvailability:
    available_bytes: int
    total_bytes: int
    source: str

    def __post_init__(self) -> None:
        if (
            self.total_bytes <= 0
            or not 0 <= self.available_bytes <= self.total_bytes
        ):
            raise ValueError("invalid memory availability")

    @property
    def fraction(self) -> float:
        return self.available_bytes / self.total_bytes


def read_memory_availability(
    *,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> MemoryAvailability:
    """Return the most constrained available-memory fraction visible locally."""

    measurements: list[MemoryAvailability] = []
    host = _linux_host_memory(proc_root / "meminfo")
    if host is not None:
        measurements.append(host)
    cgroup = _linux_cgroup_memory(proc_root / "self" / "cgroup", cgroup_root)
    if cgroup is not None:
        measurements.append(cgroup)
    if not measurements and os.name == "nt":  # pragma: no cover - Windows
        windows = _windows_memory()
        if windows is not None:
            measurements.append(windows)
    if not measurements:
        raise OSError("no supported available-memory measurement")
    return min(measurements, key=lambda item: item.fraction)


def _linux_host_memory(path: Path) -> MemoryAvailability | None:
    try:
        values = _read_meminfo(path)
        total = values["MemTotal"]
        available = values["MemAvailable"]
    except (OSError, KeyError, ValueError):
        return None
    return _measurement(available, total, "proc_meminfo")


def _read_meminfo(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        name, separator, raw = line.partition(":")
        if not separator:
            continue
        fields = raw.split()
        if not fields:
            continue
        value = int(fields[0])
        if len(fields) > 1 and fields[1] == "kB":
            value *= 1024
        values[name] = value
    return values


def _linux_cgroup_memory(
    membership_path: Path,
    cgroup_root: Path,
) -> MemoryAvailability | None:
    try:
        memberships = membership_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    unified: str | None = None
    legacy: str | None = None
    for line in memberships:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, path = fields
        if not controllers:
            unified = path
        elif "memory" in controllers.split(","):
            legacy = path
    if unified is not None:
        root = cgroup_root / unified.lstrip("/")
        measurement = _cgroup_measurement(
            root / "memory.current",
            root / "memory.max",
            "cgroup_v2",
        )
        if measurement is not None:
            return measurement
    if legacy is not None:
        root = cgroup_root / "memory" / legacy.lstrip("/")
        return _cgroup_measurement(
            root / "memory.usage_in_bytes",
            root / "memory.limit_in_bytes",
            "cgroup_v1",
        )
    return None


def _cgroup_measurement(
    current_path: Path,
    limit_path: Path,
    source: str,
) -> MemoryAvailability | None:
    try:
        raw_limit = limit_path.read_text(encoding="ascii").strip()
        if raw_limit == "max":
            return None
        limit = int(raw_limit)
        current = int(current_path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None
    # Linux v1 represents an unlimited cgroup with a near-int64 sentinel.
    if limit >= 1 << 60:
        return None
    return _measurement(max(0, limit - current), limit, source)


def _measurement(
    available: int,
    total: int,
    source: str,
) -> MemoryAvailability | None:
    if total <= 0 or available < 0:
        return None
    return MemoryAvailability(min(available, total), total, source)


def _windows_memory() -> MemoryAvailability | None:  # pragma: no cover - Windows
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return _measurement(
            int(status.available_physical),
            int(status.total_physical),
            "windows_global_memory_status",
        )
    except (AttributeError, OSError, ValueError):
        return None
