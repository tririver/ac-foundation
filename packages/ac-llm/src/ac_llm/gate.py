"""Repository-scoped provider concurrency gate and small durable circuit."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ac_jobs import (
    AtomicStateStore,
    BoundedLease,
    BoundedLeasePool,
    CorruptStateError,
    FileLease,
    RevisionConflictError,
    RunBusyError,
    StateConflictError,
    atomic_write_json,
)

from .errors import FailureCategory, ProviderFailure
from .memory import MemoryAvailability, read_memory_availability
from .request import ProviderGateOptions

_CIRCUIT_SCHEMA_VERSION = "ac.llm.provider_circuit.v1"
_CIRCUIT_FAILURES = {
    FailureCategory.AUTHENTICATION,
    FailureCategory.QUOTA,
    FailureCategory.RATE_LIMIT,
    FailureCategory.TRANSPORT,
    FailureCategory.UNAVAILABLE,
}
_IMMEDIATE_CIRCUIT_FAILURES = {
    FailureCategory.AUTHENTICATION,
    FailureCategory.QUOTA,
    FailureCategory.RATE_LIMIT,
}


@dataclass(frozen=True)
class _CircuitState:
    revision: int
    consecutive_failures: int
    epoch: int
    opened_until: float | None
    category: str | None


@dataclass(frozen=True)
class _CircuitToken:
    epoch: int
    half_open_probe: bool


class _CircuitContract:
    schema_version = _CIRCUIT_SCHEMA_VERSION

    def encode(self, value: _CircuitState) -> Mapping[str, Any]:
        return {
            "revision": value.revision,
            "consecutive_failures": value.consecutive_failures,
            "epoch": value.epoch,
            "opened_until": value.opened_until,
            "category": value.category,
        }

    def decode(self, document: Mapping[str, Any]) -> _CircuitState:
        expected = {
            "revision",
            "consecutive_failures",
            "epoch",
            "opened_until",
            "category",
        }
        if set(document) != expected:
            raise CorruptStateError("invalid provider circuit fields")
        revision = _nonnegative_int(document["revision"], "revision")
        failures = _nonnegative_int(
            document["consecutive_failures"], "consecutive failures"
        )
        epoch = _nonnegative_int(document["epoch"], "epoch")
        opened_until = document["opened_until"]
        if not (
            opened_until is None
            or (
                isinstance(opened_until, (int, float))
                and not isinstance(opened_until, bool)
            )
        ):
            raise CorruptStateError("invalid provider circuit deadline")
        category = document["category"]
        if category is not None and not isinstance(category, str):
            raise CorruptStateError("invalid provider circuit category")
        return _CircuitState(
            revision,
            failures,
            epoch,
            None if opened_until is None else float(opened_until),
            category,
        )

    def validate_transition(
        self, previous: _CircuitState | None, next: _CircuitState
    ) -> None:
        if previous is None:
            if next.revision != 0:
                raise ValueError("Initial circuit revision must be zero.")
            return
        if next.revision != previous.revision + 1:
            raise ValueError("Circuit revision must increase by one.")
        if next.epoch < previous.epoch:
            raise ValueError("Circuit epoch cannot decrease.")


class ProviderCallPermit:
    def __init__(
        self,
        gate: "ProviderCallGate",
        provider: str,
        token: _CircuitToken,
        leases: tuple[BoundedLease, ...],
        warnings: tuple[Mapping[str, Any], ...] = (),
    ) -> None:
        self._gate = gate
        self._provider = provider
        self._token = token
        self._leases = leases
        self.warnings = warnings
        self._recorded = False
        self.record_error: Exception | None = None

    def record_success(self) -> None:
        try:
            self._gate._record_success(self._provider, self._token)
        except Exception as exc:
            self.record_error = exc
        finally:
            self._recorded = True

    def record_failure(self, failure: ProviderFailure) -> None:
        try:
            self._gate._record_failure(self._provider, self._token, failure)
        except Exception as exc:
            self.record_error = exc
        finally:
            self._recorded = True

    def release(self) -> None:
        for lease in reversed(self._leases):
            lease.release()
        self._leases = ()

    def __enter__(self) -> "ProviderCallPermit":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class ProviderCallGate:
    """Bound provider calls across all runs in one explicit repository."""

    def __init__(
        self,
        root: Path,
        options: ProviderGateOptions,
        *,
        clock: Any = time.time,
        memory_probe: Callable[[], MemoryAvailability] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.root = root.resolve()
        self.options = options
        self._clock = clock
        self._memory_probe = memory_probe or read_memory_availability
        self._sleep = sleeper

    def acquire(
        self,
        provider: str,
        *,
        checkpoint: Any,
        observe: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> ProviderCallPermit:
        if not self.options.enabled:
            return ProviderCallPermit(
                self,
                provider,
                _CircuitToken(0, False),
                (),
            )
        throttled_at: float | None = None
        while True:
            leases: list[BoundedLease] = []
            try:
                provider_limit = self.options.provider_limits.get(provider)
                if provider_limit is not None:
                    leases.append(
                        BoundedLeasePool(
                            self.root / "providers" / _provider_namespace(provider),
                            provider_limit,
                        ).acquire(limit=provider_limit, checkpoint=checkpoint)
                    )
                leases.append(
                    BoundedLeasePool(
                        self.root / "global",
                        self.options.global_limit,
                    ).acquire(
                        limit=self.options.global_limit,
                        checkpoint=checkpoint,
                    )
                )
                admitted, measurement, warnings = self._memory_admission(
                    checkpoint=checkpoint
                )
                if not admitted:
                    if throttled_at is None:
                        throttled_at = float(self._clock())
                        _observe_memory(
                            observe,
                            "llm_memory_throttled",
                            measurement,
                            self.options.minimum_available_memory_fraction,
                        )
                    for lease in reversed(leases):
                        lease.release()
                    leases = []
                    checkpoint()
                    self._sleep(self.options.memory_poll_interval_seconds)
                    continue
                if throttled_at is not None:
                    details = _memory_details(
                        measurement,
                        self.options.minimum_available_memory_fraction,
                    )
                    details["wait_seconds"] = max(
                        0.0, float(self._clock()) - throttled_at
                    )
                    if observe is not None:
                        observe("llm_memory_resumed", details)
                for warning in warnings:
                    if observe is not None:
                        observe("llm_memory_guard_warning", warning)
                token, probe_lease = self._before_call(provider)
                if probe_lease is not None:
                    leases.append(probe_lease)
                return ProviderCallPermit(
                    self,
                    provider,
                    token,
                    tuple(leases),
                    warnings,
                )
            except BaseException:
                for lease in reversed(leases):
                    lease.release()
                raise

    def _memory_admission(
        self,
        *,
        checkpoint: Callable[[], None],
    ) -> tuple[bool, MemoryAvailability | None, tuple[Mapping[str, Any], ...]]:
        threshold = self.options.minimum_available_memory_fraction
        if threshold is None:
            return True, None, ()
        lease = self._acquire_memory_launch_lease(checkpoint)
        try:
            measurement, warnings = self._probe_available_memory()
            if warnings:
                return True, None, warnings
            assert measurement is not None
            if measurement.fraction < threshold:
                return False, measurement, ()
            self._pace_memory_launch(checkpoint)
            measurement, warnings = self._probe_available_memory()
            if warnings:
                return True, None, warnings
            assert measurement is not None
            if measurement.fraction < threshold:
                return False, measurement, ()
            atomic_write_json(
                self.root / "memory-admission" / "last-launch.json",
                {"launched_at": float(self._clock())},
            )
            return True, measurement, ()
        finally:
            lease.release()

    def _probe_available_memory(
        self,
    ) -> tuple[
        MemoryAvailability | None,
        tuple[Mapping[str, Any], ...],
    ]:
        try:
            measurement = self._memory_probe()
            if not isinstance(measurement, MemoryAvailability):
                raise TypeError("memory probe returned an invalid measurement")
        except Exception as exc:
            return (
                None,
                (
                    {
                        "code": "memory_guard_unavailable",
                        "message": (
                            "Available memory could not be measured; provider "
                            "admission proceeded without memory protection."
                        ),
                        "error_type": type(exc).__name__,
                    },
                ),
            )
        return measurement, ()

    def _acquire_memory_launch_lease(
        self,
        checkpoint: Callable[[], None],
    ) -> FileLease:
        path = self.root / "memory-admission" / "launch.lock"
        while True:
            checkpoint()
            try:
                return FileLease(path).acquire()
            except RunBusyError:
                self._sleep(min(0.05, self.options.memory_poll_interval_seconds))

    def _pace_memory_launch(self, checkpoint: Callable[[], None]) -> None:
        path = self.root / "memory-admission" / "last-launch.json"
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            launched_at = float(document["launched_at"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        elapsed = max(0.0, float(self._clock()) - launched_at)
        delay = self.options.memory_launch_interval_seconds - elapsed
        if delay > 0:
            checkpoint()
            self._sleep(delay)
            checkpoint()

    def _before_call(
        self, provider: str
    ) -> tuple[_CircuitToken, BoundedLease | None]:
        store = self._store(provider)
        state = store.read()
        if state is None or state.opened_until is None:
            return _CircuitToken(0 if state is None else state.epoch, False), None
        now = float(self._clock())
        if state.opened_until > now:
            raise _open_failure(state, now)
        try:
            probe = BoundedLeasePool(
                self.root / "circuits" / f"{_provider_namespace(provider)}.probe",
                1,
            ).acquire(blocking=False)
        except RunBusyError:
            raise _open_failure(state, now) from None
        try:
            # State may have changed while the crash-released probe lease was
            # being acquired. Bind the token to the current epoch and reapply
            # a newly opened cooldown before admitting the call.
            current = store.read()
            if current is None or current.opened_until is None:
                probe.release()
                return (
                    _CircuitToken(0 if current is None else current.epoch, False),
                    None,
                )
            now = float(self._clock())
            if current.opened_until > now:
                probe.release()
                raise _open_failure(current, now)
            return _CircuitToken(current.epoch, True), probe
        except BaseException:
            probe.release()
            raise

    def _record_success(self, provider: str, token: _CircuitToken) -> None:
        if not self.options.enabled:
            return
        store = self._store(provider)
        while True:
            state = store.read()
            if state is None or state.epoch != token.epoch:
                return
            if (
                state.consecutive_failures == 0
                and state.opened_until is None
            ):
                return
            closed = _CircuitState(
                state.revision + 1,
                0,
                state.epoch,
                None,
                None,
            )
            try:
                store.compare_and_swap(state.revision, closed)
                return
            except RevisionConflictError:
                continue

    def _record_failure(
        self,
        provider: str,
        token: _CircuitToken,
        failure: ProviderFailure,
    ) -> None:
        if not self.options.enabled:
            return
        if failure.category not in _CIRCUIT_FAILURES:
            return
        store = self._store(provider)
        while True:
            state = store.read()
            failures = 1 if state is None else state.consecutive_failures + 1
            should_open = (
                failure.category in _IMMEDIATE_CIRCUIT_FAILURES
                or token.half_open_probe
                or failures >= self.options.circuit_failure_threshold
            )
            now = float(self._clock())
            retry_after = (
                None
                if failure.retry_after_seconds is None
                else min(3600.0, max(1.0, failure.retry_after_seconds))
            )
            opened_until = (
                now
                + (
                    self.options.circuit_cooldown_seconds
                    if retry_after is None
                    else retry_after
                )
                if should_open
                else None if state is None else state.opened_until
            )
            epoch = (
                (0 if state is None else state.epoch) + 1
                if should_open
                else 0 if state is None else state.epoch
            )
            next_state = _CircuitState(
                0 if state is None else state.revision + 1,
                failures,
                epoch,
                opened_until,
                failure.category.value,
            )
            try:
                if state is None:
                    store.create(next_state)
                else:
                    store.compare_and_swap(state.revision, next_state)
                return
            except (RevisionConflictError, StateConflictError):
                continue

    def _store(self, provider: str) -> AtomicStateStore[_CircuitState]:
        return AtomicStateStore(
            self.root / "circuits" / f"{_provider_namespace(provider)}.json",
            _CircuitContract(),
        )


def _memory_details(
    measurement: MemoryAvailability | None,
    threshold: float | None,
) -> dict[str, Any]:
    details: dict[str, Any] = {"threshold_fraction": threshold}
    if measurement is not None:
        details.update(
            {
                "available_bytes": measurement.available_bytes,
                "total_bytes": measurement.total_bytes,
                "available_fraction": measurement.fraction,
                "source": measurement.source,
            }
        )
    return details


def _observe_memory(
    observe: Callable[[str, Mapping[str, Any]], None] | None,
    event: str,
    measurement: MemoryAvailability | None,
    threshold: float | None,
) -> None:
    if observe is not None:
        observe(event, _memory_details(measurement, threshold))


def _open_failure(state: _CircuitState, now: float) -> ProviderFailure:
    category = (
        FailureCategory(state.category)
        if state.category in {item.value for item in _CIRCUIT_FAILURES}
        else FailureCategory.UNAVAILABLE
    )
    return ProviderFailure(
        "Provider circuit is open.",
        category=category,
        retryable=True,
        details={
            "code": "provider_circuit_open",
            "retry_after_seconds": max(0.0, (state.opened_until or now) - now),
        },
    )


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CorruptStateError(f"invalid {name}")
    return value


def _provider_namespace(provider: str) -> str:
    return hashlib.sha256(provider.encode("utf-8")).hexdigest()
