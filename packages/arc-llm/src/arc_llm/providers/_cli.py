"""Shared private helpers for command-line adapters."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..errors import DeliveryState, FailureCategory, ProviderFailure
from ..output import CandidateMaterial
from .base import (
    NativeResumeHandle,
    ProviderExecution,
    ProviderObserver,
    ProviderTerminalKind,
    ProviderUsage,
)
from .process import ProcessRunner


@dataclass
class EventAccumulator:
    provider: str
    observer: ProviderObserver
    parse_event: Callable[
        [Mapping[str, Any]],
        tuple[CandidateMaterial | None, str | None, ProviderUsage | None],
    ]
    extract_failure: Callable[[Mapping[str, Any]], ProviderFailure | None] | None = None

    def __post_init__(self) -> None:
        self.buffer = b""
        self.candidates: list[CandidateMaterial] = []
        self.handle: NativeResumeHandle | None = None
        self.usage: ProviderUsage | None = None
        self.failure: ProviderFailure | None = None
        self.raw_events: list[Mapping[str, Any] | str] = []
        self.raw_bytes = 0
        self.raw_truncated = False

    def feed(self, chunk: bytes) -> None:
        self.buffer += chunk
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            self._line(line)

    def finish(self, *, validate_terminal: bool = True) -> None:
        if self.buffer.strip():
            self._line(self.buffer)
        self.buffer = b""
        if not validate_terminal or self.failure is not None:
            return
        terminal = [item for item in self.candidates if item.terminal]
        if len(terminal) > 1:
            raise ProviderFailure(
                "Provider event stream has multiple terminal responses.",
                category=FailureCategory.SCHEMA,
                delivery=DeliveryState.MAY_HAVE_RUN,
                details={"code": "invalid_terminal_closure"},
            )
        if not terminal:
            raise ProviderFailure(
                "Provider event stream ended without terminal material.",
                category=FailureCategory.SCHEMA,
                delivery=DeliveryState.MAY_HAVE_RUN,
                details={"code": "incomplete_terminal_closure"},
            )

    def _line(self, raw: bytes) -> None:
        text = raw.decode("utf-8", "replace").strip()
        if not text:
            return
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            raw: Mapping[str, Any] | str = {
                "kind": "unparsed",
                "text": text[:4096],
            }
            self._record_raw(raw)
            return
        if not isinstance(event, Mapping):
            self._record_raw({"kind": "value"})
            return
        self._record_raw(event)
        if self.failure is None and self.extract_failure is not None:
            self.failure = self.extract_failure(event)
        candidate, handle, usage = self.parse_event(event)
        if handle is not None and (self.handle is None or self.handle.value != handle):
            self.handle = NativeResumeHandle(self.provider, handle)
            self.observer.native_handle(self.handle)
        if candidate is not None:
            self.candidates.append(candidate)
        if usage is not None:
            self.usage = usage

    def _record_raw(self, event: Mapping[str, Any] | str) -> None:
        safe_event = _redact_value(event)
        self.observer.raw_event(safe_event)
        size = len(repr(safe_event).encode("utf-8", "replace"))
        if self.raw_bytes + size > 256 * 1024:
            self.raw_truncated = True
            return
        self.raw_events.append(safe_event)
        self.raw_bytes += size

    def diagnostics(self) -> dict[str, Any]:
        return {
            "raw_events": list(self.raw_events),
            "raw_events_truncated": self.raw_truncated,
        }


def run_cli(
    *,
    provider: str,
    argv: Sequence[str],
    prompt: str,
    observer: ProviderObserver,
    stop: Any,
    timeout: float,
    parse_event: Callable[
        [Mapping[str, Any]],
        tuple[CandidateMaterial | None, str | None, ProviderUsage | None],
    ],
    runner: ProcessRunner,
    env: Mapping[str, str] | None,
    validate_terminal: bool = True,
    extract_failure: (
        Callable[[Mapping[str, Any]], ProviderFailure | None] | None
    ) = None,
) -> ProviderExecution:
    accumulator = EventAccumulator(
        provider,
        observer,
        parse_event,
        extract_failure=extract_failure,
    )
    result = runner.run(
        argv,
        stdin=prompt.encode("utf-8"),
        env=os.environ if env is None else env,
        idle_timeout_seconds=timeout,
        before_stdin=observer.before_delivery,
        stop_check=stop.raise_if_requested,
        on_stdout=accumulator.feed,
    )
    # Still consume a final non-newline event for diagnostics, but terminal
    # shape validation must not replace a typed provider failure.
    accumulator.finish(
        validate_terminal=(
            validate_terminal
            and result.returncode == 0
            and accumulator.failure is None
        )
    )
    if accumulator.failure is not None:
        return ProviderExecution(
            ProviderTerminalKind.FAILED,
            candidates=tuple(accumulator.candidates),
            native_handle=accumulator.handle,
            usage=accumulator.usage,
            failure=accumulator.failure,
            diagnostics={
                "returncode": result.returncode,
                **accumulator.diagnostics(),
            },
        )
    if result.returncode != 0:
        failure = classify_cli_failure(result.stderr.decode("utf-8", "replace"))
        return ProviderExecution(
            ProviderTerminalKind.FAILED,
            candidates=tuple(accumulator.candidates),
            native_handle=accumulator.handle,
            usage=accumulator.usage,
            failure=failure,
            diagnostics={
                "returncode": result.returncode,
                **accumulator.diagnostics(),
            },
        )
    return ProviderExecution(
        ProviderTerminalKind.COMPLETED,
        candidates=tuple(accumulator.candidates),
        native_handle=accumulator.handle,
        usage=accumulator.usage,
        diagnostics=accumulator.diagnostics(),
    )


def executable_diagnostic(provider: str, binary: str) -> tuple[bool, str | None]:
    path = shutil.which(binary)
    return path is not None, path


def classify_cli_failure(stderr: str) -> ProviderFailure:
    lowered = stderr.lower()
    if "auth" in lowered or "unauthorized" in lowered or "401" in lowered:
        category = FailureCategory.AUTHENTICATION
    elif "quota" in lowered or "insufficient" in lowered:
        category = FailureCategory.QUOTA
    elif "rate limit" in lowered or "429" in lowered:
        category = FailureCategory.RATE_LIMIT
    elif "invalid request" in lowered or "schema" in lowered:
        category = FailureCategory.INVALID_REQUEST
    else:
        category = FailureCategory.TRANSPORT
    retry_after = None
    match = re.search(
        r"(?i)retry[- ]after\s*[:=]?\s*(\d+(?:\.\d+)?)",
        stderr,
    )
    if match is not None:
        retry_after = float(match.group(1))
    return ProviderFailure(
        "Provider command failed.",
        category=category,
        delivery=DeliveryState.MAY_HAVE_RUN,
        retryable=category in {FailureCategory.RATE_LIMIT, FailureCategory.TRANSPORT},
        retry_after_seconds=retry_after,
        details={"diagnostic": _redact_text(stderr[:4096])},
    )


_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
}


def _redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if str(key).lower() in _SECRET_KEYS
                else _redact_value(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(child) for child in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _redact_text(value: str) -> str:
    value = re.sub(
        r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]+",
        r"\1 [REDACTED]",
        value,
    )
    value = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "sk-[REDACTED]", value)
    return value
