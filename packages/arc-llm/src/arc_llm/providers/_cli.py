"""Shared private helpers for command-line adapters."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..errors import FailureCategory, ProviderFailure
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
    extract_message: Callable[[Mapping[str, Any]], str | None] | None = None

    def __post_init__(self) -> None:
        self.buffer = b""
        self.candidates: list[CandidateMaterial] = []
        self.handle: NativeResumeHandle | None = None
        self.usage: ProviderUsage | None = None
        self.failure: ProviderFailure | None = None
        self.last_terminal_evidence: str | None = None
        self.raw_events: list[Mapping[str, Any] | str] = []
        self.raw_bytes = 0
        self.raw_truncated = False
        self.event_count = 0
        self.terminal_event_types: list[str] = []

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
                details={"code": "invalid_terminal_closure"},
            )
        if not terminal:
            raise ProviderFailure(
                "Provider event stream ended without terminal material.",
                category=FailureCategory.TRANSPORT,
                details={"code": "incomplete_terminal_closure"},
            )

    @property
    def has_success_evidence(self) -> bool:
        return self.last_terminal_evidence == "success"

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
        try:
            candidate, handle, usage = self.parse_event(event)
        except Exception as exc:
            self.failure = ProviderFailure(
                "Provider event could not be normalized.",
                category=FailureCategory.SCHEMA,
                details={
                    "code": "provider_event_parse_failed",
                    "error_type": type(exc).__name__,
                },
            )
            self.last_terminal_evidence = "failure"
            return
        if handle is not None and (self.handle is None or self.handle.value != handle):
            self.handle = NativeResumeHandle(self.provider, handle)
            self.observer.native_handle(self.handle)
        if candidate is not None:
            self.candidates.append(candidate)
            if candidate.terminal:
                self.last_terminal_evidence = "success"
                self.failure = None
        if usage is not None:
            self.usage = usage
        event_type = event.get("type")
        if event_type == "turn.completed":
            self.last_terminal_evidence = "success"
            self.failure = None
        event_failure = (
            None
            if self.extract_failure is None
            else self.extract_failure(event)
        )
        if event_failure is not None:
            self.failure = event_failure
            self.last_terminal_evidence = "failure"
        if self.extract_message is not None:
            message = self.extract_message(event)
            if message:
                _safe_progress(
                    self.observer,
                    "llm_message",
                    {
                        "direction": "response",
                        "message_kind": "assistant",
                        **_preview_document(message),
                    },
                )

    def _record_raw(self, event: Mapping[str, Any] | str) -> None:
        safe_event = _redact_value(event)
        self.event_count += 1
        if isinstance(event, Mapping):
            event_type = event.get("type")
            if (
                isinstance(event_type, str)
                and event_type in {"error", "turn.completed", "turn.failed"}
            ):
                self.terminal_event_types.append(event_type)
        size = len(repr(safe_event).encode("utf-8", "replace"))
        if self.raw_bytes + size > 256 * 1024:
            self.raw_truncated = True
            return
        self.raw_events.append(safe_event)
        self.raw_bytes += size

    def diagnostics(self) -> dict[str, Any]:
        observation_errors = getattr(
            self.observer, "observation_errors", ()
        )
        return {
            "raw_events": list(self.raw_events),
            "raw_events_truncated": self.raw_truncated,
            "terminal_event_types": list(self.terminal_event_types),
            "last_terminal_evidence": self.last_terminal_evidence,
            "event_count": self.event_count,
            "observation_errors": (
                list(observation_errors)
                if isinstance(observation_errors, (list, tuple))
                else []
            ),
        }


def run_cli(
    *,
    provider: str,
    argv: Sequence[str],
    prompt: str,
    observer: ProviderObserver,
    stop: Any,
    timeout: float | None,
    parse_event: Callable[
        [Mapping[str, Any]],
        tuple[CandidateMaterial | None, str | None, ProviderUsage | None],
    ],
    runner: ProcessRunner,
    env: Mapping[str, str] | None,
    cwd: Path,
    validate_terminal: bool = True,
    extract_failure: (
        Callable[[Mapping[str, Any]], ProviderFailure | None] | None
    ) = None,
    extract_message: Callable[[Mapping[str, Any]], str | None] | None = None,
) -> ProviderExecution:
    accumulator = EventAccumulator(
        provider,
        observer,
        parse_event,
        extract_failure=extract_failure,
        extract_message=extract_message,
    )
    _safe_progress(observer, "llm_provider_started", {})
    activity_started = False
    last_activity_event = 0.0
    stdout_bytes = 0
    stderr_bytes = 0

    def activity(stream: str, chunk: bytes) -> None:
        nonlocal activity_started, last_activity_event, stdout_bytes, stderr_bytes
        if stream == "stdout":
            stdout_bytes += len(chunk)
        else:
            stderr_bytes += len(chunk)
        now = time.monotonic()
        if not activity_started or now - last_activity_event >= 20.0:
            activity_started = True
            last_activity_event = now
            _safe_progress(
                observer,
                "llm_pipe_activity",
                {
                    "stream": stream,
                    "stdout_bytes": stdout_bytes,
                    "stderr_bytes": stderr_bytes,
                },
            )

    def stdout(chunk: bytes) -> None:
        activity("stdout", chunk)
        accumulator.feed(chunk)

    def stderr(chunk: bytes) -> None:
        activity("stderr", chunk)

    try:
        result = runner.run(
            argv,
            stdin=prompt.encode("utf-8"),
            env=os.environ if env is None else env,
            cwd=cwd,
            idle_timeout_seconds=timeout,
            stop_check=stop.raise_if_requested,
            on_stdout=stdout,
            on_stderr=stderr,
        )
    except ProviderFailure as runner_failure:
        accumulator.finish(validate_terminal=False)
        terminal_candidates = [
            candidate
            for candidate in accumulator.candidates
            if candidate.terminal
        ]
        if len(terminal_candidates) > 1:
            accumulator.failure = ProviderFailure(
                "Provider event stream has multiple terminal responses.",
                category=FailureCategory.SCHEMA,
                details={"code": "invalid_terminal_closure"},
            )
            accumulator.last_terminal_evidence = "failure"
        if (
            runner_failure.category is FailureCategory.TIMEOUT
            and accumulator.has_success_evidence
        ):
            warning = _completion_warning(
                provider,
                code="provider_idle_timeout_with_valid_output",
                message=(
                    "The configured idle timeout elapsed after the provider "
                    "wrote a complete terminal response."
                ),
                returncode=_process_detail_int(runner_failure, "returncode"),
            )
            diagnostics = {
                "returncode": _process_detail_int(
                    runner_failure, "returncode"
                ),
                "runner_failure": True,
                **_process_failure_diagnostics(runner_failure),
                **accumulator.diagnostics(),
                "warnings": [warning],
            }
            _safe_progress(
                observer,
                "llm_provider_finished",
                {"warning_code": warning["code"]},
            )
            return ProviderExecution(
                ProviderTerminalKind.COMPLETED,
                candidates=tuple(accumulator.candidates),
                native_handle=accumulator.handle,
                usage=accumulator.usage,
                diagnostics=diagnostics,
            )
        failure = _prefer_definitive_failure(
            runner_failure,
            accumulator.failure,
        )
        assert failure is not None
        execution = ProviderExecution(
            ProviderTerminalKind.FAILED,
            candidates=tuple(accumulator.candidates),
            native_handle=accumulator.handle,
            usage=accumulator.usage,
            failure=failure,
            diagnostics={
                "returncode": None,
                "runner_failure": True,
                **_process_failure_diagnostics(runner_failure),
                **accumulator.diagnostics(),
            },
        )
        _safe_progress(
            observer,
            "llm_provider_failed",
            {"category": failure.category.value},
        )
        return execution
    # Still consume a final non-newline event for diagnostics, but terminal
    # shape validation must not replace a typed provider failure.
    try:
        accumulator.finish(
            validate_terminal=(
                validate_terminal
                and accumulator.failure is None
                and (
                    result.returncode == 0
                    or accumulator.has_success_evidence
                )
            )
        )
    except ProviderFailure as closure_failure:
        accumulator.failure = closure_failure
        accumulator.last_terminal_evidence = "failure"
    failure = accumulator.failure
    if (
        validate_terminal
        and failure is None
        and not accumulator.has_success_evidence
    ):
        failure = ProviderFailure(
            "Provider event stream ended without terminal material.",
            category=FailureCategory.TRANSPORT,
            details={"code": "incomplete_terminal_closure"},
        )
        accumulator.failure = failure
        accumulator.last_terminal_evidence = "failure"
    if result.returncode != 0:
        exit_failure = classify_cli_failure(
            result.stderr.decode("utf-8", "replace")
        )
        if accumulator.has_success_evidence:
            warning = _completion_warning(
                provider,
                code="provider_nonzero_exit_with_valid_output",
                message=(
                    "The provider returned a nonzero exit after writing a "
                    "complete terminal response."
                ),
                returncode=result.returncode,
            )
            diagnostics = {
                "returncode": result.returncode,
                **_process_result_diagnostics(result),
                **accumulator.diagnostics(),
                "warnings": [warning],
            }
            _safe_progress(
                observer,
                "llm_provider_finished",
                {"warning_code": warning["code"]},
            )
            return ProviderExecution(
                ProviderTerminalKind.COMPLETED,
                candidates=tuple(accumulator.candidates),
                native_handle=accumulator.handle,
                usage=accumulator.usage,
                diagnostics=diagnostics,
            )
        failure = _prefer_definitive_failure(failure, exit_failure)
    if failure is not None:
        execution = ProviderExecution(
            ProviderTerminalKind.FAILED,
            candidates=tuple(accumulator.candidates),
            native_handle=accumulator.handle,
            usage=accumulator.usage,
            failure=failure,
            diagnostics={
                "returncode": result.returncode,
                **_process_result_diagnostics(result),
                **accumulator.diagnostics(),
            },
        )
        _safe_progress(
            observer,
            "llm_provider_failed",
            {"category": failure.category.value},
        )
        return execution
    execution = ProviderExecution(
        ProviderTerminalKind.COMPLETED,
        candidates=tuple(accumulator.candidates),
        native_handle=accumulator.handle,
        usage=accumulator.usage,
        diagnostics={
            **_process_result_diagnostics(result),
            **accumulator.diagnostics(),
        },
    )
    _safe_progress(observer, "llm_provider_finished", {})
    return execution


def _prefer_definitive_failure(
    current: ProviderFailure | None,
    candidate: ProviderFailure | None,
) -> ProviderFailure | None:
    """Replace a generic transport failure only with definitive evidence."""

    if current is None:
        return candidate
    if (
        candidate is not None
        and current.category is FailureCategory.TRANSPORT
        and candidate.category is not FailureCategory.TRANSPORT
    ):
        return candidate
    return current


def executable_diagnostic(provider: str, binary: str) -> tuple[bool, str | None]:
    path = shutil.which(binary)
    return path is not None, path


def classify_cli_failure(stderr: str) -> ProviderFailure:
    lowered = stderr.lower()
    if re.search(
        r"(?:^|\n)\s*(?:401|403)(?:\b|:)"
        r"|\bhttp(?:/\d(?:\.\d)?)?\s*(?:401|403)(?:\b|:)"
        r"|\bunauthori[sz]ed\b|\bauthentication\s+(?:failed|required|error)\b",
        lowered,
    ):
        category = FailureCategory.AUTHENTICATION
    elif re.search(
        r"\b(?:quota\s+(?:exceeded|exhausted)|insufficient_quota)\b",
        lowered,
    ):
        category = FailureCategory.QUOTA
    elif re.search(
        r"(?:^|\b)(?:http(?:/\d(?:\.\d)?)?\s*)?429(?:\b|:)"
        r"|\brate[- ]limit(?:ed|ing)?\b",
        lowered,
    ):
        category = FailureCategory.RATE_LIMIT
    elif re.search(r"\binvalid[- ]request\b", lowered):
        category = FailureCategory.INVALID_REQUEST
    else:
        category = FailureCategory.TRANSPORT
    retry_after = None
    match = re.search(
        r"(?i)retry[- ]after\s*[:=]?\s*(\d+(?:\.\d+)?)",
        stderr,
    )
    if match is not None:
        retry_after = min(3600.0, max(1.0, float(match.group(1))))
    return ProviderFailure(
        "Provider command failed.",
        category=category,
        retryable=category in {FailureCategory.RATE_LIMIT, FailureCategory.TRANSPORT},
        retry_after_seconds=retry_after,
        details={"diagnostic": _redact_text(stderr[:4096])},
    )


def _completion_warning(
    provider: str,
    *,
    code: str,
    message: str,
    returncode: int | None,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "provider": provider,
        "returncode": returncode,
    }


def _process_detail_int(failure: ProviderFailure, key: str) -> int | None:
    value = failure.details.get(key)
    return value if type(value) is int else None


def _process_failure_diagnostics(failure: ProviderFailure) -> dict[str, Any]:
    diagnostics = {
        key: failure.details[key]
        for key in (
            "stdout_bytes",
            "stderr_bytes",
            "stdout_truncated",
            "stderr_truncated",
            "stderr_tail",
            "last_activity_at",
            "termination_reason",
        )
        if key in failure.details
    }
    stderr_tail = diagnostics.get("stderr_tail")
    if isinstance(stderr_tail, str):
        diagnostics["stderr_tail"] = _redact_text(stderr_tail)
    return diagnostics


def _process_result_diagnostics(result: Any) -> dict[str, Any]:
    return {
        "stdout_bytes": (
            len(result.stdout)
            if result.stdout_bytes is None
            else result.stdout_bytes
        ),
        "stderr_bytes": (
            len(result.stderr)
            if result.stderr_bytes is None
            else result.stderr_bytes
        ),
        "stdout_truncated": bool(result.stdout_truncated),
        "stderr_truncated": bool(result.stderr_truncated),
        "stderr_tail": _redact_text(
            result.stderr.decode("utf-8", "replace")
        ),
        "last_activity_at": result.last_activity_at,
        "termination_reason": None,
    }


def _preview_document(value: str) -> dict[str, Any]:
    normalized = " ".join(_redact_text(value).split())
    return {
        "preview": normalized[:100],
        "truncated": len(normalized) > 100,
    }


def _safe_progress(
    observer: ProviderObserver,
    kind: str,
    data: Mapping[str, Any],
) -> None:
    try:
        observer.progress(kind, data)
    except Exception:
        # Observation must never abort an already-running paid provider call.
        return


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
