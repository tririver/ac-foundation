"""Safe provider observer projection."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .diagnostics import redact_text
from .providers.base import NativeResumeHandle


def message_preview(value: str) -> dict[str, Any]:
    """Return a credential-redacted preview of one complete logical message."""

    normalized = " ".join(redact_text(value).split())
    return {
        "preview": normalized[:100],
        "truncated": len(normalized) > 100,
    }


class DurableProviderObserver:
    """Project provider progress and native handles into durable task state."""

    def __init__(
        self,
        *,
        context: Any,
        on_handle: Callable[[NativeResumeHandle], None],
        task_id: str,
        provider: str,
        generation: int,
        host_turn_round: int,
    ) -> None:
        self.context = context
        self.on_handle = on_handle
        self.metadata = {
            "task_id": task_id,
            "provider": provider,
            "generation": generation,
            "host_turn_round": host_turn_round,
        }
        self.observation_errors: list[str] = []

    def native_handle(self, handle: NativeResumeHandle) -> None:
        try:
            self.on_handle(handle)
        except Exception as exc:
            self._record_observation_error(exc)

    def progress(self, kind: str, data: Mapping[str, Any]) -> None:
        try:
            self.context.events.emit(kind, {**dict(data), **self.metadata})
        except Exception as exc:
            self._record_observation_error(exc)

    def _record_observation_error(self, exc: Exception) -> None:
        if len(self.observation_errors) < 8:
            self.observation_errors.append(type(exc).__name__)
