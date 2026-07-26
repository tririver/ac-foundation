"""Safe provider observer projection."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from arc_jobs import validate_progress_data

from .providers.base import NativeResumeHandle


class DurableProviderObserver:
    """Project provider progress and native handles into durable task state."""

    def __init__(
        self,
        *,
        context: Any,
        on_handle: Callable[[NativeResumeHandle], None],
        raw_limit_bytes: int = 256 * 1024,
    ) -> None:
        self.context = context
        self.on_handle = on_handle
        self.raw_limit_bytes = raw_limit_bytes
        self.raw_events: list[Mapping[str, Any] | str] = []
        self.raw_bytes = 0
        self.truncated = False

    def native_handle(self, handle: NativeResumeHandle) -> None:
        self.on_handle(handle)

    def raw_event(self, event: Mapping[str, Any] | str) -> None:
        size = len(repr(event).encode("utf-8", "replace"))
        if self.raw_bytes + size > self.raw_limit_bytes:
            self.truncated = True
            return
        self.raw_events.append(event)
        self.raw_bytes += size

    def progress(self, kind: str, data: Mapping[str, Any]) -> None:
        validate_progress_data(data)
        self.context.events.emit(kind, dict(data))
