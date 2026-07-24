"""Safe provider observer projection."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from arc_jobs import validate_progress_data

from .providers.base import NativeResumeHandle


class DurableProviderObserver:
    """Connect a provider adapter to arc-jobs effects and artifacts."""

    def __init__(
        self,
        *,
        context: Any,
        effect_id: str,
        on_handle: Callable[[NativeResumeHandle], None],
        raw_limit_bytes: int = 256 * 1024,
    ) -> None:
        self.context = context
        self.effect_id = effect_id
        self.on_handle = on_handle
        self.raw_limit_bytes = raw_limit_bytes
        self.raw_events: list[Mapping[str, Any] | str] = []
        self.raw_bytes = 0
        self.truncated = False
        self.delivered = False

    def before_delivery(self) -> None:
        if not self.delivered:
            self.context.effects.mark_may_have_run(self.effect_id)
            self.delivered = True

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

    def response_saved(self, ref: Any) -> None:
        self.context.effects.save_output(self.effect_id, ref)
