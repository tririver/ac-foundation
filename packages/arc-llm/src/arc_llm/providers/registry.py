"""Provider adapter registry."""

from __future__ import annotations

from collections.abc import Callable

from ..errors import InvalidRequestError
from .base import ProviderAdapter

ProviderFactory = Callable[[], ProviderAdapter]


class ProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        if not name or name in self._factories:
            raise ValueError(f"Provider is already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str) -> ProviderAdapter:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise InvalidRequestError(f"Unknown provider: {name}") from exc
        adapter = factory()
        if adapter.name != name:
            raise RuntimeError("Provider factory returned an adapter with the wrong name.")
        return adapter

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

def default_registry() -> ProviderRegistry:
    from .claude import ClaudeAdapter
    from .codex import CodexAdapter
    from .dsh import DshAdapter
    from .kimi import KimiAdapter

    registry = ProviderRegistry()
    registry.register("codex", CodexAdapter)
    registry.register("claude", ClaudeAdapter)
    registry.register("kimi", KimiAdapter)
    registry.register("dsh", DshAdapter)
    return registry
