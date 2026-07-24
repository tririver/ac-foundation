"""Provider adapter registry."""

from __future__ import annotations

from collections.abc import Callable

from ..errors import InvalidRequestError
from .base import InputDeliveryMode, ProviderAdapter

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

    def delivery_modes(
        self,
        name: str,
        media_types: tuple[str, ...],
    ) -> tuple[InputDeliveryMode, ...]:
        mapping = self.create(name).capabilities().input_delivery
        return tuple(
            mapping.get(media_type, InputDeliveryMode.UNSUPPORTED)
            for media_type in media_types
        )

    def supporting(self, media_types: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.names()
            if all(
                mode is not InputDeliveryMode.UNSUPPORTED
                for mode in self.delivery_modes(name, media_types)
            )
        )


def default_registry() -> ProviderRegistry:
    from .claude import ClaudeAdapter
    from .codex import CodexAdapter
    from .kimi import KimiAdapter

    registry = ProviderRegistry()
    registry.register("codex", CodexAdapter)
    registry.register("claude", ClaudeAdapter)
    registry.register("kimi", KimiAdapter)
    return registry
