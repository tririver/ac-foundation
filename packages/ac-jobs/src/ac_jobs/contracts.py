from __future__ import annotations

from typing import Mapping, Protocol, TypeVar

from .models import JsonValue

T = TypeVar("T")


class StateContract(Protocol[T]):
    schema_version: str

    def encode(self, value: T) -> Mapping[str, JsonValue]: ...

    def decode(self, document: Mapping[str, JsonValue]) -> T: ...

    def validate_transition(self, previous: T | None, next: T) -> None: ...
