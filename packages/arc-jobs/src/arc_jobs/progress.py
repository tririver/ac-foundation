"""Shared validation for body-free progress metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_FORBIDDEN_PROGRESS_KEYS = frozenset(
    {
        "text",
        "token",
        "content",
        "output",
        "delta",
        "prompt",
        "candidate",
        "result",
    }
)


def validate_progress_data(value: Any) -> None:
    """Reject recursively nested fields that could carry response bodies."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_PROGRESS_KEYS:
                raise ValueError(f"progress contains forbidden field {key!r}")
            validate_progress_data(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            validate_progress_data(child)
