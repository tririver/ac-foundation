from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def validate_json_value(value: Any) -> None:
    """Require values representable by AC Foundation's exact JSON data model."""

    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            validate_json_value(child)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        if not isinstance(value, list):
            raise ValueError("JSON arrays must be lists")
        for child in value:
            validate_json_value(child)
        return
    raise ValueError(f"value of type {type(value).__name__} is not valid JSON")
