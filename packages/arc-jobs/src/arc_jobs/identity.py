from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .errors import InvalidRunIdError
from .models import SemanticKeyDigest

_SIMPLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def semantic_key(value: Any) -> SemanticKeyDigest:
    return SemanticKeyDigest(hashlib.sha256(canonical_json_bytes(value)).hexdigest())


def validate_simple_id(value: str, *, label: str = "identifier") -> str:
    if not isinstance(value, str) or _SIMPLE_ID.fullmatch(value) is None:
        raise InvalidRunIdError(f"invalid {label}: {value!r}")
    return value


def validate_artifact_id(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 512:
        raise InvalidRunIdError("artifact id must contain 1..512 ASCII characters")
    if "\\" in value or value.startswith("/") or value.endswith("/"):
        raise InvalidRunIdError(f"invalid artifact id: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} or _SIMPLE_ID.fullmatch(part) is None for part in parts):
        raise InvalidRunIdError(f"invalid artifact id: {value!r}")
    return value
