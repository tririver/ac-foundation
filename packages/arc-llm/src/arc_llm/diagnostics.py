"""Small shared credential redaction for local diagnostics and previews."""

from __future__ import annotations

import re
from typing import Any, Mapping


_SECRET_KEY_NAMES = {
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "passwd",
    "refreshtoken",
    "secret",
    "setcookie",
    "token",
    "xapikey",
    "accesstoken",
}
_KEY_VALUE_SECRET = re.compile(
    r"(?i)\b("
    r"api[-_]?key|authorization|cookie|credential|password|passwd|"
    r"refresh[-_]?token|secret|set-cookie|token|x-api-key|access[-_]?token"
    r")(\s*[:=]\s*)([^\s,;]+)"
)


def redact_text(value: str) -> str:
    value = re.sub(
        r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]+",
        r"\1 [REDACTED]",
        value,
    )
    value = _KEY_VALUE_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        value,
    )
    patterns = (
        r"\bsk-[A-Za-z0-9_-]{8,}\b",
        r"\b(?:gh[opusr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,})\b",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bAIza[0-9A-Za-z_-]{30,}\b",
        r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
    )
    for pattern in patterns:
        value = re.sub(pattern, "[REDACTED]", value)
    return value


def redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if _secret_key(key)
                else redact_value(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [redact_value(child) for child in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _secret_key(value: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(value).lower())
    return normalized in _SECRET_KEY_NAMES
