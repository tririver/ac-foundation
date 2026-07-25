"""Operational accounting for scoped interaction resolvers."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Mapping
from typing import Any

from .request import (
    InteractionRequest,
    InteractionResolver,
    InteractionResponse,
)


class ScopedInteractionLedger:
    """Share one resolver while accounting for requests by scope.

    The ledger observes requests only. It does not cache responses, change
    admission order, or alter the shared resolver's global request budget.
    """

    def __init__(
        self,
        resolver: InteractionResolver,
        scope_ids: Iterable[str],
    ) -> None:
        self.resolver = resolver
        self._lock = threading.Lock()
        self._seen_signatures: set[str] = set()
        self._per_scope: dict[str, dict[str, Any]] = {}
        for scope_id in scope_ids:
            if not isinstance(scope_id, str) or not scope_id:
                raise ValueError("scope IDs must be non-empty strings")
            if scope_id in self._per_scope:
                raise ValueError(f"duplicate scope ID: {scope_id}")
            self._per_scope[scope_id] = {
                "request_count": 0,
                "repeated_request_count": 0,
                "error_counts": {},
            }
        if not self._per_scope:
            raise ValueError("scope_ids must not be empty")

    def scoped(self, scope_id: str) -> InteractionResolver:
        if scope_id not in self._per_scope:
            raise ValueError(f"unknown scope ID: {scope_id}")
        return _ScopedInteractionResolver(self, scope_id)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                scope_id: {
                    "request_count": counts["request_count"],
                    "repeated_request_count": counts[
                        "repeated_request_count"
                    ],
                    "error_counts": dict(counts["error_counts"]),
                }
                for scope_id, counts in self._per_scope.items()
            }

    def _resolve(
        self,
        scope_id: str,
        request: InteractionRequest,
    ) -> InteractionResponse:
        signature = _request_signature(request)
        with self._lock:
            counts = self._per_scope[scope_id]
            counts["request_count"] += 1
            if signature in self._seen_signatures:
                counts["repeated_request_count"] += 1
            else:
                self._seen_signatures.add(signature)

        response = self.resolver.resolve(request)
        error = response.error
        if isinstance(error, Mapping):
            code = error.get("code")
            if isinstance(code, str) and code:
                with self._lock:
                    error_counts = self._per_scope[scope_id]["error_counts"]
                    error_counts[code] = error_counts.get(code, 0) + 1
        return response


class _ScopedInteractionResolver:
    def __init__(
        self,
        ledger: ScopedInteractionLedger,
        scope_id: str,
    ) -> None:
        self._ledger = ledger
        self._scope_id = scope_id

    def resolve(self, request: InteractionRequest) -> InteractionResponse:
        return self._ledger._resolve(self._scope_id, request)


def _request_signature(request: InteractionRequest) -> str:
    return json.dumps(
        {
            "operation": request.operation,
            "arguments": dict(request.arguments),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = ["ScopedInteractionLedger"]
