from __future__ import annotations

import pytest

from arc_llm import (
    InteractionRequest,
    InteractionResponse,
    ScopedInteractionLedger,
)


class _Resolver:
    def __init__(self) -> None:
        self.requests: list[InteractionRequest] = []

    def resolve(self, request: InteractionRequest) -> InteractionResponse:
        self.requests.append(request)
        if request.operation == "fail":
            return InteractionResponse(
                request_id=request.request_id,
                error={"code": "expected_failure", "message": "failed"},
            )
        return InteractionResponse(
            request_id=request.request_id,
            result={"ok": True},
        )


def test_scoped_ledger_observes_without_changing_shared_resolver() -> None:
    resolver = _Resolver()
    ledger = ScopedInteractionLedger(resolver, ["loop-a", "loop-b"])
    loop_a = ledger.scoped("loop-a")
    loop_b = ledger.scoped("loop-b")

    loop_a.resolve(InteractionRequest("a-1", "search", {"query": "same"}))
    loop_b.resolve(InteractionRequest("b-1", "search", {"query": "same"}))
    loop_b.resolve(InteractionRequest("b-2", "fail", {}))

    assert len(resolver.requests) == 3
    assert ledger.snapshot() == {
        "loop-a": {
            "request_count": 1,
            "repeated_request_count": 0,
            "error_counts": {},
        },
        "loop-b": {
            "request_count": 2,
            "repeated_request_count": 1,
            "error_counts": {"expected_failure": 1},
        },
    }


@pytest.mark.parametrize("scope_ids", [[], ["same", "same"], [""]])
def test_scoped_ledger_rejects_invalid_scope_ids(
    scope_ids: list[str],
) -> None:
    with pytest.raises(ValueError):
        ScopedInteractionLedger(_Resolver(), scope_ids)
