from __future__ import annotations

import pytest

from ac_proposer_reviewer.models import REVIEW_SCHEMA_VERSION
from ac_proposer_reviewer.validation import RequestValidationError, decode_review


def valid_review() -> dict[str, object]:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "action": "continue",
        "reason": "Both proposals need one refinement.",
        "feedback": {"a": "Tighten A.", "b": "Clarify B."},
        "payload": {"winner": "a"},
    }


def test_review_feedback_exactly_matches_successful_active_proposers() -> None:
    review = decode_review(
        valid_review(),  # type: ignore[arg-type]
        active_proposer_ids=("a", "b"),
        validate_payload=lambda value: None,
    )
    assert tuple(review.feedback) == ("a", "b")


@pytest.mark.parametrize(
    "feedback",
    [
        {"a": "Tighten A."},
        {"a": "Tighten A.", "b": "Clarify B.", "failed": "Try again."},
        {"a": "", "b": "Clarify B."},
    ],
)
def test_review_rejects_missing_extra_or_empty_feedback(
    feedback: dict[str, str],
) -> None:
    document = valid_review()
    document["feedback"] = feedback
    with pytest.raises(RequestValidationError):
        decode_review(
            document,  # type: ignore[arg-type]
            active_proposer_ids=("a", "b"),
            validate_payload=lambda value: None,
        )


def test_review_is_closed_and_payload_validation_is_delegated() -> None:
    document = valid_review()
    document["diagnostics"] = {}
    with pytest.raises(RequestValidationError, match="unknown field"):
        decode_review(
            document,  # type: ignore[arg-type]
            active_proposer_ids=("a", "b"),
            validate_payload=lambda value: None,
        )

    del document["diagnostics"]
    with pytest.raises(RuntimeError, match="payload rejected"):
        decode_review(
            document,  # type: ignore[arg-type]
            active_proposer_ids=("a", "b"),
            validate_payload=lambda value: (_ for _ in ()).throw(
                RuntimeError("payload rejected")
            ),
        )
