from __future__ import annotations

import pytest

from arc_jobs import ArtifactDigest, ArtifactRef, encode_artifact_ref
from arc_proposer_reviewer.artifacts import (
    artifact_ref_from_document,
    artifact_ref_to_document,
)
from arc_proposer_reviewer.dialogue import (
    TranscriptTurn,
    decode_transcript_turn,
    encode_transcript_turn,
)


def _ref() -> ArtifactRef:
    return ArtifactRef(
        "proposal",
        ArtifactDigest("sha256", "a" * 64, 4),
        "application/json",
        "objects/sha256/aa/" + "a" * 64,
    )


def test_proposer_reviewer_artifact_codec_uses_the_shared_document() -> None:
    assert artifact_ref_to_document(_ref()) == encode_artifact_ref(_ref())
    assert artifact_ref_from_document(artifact_ref_to_document(_ref())) == _ref()
    turn = encode_transcript_turn(TranscriptTurn("proposer", "worker", 1, _ref()))
    assert turn["content_ref"] == encode_artifact_ref(_ref())


def test_transcript_turn_codec_is_closed_and_round_trips() -> None:
    turn = TranscriptTurn("proposer", "worker", 1, _ref(), ("reviewer",))
    document = encode_transcript_turn(turn)

    assert decode_transcript_turn(document) == turn

    document["unknown"] = None
    with pytest.raises(ValueError, match="closed shape"):
        decode_transcript_turn(document)


@pytest.mark.parametrize(
    "value",
    (
        {
            "artifact_id": "proposal",
            "digest": {},
            "media_type": "application/json",
            "relative_path": "path",
        },
        {
            "artifact_id": "proposal",
            "digest": {"algorithm": "sha256", "value": "A" * 64, "size_bytes": 4},
            "media_type": "application/json",
            "relative_path": "path",
        },
        {
            "artifact_id": "proposal",
            "digest": {"algorithm": "sha256", "value": "a" * 64, "size_bytes": True},
            "media_type": "application/json",
            "relative_path": "path",
        },
    ),
)
def test_proposer_reviewer_codec_translates_shared_errors_to_value_error(value) -> None:
    with pytest.raises(ValueError, match="invalid artifact reference"):
        artifact_ref_from_document(value)
