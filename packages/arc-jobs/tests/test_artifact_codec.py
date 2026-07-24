from __future__ import annotations

import pytest

from arc_jobs import (
    ArtifactDigest,
    ArtifactRef,
    CorruptStateError,
    decode_artifact_digest,
    decode_artifact_ref,
    encode_artifact_digest,
    encode_artifact_ref,
)
from arc_jobs.effects import _decode_ref as decode_effect_ref
from arc_jobs.engine import _decode_ref as decode_snapshot_ref
from arc_jobs.groups import _artifact_value
from arc_jobs.storage import ImmutableArtifactStore


def _ref() -> ArtifactRef:
    return ArtifactRef(
        "result",
        ArtifactDigest("sha256", "a" * 64, 3),
        "application/json",
        "artifacts/objects/sha256/aa/" + "a" * 64,
    )


def test_artifact_codecs_round_trip_and_are_reused_for_group_values() -> None:
    digest = _ref().digest
    assert decode_artifact_digest(encode_artifact_digest(digest)) == digest
    assert decode_artifact_ref(encode_artifact_ref(_ref())) == _ref()
    assert _artifact_value(_ref()) == encode_artifact_ref(_ref())


@pytest.mark.parametrize(
    "value",
    (
        None,
        [],
        {"algorithm": "sha256", "value": "a" * 64},
        {"algorithm": "sha256", "value": "a" * 64, "size_bytes": 0, "x": 1},
        {"algorithm": "sha256", "value": "A" * 64, "size_bytes": 0},
        {"algorithm": "sha256", "value": "g" * 64, "size_bytes": 0},
        {"algorithm": "sha256", "value": "a" * 63, "size_bytes": 0},
        {"algorithm": "sha256", "value": "a" * 64, "size_bytes": -1},
        {"algorithm": "sha256", "value": "a" * 64, "size_bytes": True},
    ),
)
def test_artifact_digest_codec_rejects_invalid_closed_shapes(value) -> None:
    with pytest.raises(ValueError):
        decode_artifact_digest(value)


@pytest.mark.parametrize(
    "value",
    (
        None,
        [],
        {"artifact_id": "result", "digest": {}, "media_type": "application/json"},
        {
            "artifact_id": "result",
            "digest": encode_artifact_digest(_ref().digest),
            "media_type": "application/json",
            "relative_path": "objects/a",
            "unknown": None,
        },
        {
            "artifact_id": 1,
            "digest": encode_artifact_digest(_ref().digest),
            "media_type": "application/json",
            "relative_path": "objects/a",
        },
        {
            "artifact_id": "result",
            "digest": encode_artifact_digest(_ref().digest),
            "media_type": None,
            "relative_path": "objects/a",
        },
        {
            "artifact_id": "result",
            "digest": encode_artifact_digest(_ref().digest),
            "media_type": "application/json",
            "relative_path": False,
        },
    ),
)
def test_artifact_ref_codec_rejects_invalid_closed_shapes(value) -> None:
    with pytest.raises(ValueError):
        decode_artifact_ref(value)


@pytest.mark.parametrize(
    "digest",
    (
        {"algorithm": "sha256", "value": "A" * 64, "size_bytes": 0},
        {"algorithm": "sha256", "value": "g" * 64, "size_bytes": 0},
        {"algorithm": "sha256", "value": "a" * 64, "size_bytes": True},
    ),
)
def test_jobs_durable_consumers_reject_the_same_invalid_digest(digest) -> None:
    document = {
        "artifact_id": "result",
        "digest": digest,
        "media_type": "application/json",
        "relative_path": "objects/a",
    }
    for decode in (decode_snapshot_ref, decode_effect_ref):
        with pytest.raises(CorruptStateError):
            decode(document)
    with pytest.raises(CorruptStateError):
        ImmutableArtifactStore._digest_from_json(digest)
