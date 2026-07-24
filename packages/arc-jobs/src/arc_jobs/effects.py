from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping, Protocol

from .cancellation import CancellationToken
from .artifacts import decode_artifact_ref, encode_artifact_ref
from .contracts import StateContract
from .errors import (
    CorruptStateError,
    InvalidTransitionError,
    StateConflictError,
    UnsafeEffectRecoveryError,
)
from .identity import validate_simple_id
from .models import (
    ArtifactRef,
    EffectRecord,
    EffectRequestDigest,
    EffectStage,
    JsonValue,
    RecoveryDecision,
)
from .storage import AtomicStateStore, ImmutableArtifactStore, require_fields


class EffectRecoveryPolicy(Protocol):
    def classify(self, record: EffectRecord) -> RecoveryDecision: ...


def _ref_json(value: ArtifactRef) -> dict[str, JsonValue]:
    return encode_artifact_ref(value)


def _decode_ref(value: JsonValue) -> ArtifactRef | None:
    if value is None:
        return None
    try:
        return decode_artifact_ref(value)
    except ValueError as exc:
        raise CorruptStateError("invalid artifact ref") from exc


class _EffectContract(StateContract[EffectRecord]):
    schema_version = "arc.jobs.effect.v1"

    def encode(self, value: EffectRecord) -> Mapping[str, JsonValue]:
        return {
            "effect_id": value.effect_id,
            "effect_request_digest": {"sha256": value.effect_request_digest.sha256},
            "stage": value.stage.value,
            "details": dict(value.details),
            "external_handle": value.external_handle,
            "output_ref": _ref_json(value.output_ref) if value.output_ref else None,
            "revision": value.revision,
        }

    def decode(self, document: Mapping[str, JsonValue]) -> EffectRecord:
        require_fields(
            document,
            required={
                "effect_id",
                "effect_request_digest",
                "stage",
                "details",
                "external_handle",
                "output_ref",
                "revision",
            },
        )
        digest = document["effect_request_digest"]
        if not isinstance(digest, dict) or set(digest) != {"sha256"} or not isinstance(
            digest["sha256"], str
        ):
            raise CorruptStateError("invalid effect request digest")
        if len(digest["sha256"]) != 64 or any(
            character not in "0123456789abcdef"
            for character in digest["sha256"]
        ):
            raise CorruptStateError("invalid effect request digest")
        details = document["details"]
        if not isinstance(details, dict):
            raise CorruptStateError("effect details must be an object")
        try:
            stage = EffectStage(str(document["stage"]))
        except ValueError as exc:
            raise CorruptStateError("invalid effect stage") from exc
        effect_id = document["effect_id"]
        revision = document["revision"]
        external_handle = document["external_handle"]
        if (
            not isinstance(effect_id, str)
            or not isinstance(revision, int)
            or isinstance(revision, bool)
            or revision < 0
            or not (external_handle is None or isinstance(external_handle, str))
        ):
            raise CorruptStateError("invalid effect fields")
        record = EffectRecord(
            effect_id,
            revision,
            EffectRequestDigest(digest["sha256"]),
            stage,
            details,
            external_handle,
            _decode_ref(document["output_ref"]),
        )
        self._validate_shape(record)
        return record

    @staticmethod
    def _validate_shape(value: EffectRecord) -> None:
        digest = value.effect_request_digest.sha256
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise InvalidTransitionError("effect request digest must be SHA-256")
        if value.stage is EffectStage.PREPARED and (
            value.external_handle is not None or value.output_ref is not None
        ):
            raise InvalidTransitionError("PREPARED effect cannot contain execution output")
        if value.stage is EffectStage.MAY_HAVE_RUN and value.output_ref is not None:
            raise InvalidTransitionError("MAY_HAVE_RUN effect cannot contain saved output")
        if value.stage in {EffectStage.OUTPUT_SAVED, EffectStage.COMMITTED} and (
            value.output_ref is None
        ):
            raise InvalidTransitionError(f"{value.stage.value} effect requires output_ref")

    def validate_transition(
        self, previous: EffectRecord | None, next: EffectRecord
    ) -> None:
        validate_simple_id(next.effect_id, label="effect id")
        self._validate_shape(next)
        if previous is None:
            if next.revision != 0 or next.stage is not EffectStage.PREPARED:
                raise InvalidTransitionError("effect must start PREPARED at revision 0")
            return
        if next.effect_id != previous.effect_id:
            raise InvalidTransitionError("effect id cannot change")
        if next.effect_request_digest != previous.effect_request_digest:
            raise InvalidTransitionError("effect request digest cannot change")
        order = list(EffectStage)
        if order.index(next.stage) != order.index(previous.stage) + 1:
            raise InvalidTransitionError(
                f"invalid effect transition {previous.stage} -> {next.stage}"
            )


class EffectJournal:
    def __init__(
        self,
        directory: Path,
        *,
        artifacts: ImmutableArtifactStore,
        cancel: CancellationToken,
    ):
        self.directory = directory
        self.artifacts = artifacts
        self.cancel = cancel

    def _store(self, effect_id: str) -> AtomicStateStore[EffectRecord]:
        validate_simple_id(effect_id, label="effect id")
        return AtomicStateStore(self.directory / f"{effect_id}.json", _EffectContract())

    def read(self, effect_id: str) -> EffectRecord | None:
        return self._store(effect_id).read()

    def prepare(
        self,
        effect_id: str,
        *,
        effect_request_digest: EffectRequestDigest,
        details: Mapping[str, JsonValue] | None = None,
    ) -> EffectRecord:
        self.cancel.raise_if_requested()
        record = EffectRecord(
            effect_id,
            0,
            effect_request_digest,
            EffectStage.PREPARED,
            dict(details or {}),
        )
        try:
            return self._store(effect_id).create(record)
        except StateConflictError:
            current = self._store(effect_id).read()
            if current is not None and current.effect_request_digest == effect_request_digest:
                return current
            raise

    def _advance(self, effect_id: str, stage: EffectStage, **changes: object) -> EffectRecord:
        self.cancel.raise_if_requested()
        store = self._store(effect_id)
        current = store.read()
        if current is None:
            raise InvalidTransitionError("effect has not been prepared")
        if current.stage is stage:
            for name, expected in changes.items():
                if getattr(current, name) != expected:
                    raise StateConflictError(
                        f"idempotent effect transition changed {name}"
                    )
            return current
        next_value = replace(current, revision=current.revision + 1, stage=stage, **changes)
        return store.compare_and_swap(current.revision, next_value)

    def mark_may_have_run(
        self, effect_id: str, *, external_handle: str | None = None
    ) -> EffectRecord:
        return self._advance(
            effect_id, EffectStage.MAY_HAVE_RUN, external_handle=external_handle
        )

    def save_output(self, effect_id: str, output_ref: ArtifactRef) -> EffectRecord:
        self.artifacts.verify(output_ref)
        return self._advance(effect_id, EffectStage.OUTPUT_SAVED, output_ref=output_ref)

    def commit(self, effect_id: str) -> EffectRecord:
        return self._advance(effect_id, EffectStage.COMMITTED)

    def recover(
        self, effect_id: str, policy: EffectRecoveryPolicy | None = None
    ) -> RecoveryDecision:
        record = self.read(effect_id)
        if record is None:
            raise InvalidTransitionError("effect has not been prepared")
        if record.stage in {EffectStage.OUTPUT_SAVED, EffectStage.COMMITTED}:
            assert record.output_ref is not None
            self.artifacts.verify(record.output_ref)
            return RecoveryDecision.REPLAY_OUTPUT
        if record.stage is EffectStage.PREPARED:
            return RecoveryDecision.RETRY_VERIFIED_NOT_RUN
        if policy is None:
            raise UnsafeEffectRecoveryError(
                f"effect {effect_id!r} may have run and requires supervision",
                effect_id=effect_id,
            )
        decision = policy.classify(record)
        if not isinstance(decision, RecoveryDecision):
            raise TypeError("effect recovery policy returned an invalid decision")
        if decision in {
            RecoveryDecision.RETRY_VERIFIED_NOT_RUN,
            RecoveryDecision.RESUME_EXTERNALLY,
        }:
            return decision
        raise UnsafeEffectRecoveryError(
            f"effect {effect_id!r} recovery decision {decision.value!r} requires supervision",
            effect_id=effect_id,
        )
