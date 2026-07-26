"""Reusable durable facade for proposer-reviewer batches."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from arc_jobs import (
    ArtifactDigest,
    ArtifactSourceRef,
    EventSink,
    ImmutableArtifactStore,
    JsonValue,
    RunEngine,
    RunRepository,
    RunSnapshot,
    RunSpec,
    RunView,
)
from arc_llm import LLMInputArtifact, LLMTaskService

from .handler import ProposerReviewerHandler
from .identity import derive_batch_run_id
from .models import BatchRequest, ExecutionOptions
from .projection import BatchProjection
from .protocol import decode_batch_request, encode_batch_request
from .service import ProposerReviewerService
from .validation import validate_execution_options


_RunRoot = RunRepository | str | Path


@dataclass(frozen=True)
class BatchInputPayload:
    """One whole-file input to materialize inside a batch run before execution."""

    input_id: str
    media_type: str
    content: bytes


class BatchRunner:
    """Prepare, execute, resume, and inspect one durable batch lineage."""

    def __init__(self, task_service: LLMTaskService | None = None) -> None:
        self._task_service = task_service

    def prepare(
        self,
        request: BatchRequest,
        run_root: _RunRoot,
        run_id: str | None = None,
        *,
        input_payloads: Sequence[BatchInputPayload] = (),
    ) -> RunSnapshot:
        repository = _repository(run_root)
        resolved_run_id = run_id or derive_batch_run_id(request.batch_id)
        materialized = _materialize_inputs(
            request,
            run_id=resolved_run_id,
            payloads=input_payloads,
        )
        _publish_inputs(repository, resolved_run_id, materialized, input_payloads)
        return repository.create(_run_spec(materialized, resolved_run_id))

    def run(
        self,
        request: BatchRequest,
        run_root: _RunRoot,
        run_id: str | None = None,
        options: ExecutionOptions = ExecutionOptions(),
        *,
        input_payloads: Sequence[BatchInputPayload] = (),
        event_sink: EventSink | None = None,
    ) -> RunSnapshot:
        validate_execution_options(options)
        repository = _repository(run_root)
        snapshot = self.prepare(
            request,
            repository,
            run_id,
            input_payloads=input_payloads,
        )
        spec = repository.read_spec(snapshot.run_id)
        return RunEngine(repository).execute(
            spec,
            self._handler(options),
            event_sink=event_sink,
        )

    def resume(
        self,
        run_root: _RunRoot,
        run_id: str,
        input: Mapping[str, JsonValue] | None = None,
        options: ExecutionOptions = ExecutionOptions(),
        *,
        event_sink: EventSink | None = None,
    ) -> RunSnapshot:
        validate_execution_options(options)
        repository = _repository(run_root)
        _read_request(repository, run_id)
        return RunEngine(repository).resume(
            run_id,
            self._handler(options),
            input=input,
            event_sink=event_sink,
        )

    def stop(
        self,
        run_root: _RunRoot,
        run_id: str,
        reason: str | None = None,
    ) -> RunView:
        repository = _repository(run_root)
        repository.inspect(run_id)
        _read_request(repository, run_id)
        return repository.request_stop(run_id, reason=reason)

    def inspect(self, run_root: _RunRoot, run_id: str) -> RunView:
        repository = _repository(run_root)
        view = repository.inspect(run_id)
        _read_request(repository, run_id)
        return view

    def projection(self, run_root: _RunRoot, run_id: str) -> BatchProjection:
        return BatchProjection(_repository(run_root), run_id)

    def read_request(self, run_root: _RunRoot, run_id: str) -> BatchRequest:
        return _read_request(_repository(run_root), run_id)

    def _handler(self, options: ExecutionOptions) -> ProposerReviewerHandler:
        task_service = (
            self._task_service
            if self._task_service is not None
            else LLMTaskService()
        )
        return ProposerReviewerHandler(
            ProposerReviewerService(task_service),
            options=options,
        )


def _repository(run_root: _RunRoot) -> RunRepository:
    if isinstance(run_root, RunRepository):
        return run_root
    return RunRepository(run_root)


def _run_spec(request: BatchRequest, run_id: str | None) -> RunSpec:
    resolved_run_id = run_id or derive_batch_run_id(request.batch_id)
    return RunSpec(
        resolved_run_id,
        ProposerReviewerHandler.name,
        encode_batch_request(request),
    )


def _materialize_inputs(
    request: BatchRequest,
    *,
    run_id: str,
    payloads: Sequence[BatchInputPayload],
) -> BatchRequest:
    if not payloads:
        return request
    if request.inputs:
        raise ValueError("input payloads cannot be combined with persisted batch inputs")
    inputs: list[LLMInputArtifact] = []
    for index, payload in enumerate(payloads):
        if not isinstance(payload, BatchInputPayload):
            raise TypeError("input_payloads must contain BatchInputPayload values")
        if not isinstance(payload.content, bytes):
            raise TypeError("batch input content must be bytes")
        digest = ArtifactDigest(
            "sha256",
            hashlib.sha256(payload.content).hexdigest(),
            len(payload.content),
        )
        inputs.append(
            LLMInputArtifact(
                payload.input_id,
                ArtifactSourceRef(
                    run_id,
                    _input_artifact_id(index, payload.input_id),
                    digest,
                ),
                payload.media_type,
            )
        )
    if len({item.input_id for item in inputs}) != len(inputs):
        raise ValueError("batch input IDs must be unique")
    return replace(request, inputs=tuple(inputs))


def _publish_inputs(
    repository: RunRepository,
    run_id: str,
    request: BatchRequest,
    payloads: Sequence[BatchInputPayload],
) -> None:
    store = ImmutableArtifactStore(
        repository.root / "runs" / run_id,
        repository_root=repository.root,
    )
    if payloads:
        if len(payloads) != len(request.inputs):
            raise ValueError("batch input payload count differs from persisted inputs")
        for index, (payload, item) in enumerate(
            zip(payloads, request.inputs, strict=True)
        ):
            ref = store.publish_bytes(
                _input_artifact_id(index, payload.input_id),
                payload.content,
                media_type=payload.media_type,
            )
            if (
                ref.digest != item.source.expected_digest
                or ref.media_type != item.media_type
                or ref.artifact_id != item.source.source_artifact_id
                or item.source.source_run_id != run_id
            ):
                raise ValueError(
                    "materialized batch input differs from its persisted reference"
                )
    _verify_inputs(store, request.inputs)


def _verify_inputs(
    store: ImmutableArtifactStore,
    inputs: tuple[LLMInputArtifact, ...],
) -> None:
    for item in inputs:
        verified = store.read_source(item.source)
        if (
            verified.digest != item.source.expected_digest
            or verified.media_type != item.media_type
        ):
            raise ValueError("verified batch input differs from its durable reference")


def _input_artifact_id(index: int, input_id: str) -> str:
    return f"proposer-reviewer/inputs/source/{index:04d}-{input_id}"


def _read_request(repository: RunRepository, run_id: str) -> BatchRequest:
    spec = repository.read_spec(run_id)
    if spec.handler != ProposerReviewerHandler.name:
        raise ValueError("run is not a proposer-reviewer batch")
    return decode_batch_request(spec.semantic_input)


__all__ = ["BatchInputPayload", "BatchRunner"]
