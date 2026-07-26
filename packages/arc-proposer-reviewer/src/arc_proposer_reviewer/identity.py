from __future__ import annotations

from typing import Literal, Mapping

from arc_jobs import JsonValue, SemanticKeyDigest, semantic_key
from arc_llm import LLMInputArtifact

from .models import LoopSpec, WorkerSpec


WORKER_SEMANTIC_KEY_SCHEMA = "arc.proposer_reviewer.worker_semantic_key.v6"
LOOP_SEMANTIC_KEY_SCHEMA = "arc.proposer_reviewer.loop_semantic_key.v6"


def worker_contract_document(worker: WorkerSpec) -> dict[str, JsonValue]:
    return {
        "worker_id": worker.worker_id,
        "instructions": worker.instructions,
        "output_schema": dict(worker.output_schema),
        "model": {
            "provider": worker.model.provider,
            "model": worker.model.model,
            "tier": worker.model.tier,
        },
    }


def input_artifact_documents(
    inputs: tuple[LLMInputArtifact, ...],
) -> list[dict[str, JsonValue]]:
    return [
        {
            "input_id": item.input_id,
            "media_type": item.media_type,
            "digest": {
                "algorithm": item.source.expected_digest.algorithm,
                "value": item.source.expected_digest.value,
                "size_bytes": item.source.expected_digest.size_bytes,
            },
        }
        for item in inputs
    ]


def loop_semantic_projection(
    loop: LoopSpec,
    *,
    inputs: tuple[LLMInputArtifact, ...] = (),
) -> dict[str, JsonValue]:
    return {
        "semantic_key_schema": LOOP_SEMANTIC_KEY_SCHEMA,
        "loop_id": loop.loop_id,
        "context": dict(loop.context),
        "inputs": input_artifact_documents(inputs),
        "proposers": [
            worker_contract_document(worker) for worker in loop.proposers
        ],
        "reviewer": worker_contract_document(loop.reviewer),
        "max_rounds": loop.max_rounds,
        "allow_early_stop": loop.allow_early_stop,
        "on_proposer_failure": loop.on_proposer_failure.value,
    }


def worker_semantic_projection(
    *,
    role: Literal["proposer", "reviewer"],
    loop: LoopSpec,
    round_number: int,
    worker: WorkerSpec,
    upstream_digests: Mapping[str, str],
    inputs: tuple[LLMInputArtifact, ...] = (),
) -> dict[str, JsonValue]:
    return {
        "semantic_key_schema": WORKER_SEMANTIC_KEY_SCHEMA,
        "role": role,
        "loop_id": loop.loop_id,
        "round": round_number,
        "worker_id": worker.worker_id,
        "worker_contract": worker_contract_document(worker),
        "loop_context": dict(loop.context),
        "inputs": input_artifact_documents(inputs),
        "upstream_content_digests": dict(sorted(upstream_digests.items())),
    }


def worker_semantic_key(
    *,
    role: Literal["proposer", "reviewer"],
    loop: LoopSpec,
    round_number: int,
    worker: WorkerSpec,
    upstream_digests: Mapping[str, str],
    inputs: tuple[LLMInputArtifact, ...] = (),
) -> SemanticKeyDigest:
    return semantic_key(
        worker_semantic_projection(
            role=role,
            loop=loop,
            round_number=round_number,
            worker=worker,
            upstream_digests=upstream_digests,
            inputs=inputs,
        )
    )


def worker_task_id(
    *,
    role: Literal["proposer", "reviewer"],
    loop: LoopSpec,
    round_number: int,
    worker: WorkerSpec,
    upstream_digests: Mapping[str, str],
    inputs: tuple[LLMInputArtifact, ...] = (),
) -> str:
    digest = worker_semantic_key(
        role=role,
        loop=loop,
        round_number=round_number,
        worker=worker,
        upstream_digests=upstream_digests,
        inputs=inputs,
    )
    return f"pr-{role}-{digest.sha256}"


def derive_batch_run_id(batch_id: str) -> str:
    digest = semantic_key(
        {
            "semantic_key_schema": "arc.proposer_reviewer.run_id.v1",
            "handler": "arc.proposer_reviewer.batch.v4",
            "batch_id": batch_id,
        }
    )
    return f"pr-{digest.sha256[:24]}"
