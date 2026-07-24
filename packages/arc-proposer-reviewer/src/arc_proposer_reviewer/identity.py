from __future__ import annotations

from typing import Literal, Mapping

from arc_jobs import JsonValue, SemanticKeyDigest, semantic_key

from .models import LoopSpec, WorkerSpec


WORKER_SEMANTIC_KEY_SCHEMA = "arc.proposer_reviewer.worker_semantic_key.v1"
LOOP_SEMANTIC_KEY_SCHEMA = "arc.proposer_reviewer.loop_semantic_key.v1"


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
        "capabilities": {
            "internet": worker.capabilities.internet,
            "inherit_host_config": worker.capabilities.inherit_host_config,
            "allowed_tools": list(worker.capabilities.allowed_tools),
        },
    }


def loop_semantic_projection(loop: LoopSpec) -> dict[str, JsonValue]:
    return {
        "semantic_key_schema": LOOP_SEMANTIC_KEY_SCHEMA,
        "loop_id": loop.loop_id,
        "context": dict(loop.context),
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
) -> dict[str, JsonValue]:
    return {
        "semantic_key_schema": WORKER_SEMANTIC_KEY_SCHEMA,
        "role": role,
        "loop_id": loop.loop_id,
        "round": round_number,
        "worker_id": worker.worker_id,
        "worker_contract": worker_contract_document(worker),
        "loop_context": dict(loop.context),
        "upstream_content_digests": dict(sorted(upstream_digests.items())),
    }


def worker_semantic_key(
    *,
    role: Literal["proposer", "reviewer"],
    loop: LoopSpec,
    round_number: int,
    worker: WorkerSpec,
    upstream_digests: Mapping[str, str],
) -> SemanticKeyDigest:
    return semantic_key(
        worker_semantic_projection(
            role=role,
            loop=loop,
            round_number=round_number,
            worker=worker,
            upstream_digests=upstream_digests,
        )
    )


def worker_task_id(
    *,
    role: Literal["proposer", "reviewer"],
    loop: LoopSpec,
    round_number: int,
    worker: WorkerSpec,
    upstream_digests: Mapping[str, str],
) -> str:
    digest = worker_semantic_key(
        role=role,
        loop=loop,
        round_number=round_number,
        worker=worker,
        upstream_digests=upstream_digests,
    )
    return f"pr-{role}-{digest.sha256}"


def derive_batch_run_id(batch_id: str) -> str:
    digest = semantic_key(
        {
            "semantic_key_schema": "arc.proposer_reviewer.run_id.v1",
            "handler": "arc.proposer_reviewer.batch.v1",
            "batch_id": batch_id,
        }
    )
    return f"pr-{digest.sha256[:24]}"
