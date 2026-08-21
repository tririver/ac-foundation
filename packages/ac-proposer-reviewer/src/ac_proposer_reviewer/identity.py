from __future__ import annotations

from typing import Literal, Mapping

from ac_jobs import JsonValue, SemanticKeyDigest, semantic_key, validate_simple_id
from ac_llm import LLMInputArtifact

from .models import LoopSpec, RevisionContextMode, WorkerSpec
from .prompts import PROMPT_CONTRACT


WORKER_SEMANTIC_KEY_SCHEMA = "ac.proposer_reviewer.worker_semantic_key.v7"
LOOP_SEMANTIC_KEY_SCHEMA = "ac.proposer_reviewer.loop_semantic_key.v7"
EXECUTION_SCOPE_SCHEMA = "ac.proposer_reviewer.execution_scope.v1"


def normalize_execution_scope(execution_scope: str | None) -> str | None:
    """Validate an optional operational scope without changing semantic inputs."""

    if execution_scope is None:
        return None
    return validate_simple_id(execution_scope, label="execution scope")


def execution_scope_token(execution_scope: str | None) -> str | None:
    """Return a bounded namespace token for a non-default execution scope."""

    normalized = normalize_execution_scope(execution_scope)
    if normalized is None:
        return None
    return semantic_key(
        {
            "semantic_key_schema": EXECUTION_SCOPE_SCHEMA,
            "execution_scope": normalized,
        }
    ).sha256[:32]


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
    projection: dict[str, JsonValue] = {
        "semantic_key_schema": LOOP_SEMANTIC_KEY_SCHEMA,
        "prompt_contract": PROMPT_CONTRACT,
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
    # Omitting this default preserves stable task identities across releases.
    if not loop.review_final_round:
        projection["review_final_round"] = False
    # This default does not alter a revision prompt; omit it from identity.
    if loop.revision_context_mode is not RevisionContextMode.FEEDBACK_ONLY:
        projection["revision_context_mode"] = loop.revision_context_mode.value
    return projection


def worker_semantic_projection(
    *,
    role: Literal["proposer", "reviewer"],
    loop: LoopSpec,
    round_number: int,
    worker: WorkerSpec,
    upstream_digests: Mapping[str, str],
    inputs: tuple[LLMInputArtifact, ...] = (),
) -> dict[str, JsonValue]:
    projection: dict[str, JsonValue] = {
        "semantic_key_schema": WORKER_SEMANTIC_KEY_SCHEMA,
        "prompt_contract": PROMPT_CONTRACT,
        "role": role,
        "loop_id": loop.loop_id,
        "round": round_number,
        "worker_id": worker.worker_id,
        "worker_contract": worker_contract_document(worker),
        "loop_context": dict(loop.context),
        "inputs": input_artifact_documents(inputs),
        "upstream_content_digests": dict(sorted(upstream_digests.items())),
    }
    # This changes only a delta proposer prompt. Keep initial proposer and
    # reviewer task identities unchanged, including for durable resumed runs.
    if (
        role == "proposer"
        and "previous_proposal" in upstream_digests
        and loop.revision_context_mode is not RevisionContextMode.FEEDBACK_ONLY
    ):
        projection["revision_context_mode"] = loop.revision_context_mode.value
    return projection


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
    execution_scope: str | None = None,
) -> str:
    digest = worker_semantic_key(
        role=role,
        loop=loop,
        round_number=round_number,
        worker=worker,
        upstream_digests=upstream_digests,
        inputs=inputs,
    )
    scope_token = execution_scope_token(execution_scope)
    if scope_token is None:
        return f"pr-{role}-{digest.sha256}"
    return f"pr-{role}-{scope_token}-{digest.sha256}"


def derive_batch_run_id(batch_id: str) -> str:
    digest = semantic_key(
        {
            "semantic_key_schema": "ac.proposer_reviewer.run_id.v1",
            "handler": "ac.proposer_reviewer.batch.v4",
            "batch_id": batch_id,
        }
    )
    return f"pr-{digest.sha256[:24]}"
