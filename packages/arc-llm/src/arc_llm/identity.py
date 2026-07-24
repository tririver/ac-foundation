"""Typed semantic and execution identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from arc_jobs import ExecutionFingerprint, SemanticKeyDigest

from .request import (
    LLMRequest,
    ModelSelection,
    ResumeInput,
    encode_output_contract,
)

SEMANTIC_KEY_SCHEMA = "arc.llm.semantic_key.v2"
EXECUTION_RECIPE_SCHEMA = "arc.llm.execution_recipe.v2"
_RESUME_KEY_DIGEST_LENGTH = 24


@dataclass(frozen=True)
class AdoptionAuthorization:
    source_semantic_key: SemanticKeyDigest
    target_semantic_key: SemanticKeyDigest
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Adoption authorization requires a reason.")
        for name, digest in (
            ("source_semantic_key", self.source_semantic_key),
            ("target_semantic_key", self.target_semantic_key),
        ):
            if not isinstance(digest.sha256, str) or len(digest.sha256) != 64:
                raise ValueError(f"{name} must contain a SHA-256 digest.")
            try:
                int(digest.sha256, 16)
            except ValueError as exc:
                raise ValueError(f"{name} must contain a SHA-256 digest.") from exc


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def document_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def semantic_document(request: LLMRequest) -> dict[str, Any]:
    model = _semantic_model(request.model)
    return {
        "schema_version": SEMANTIC_KEY_SCHEMA,
        "task_id": request.task_id,
        "request": {"prompt": request.prompt},
        "output_contract": encode_output_contract(request.output),
        "model_requirement": model,
        "capability_requirements": {
            "internet": request.capabilities.internet,
            "inherit_host_config": request.capabilities.inherit_host_config,
            "allowed_tools": list(request.capabilities.allowed_tools),
        },
        "session_accepted_prefix_sha256": (
            None if request.session is None else request.session.accepted_prefix_sha256
        ),
        "inputs": input_identity_document(request),
    }


def semantic_key(request: LLMRequest) -> SemanticKeyDigest:
    return SemanticKeyDigest(document_sha256(semantic_document(request)))


def resume_input_matches(request: LLMRequest, resume_input: ResumeInput) -> bool:
    """Return whether a resume input belongs to this request's task namespace.

    Resume keys are opaque outside ``arc-llm``. Higher-level workflows can use
    this predicate to route one parent-run resume input among several child LLM
    tasks without depending on the private key format.
    """

    return resume_input.resume_key.startswith(_resume_key_prefix(semantic_key(request)))


def _make_resume_key(semantic: SemanticKeyDigest, revision: int) -> str:
    """Construct an opaque resume key for one durable task revision."""

    if revision < 1:
        raise ValueError("Resume-key revision must be positive.")
    return f"{_resume_key_prefix(semantic)}{revision}"


def _resume_key_prefix(semantic: SemanticKeyDigest) -> str:
    return f"resume-{semantic.sha256[:_RESUME_KEY_DIGEST_LENGTH]}-"


def input_identity_document(request: LLMRequest) -> list[dict[str, Any]]:
    """Return ordered content identity without physical source locators."""

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
        for item in request.inputs
    ]


def execution_document(
    *,
    provider: str,
    model: str,
    capabilities: Mapping[str, Any],
    adapter_compatibility_version: str,
    session_compatibility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": EXECUTION_RECIPE_SCHEMA,
        "provider": provider,
        "model": model,
        "capabilities": dict(capabilities),
        "adapter_compatibility_version": adapter_compatibility_version,
        "session_compatibility": dict(session_compatibility or {}),
    }


def execution_fingerprint(document: Mapping[str, Any]) -> ExecutionFingerprint:
    if document.get("schema_version") != EXECUTION_RECIPE_SCHEMA:
        raise ValueError("Execution document uses an unsupported schema.")
    return ExecutionFingerprint(EXECUTION_RECIPE_SCHEMA, document_sha256(document))


def derive_run_id(handler: str, task_id: str) -> str:
    digest = hashlib.sha256(f"{handler}\0{task_id}".encode()).hexdigest()[:24]
    return f"llm-{digest}"


def _semantic_model(model: ModelSelection) -> dict[str, Any]:
    if model.provider == "auto":
        return {"provider": "auto", "model": None, "tier": model.tier}
    return {"provider": model.provider, "model": model.model, "tier": model.tier}
