from __future__ import annotations

import json
from typing import Mapping

from arc_jobs import JsonValue

from .models import LoopSpec, WorkerSpec


PROMPT_CONTRACT = "arc.proposer_reviewer.prompt.v2"

_PROPOSER_PROTOCOL = """You are one proposer in a typed proposer-reviewer round.
Return one complete JSON value that satisfies the output contract. Do not return
diagnostics, provider metadata, a patch, or a wrapper around the requested value."""

_REVIEWER_PROTOCOL = """You are the sole reviewer in a typed proposer-reviewer round.
Independently review all current proposals. Return only the closed review envelope:
schema_version, action, reason, feedback, and payload. Feedback must contain exactly
one non-empty entry for each active proposer.

Choose continue only when another round has a concrete expected scientific
contribution and the feedback gives each proposer an actionable way to obtain
it. Choose stop when the objective is satisfied or no concrete improvement path
remains. Do not continue merely because additional rounds are available."""

_WORKSPACE_INPUT_PROTOCOL = (
    "When host/control.json declares workspace inputs, read the complete verified "
    "files at those relative paths before answering."
)

_PROPOSER_PROTOCOL = f"{_PROPOSER_PROTOCOL}\n{_WORKSPACE_INPUT_PROTOCOL}"
_REVIEWER_PROTOCOL = f"{_REVIEWER_PROTOCOL}\n{_WORKSPACE_INPUT_PROTOCOL}"


def render_initial_proposer_prompt(
    *, loop: LoopSpec, worker: WorkerSpec, round_number: int
) -> str:
    return _sections(
        _PROPOSER_PROTOCOL,
        worker.instructions,
        worker.output_schema,
        loop.context,
        {
            "kind": "initial_proposal",
            "loop_id": loop.loop_id,
            "round": round_number,
            "instruction": "Produce a complete independent proposal.",
        },
    )


def render_delta_proposer_prompt(
    *,
    loop: LoopSpec,
    worker: WorkerSpec,
    round_number: int,
    previous_proposal: JsonValue,
    targeted_feedback: str,
    transcript_refs: tuple[Mapping[str, JsonValue], ...] = (),
) -> str:
    return _sections(
        _PROPOSER_PROTOCOL,
        worker.instructions,
        worker.output_schema,
        loop.context,
        {
            "kind": "revised_proposal",
            "loop_id": loop.loop_id,
            "round": round_number,
            "previous_proposal": previous_proposal,
            "targeted_feedback": targeted_feedback,
            "transcript_refs": list(transcript_refs),
            "instruction": (
                "Recompute and return a complete standalone proposal. "
                "Do not return a patch or only the changed fields."
            ),
        },
    )


def render_reviewer_prompt(
    *,
    loop: LoopSpec,
    round_number: int,
    proposals: Mapping[str, JsonValue],
    previous_review: JsonValue | None,
    failed_proposer_ids: tuple[str, ...] = (),
    transcript_refs: tuple[Mapping[str, JsonValue], ...] = (),
) -> str:
    active_ids = tuple(proposals)
    envelope_schema = reviewer_envelope_schema(
        payload_schema=loop.reviewer.output_schema,
        active_proposer_ids=active_ids,
    )
    return _sections(
        _REVIEWER_PROTOCOL,
        loop.reviewer.instructions,
        envelope_schema,
        loop.context,
        {
            "kind": "independent_review",
            "loop_id": loop.loop_id,
            "round": round_number,
            "active_proposer_ids": list(active_ids),
            "current_proposals": dict(proposals),
            "previous_review": previous_review,
            "failed_proposer_ids": list(failed_proposer_ids),
            "transcript_refs": list(transcript_refs),
            "instruction": (
                "Perform a complete review of the current proposals. "
                "Do not patch or merely endorse the previous review."
            ),
        },
    )


def reviewer_envelope_schema(
    *, payload_schema: Mapping[str, JsonValue], active_proposer_ids: tuple[str, ...]
) -> dict[str, JsonValue]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "action", "reason", "feedback", "payload"],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": "arc.proposer_reviewer.review.v1",
            },
            "action": {"type": "string", "enum": ["continue", "stop"]},
            "reason": {"type": "string", "minLength": 1},
            "feedback": {
                "type": "object",
                "additionalProperties": False,
                "required": list(active_proposer_ids),
                "properties": {
                    worker_id: {"type": "string", "minLength": 1}
                    for worker_id in active_proposer_ids
                },
            },
            "payload": dict(payload_schema),
        },
    }


def _sections(
    protocol: str,
    instructions: str,
    output_contract: Mapping[str, JsonValue],
    caller_context: Mapping[str, JsonValue],
    round_task: Mapping[str, JsonValue],
) -> str:
    parts = (
        ("Package protocol", protocol),
        ("Worker instructions", instructions),
        ("Output contract", _canonical_json(output_contract)),
        ("Caller context", _canonical_json(caller_context)),
        ("Round task", _canonical_json(round_task)),
    )
    return "\n\n".join(f"## {heading}\n{body}" for heading, body in parts) + "\n"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
