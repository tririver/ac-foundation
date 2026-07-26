from __future__ import annotations

from dataclasses import replace

from arc_proposer_reviewer.identity import (
    LOOP_SEMANTIC_KEY_SCHEMA,
    WORKER_SEMANTIC_KEY_SCHEMA,
    worker_contract_document,
    worker_semantic_key,
)
from arc_jobs import ArtifactDigest, ArtifactSourceRef
from arc_llm import LLMInputArtifact
from arc_proposer_reviewer.models import LoopSpec, WorkerSpec
from arc_proposer_reviewer.prompts import (
    render_delta_proposer_prompt,
    render_initial_proposer_prompt,
    render_reviewer_prompt,
)


SCHEMA = {"type": "object", "additionalProperties": True}


def loop(loop_id: str = "loop-a") -> LoopSpec:
    proposer = WorkerSpec("p", "Propose.", SCHEMA)
    return LoopSpec(
        loop_id=loop_id,
        context={"question": "Q"},
        proposers=(proposer,),
        reviewer=WorkerSpec("r", "Review.", SCHEMA),
        max_rounds=2,
    )


def test_initial_prompt_has_fixed_section_order() -> None:
    value = loop()
    prompt = render_initial_proposer_prompt(
        loop=value, worker=value.proposers[0], round_number=1
    )
    headings = [
        "## Package protocol",
        "## Worker instructions",
        "## Output contract",
        "## Caller context",
        "## Round task",
    ]
    assert [prompt.index(heading) for heading in headings] == sorted(
        prompt.index(heading) for heading in headings
    )
    changed = replace(value, context={"question": "different variable context"})
    changed_prompt = render_initial_proposer_prompt(
        loop=changed, worker=changed.proposers[0], round_number=1
    )
    assert prompt.split("## Caller context", 1)[0] == changed_prompt.split(
        "## Caller context", 1
    )[0]


def test_delta_and_reviewer_prompts_keep_only_business_context() -> None:
    value = loop()
    delta = render_delta_proposer_prompt(
        loop=value,
        worker=value.proposers[0],
        round_number=2,
        previous_proposal={"answer": "old"},
        targeted_feedback="Address this exact issue.",
    )
    assert "complete standalone proposal" in delta
    assert "Address this exact issue." in delta
    reviewer = render_reviewer_prompt(
        loop=value,
        round_number=1,
        proposals={"p": {"answer": "candidate"}},
        previous_review={"reason": "An earlier review."},
    )
    assert '"answer":"candidate"' in reviewer
    assert "provider" not in reviewer.lower()
    assert "Independently review all current proposals" in reviewer


def test_worker_identity_covers_only_current_worker_contract() -> None:
    assert WORKER_SEMANTIC_KEY_SCHEMA.endswith("v5")
    assert LOOP_SEMANTIC_KEY_SCHEMA.endswith("v5")
    value = loop()
    worker = value.proposers[0]
    base = worker_semantic_key(
        role="proposer", loop=value, round_number=1, worker=worker, upstream_digests={}
    )
    assert base != worker_semantic_key(
        role="proposer",
        loop=replace(value, context={"question": "changed"}),
        round_number=1,
        worker=worker,
        upstream_digests={},
    )
    workspace_input = LLMInputArtifact(
        "domain-markdown-001",
        ArtifactSourceRef("run-a", "proposer-reviewer/inputs/source/0000-domain-markdown-001", ArtifactDigest("sha256", "a" * 64, 8)),
        "text/markdown",
    )
    assert base != worker_semantic_key(
        role="proposer",
        loop=value,
        round_number=1,
        worker=worker,
        upstream_digests={},
        inputs=(workspace_input,),
    )
    assert base != worker_semantic_key(
        role="proposer",
        loop=value,
        round_number=1,
        worker=replace(worker, instructions="Changed."),
        upstream_digests={},
    )
    assert set(worker_contract_document(worker)) == {
        "worker_id", "instructions", "output_schema", "model"
    }
