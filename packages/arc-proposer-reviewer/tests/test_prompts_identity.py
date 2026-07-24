from __future__ import annotations

from dataclasses import replace

from arc_proposer_reviewer.identity import worker_semantic_key
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


def test_delta_prompt_requires_complete_recomputation_and_targeted_feedback() -> None:
    value = loop()
    prompt = render_delta_proposer_prompt(
        loop=value,
        worker=value.proposers[0],
        round_number=2,
        previous_proposal={"answer": "old"},
        targeted_feedback="Address this exact issue.",
    )
    assert "complete standalone proposal" in prompt
    assert "Address this exact issue." in prompt
    assert '"answer":"old"' in prompt


def test_reviewer_prompt_contains_business_values_not_diagnostics() -> None:
    value = loop()
    prompt = render_reviewer_prompt(
        loop=value,
        round_number=1,
        proposals={"p": {"answer": "candidate"}},
        previous_review=None,
    )
    assert '"answer":"candidate"' in prompt
    assert "provider" not in prompt.lower()
    assert "usage" not in prompt.lower()
    assert "call_record" not in prompt
    assert "complete review" in prompt


def test_worker_identity_ignores_outer_run_path_concurrency_and_other_loops() -> None:
    value = loop()
    worker = value.proposers[0]
    base = worker_semantic_key(
        role="proposer",
        loop=value,
        round_number=1,
        worker=worker,
        upstream_digests={},
    )
    # No API parameter exists for outer run ID, path, batch concurrency, or other loops.
    assert base == worker_semantic_key(
        role="proposer",
        loop=value,
        round_number=1,
        worker=worker,
        upstream_digests={},
    )
    assert base != worker_semantic_key(
        role="proposer",
        loop=replace(value, context={"question": "changed"}),
        round_number=1,
        worker=worker,
        upstream_digests={},
    )
    assert base != worker_semantic_key(
        role="proposer",
        loop=value,
        round_number=1,
        worker=replace(worker, instructions="Changed."),
        upstream_digests={},
    )
    assert base != worker_semantic_key(
        role="proposer",
        loop=value,
        round_number=1,
        worker=replace(
            worker,
            output_schema={
                "type": "object",
                "required": ["changed"],
                "properties": {"changed": {"type": "boolean"}},
            },
        ),
        upstream_digests={},
    )
    assert base != worker_semantic_key(
        role="proposer",
        loop=value,
        round_number=1,
        worker=worker,
        upstream_digests={"prior": "abc"},
    )
