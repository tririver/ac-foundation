from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Mapping

from arc_jobs import JsonValue, RunError
from arc_llm import LLMExecutionOptions, LLMInputArtifact, ModelSelection


BATCH_SCHEMA_VERSION = "arc.proposer_reviewer.batch.v4"
REVIEW_SCHEMA_VERSION = "arc.proposer_reviewer.review.v1"
RESULT_SCHEMA_VERSION = "arc.proposer_reviewer.result.v1"


class ProposerFailurePolicy(StrEnum):
    FAIL_LOOP = "fail_loop"
    CONTINUE_IF_ANY = "continue_if_any"


class BatchFailurePolicy(StrEnum):
    COLLECT = "collect"
    FAIL_FAST = "fail_fast"


class LoopTermination(StrEnum):
    REVIEWER_STOP = "reviewer_stop"
    ROUND_LIMIT = "round_limit"
    FAILED = "failed"


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    instructions: str
    output_schema: Mapping[str, JsonValue]
    model: ModelSelection = field(default_factory=ModelSelection)


@dataclass(frozen=True)
class LoopSpec:
    loop_id: str
    context: Mapping[str, JsonValue]
    proposers: tuple[WorkerSpec, ...]
    reviewer: WorkerSpec
    max_rounds: int
    allow_early_stop: bool = True
    on_proposer_failure: ProposerFailurePolicy = ProposerFailurePolicy.FAIL_LOOP


@dataclass(frozen=True)
class BatchRequest:
    schema_version: Literal["arc.proposer_reviewer.batch.v4"]
    batch_id: str
    loops: tuple[LoopSpec, ...]
    failure_policy: BatchFailurePolicy = BatchFailurePolicy.COLLECT
    inputs: tuple[LLMInputArtifact, ...] = ()


@dataclass(frozen=True)
class ExecutionOptions:
    max_concurrent_loops: int = 1
    max_concurrent_workers: int = 1
    llm: LLMExecutionOptions = field(default_factory=LLMExecutionOptions)
    progress_callback: Callable[[Mapping[str, JsonValue]], None] | None = field(
        default=None,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True)
class Review:
    schema_version: Literal["arc.proposer_reviewer.review.v1"]
    action: Literal["continue", "stop"]
    reason: str
    feedback: Mapping[str, str]
    payload: JsonValue


@dataclass(frozen=True)
class LoopResult:
    loop_id: str
    termination: LoopTermination
    rounds_completed: int
    final_proposals: Mapping[str, JsonValue]
    final_review: JsonValue | None
    error: RunError | None


@dataclass(frozen=True)
class BatchResult:
    schema_version: Literal["arc.proposer_reviewer.result.v1"]
    loops: tuple[LoopResult, ...]
