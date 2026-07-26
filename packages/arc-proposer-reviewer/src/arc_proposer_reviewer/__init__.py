"""Typed proposer-reviewer orchestration over arc-jobs and arc-llm."""

from .handler import ProposerReviewerHandler
from .models import (
    BatchFailurePolicy,
    BatchRequest,
    BatchResult,
    ExecutionOptions,
    LoopResult,
    LoopSpec,
    LoopTermination,
    ProposerFailurePolicy,
    WorkerSpec,
)
from .service import ProposerReviewerService
from .runner import BatchInputPayload, BatchRunner
from .projection import (
    BatchInspection,
    BatchProjection,
    BatchProjectionIntegrityError,
    BatchTrace,
    CommittedRound,
    CommittedRoundNotFoundError,
    inspect_batch,
    read_batch_round,
    read_batch_trace,
)

__version__ = "1.0.1"

__all__ = [
    "BatchFailurePolicy",
    "BatchInputPayload",
    "BatchInspection",
    "BatchProjection",
    "BatchProjectionIntegrityError",
    "BatchRequest",
    "BatchResult",
    "BatchRunner",
    "BatchTrace",
    "CommittedRound",
    "CommittedRoundNotFoundError",
    "ExecutionOptions",
    "LoopResult",
    "LoopSpec",
    "LoopTermination",
    "ProposerFailurePolicy",
    "ProposerReviewerHandler",
    "ProposerReviewerService",
    "inspect_batch",
    "read_batch_round",
    "read_batch_trace",
    "WorkerSpec",
]
