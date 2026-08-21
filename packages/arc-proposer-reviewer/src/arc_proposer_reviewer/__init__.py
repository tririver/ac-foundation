"""Typed proposer-reviewer orchestration over arc-jobs and arc-llm."""

from .handler import ProposerReviewerHandler
from .models import (
    BATCH_SCHEMA_VERSION,
    BatchFailurePolicy,
    BatchRequest,
    BatchResult,
    ExecutionOptions,
    LoopResult,
    LoopSpec,
    LoopTermination,
    ProposerFailurePolicy,
    RevisionContextMode,
    WorkerSpec,
)
from .protocol import decode_batch_result
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

__version__ = "1.1.1"

__all__ = [
    "BatchFailurePolicy",
    "BATCH_SCHEMA_VERSION",
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
    "RevisionContextMode",
    "ProposerReviewerHandler",
    "ProposerReviewerService",
    "inspect_batch",
    "decode_batch_result",
    "read_batch_round",
    "read_batch_trace",
    "WorkerSpec",
]
