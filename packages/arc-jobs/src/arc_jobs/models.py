from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Mapping, TypeAlias

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ResumeReason(StrEnum):
    INTERACTION_REQUIRED = "interaction_required"
    SUPERVISION_REQUIRED = "supervision_required"
    EXECUTION_BUDGET_EXHAUSTED = "execution_budget_exhausted"
    EXTERNAL_CONDITION = "external_condition"
    EXECUTION_INTERRUPTED = "execution_interrupted"
    EXECUTION_STOPPED = "execution_stopped"


@dataclass(frozen=True)
class ArtifactDigest:
    algorithm: Literal["sha256"]
    value: str
    size_bytes: int


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    digest: ArtifactDigest
    media_type: str
    relative_path: str


@dataclass(frozen=True)
class ArtifactSourceRef:
    source_run_id: str
    source_artifact_id: str
    expected_digest: ArtifactDigest


@dataclass(frozen=True)
class VerifiedArtifact:
    source: ArtifactSourceRef
    digest: ArtifactDigest
    media_type: str
    content: bytes


@dataclass(frozen=True)
class SemanticKeyDigest:
    sha256: str


@dataclass(frozen=True)
class ExecutionFingerprint:
    schema_version: str
    sha256: str


@dataclass(frozen=True)
class EffectRequestDigest:
    sha256: str


@dataclass(frozen=True)
class Awaiting:
    reason: ResumeReason
    resume_key: str
    input_required: bool
    request_ref: ArtifactRef | None = None
    response_contract: str | None = None
    details: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class RunError:
    code: str
    message: str
    details: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    handler: str
    semantic_input: Mapping[str, JsonValue]


@dataclass(frozen=True)
class RunSnapshot:
    run_id: str
    revision: int
    status: RunStatus
    attempt: int
    created_at: str
    updated_at: str
    awaiting: Awaiting | None = None
    result_ref: ArtifactRef | None = None
    error: RunError | None = None
    interrupted: bool = False


@dataclass(frozen=True)
class StopRequest:
    target_attempt: int
    requested_at: str
    reason: str | None


@dataclass(frozen=True)
class RunView:
    snapshot: RunSnapshot
    stop_request: StopRequest | None


@dataclass(frozen=True)
class Succeeded:
    result_ref: ArtifactRef | None = None


@dataclass(frozen=True)
class Paused:
    awaiting: Awaiting


@dataclass(frozen=True)
class Failed:
    error: RunError


RunOutcome: TypeAlias = Succeeded | Paused | Failed


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    path: tuple[str | int, ...] = ()


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class ExecutionSlice:
    monotonic_deadline: float | None = None


class EffectStage(StrEnum):
    PREPARED = "prepared"
    MAY_HAVE_RUN = "may_have_run"
    OUTPUT_SAVED = "output_saved"
    COMMITTED = "committed"


@dataclass(frozen=True)
class EffectRecord:
    effect_id: str
    revision: int
    effect_request_digest: EffectRequestDigest
    stage: EffectStage
    details: Mapping[str, JsonValue] = field(default_factory=dict)
    external_handle: str | None = None
    output_ref: ArtifactRef | None = None


class RecoveryDecision(StrEnum):
    REPLAY_OUTPUT = "replay_output"
    RETRY_VERIFIED_NOT_RUN = "retry_verified_not_run"
    RESUME_EXTERNALLY = "resume_externally"
    PAUSE_UNCERTAIN = "pause_uncertain"


class FailureMode(StrEnum):
    COLLECT = "collect"
    FAIL_FAST = "fail_fast"


@dataclass(frozen=True)
class WorkUnit:
    unit_id: str
    semantic_input: JsonValue


@dataclass(frozen=True)
class UnitResult:
    unit_id: str
    status: Literal["succeeded", "failed"]
    value: JsonValue = None
    error: RunError | None = None
    replayed: bool = False


@dataclass(frozen=True)
class GroupUnitView:
    unit_id: str
    status: Literal["pending", "succeeded", "failed"]
    value: JsonValue = None
    error: RunError | None = None


@dataclass(frozen=True)
class GroupView:
    group_id: str
    units: tuple[GroupUnitView, ...]


@dataclass(frozen=True)
class GroupResult:
    group_id: str
    units: tuple[UnitResult, ...]


GroupExecutionResult: TypeAlias = GroupResult | Paused
