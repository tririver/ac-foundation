"""Stable AC Foundation LLM error taxonomy."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping


class ErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    INVALID_SCHEMA = "invalid_schema"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    RESUME_KEY_MISMATCH = "resume_key_mismatch"
    RESUME_INPUT_CONFLICT = "resume_input_conflict"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_AUTHENTICATION = "provider_authentication"
    PROVIDER_QUOTA = "provider_quota"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_INVALID_REQUEST = "provider_invalid_request"
    PROVIDER_TRANSPORT = "provider_transport"
    PROVIDER_TIMEOUT = "provider_timeout"
    OUTPUT_INVALID = "output_invalid"
    CANDIDATE_CONFLICT = "candidate_selection_required"
    EXECUTION_MISMATCH = "execution_mismatch"
    CORRUPT_STATE = "corrupt_state"
    LOCAL_IO = "local_io"
    ADOPTION_CONFLICT = "adoption_conflict"
    ADOPTION_NOT_AUTHORIZED = "adoption_not_authorized"
    STOPPED = "stopped"


class FailureCategory(StrEnum):
    AUTHENTICATION = "authentication"
    QUOTA = "quota"
    RATE_LIMIT = "rate_limit"
    UNAVAILABLE = "unavailable"
    INVALID_REQUEST = "invalid_request"
    SCHEMA = "schema"
    TRANSPORT = "transport"
    TIMEOUT = "timeout"
    STOPPED = "stopped"
    LOCAL_IO = "local_io"
    INTERNAL = "internal"


class AcLLMError(RuntimeError):
    """Base exception with a stable machine-readable code."""

    code: ErrorCode

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


class InvalidRequestError(AcLLMError):
    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.INVALID_REQUEST, message, details=details)


class InvalidSchemaError(AcLLMError):
    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(ErrorCode.INVALID_SCHEMA, message, details=details)


class IdempotencyConflictError(AcLLMError):
    def __init__(self, message: str = "The task ID is bound to different semantic input.") -> None:
        super().__init__(ErrorCode.IDEMPOTENCY_CONFLICT, message)


class ResumeKeyMismatchError(AcLLMError):
    def __init__(self) -> None:
        super().__init__(
            ErrorCode.RESUME_KEY_MISMATCH,
            "The resume input does not target the current pause.",
        )


class ResumeInputConflictError(AcLLMError):
    def __init__(self) -> None:
        super().__init__(
            ErrorCode.RESUME_INPUT_CONFLICT,
            "The resume key is already bound to different input.",
        )


class OutputInvalidError(AcLLMError):
    def __init__(self, message: str = "No candidate satisfies the output contract.") -> None:
        super().__init__(ErrorCode.OUTPUT_INVALID, message)


class CandidateConflictError(AcLLMError):
    def __init__(self, candidate_digests: tuple[str, ...]) -> None:
        super().__init__(
            ErrorCode.CANDIDATE_CONFLICT,
            "Multiple non-equivalent valid output candidates require selection.",
            details={"candidate_digests": list(candidate_digests)},
        )
        self.candidate_digests = candidate_digests


class ProviderFailure(AcLLMError):
    """A normalized provider failure."""

    def __init__(
        self,
        message: str,
        *,
        category: FailureCategory,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        code = {
            FailureCategory.AUTHENTICATION: ErrorCode.PROVIDER_AUTHENTICATION,
            FailureCategory.QUOTA: ErrorCode.PROVIDER_QUOTA,
            FailureCategory.RATE_LIMIT: ErrorCode.PROVIDER_RATE_LIMIT,
            FailureCategory.UNAVAILABLE: ErrorCode.PROVIDER_UNAVAILABLE,
            FailureCategory.INVALID_REQUEST: ErrorCode.PROVIDER_INVALID_REQUEST,
            FailureCategory.SCHEMA: ErrorCode.INVALID_SCHEMA,
            FailureCategory.TRANSPORT: ErrorCode.PROVIDER_TRANSPORT,
            FailureCategory.TIMEOUT: ErrorCode.PROVIDER_TIMEOUT,
            FailureCategory.STOPPED: ErrorCode.STOPPED,
            FailureCategory.LOCAL_IO: ErrorCode.LOCAL_IO,
            FailureCategory.INTERNAL: ErrorCode.PROVIDER_TRANSPORT,
        }[category]
        super().__init__(code, message, details=details)
        self.category = category
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds


class ExecutionMismatchError(AcLLMError):
    def __init__(self) -> None:
        super().__init__(
            ErrorCode.EXECUTION_MISMATCH,
            "The active provider session is incompatible with this execution recipe.",
        )


class CorruptTaskStateError(AcLLMError):
    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(ErrorCode.CORRUPT_STATE, message, details=details)


class AdoptionConflictError(AcLLMError):
    def __init__(self, message: str) -> None:
        super().__init__(ErrorCode.ADOPTION_CONFLICT, message)


class AdoptionAuthorizationError(AcLLMError):
    def __init__(self) -> None:
        super().__init__(
            ErrorCode.ADOPTION_NOT_AUTHORIZED,
            "Adoption across semantic tasks requires explicit authorization.",
        )
