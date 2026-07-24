from __future__ import annotations


class ArcJobsError(Exception):
    """Base error for the durable run kernel."""


class InvalidRunIdError(ArcJobsError):
    pass


class InvalidStateError(ArcJobsError):
    pass


class UnsupportedSchemaError(InvalidStateError):
    pass


class CorruptStateError(InvalidStateError):
    pass


class InvalidTransitionError(InvalidStateError):
    pass


class StateConflictError(ArcJobsError):
    pass


class RevisionConflictError(StateConflictError):
    pass


class ArtifactConflictError(StateConflictError):
    pass


class IdempotencyConflictError(StateConflictError):
    pass


class ResumeInputConflictError(StateConflictError):
    pass


class RunNotFoundError(ArcJobsError):
    pass


class RunBusyError(ArcJobsError):
    pass


class ResumeMismatchError(ArcJobsError):
    pass


class CancelledError(ArcJobsError):
    pass


class UnsafeEffectRecoveryError(ArcJobsError):
    pass
