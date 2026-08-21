from __future__ import annotations


class AcJobsError(Exception):
    """Base error for the durable run kernel."""


class InvalidRunIdError(AcJobsError):
    pass


class InvalidStateError(AcJobsError):
    pass


class UnsupportedSchemaError(InvalidStateError):
    pass


class CorruptStateError(InvalidStateError):
    pass


class InvalidTransitionError(InvalidStateError):
    pass


class StateConflictError(AcJobsError):
    pass


class RevisionConflictError(StateConflictError):
    pass


class ArtifactConflictError(StateConflictError):
    pass


class IdempotencyConflictError(StateConflictError):
    pass


class ResumeInputConflictError(StateConflictError):
    pass


class RunNotFoundError(AcJobsError):
    pass


class RunBusyError(AcJobsError):
    pass


class ResumeMismatchError(AcJobsError):
    pass


class StoppedError(AcJobsError):
    pass
