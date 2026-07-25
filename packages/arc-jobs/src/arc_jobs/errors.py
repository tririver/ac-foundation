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


class StoppedError(ArcJobsError):
    pass


class UnsafeEffectRecoveryError(ArcJobsError):
    def __init__(self, message: str, *, effect_id: str):
        """Pause an identified effect recovery that requires supervision."""
        self.effect_id = effect_id
        super().__init__(message)
