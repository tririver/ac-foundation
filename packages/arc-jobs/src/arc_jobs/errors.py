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
    def __init__(self, *args: object, effect_id: str | None = None):
        """Pause an effect recovery that requires human supervision.

        Preserve the base exception's positional argument contract while
        allowing recovery code to attach an optional effect identifier.
        """

        self.effect_id = effect_id
        super().__init__(*args)
