from __future__ import annotations

from ..sources import SourceArtifact


class ParseError(RuntimeError):
    """A typed deterministic source-parsing failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        artifact: SourceArtifact,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.artifact = artifact


__all__ = ["ParseError"]
