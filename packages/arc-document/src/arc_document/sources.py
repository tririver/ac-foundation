from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .parse.models import ParsedDocument


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class SourceFormat(str, Enum):
    HTML = "html"
    MARKDOWN = "markdown"
    TEX = "tex"
    PDF = "pdf"


class SourceOriginKind(str, Enum):
    LOCAL_IMPORT = "local_import"
    REMOTE_PROVIDER = "remote_provider"
    REPOSITORY = "repository"


class ValidationPolicy(str, Enum):
    NONE = "none"
    DETERMINISTIC_ONLY = "deterministic_only"
    VISUAL_ALL_PAGES = "visual_all_pages"


class ReconciliationStatus(str, Enum):
    VERIFIED = "verified"
    MISMATCH = "mismatch"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    UNREVIEWED = "unreviewed"


@dataclass(frozen=True)
class SourceOrigin:
    """Provenance for one acquisition, excluded from source content identity."""

    kind: SourceOriginKind
    provider: str = ""
    locator: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = SourceOriginKind(self.kind)
        provider = self.provider.strip()
        if kind is SourceOriginKind.REMOTE_PROVIDER and not provider:
            raise ValueError("remote source origin requires a provider")
        normalized = {
            str(key): str(value)
            for key, value in sorted(self.metadata.items(), key=lambda item: str(item[0]))
        }
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "metadata", MappingProxyType(normalized))


@dataclass(frozen=True)
class SourceArtifact:
    """An immutable source stored and verified by a SourceRepository."""

    source_format: SourceFormat
    artifact_digest: str
    size: int
    media_type: str
    origin: SourceOrigin

    def __post_init__(self) -> None:
        source_format = SourceFormat(self.source_format)
        digest = self.artifact_digest.casefold()
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("artifact_digest must be a lowercase SHA-256 digest")
        if self.size < 0:
            raise ValueError("source artifact size cannot be negative")
        if not self.media_type or ";" in self.media_type:
            raise ValueError("media_type must be a normalized MIME type")
        object.__setattr__(self, "source_format", source_format)
        object.__setattr__(self, "artifact_digest", digest)
        object.__setattr__(self, "media_type", self.media_type.casefold())

    @property
    def content_identity(self) -> tuple[str, str, str, int]:
        """Stable semantic identity; origin and repository paths are locators."""

        return (
            self.source_format.value,
            self.media_type,
            self.artifact_digest,
            self.size,
        )


@dataclass(frozen=True)
class SourceBundle:
    """One authoritative primary plus independently evaluated validators."""

    primary: SourceArtifact
    validators: tuple[SourceArtifact, ...] = ()

    def __post_init__(self) -> None:
        validators = tuple(sorted(self.validators, key=lambda item: item.content_identity))
        identities = [item.content_identity for item in validators]
        if len(identities) != len(set(identities)):
            raise ValueError("source bundle contains a duplicate validator")
        if self.primary.content_identity in identities:
            raise ValueError("source bundle primary cannot also be a validator")
        object.__setattr__(self, "validators", validators)


@dataclass(frozen=True)
class ReconciliationEntry:
    validator: SourceArtifact
    status: ReconciliationStatus
    subject_id: str = ""
    message: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ReconciliationStatus(self.status))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True)
class ReconciliationReport:
    primary: SourceArtifact
    policy: ValidationPolicy
    entries: tuple[ReconciliationEntry, ...] = ()

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        for entry in entries:
            if entry.validator.content_identity == self.primary.content_identity:
                raise ValueError("reconciliation entry cannot validate the primary against itself")
        object.__setattr__(self, "policy", ValidationPolicy(self.policy))
        object.__setattr__(self, "entries", entries)


@dataclass(frozen=True)
class ParseOutcome:
    """A successful primary parse, including non-fatal validation evidence."""

    document: "ParsedDocument"
    report: ReconciliationReport
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        from .parse.models import ParsedDocument

        document = self.document
        if not isinstance(document, ParsedDocument):
            raise TypeError("ParseOutcome.document must be a ParsedDocument")
        if document.source.content_identity != self.report.primary.content_identity:
            raise ValueError("parsed document source does not match report primary")
        object.__setattr__(self, "document", document)
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))


__all__ = [
    "ParseOutcome",
    "ReconciliationEntry",
    "ReconciliationReport",
    "ReconciliationStatus",
    "SourceArtifact",
    "SourceBundle",
    "SourceFormat",
    "SourceOrigin",
    "SourceOriginKind",
    "ValidationPolicy",
]
