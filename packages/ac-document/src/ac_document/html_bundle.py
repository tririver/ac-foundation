"""Portable, network-free HTML source bundle contracts and materialization."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Protocol, Sequence
from urllib.parse import unquote, urlsplit, urlunsplit

from ac_jobs import atomic_write_bytes
from bs4 import BeautifulSoup

from ._file_lock import exclusive_file_lock
from .source_repository import SourceRepository, SourceRepositoryError
from .sources import SourceArtifact, SourceFormat, SourceOrigin, SourceOriginKind


HTML_SOURCE_BUNDLE_SCHEMA = "ac.document.html_source_bundle.v1"
HTML_SOURCE_EXPORT_SCHEMA = "ac.document.html_source_export.v1"
HTML_SOURCE_BUNDLE_CACHE_SCHEMA = "ac.document.html_source_bundle_cache.v1"

DEFAULT_MAX_PRIMARY_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_DEPENDENCY_COUNT = 256
DEFAULT_MAX_DEPENDENCY_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_TOTAL_DEPENDENCY_BYTES = 200 * 1024 * 1024
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_TIMEOUT_SECONDS = 30.0

SUPPORTED_DEPENDENCY_MEDIA_TYPES = frozenset(
    {
        "image/avif",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/svg+xml",
        "image/webp",
    }
)
_EXTENSION_MEDIA_TYPES = {
    ".avif": "image/avif",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
}
_SHA256_RE = frozenset("0123456789abcdef")
_POLICY_FIELDS = {
    "max_primary_bytes",
    "max_dependency_count",
    "max_dependency_bytes",
    "max_total_dependency_bytes",
    "max_redirects",
    "timeout_seconds",
    "same_origin_dependencies",
    "allowed_origins",
}
_PRIMARY_FIELDS = {
    "source_format",
    "artifact_digest",
    "size",
    "media_type",
    "origin",
}
_ORIGIN_FIELDS = {"kind", "provider", "locator", "metadata"}
_DEPENDENCY_FIELDS = {
    "ordinal",
    "element",
    "attribute",
    "authored_target",
    "request_url",
    "resolved_url",
    "declared_media_type",
    "availability",
    "materialization_path",
    "media_type",
    "artifact_digest",
    "size",
    "error_code",
    "error_message",
}
_WARNING_FIELDS = {
    "code",
    "message",
    "dependency_ordinal",
    "element",
    "attribute",
    "authored_target",
}
_BUNDLE_FIELDS = {
    "schema_version",
    "primary",
    "requested_url",
    "final_url",
    "base_url",
    "acquisition_policy",
    "dependencies",
    "warnings",
    "bundle_digest",
}
_EXPORT_FIELDS = {
    "schema_version",
    "bundle",
    "materialized_source",
    "resources",
    "rewrites",
}
_MATERIALIZED_SOURCE_FIELDS = {"path", "artifact_digest", "size"}
_EXPORT_RESOURCE_FIELDS = {
    "dependency_ordinal",
    "path",
    "artifact_digest",
    "media_type",
    "size",
}
_REWRITE_FIELDS = {
    "dependency_ordinal",
    "element",
    "attribute",
    "authored_target",
    "materialized_target",
}


class HTMLSourceBundleError(RuntimeError):
    """Typed failure for portable HTML bundle contracts and durable state."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class StoredHTMLDependency:
    """Portable content identity returned by an HTML bundle storage adapter."""

    artifact_digest: str
    media_type: str
    size: int

    def __post_init__(self) -> None:
        if not _is_digest(self.artifact_digest):
            raise ValueError("stored HTML dependency digest is invalid")
        if _normalize_media_type(self.media_type) != self.media_type:
            raise ValueError("stored HTML dependency media type is invalid")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("stored HTML dependency size is invalid")


class HTMLSourceBundleStorage(Protocol):
    """Storage seam shared by acquisition, replay verification, and export."""

    def read_primary(self, primary: SourceArtifact) -> bytes: ...

    def store_dependency(
        self, payload: bytes, *, media_type: str
    ) -> StoredHTMLDependency: ...

    def read_dependency(
        self, *, artifact_digest: str, media_type: str, size: int
    ) -> bytes: ...


class SourceRepositoryHTMLSourceBundleStorage:
    """Default HTML bundle storage backed by AC Foundation SourceRepository."""

    def __init__(self, repository: SourceRepository):
        self.repository = repository

    def read_primary(self, primary: SourceArtifact) -> bytes:
        return self.repository.read_bytes(primary)

    def store_dependency(
        self, payload: bytes, *, media_type: str
    ) -> StoredHTMLDependency:
        asset = self.repository.store_asset_bytes(payload, media_type=media_type)
        return StoredHTMLDependency(
            artifact_digest=asset.artifact_digest,
            media_type=asset.media_type,
            size=asset.size,
        )

    def read_dependency(
        self, *, artifact_digest: str, media_type: str, size: int
    ) -> bytes:
        asset = self.repository.get_asset(artifact_digest)
        if asset.content_identity != (media_type, artifact_digest, size):
            raise SourceRepositoryError(
                "asset_artifact_mismatch", "asset metadata does not match bundle identity"
            )
        return self.repository.read_asset_bytes(asset)


@dataclass(frozen=True)
class HTMLAcquisitionPolicy:
    """Bounded acquisition policy recorded in every bundle identity."""

    max_primary_bytes: int = DEFAULT_MAX_PRIMARY_BYTES
    max_dependency_count: int = DEFAULT_MAX_DEPENDENCY_COUNT
    max_dependency_bytes: int = DEFAULT_MAX_DEPENDENCY_BYTES
    max_total_dependency_bytes: int = DEFAULT_MAX_TOTAL_DEPENDENCY_BYTES
    max_redirects: int = DEFAULT_MAX_REDIRECTS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    same_origin_dependencies: bool = True
    allowed_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        integers = (
            self.max_primary_bytes,
            self.max_dependency_count,
            self.max_dependency_bytes,
            self.max_total_dependency_bytes,
            self.max_redirects,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise ValueError("HTML acquisition limits must be integers")
        if (
            self.max_primary_bytes <= 0
            or self.max_dependency_count <= 0
            or self.max_dependency_bytes <= 0
            or self.max_total_dependency_bytes <= 0
            or self.max_redirects < 0
            or self.max_total_dependency_bytes < self.max_dependency_bytes
        ):
            raise ValueError("HTML acquisition limits are invalid")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("HTML acquisition timeout must be positive")
        if not isinstance(self.same_origin_dependencies, bool):
            raise ValueError("same_origin_dependencies must be a boolean")
        origins = _normalize_allowed_origins(self.allowed_origins)
        if not self.same_origin_dependencies and not origins:
            raise ValueError(
                "same_origin_dependencies=False requires explicit allowed origins"
            )
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        object.__setattr__(self, "allowed_origins", origins)

    def to_document(self) -> dict[str, Any]:
        return {
            "max_primary_bytes": self.max_primary_bytes,
            "max_dependency_count": self.max_dependency_count,
            "max_dependency_bytes": self.max_dependency_bytes,
            "max_total_dependency_bytes": self.max_total_dependency_bytes,
            "max_redirects": self.max_redirects,
            "timeout_seconds": self.timeout_seconds,
            "same_origin_dependencies": self.same_origin_dependencies,
            "allowed_origins": list(self.allowed_origins),
        }

    @classmethod
    def from_document(cls, value: Any) -> "HTMLAcquisitionPolicy":
        if not isinstance(value, Mapping) or set(value) != _POLICY_FIELDS:
            raise ValueError("HTML source bundle acquisition policy is invalid")
        origins = value.get("allowed_origins")
        if not isinstance(origins, (list, tuple)) or not all(
            isinstance(item, str) for item in origins
        ):
            raise ValueError("HTML source bundle allowed origins are invalid")
        return cls(
            max_primary_bytes=value["max_primary_bytes"],
            max_dependency_count=value["max_dependency_count"],
            max_dependency_bytes=value["max_dependency_bytes"],
            max_total_dependency_bytes=value["max_total_dependency_bytes"],
            max_redirects=value["max_redirects"],
            timeout_seconds=value["timeout_seconds"],
            same_origin_dependencies=value["same_origin_dependencies"],
            allowed_origins=tuple(origins),
        )


@dataclass(frozen=True)
class HTMLSourceWarning:
    code: str
    message: str
    dependency_ordinal: int | None = None
    element: str = ""
    attribute: str = ""
    authored_target: str = ""

    def __post_init__(self) -> None:
        strings = (self.code, self.message, self.element, self.attribute, self.authored_target)
        if not all(isinstance(item, str) for item in strings) or not self.code or not self.message:
            raise ValueError("HTML source warning requires nonempty string code and message")
        if self.dependency_ordinal is not None and (
            isinstance(self.dependency_ordinal, bool)
            or not isinstance(self.dependency_ordinal, int)
            or self.dependency_ordinal < 0
        ):
            raise ValueError("HTML source warning ordinal cannot be negative")

    def __str__(self) -> str:
        suffix = f": {self.authored_target}" if self.authored_target else ""
        return f"{self.code}: {self.message}{suffix}"


@dataclass(frozen=True)
class HTMLSourceDependency:
    ordinal: int
    element: str
    attribute: str
    authored_target: str
    request_url: str = ""
    resolved_url: str = ""
    declared_media_type: str = ""
    availability: str = "unavailable"
    materialization_path: str = ""
    media_type: str = ""
    artifact_digest: str = ""
    size: int = 0
    error_code: str = ""
    error_message: str = ""

    def __post_init__(self) -> None:
        strings = (
            self.element,
            self.attribute,
            self.authored_target,
            self.request_url,
            self.resolved_url,
            self.declared_media_type,
            self.availability,
            self.materialization_path,
            self.media_type,
            self.artifact_digest,
            self.error_code,
            self.error_message,
        )
        if not all(isinstance(item, str) for item in strings):
            raise TypeError("HTML dependency fields must be strings")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise ValueError("HTML dependency ordinal cannot be negative")
        if self.element not in {"img", "object", "source"} or self.attribute not in {"src", "data", "srcset"}:
            raise ValueError("HTML dependency element or attribute is unsupported")
        if self.availability not in {"available", "unavailable"}:
            raise ValueError("HTML dependency availability is invalid")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("HTML dependency size cannot be negative")
        for media_type in (self.declared_media_type, self.media_type):
            if media_type and _normalize_media_type(media_type) != media_type:
                raise ValueError("HTML dependency media type must be normalized")
        if self.availability == "available":
            if (
                not self.request_url
                or not self.resolved_url
                or not _safe_materialization_path(self.materialization_path)
                or self.media_type not in SUPPORTED_DEPENDENCY_MEDIA_TYPES
                or not _is_digest(self.artifact_digest)
                or self.error_code
                or self.error_message
            ):
                raise ValueError("available HTML dependency metadata is incomplete")
        elif (
            self.materialization_path
            or self.media_type
            or self.artifact_digest
            or self.size != 0
            or not self.error_code
            or not self.error_message
        ):
            raise ValueError("unavailable HTML dependency metadata is inconsistent")


@dataclass(frozen=True)
class HTMLSourceBundle:
    """Portable identity of one HTML primary and its optional local assets."""

    primary: SourceArtifact
    requested_url: str
    final_url: str
    base_url: str
    acquisition_policy: Mapping[str, Any]
    dependencies: tuple[HTMLSourceDependency, ...] = ()
    warnings: tuple[HTMLSourceWarning, ...] = ()
    bundle_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.primary.source_format is not SourceFormat.HTML:
            raise ValueError("HTML source bundle primary must be HTML")
        requested = normalize_https_url(self.requested_url)
        final = normalize_https_url(self.final_url)
        base = normalize_https_url(self.base_url)
        policy = HTMLAcquisitionPolicy.from_document(self.acquisition_policy)
        dependencies = tuple(self.dependencies)
        if [item.ordinal for item in dependencies] != list(range(len(dependencies))):
            raise ValueError("HTML dependency ordinals must be contiguous")
        warnings = tuple(self.warnings)
        by_ordinal = {
            item.dependency_ordinal: item
            for item in warnings
            if item.dependency_ordinal is not None
        }
        if len(by_ordinal) != sum(item.dependency_ordinal is not None for item in warnings):
            raise ValueError("HTML source bundle has duplicate dependency warnings")
        for ordinal, warning in by_ordinal.items():
            if ordinal >= len(dependencies):
                raise ValueError("HTML dependency warning ordinal is out of range")
            dependency = dependencies[ordinal]
            if (
                dependency.availability != "unavailable"
                or warning.code != dependency.error_code
                or warning.element != dependency.element
                or warning.attribute != dependency.attribute
                or warning.authored_target != dependency.authored_target
            ):
                raise ValueError("HTML dependency warning does not match its record")
        if any(
            item.availability == "unavailable" and item.ordinal not in by_ordinal
            for item in dependencies
        ):
            raise ValueError("unavailable HTML dependency requires a warning")
        object.__setattr__(self, "requested_url", requested)
        object.__setattr__(self, "final_url", final)
        object.__setattr__(self, "base_url", base)
        internal_policy = policy.to_document()
        internal_policy["allowed_origins"] = policy.allowed_origins
        object.__setattr__(self, "acquisition_policy", MappingProxyType(internal_policy))
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(
            self,
            "bundle_digest",
            hashlib.sha256(_json_bytes(_bundle_identity_document(self))).hexdigest(),
        )


@dataclass(frozen=True)
class HTMLSourceBundleMaterialization:
    source_path: Path
    manifest_path: Path
    resource_paths: tuple[Path, ...]
    bundle_digest: str
    warnings: tuple[str, ...]


def html_source_bundle_export_to_document(
    bundle: HTMLSourceBundle,
    *,
    materialized_source: bytes,
) -> dict[str, Any]:
    resources = [
        {
            "dependency_ordinal": dependency.ordinal,
            "path": dependency.materialization_path,
            "artifact_digest": dependency.artifact_digest,
            "media_type": dependency.media_type,
            "size": dependency.size,
        }
        for dependency in bundle.dependencies
        if dependency.availability == "available"
    ]
    rewrites = [
        {
            "dependency_ordinal": dependency.ordinal,
            "element": dependency.element,
            "attribute": dependency.attribute,
            "authored_target": dependency.authored_target,
            "materialized_target": dependency.materialization_path,
        }
        for dependency in bundle.dependencies
        if dependency.availability == "available"
        and dependency.materialization_path != dependency.authored_target
    ]
    document = {
        "schema_version": HTML_SOURCE_EXPORT_SCHEMA,
        "bundle": html_source_bundle_to_document(bundle),
        "materialized_source": {
            "path": "source.html",
            "artifact_digest": hashlib.sha256(materialized_source).hexdigest(),
            "size": len(materialized_source),
        },
        "resources": resources,
        "rewrites": rewrites,
    }
    html_source_bundle_export_from_document(document)
    return document


def html_source_bundle_export_from_document(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _EXPORT_FIELDS:
        raise ValueError("HTML source bundle export has invalid fields")
    if value.get("schema_version") != HTML_SOURCE_EXPORT_SCHEMA:
        raise ValueError("HTML source bundle export schema is unsupported")
    bundle = html_source_bundle_from_document(value.get("bundle"))
    source = value.get("materialized_source")
    if not isinstance(source, Mapping) or set(source) != _MATERIALIZED_SOURCE_FIELDS:
        raise ValueError("HTML source bundle materialized source is invalid")
    if source.get("path") != "source.html" or not _is_digest(source.get("artifact_digest")) or not _is_nonnegative_int(source.get("size")):
        raise ValueError("HTML source bundle materialized source identity is invalid")
    resources = value.get("resources")
    rewrites = value.get("rewrites")
    if not isinstance(resources, list) or not isinstance(rewrites, list):
        raise ValueError("HTML source bundle export resources or rewrites are invalid")
    expected_resources = [
        {
            "dependency_ordinal": dependency.ordinal,
            "path": dependency.materialization_path,
            "artifact_digest": dependency.artifact_digest,
            "media_type": dependency.media_type,
            "size": dependency.size,
        }
        for dependency in bundle.dependencies
        if dependency.availability == "available"
    ]
    expected_rewrites = [
        {
            "dependency_ordinal": dependency.ordinal,
            "element": dependency.element,
            "attribute": dependency.attribute,
            "authored_target": dependency.authored_target,
            "materialized_target": dependency.materialization_path,
        }
        for dependency in bundle.dependencies
        if dependency.availability == "available"
        and dependency.materialization_path != dependency.authored_target
    ]
    if resources != expected_resources or rewrites != expected_rewrites:
        raise ValueError("HTML source bundle export does not match its bundle")
    return {
        "schema_version": HTML_SOURCE_EXPORT_SCHEMA,
        "bundle": html_source_bundle_to_document(bundle),
        "materialized_source": dict(source),
        "resources": [dict(item) for item in resources],
        "rewrites": [dict(item) for item in rewrites],
    }


def normalize_https_url(url: str, *, allowed_origins: Sequence[str] = ()) -> str:
    """Normalize an HTTPS URL without performing a DNS or network lookup."""

    text = str(url or "").strip()
    if not text or any(ord(character) < 0x20 or ord(character) == 0x7F for character in text):
        raise HTMLSourceBundleError("remote_url_invalid", "remote URL contains invalid characters")
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError as exc:
        raise HTMLSourceBundleError("remote_url_invalid", "remote URL is malformed") from exc
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise HTMLSourceBundleError("remote_url_invalid", "remote URL must use HTTPS with a host")
    if parsed.username is not None or parsed.password is not None:
        raise HTMLSourceBundleError("remote_url_invalid", "remote URL cannot include credentials")
    if port not in (None, 443):
        raise HTMLSourceBundleError("remote_url_invalid", "remote URL cannot use a nondefault port")
    if parsed.fragment:
        raise HTMLSourceBundleError("remote_url_invalid", "remote URL cannot include a fragment")
    try:
        hostname = parsed.hostname.encode("idna").decode("ascii").casefold()
    except UnicodeError as exc:
        raise HTMLSourceBundleError("remote_url_invalid", "remote URL hostname is invalid") from exc
    netloc = f"[{hostname}]" if ":" in hostname else hostname
    normalized = urlunsplit(("https", netloc, parsed.path or "/", parsed.query, ""))
    permitted = _normalize_allowed_origins(allowed_origins)
    if permitted and _origin(normalized) not in permitted:
        raise HTMLSourceBundleError("remote_origin_not_allowed", "remote URL is outside the allowed origins")
    return normalized


def html_source_bundle_to_document(bundle: HTMLSourceBundle) -> dict[str, Any]:
    return {
        **_bundle_identity_document(bundle),
        "bundle_digest": bundle.bundle_digest,
    }


def html_source_bundle_from_document(value: Any) -> HTMLSourceBundle:
    if not isinstance(value, Mapping) or set(value) != _BUNDLE_FIELDS:
        raise ValueError("HTML source bundle document has invalid fields")
    if value.get("schema_version") != HTML_SOURCE_BUNDLE_SCHEMA:
        raise ValueError("HTML source bundle schema version is unsupported")
    dependencies = value.get("dependencies")
    warnings = value.get("warnings")
    if not isinstance(dependencies, list) or not isinstance(warnings, list):
        raise ValueError("HTML source bundle dependencies or warnings are invalid")
    bundle = HTMLSourceBundle(
        primary=_source_artifact_from_document(value.get("primary")),
        requested_url=_required_string(value, "requested_url"),
        final_url=_required_string(value, "final_url"),
        base_url=_required_string(value, "base_url"),
        acquisition_policy=value.get("acquisition_policy"),
        dependencies=tuple(_dependency_from_document(item) for item in dependencies),
        warnings=tuple(_warning_from_document(item) for item in warnings),
    )
    if value.get("bundle_digest") != bundle.bundle_digest:
        raise ValueError("HTML source bundle digest does not match its contents")
    return bundle


class HTMLSourceBundleCache:
    """Durable cache for bundle manifests with repository-integrity replay."""

    def __init__(self, root: str | Path, storage: HTMLSourceBundleStorage):
        self.root = Path(root)
        self.storage = storage

    def request_key(self, requested_url: str, policy: Mapping[str, Any]) -> str:
        normalized = normalize_https_url(requested_url)
        normalized_policy = HTMLAcquisitionPolicy.from_document(policy).to_document()
        return hashlib.sha256(
            _json_bytes(
                {
                    "schema_version": HTML_SOURCE_BUNDLE_CACHE_SCHEMA,
                    "requested_url": normalized,
                    "acquisition_policy": normalized_policy,
                }
            )
        ).hexdigest()

    def lookup(self, request_key: str) -> HTMLSourceBundle | None:
        if not _is_digest(request_key):
            raise HTMLSourceBundleError("html_bundle_cache_key_invalid", "cache request key is invalid")
        path = self._request_path(request_key)
        with self._lock("request", request_key):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise HTMLSourceBundleError(
                    "html_bundle_cache_corrupt", "cached HTML bundle request index is unreadable"
                ) from exc
        if (
            not isinstance(value, Mapping)
            or set(value) != {"schema_version", "bundle_digest"}
            or value.get("schema_version") != HTML_SOURCE_BUNDLE_CACHE_SCHEMA
            or not isinstance(value.get("bundle_digest"), str)
            or not _is_digest(value["bundle_digest"])
        ):
            raise HTMLSourceBundleError(
                "html_bundle_cache_corrupt", "cached HTML bundle request index is invalid"
            )
        bundle = self.load(value["bundle_digest"])
        if self.request_key(bundle.requested_url, bundle.acquisition_policy) != request_key:
            raise HTMLSourceBundleError(
                "html_bundle_cache_corrupt", "cached request index does not match the bundle identity"
            )
        return bundle

    def store(self, bundle: HTMLSourceBundle, *, request_key: str | None = None) -> None:
        document = html_source_bundle_to_document(bundle)
        digest = bundle.bundle_digest
        with self._lock("bundle", digest):
            path = self._bundle_path(digest)
            if path.exists():
                existing = self._load_document(path, digest)
                if existing != bundle:
                    raise HTMLSourceBundleError(
                        "html_bundle_cache_corrupt", "cached HTML bundle digest maps to conflicting content"
                    )
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(path, _json_bytes(document) + b"\n")
        if request_key is not None:
            if not _is_digest(request_key):
                raise HTMLSourceBundleError("html_bundle_cache_key_invalid", "cache request key is invalid")
            expected_request_key = self.request_key(
                bundle.requested_url, bundle.acquisition_policy
            )
            if request_key != expected_request_key:
                raise HTMLSourceBundleError(
                    "html_bundle_cache_corrupt", "cache request key does not match the bundle identity"
                )
            with self._lock("request", request_key):
                path = self._request_path(request_key)
                if path.exists():
                    current = self._load_request_digest(path)
                    if current != digest:
                        raise HTMLSourceBundleError(
                            "html_bundle_cache_corrupt", "cache request key maps to conflicting bundle"
                        )
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write_bytes(
                        path,
                        _json_bytes(
                            {
                                "schema_version": HTML_SOURCE_BUNDLE_CACHE_SCHEMA,
                                "bundle_digest": digest,
                            }
                        )
                        + b"\n",
                    )

    def load(self, bundle_digest: str) -> HTMLSourceBundle:
        if not _is_digest(bundle_digest):
            raise HTMLSourceBundleError("html_bundle_cache_key_invalid", "bundle digest is invalid")
        with self._lock("bundle", bundle_digest):
            bundle = self._load_document(self._bundle_path(bundle_digest), bundle_digest)
        self._verify_repository_content(bundle)
        return bundle

    def _load_document(self, path: Path, expected_digest: str) -> HTMLSourceBundle:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise HTMLSourceBundleError("html_bundle_cache_miss", "cached HTML bundle is absent") from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise HTMLSourceBundleError(
                "html_bundle_cache_corrupt", "cached HTML bundle manifest is unreadable"
            ) from exc
        try:
            bundle = html_source_bundle_from_document(value)
        except (TypeError, ValueError, HTMLSourceBundleError) as exc:
            raise HTMLSourceBundleError(
                "html_bundle_cache_corrupt", "cached HTML bundle manifest is invalid"
            ) from exc
        if bundle.bundle_digest != expected_digest:
            raise HTMLSourceBundleError(
                "html_bundle_cache_corrupt", "cached HTML bundle manifest identity does not match its key"
            )
        return bundle

    def _verify_repository_content(self, bundle: HTMLSourceBundle) -> None:
        try:
            primary_payload = self.storage.read_primary(bundle.primary)
            if (
                len(primary_payload) != bundle.primary.size
                or hashlib.sha256(primary_payload).hexdigest()
                != bundle.primary.artifact_digest
            ):
                raise SourceRepositoryError(
                    "source_artifact_mismatch", "cached primary does not match bundle metadata"
                )
            for dependency in bundle.dependencies:
                if dependency.availability != "available":
                    continue
                payload = self.storage.read_dependency(
                    artifact_digest=dependency.artifact_digest,
                    media_type=dependency.media_type,
                    size=dependency.size,
                )
                if len(payload) != dependency.size or hashlib.sha256(payload).hexdigest() != dependency.artifact_digest:
                    raise SourceRepositoryError(
                        "asset_artifact_mismatch", "cached asset does not match bundle metadata"
                    )
        except (OSError, SourceRepositoryError, ValueError) as exc:
            raise HTMLSourceBundleError(
                "html_bundle_cache_corrupt", "cached HTML bundle repository content is missing or corrupt"
            ) from exc

    def _load_request_digest(self, path: Path) -> str:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise HTMLSourceBundleError(
                "html_bundle_cache_corrupt", "cached HTML bundle request index is unreadable"
            ) from exc
        if (
            not isinstance(value, Mapping)
            or set(value) != {"schema_version", "bundle_digest"}
            or value.get("schema_version") != HTML_SOURCE_BUNDLE_CACHE_SCHEMA
            or not isinstance(value.get("bundle_digest"), str)
            or not _is_digest(value["bundle_digest"])
        ):
            raise HTMLSourceBundleError(
                "html_bundle_cache_corrupt", "cached HTML bundle request index is invalid"
            )
        return value["bundle_digest"]

    def _bundle_path(self, digest: str) -> Path:
        return self.root / "html-source-bundles" / "v1" / "bundle" / "sha256" / digest[:2] / digest / "manifest.json"

    def _request_path(self, digest: str) -> Path:
        return self.root / "html-source-bundles" / "v1" / "request" / "sha256" / digest[:2] / f"{digest}.json"

    @contextmanager
    def _lock(self, scope: str, digest: str) -> Iterator[None]:
        with exclusive_file_lock(
            self.root / "html-source-bundles" / "v1" / "locks" / scope / f"{digest}.lock"
        ):
            yield


def materialize_html_source_bundle(
    bundle: HTMLSourceBundle,
    storage: HTMLSourceBundleStorage,
    output_dir: str | Path,
) -> HTMLSourceBundleMaterialization:
    """Publish a self-contained HTML bundle without replacing existing output."""

    output = Path(output_dir)
    try:
        source_payload = storage.read_primary(bundle.primary)
        if (
            len(source_payload) != bundle.primary.size
            or hashlib.sha256(source_payload).hexdigest()
            != bundle.primary.artifact_digest
        ):
            raise HTMLSourceBundleError(
                "html_bundle_materialization_failed",
                "bundle primary identity does not match its storage bytes",
            )
        resources = _bundle_resources(bundle, storage)
    except SourceRepositoryError as exc:
        raise HTMLSourceBundleError(
            "html_bundle_materialization_failed", "bundle repository content is missing or corrupt"
        ) from exc
    rendered, _rewrites = _render_materialized_source(source_payload, bundle.dependencies)
    manifest = html_source_bundle_export_to_document(
        bundle, materialized_source=rendered
    )
    _publish_materialization(
        output,
        source_payload=rendered,
        manifest_payload=_json_bytes(manifest) + b"\n",
        manifest_document=manifest,
        resources=resources,
    )
    return HTMLSourceBundleMaterialization(
        source_path=output / "source.html",
        manifest_path=output / "manifest.json",
        resource_paths=tuple(output / path for path in sorted(resources)),
        bundle_digest=bundle.bundle_digest,
        warnings=tuple(str(item) for item in bundle.warnings),
    )


def verify_html_source_bundle_export(output_dir: str | Path) -> dict[str, Any]:
    """Load and verify one materialized export manifest and every referenced file."""

    output = Path(output_dir)
    try:
        document = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        _verify_materialization_tree(output, document)
    except HTMLSourceBundleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HTMLSourceBundleError(
            "html_bundle_export_invalid", "materialized HTML bundle manifest is unreadable"
        ) from exc
    return html_source_bundle_export_from_document(document)


def dependency_media_type_for_url(url: str) -> str | None:
    suffix = PurePosixPath(unquote(urlsplit(url).path)).suffix.casefold()
    if not suffix:
        return None
    try:
        return _EXTENSION_MEDIA_TYPES[suffix]
    except KeyError as exc:
        raise HTMLSourceBundleError(
            "html_dependency_extension_unsupported", f"dependency extension is unsupported: {suffix}"
        ) from exc


def local_path_for_url(url: str) -> str:
    normalized = normalize_https_url(url)
    parsed = urlsplit(normalized)
    raw_path = unquote(parsed.path).lstrip("/")
    parts = raw_path.split("/") if raw_path else ["index"]
    if any(not part or part in {".", ".."} or "\\" in part or "\x00" in part for part in parts):
        raise HTMLSourceBundleError("html_dependency_path_invalid", "dependency URL does not map to a safe local path")
    if parsed.query:
        query_suffix = hashlib.sha256(parsed.query.encode("utf-8")).hexdigest()[:12]
        filename = parts[-1]
        suffix = PurePosixPath(filename).suffix
        stem = filename[: -len(suffix)] if suffix else filename
        parts[-1] = f"{stem}.{query_suffix}{suffix}"
    path = PurePosixPath("assets", parsed.hostname or "", *parts)
    if not _safe_materialization_path(path.as_posix()):
        raise HTMLSourceBundleError("html_dependency_path_invalid", "dependency URL does not map to a safe local path")
    return path.as_posix()


def materialization_path_for_target(authored_target: str, resolved_url: str) -> str:
    """Keep safe relative authored paths byte-faithful; isolate all others."""

    target = str(authored_target or "")
    try:
        parsed = urlsplit(target)
    except ValueError:
        parsed = None
    if (
        parsed is not None
        and not parsed.scheme
        and not parsed.netloc
        and not parsed.query
        and not parsed.fragment
        and _safe_authored_relative_target(parsed.path)
    ):
        return parsed.path
    return local_path_for_url(resolved_url)


def _bundle_resources(
    bundle: HTMLSourceBundle,
    storage: HTMLSourceBundleStorage,
) -> dict[str, tuple[HTMLSourceDependency, bytes]]:
    resources: dict[str, tuple[HTMLSourceDependency, bytes]] = {}
    for dependency in bundle.dependencies:
        if dependency.availability != "available":
            continue
        payload = storage.read_dependency(
            artifact_digest=dependency.artifact_digest,
            media_type=dependency.media_type,
            size=dependency.size,
        )
        if len(payload) != dependency.size or hashlib.sha256(payload).hexdigest() != dependency.artifact_digest:
            raise HTMLSourceBundleError(
                "html_bundle_materialization_failed", "bundle asset identity does not match its manifest"
            )
        current = resources.get(dependency.materialization_path)
        if current is not None:
            previous, previous_payload = current
            if (
                previous.artifact_digest != dependency.artifact_digest
                or previous.media_type != dependency.media_type
                or previous.size != dependency.size
                or previous_payload != payload
            ):
                raise HTMLSourceBundleError(
                    "html_bundle_path_collision", "bundle assets map different content to one local path"
                )
            continue
        resources[dependency.materialization_path] = (dependency, payload)
    return resources


def _render_materialized_source(
    source_payload: bytes, dependencies: Sequence[HTMLSourceDependency]
) -> tuple[bytes, list[dict[str, Any]]]:
    rewritable = tuple(
        item
        for item in dependencies
        if item.availability == "available" and item.materialization_path != item.authored_target
    )
    if not rewritable:
        return source_payload, []
    try:
        soup = BeautifulSoup(source_payload, "lxml")
    except Exception as exc:
        raise HTMLSourceBundleError(
            "html_bundle_materialization_failed", "HTML source cannot be rendered for materialization"
        ) from exc
    candidates: list[tuple[str, str, str, Any]] = []
    root = soup.select_one("article.ltx_document") or soup
    for node in root.find_all(("object", "img", "source")):
        element = str(node.name).casefold()
        for attribute in ("data",) if element == "object" else ("src", "srcset"):
            if node.has_attr(attribute):
                candidates.append((element, attribute, str(node.get(attribute) or ""), node))
    if len(candidates) < len(dependencies):
        raise HTMLSourceBundleError(
            "html_bundle_source_mismatch", "HTML source no longer matches bundle dependency records"
        )
    rewrites: list[dict[str, Any]] = []
    for dependency in dependencies:
        element, attribute, authored_target, node = candidates[dependency.ordinal]
        if (
            element != dependency.element
            or attribute != dependency.attribute
            or authored_target != dependency.authored_target
        ):
            raise HTMLSourceBundleError(
                "html_bundle_source_mismatch", "HTML source no longer matches bundle dependency records"
            )
        if dependency.availability == "available":
            if dependency.materialization_path != dependency.authored_target:
                node[attribute] = dependency.materialization_path
                rewrites.append(
                    {
                        "ordinal": dependency.ordinal,
                        "element": dependency.element,
                        "attribute": dependency.attribute,
                        "from": dependency.authored_target,
                        "to": dependency.materialization_path,
                    }
                )
    return soup.encode("utf-8", formatter="minimal"), rewrites


def _publish_materialization(
    output: Path,
    *,
    source_payload: bytes,
    manifest_payload: bytes,
    manifest_document: Mapping[str, Any],
    resources: Mapping[str, tuple[HTMLSourceDependency, bytes]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.html-bundle-", dir=output.parent))
    replaced_empty = False
    try:
        atomic_write_bytes(staging / "source.html", source_payload)
        for path, (_dependency, payload) in resources.items():
            target = staging / path
            if target.resolve(strict=False).parent != target.parent.resolve(strict=False):  # pragma: no cover
                raise HTMLSourceBundleError("html_bundle_path_invalid", "resource path escaped staging directory")
            atomic_write_bytes(target, payload)
        atomic_write_bytes(staging / "manifest.json", manifest_payload)
        _verify_materialization_tree(staging, manifest_document)
        _require_available_output(output)
        if output.exists():
            output.rmdir()
            replaced_empty = True
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if replaced_empty and not output.exists():
            output.mkdir()
        raise


def _verify_materialization_tree(output: Path, manifest_document: Mapping[str, Any]) -> None:
    """Verify exact staged bytes before the atomic directory rename publishes them."""

    try:
        manifest = html_source_bundle_export_from_document(manifest_document)
        source = manifest["materialized_source"]
        payload = (output / source["path"]).read_bytes()
        if (
            len(payload) != source["size"]
            or hashlib.sha256(payload).hexdigest() != source["artifact_digest"]
        ):
            raise ValueError("materialized source identity mismatch")
        for resource in manifest["resources"]:
            payload = (output / resource["path"]).read_bytes()
            if (
                len(payload) != resource["size"]
                or hashlib.sha256(payload).hexdigest() != resource["artifact_digest"]
            ):
                raise ValueError("materialized resource identity mismatch")
    except (OSError, TypeError, ValueError) as exc:
        raise HTMLSourceBundleError(
            "html_bundle_materialization_failed", "staged HTML bundle bytes do not match its manifest"
        ) from exc


def _require_available_output(output: Path) -> None:
    if not output.exists():
        return
    if not output.is_dir():
        raise HTMLSourceBundleError(
            "html_bundle_output_exists", f"output path exists and is not a directory: {output}"
        )
    try:
        nonempty = next(output.iterdir(), None) is not None
    except OSError as exc:
        raise HTMLSourceBundleError(
            "html_bundle_output_unreadable", f"output directory cannot be inspected: {output}"
        ) from exc
    if nonempty:
        raise HTMLSourceBundleError(
            "html_bundle_output_not_empty", f"output directory must be absent or empty: {output}"
        )


def _bundle_identity_document(bundle: HTMLSourceBundle) -> dict[str, Any]:
    return {
        "schema_version": HTML_SOURCE_BUNDLE_SCHEMA,
        "primary": _source_artifact_to_document(bundle.primary),
        "requested_url": bundle.requested_url,
        "final_url": bundle.final_url,
        "base_url": bundle.base_url,
        "acquisition_policy": HTMLAcquisitionPolicy.from_document(
            bundle.acquisition_policy
        ).to_document(),
        "dependencies": [_dependency_to_document(item) for item in bundle.dependencies],
        "warnings": [_warning_to_document(item) for item in bundle.warnings],
    }


def _source_artifact_to_document(value: SourceArtifact) -> dict[str, Any]:
    return {
        "source_format": value.source_format.value,
        "artifact_digest": value.artifact_digest,
        "size": value.size,
        "media_type": value.media_type,
        "origin": {
            "kind": value.origin.kind.value,
            "provider": value.origin.provider,
            "locator": value.origin.locator,
            "metadata": dict(value.origin.metadata),
        },
    }


def _source_artifact_from_document(value: Any) -> SourceArtifact:
    if not isinstance(value, Mapping) or set(value) != _PRIMARY_FIELDS:
        raise ValueError("HTML source bundle primary is invalid")
    origin = value.get("origin")
    if not isinstance(origin, Mapping) or set(origin) != _ORIGIN_FIELDS:
        raise ValueError("HTML source bundle primary origin is invalid")
    metadata = origin.get("metadata")
    if not isinstance(metadata, Mapping) or not all(isinstance(key, str) and isinstance(item, str) for key, item in metadata.items()):
        raise ValueError("HTML source bundle primary origin metadata is invalid")
    return SourceArtifact(
        source_format=value.get("source_format"),
        artifact_digest=_required_string(value, "artifact_digest"),
        size=_required_int(value, "size"),
        media_type=_required_string(value, "media_type"),
        origin=SourceOrigin(
            kind=origin.get("kind"),
            provider=str(origin.get("provider") or ""),
            locator=str(origin.get("locator") or ""),
            metadata=dict(metadata),
        ),
    )


def _dependency_to_document(value: HTMLSourceDependency) -> dict[str, Any]:
    return {key: getattr(value, key) for key in _DEPENDENCY_FIELDS}


def _dependency_from_document(value: Any) -> HTMLSourceDependency:
    if not isinstance(value, Mapping) or set(value) != _DEPENDENCY_FIELDS:
        raise ValueError("HTML source dependency record has invalid fields")
    return HTMLSourceDependency(**dict(value))


def _warning_to_document(value: HTMLSourceWarning) -> dict[str, Any]:
    return {key: getattr(value, key) for key in _WARNING_FIELDS}


def _warning_from_document(value: Any) -> HTMLSourceWarning:
    if not isinstance(value, Mapping) or set(value) != _WARNING_FIELDS:
        raise ValueError("HTML source warning record has invalid fields")
    return HTMLSourceWarning(**dict(value))


def _normalize_allowed_origins(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("allowed origins must be a sequence of URLs")
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise ValueError("allowed origins must be a sequence of URLs") from exc
    normalized: list[str] = []
    for item in raw:
        candidate = normalize_https_url(str(item))
        parsed = urlsplit(candidate)
        if parsed.path != "/" or parsed.query:
            raise ValueError("allowed origins must not include a path or query")
        normalized.append(_origin(candidate))
    return tuple(sorted(set(normalized)))


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _normalize_media_type(value: str) -> str:
    normalized = str(value).strip().casefold()
    if not normalized or ";" in normalized or "/" not in normalized:
        raise ValueError("media type must be normalized")
    return normalized


def _safe_materialization_path(value: str) -> bool:
    if not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _safe_authored_relative_target(value: str) -> bool:
    if not value or value.startswith("/") or "\\" in value or "\x00" in value:
        return False
    return all(part != ".." for part in unquote(value).split("/"))


def _is_digest(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in _SHA256_RE for char in value)


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"HTML source bundle {key} must be a nonempty string")
    return item


def _required_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool) or item < 0:
        raise ValueError(f"HTML source bundle {key} must be a nonnegative integer")
    return item


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


__all__ = [
    "DEFAULT_MAX_DEPENDENCY_BYTES",
    "DEFAULT_MAX_DEPENDENCY_COUNT",
    "DEFAULT_MAX_PRIMARY_BYTES",
    "DEFAULT_MAX_REDIRECTS",
    "DEFAULT_MAX_TOTAL_DEPENDENCY_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "HTMLAcquisitionPolicy",
    "HTMLSourceBundle",
    "HTMLSourceBundleCache",
    "HTMLSourceBundleError",
    "HTMLSourceBundleMaterialization",
    "HTMLSourceBundleStorage",
    "HTMLSourceDependency",
    "HTMLSourceWarning",
    "HTML_SOURCE_BUNDLE_CACHE_SCHEMA",
    "HTML_SOURCE_BUNDLE_SCHEMA",
    "HTML_SOURCE_EXPORT_SCHEMA",
    "SUPPORTED_DEPENDENCY_MEDIA_TYPES",
    "dependency_media_type_for_url",
    "html_source_bundle_from_document",
    "html_source_bundle_export_from_document",
    "html_source_bundle_export_to_document",
    "html_source_bundle_to_document",
    "local_path_for_url",
    "materialization_path_for_target",
    "materialize_html_source_bundle",
    "normalize_https_url",
    "SourceRepositoryHTMLSourceBundleStorage",
    "StoredHTMLDependency",
    "verify_html_source_bundle_export",
]
