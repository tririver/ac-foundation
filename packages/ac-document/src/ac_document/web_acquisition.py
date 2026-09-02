"""Explicit, bounded public-HTTPS acquisition for HTML source bundles."""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from ._cache_root import resolve_cache_root
from .html_bundle import (
    HTMLAcquisitionPolicy,
    HTMLSourceBundle,
    HTMLSourceBundleCache,
    HTMLSourceBundleError,
    HTMLSourceBundleStorage,
    HTMLSourceDependency,
    HTMLSourceWarning,
    SUPPORTED_DEPENDENCY_MEDIA_TYPES,
    SourceRepositoryHTMLSourceBundleStorage,
    StoredHTMLDependency,
    dependency_media_type_for_url,
    materialization_path_for_target,
    normalize_https_url,
)
from .source_repository import SourceRepository, SourceRepositoryError
from .sources import SourceArtifact, SourceFormat, SourceOrigin, SourceOriginKind


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_PRIMARY_MEDIA_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_WARNING_MESSAGES = {
    "html_base_url_invalid": "ignored an invalid HTML base URL",
    "html_base_origin_invalid": "ignored an HTML base URL outside the final document origin",
    "html_dependency_base_invalid": "dependency acquisition is unavailable because the HTML base URL is invalid",
    "html_dependency_cache_write_failed": "dependency storage write failed",
    "html_dependency_count_limit": "ignored dependency references beyond the configured count limit",
    "html_dependency_declared_media_type_mismatch": "authored dependency type does not match a supported response media type",
    "html_dependency_extension_unsupported": "dependency URL has an unsupported media extension",
    "html_dependency_fetch_failed": "dependency request failed",
    "html_dependency_media_type_invalid": "dependency response media type is unsupported",
    "html_dependency_media_type_mismatch": "dependency URL extension does not match the response media type",
    "html_dependency_origin_invalid": "dependency URL is outside the final document origin",
    "html_dependency_path_collision": "dependency materialization path collides with another URL",
    "html_dependency_path_invalid": "dependency URL does not map to a safe materialization path",
    "html_dependency_redirect_limit": "dependency redirect limit was exceeded",
    "html_dependency_redirect_invalid": "dependency redirect leaves the allowed final document origin",
    "html_dependency_srcset_unsupported": "srcset dependency acquisition is unsupported",
    "html_dependency_storage_invalid": "dependency storage returned an inconsistent content identity",
    "html_dependency_too_large": "dependency response exceeds the configured byte limit",
    "html_dependency_total_too_large": "dependency responses exceed the configured total byte limit",
    "remote_content_type_invalid": "remote response did not provide a valid media type",
    "remote_origin_not_allowed": "remote URL is outside the allowed origins",
    "remote_target_not_public": "remote URL resolves to a non-public address",
    "remote_url_invalid": "remote URL is invalid",
}


@dataclass(frozen=True)
class WebResponse:
    """A complete no-redirect HTTP response supplied by an injectable transport."""

    url: str
    status: int
    headers: Mapping[str, str]
    body: bytes

    def __post_init__(self) -> None:
        if isinstance(self.status, bool) or not isinstance(self.status, int):
            raise TypeError("web response status must be an integer")
        if not isinstance(self.body, bytes):
            raise TypeError("web response body must be bytes")
        normalized_headers = {
            str(key).casefold(): str(value)
            for key, value in self.headers.items()
        }
        object.__setattr__(self, "headers", normalized_headers)


class HTMLWebTransport(Protocol):
    def fetch(
        self,
        url: str,
        *,
        timeout_seconds: float,
        maximum_bytes: int,
        validated_addresses: Sequence[str],
    ) -> WebResponse: ...


class AddressResolver(Protocol):
    def __call__(self, hostname: str) -> Sequence[str]: ...


class StdlibHTTPSWebTransport:
    """HTTPS-only transport which never follows redirects automatically."""

    def fetch(
        self,
        url: str,
        *,
        timeout_seconds: float,
        maximum_bytes: int,
        validated_addresses: Sequence[str],
    ) -> WebResponse:
        parsed = urlsplit(url)
        host = parsed.hostname
        if host is None:  # pragma: no cover - validated by caller
            raise HTMLSourceBundleError("remote_url_invalid", "remote URL is invalid")
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        connection = _pinned_https_connection(
            host, validated_addresses, timeout_seconds
        )
        try:
            connection.request(
                "GET",
                path,
                headers={"Accept": "text/html,application/xhtml+xml,image/*;q=0.8", "User-Agent": "ac-document/2"},
            )
            response = connection.getresponse()
            headers = {key.casefold(): value for key, value in response.getheaders()}
            if response.status in _REDIRECT_STATUSES or not 200 <= response.status < 300:
                return WebResponse(
                    url=url,
                    status=response.status,
                    headers=headers,
                    body=b"",
                )
            content_length = headers.get("content-length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError as exc:
                    raise HTMLSourceBundleError(
                        "remote_content_length_invalid", "remote response has an invalid Content-Length"
                    ) from exc
                if declared_size < 0:
                    raise HTMLSourceBundleError(
                        "remote_content_length_invalid", "remote response has an invalid Content-Length"
                    )
                if declared_size > maximum_bytes:
                    raise HTMLSourceBundleError(
                        "remote_response_too_large", "remote response exceeds the configured byte limit"
                    )
            chunks: list[bytes] = []
            received = 0
            while True:
                chunk = response.read(min(1024 * 1024, maximum_bytes + 1))
                if not chunk:
                    break
                received += len(chunk)
                if received > maximum_bytes:
                    raise HTMLSourceBundleError(
                        "remote_response_too_large", "remote response exceeds the configured byte limit"
                    )
                chunks.append(chunk)
            return WebResponse(
                url=url,
                status=response.status,
                headers=headers,
                body=b"".join(chunks),
            )
        except HTMLSourceBundleError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            raise HTMLSourceBundleError("remote_fetch_failed", "remote request failed") from exc
        finally:
            connection.close()


@dataclass(frozen=True)
class _Candidate:
    ordinal: int
    element: str
    attribute: str
    authored_target: str
    declared_media_type: str


@dataclass(frozen=True)
class _FetchedDependency:
    request_url: str
    resolved_url: str = ""
    stored: StoredHTMLDependency | None = None
    error_code: str = ""


@dataclass(frozen=True)
class _ValidatedTarget:
    url: str
    addresses: tuple[str, ...]


class HTMLSourceAcquisitionService:
    """Explicit acquisition facade; local document APIs never instantiate it."""

    def __init__(
        self,
        *,
        cache_root: str | Path | None = None,
        repository: SourceRepository | None = None,
        storage: HTMLSourceBundleStorage | None = None,
        transport: HTMLWebTransport | None = None,
        resolver: AddressResolver | None = None,
    ) -> None:
        root = resolve_cache_root(cache_root, repository=repository)
        self.repository = repository or SourceRepository(root)
        self.storage = storage or SourceRepositoryHTMLSourceBundleStorage(self.repository)
        self.cache = HTMLSourceBundleCache(root, self.storage)
        self.transport = transport or StdlibHTTPSWebTransport()
        self.resolver = resolver or _resolve_addresses

    def acquire(
        self,
        url: str,
        *,
        policy: HTMLAcquisitionPolicy | None = None,
    ) -> HTMLSourceBundle:
        """Fetch one primary URL and cache a verified bundle explicitly."""

        if not isinstance(self.storage, SourceRepositoryHTMLSourceBundleStorage):
            raise HTMLSourceBundleError(
                "html_direct_acquisition_storage_unsupported",
                "direct acquisition requires the default SourceRepository storage adapter",
            )
        resolved_policy = policy or HTMLAcquisitionPolicy()
        requested_target = self._validate_public_target(url, resolved_policy)
        requested_url = requested_target.url
        request_key = self.cache.request_key(requested_url, resolved_policy.to_document())
        cached = self.cache.lookup(request_key)
        if cached is not None:
            return cached
        response, final_url = self._fetch_following(
            requested_url,
            policy=resolved_policy,
            maximum_bytes=resolved_policy.max_primary_bytes,
            too_large_code="html_primary_too_large",
            initial_target=requested_target,
        )
        media_type = _response_media_type(response)
        if media_type not in _PRIMARY_MEDIA_TYPES:
            raise HTMLSourceBundleError(
                "html_primary_media_type_invalid", "primary response media type must be HTML"
            )
        primary = self.repository.store_bytes(
            response.body,
            source_format=SourceFormat.HTML,
            media_type=media_type,
            origin=SourceOrigin(
                SourceOriginKind.REMOTE_PROVIDER,
                provider="web",
                locator=final_url,
                metadata={"requested_url": requested_url},
            ),
        )
        bundle = self.acquire_dependencies(
            primary,
            document_url=final_url,
            requested_url=requested_url,
            policy=resolved_policy,
            storage=self.storage,
        )
        self.cache.store(bundle, request_key=request_key)
        return bundle

    def acquire_dependencies(
        self,
        primary: SourceArtifact,
        *,
        document_url: str,
        requested_url: str | None = None,
        policy: HTMLAcquisitionPolicy | None = None,
        storage: HTMLSourceBundleStorage | None = None,
    ) -> HTMLSourceBundle:
        """Acquire only dependencies for a caller-supplied verified HTML primary."""

        if primary.source_format is not SourceFormat.HTML:
            raise HTMLSourceBundleError("html_primary_invalid", "dependency acquisition requires an HTML primary")
        if primary.media_type not in _PRIMARY_MEDIA_TYPES:
            raise HTMLSourceBundleError("html_primary_media_type_invalid", "primary media type must be HTML")
        resolved_policy = policy or HTMLAcquisitionPolicy()
        active_storage = storage or self.storage
        final_url = self._validate_public_url(document_url, resolved_policy)
        resolved_requested = self._validate_public_url(requested_url or final_url, resolved_policy)
        try:
            payload = active_storage.read_primary(primary)
        except (OSError, SourceRepositoryError, ValueError) as exc:
            raise HTMLSourceBundleError(
                "html_primary_unavailable", "verified primary bytes are unavailable from storage"
            ) from exc
        if len(payload) != primary.size or _digest(payload) != primary.artifact_digest:
            raise HTMLSourceBundleError(
                "html_primary_invalid", "verified primary identity does not match storage"
            )
        base_url, candidates, extraction_warnings, base_failure = self._extract_dependencies(
            payload, final_url, resolved_policy
        )
        warnings: list[HTMLSourceWarning] = list(extraction_warnings)
        dependencies: list[HTMLSourceDependency] = []
        fetched: dict[str, _FetchedDependency] = {}
        paths: dict[str, str] = {}
        counted_digests: set[str] = set()
        total_bytes = 0
        final_origin = _origin(final_url)
        dependency_origin = (
            final_origin if resolved_policy.same_origin_dependencies else None
        )
        for candidate in candidates:
            failure_request_url = ""
            failure_resolved_url = ""
            if base_failure is not None:
                dependency, warning = _unavailable(candidate, base_failure)
                dependencies.append(dependency)
                warnings.append(warning)
                continue
            if candidate.attribute == "srcset":
                dependency, warning = _unavailable(
                    candidate, "html_dependency_srcset_unsupported"
                )
                dependencies.append(dependency)
                warnings.append(warning)
                continue
            try:
                if candidate.declared_media_type and candidate.declared_media_type not in SUPPORTED_DEPENDENCY_MEDIA_TYPES:
                    raise HTMLSourceBundleError(
                        "html_dependency_declared_media_type_mismatch",
                        _WARNING_MESSAGES["html_dependency_declared_media_type_mismatch"],
                    )
                request_url = self._dependency_url(
                    base_url, candidate.authored_target, resolved_policy, final_origin
                )
                failure_request_url = request_url
                materialization_path = materialization_path_for_target(
                    candidate.authored_target, request_url
                )
                prior_url = paths.setdefault(materialization_path, request_url)
                if prior_url != request_url:
                    raise HTMLSourceBundleError(
                        "html_dependency_path_collision",
                        _WARNING_MESSAGES["html_dependency_path_collision"],
                    )
                result = fetched.get(request_url)
                if result is None:
                    result, total_bytes = self._fetch_dependency(
                        request_url,
                        policy=resolved_policy,
                        required_origin=dependency_origin,
                        storage=active_storage,
                        counted_digests=counted_digests,
                        total_bytes=total_bytes,
                    )
                    fetched[request_url] = result
                failure_resolved_url = result.resolved_url
                if result.stored is None:
                    dependency, warning = _unavailable(
                        candidate,
                        result.error_code or "html_dependency_fetch_failed",
                        request_url=result.request_url,
                        resolved_url=result.resolved_url,
                    )
                    dependencies.append(dependency)
                    warnings.append(warning)
                    continue
                if candidate.declared_media_type and candidate.declared_media_type != result.stored.media_type:
                    raise HTMLSourceBundleError(
                        "html_dependency_declared_media_type_mismatch",
                        _WARNING_MESSAGES["html_dependency_declared_media_type_mismatch"],
                    )
                dependencies.append(
                    HTMLSourceDependency(
                        ordinal=candidate.ordinal,
                        element=candidate.element,
                        attribute=candidate.attribute,
                        authored_target=candidate.authored_target,
                        request_url=result.request_url,
                        resolved_url=result.resolved_url,
                        declared_media_type=candidate.declared_media_type,
                        availability="available",
                        materialization_path=materialization_path,
                        media_type=result.stored.media_type,
                        artifact_digest=result.stored.artifact_digest,
                        size=result.stored.size,
                    )
                )
            except HTMLSourceBundleError as exc:
                dependency, warning = _unavailable(
                    candidate,
                    exc.code if exc.code in _WARNING_MESSAGES else "html_dependency_fetch_failed",
                    request_url=failure_request_url,
                    resolved_url=failure_resolved_url,
                )
                dependencies.append(dependency)
                warnings.append(warning)
        return HTMLSourceBundle(
            primary=primary,
            requested_url=resolved_requested,
            final_url=final_url,
            base_url=base_url,
            acquisition_policy=resolved_policy.to_document(),
            dependencies=tuple(dependencies),
            warnings=tuple(warnings),
        )

    def _extract_dependencies(
        self,
        payload: bytes,
        final_url: str,
        policy: HTMLAcquisitionPolicy,
    ) -> tuple[str, tuple[_Candidate, ...], tuple[HTMLSourceWarning, ...], str | None]:
        try:
            soup = BeautifulSoup(payload, "lxml")
        except Exception:
            return (
                final_url,
                (),
                (HTMLSourceWarning("html_dependency_parse_failed", "HTML dependency extraction failed"),),
                None,
            )
        base_url = final_url
        base_failure: str | None = None
        base = soup.find("base", href=True)
        if base is not None:
            raw_base = str(base.get("href") or "")
            try:
                candidate = normalize_https_url(urljoin(final_url, raw_base), allowed_origins=policy.allowed_origins)
                if policy.same_origin_dependencies and _origin(candidate) != _origin(final_url):
                    raise HTMLSourceBundleError(
                        "html_base_origin_invalid", _WARNING_MESSAGES["html_base_origin_invalid"]
                    )
                base_url = self._validate_public_url(candidate, policy)
            except HTMLSourceBundleError:
                base_failure = "html_dependency_base_invalid"
        candidates: list[_Candidate] = []
        overflow = 0
        root = soup.select_one("article.ltx_document") or soup
        for node in root.find_all(("object", "img", "source")):
            element = str(node.name).casefold()
            attributes = ("data",) if element == "object" else ("src", "srcset")
            declared = str(node.get("type") or "").split(";", 1)[0].strip().casefold()
            for attribute in attributes:
                if not node.has_attr(attribute):
                    continue
                if len(candidates) >= policy.max_dependency_count:
                    overflow += 1
                    continue
                candidates.append(
                    _Candidate(
                        ordinal=len(candidates),
                        element=element,
                        attribute=attribute,
                        authored_target=str(node.get(attribute) or ""),
                        declared_media_type=declared,
                    )
                )
        if overflow:
            extraction_warnings = (
                HTMLSourceWarning(
                    "html_dependency_count_limit",
                    (
                        f"ignored {overflow} dependency references beyond the limit of "
                        f"{policy.max_dependency_count}"
                    ),
                ),
            )
        else:
            extraction_warnings = ()
        return base_url, tuple(candidates), extraction_warnings, base_failure

    def _dependency_url(
        self,
        base_url: str,
        target: str,
        policy: HTMLAcquisitionPolicy,
        final_origin: str,
    ) -> str:
        if not target.strip():
            raise HTMLSourceBundleError("remote_url_invalid", _WARNING_MESSAGES["remote_url_invalid"])
        candidate = normalize_https_url(
            urljoin(base_url, target), allowed_origins=policy.allowed_origins
        )
        if policy.same_origin_dependencies and _origin(candidate) != final_origin:
            raise HTMLSourceBundleError(
                "html_dependency_origin_invalid", _WARNING_MESSAGES["html_dependency_origin_invalid"]
            )
        return self._validate_public_url(candidate, policy)

    def _fetch_dependency(
        self,
        request_url: str,
        *,
        policy: HTMLAcquisitionPolicy,
        required_origin: str | None,
        storage: HTMLSourceBundleStorage,
        counted_digests: set[str],
        total_bytes: int,
    ) -> tuple[_FetchedDependency, int]:
        resolved_url = request_url
        try:
            response, resolved_url = self._fetch_following(
                request_url,
                policy=policy,
                maximum_bytes=policy.max_dependency_bytes,
                too_large_code="html_dependency_too_large",
                required_origin=required_origin,
                is_dependency=True,
            )
            media_type = _response_media_type(response)
            if media_type not in SUPPORTED_DEPENDENCY_MEDIA_TYPES:
                raise HTMLSourceBundleError(
                    "html_dependency_media_type_invalid", _WARNING_MESSAGES["html_dependency_media_type_invalid"]
                )
            expected = dependency_media_type_for_url(resolved_url) or dependency_media_type_for_url(request_url)
            if expected is not None and expected != media_type:
                raise HTMLSourceBundleError(
                    "html_dependency_media_type_mismatch", _WARNING_MESSAGES["html_dependency_media_type_mismatch"]
                )
            digest = _digest(response.body)
            if digest not in counted_digests and total_bytes + len(response.body) > policy.max_total_dependency_bytes:
                raise HTMLSourceBundleError(
                    "html_dependency_total_too_large", _WARNING_MESSAGES["html_dependency_total_too_large"]
                )
            try:
                stored = storage.store_dependency(response.body, media_type=media_type)
            except Exception as exc:
                raise HTMLSourceBundleError(
                    "html_dependency_cache_write_failed",
                    _WARNING_MESSAGES["html_dependency_cache_write_failed"],
                ) from exc
            if (
                stored.artifact_digest != digest
                or stored.media_type != media_type
                or stored.size != len(response.body)
            ):
                raise HTMLSourceBundleError(
                    "html_dependency_storage_invalid", _WARNING_MESSAGES["html_dependency_storage_invalid"]
                )
            if digest not in counted_digests:
                counted_digests.add(digest)
                total_bytes += len(response.body)
            return _FetchedDependency(request_url, resolved_url, stored), total_bytes
        except HTMLSourceBundleError as exc:
            code = exc.code if exc.code in _WARNING_MESSAGES else "html_dependency_fetch_failed"
            return _FetchedDependency(
                request_url=request_url,
                resolved_url=resolved_url,
                error_code=code,
            ), total_bytes
        except (OSError, SourceRepositoryError, ValueError):
            return _FetchedDependency(
                request_url=request_url,
                resolved_url=resolved_url,
                error_code="html_dependency_fetch_failed",
            ), total_bytes

    def _fetch_following(
        self,
        url: str,
        *,
        policy: HTMLAcquisitionPolicy,
        maximum_bytes: int,
        too_large_code: str,
        required_origin: str | None = None,
        initial_target: _ValidatedTarget | None = None,
        is_dependency: bool = False,
    ) -> tuple[WebResponse, str]:
        current = initial_target or self._validate_public_target(
            url, policy, required_origin=required_origin
        )
        for redirects in range(policy.max_redirects + 1):
            try:
                response = self.transport.fetch(
                    current.url,
                    timeout_seconds=policy.timeout_seconds,
                    maximum_bytes=maximum_bytes,
                    validated_addresses=current.addresses,
                )
            except HTMLSourceBundleError as exc:
                if exc.code == "remote_response_too_large":
                    raise HTMLSourceBundleError(
                        too_large_code,
                        "remote response exceeds the configured byte limit",
                    ) from exc
                raise
            response_url = normalize_https_url(
                response.url, allowed_origins=policy.allowed_origins
            )
            if required_origin is not None and _origin(response_url) != required_origin:
                raise HTMLSourceBundleError(
                    "html_dependency_origin_invalid", _WARNING_MESSAGES["html_dependency_origin_invalid"]
                )
            if response_url != current.url:
                raise HTMLSourceBundleError(
                    "remote_response_url_invalid", "remote response URL does not match its request"
                )
            if response.status in _REDIRECT_STATUSES:
                if redirects == policy.max_redirects:
                    code = (
                        "html_dependency_redirect_limit"
                        if is_dependency
                        else "html_primary_redirect_limit"
                    )
                    raise HTMLSourceBundleError(code, "remote redirect limit was exceeded")
                location = response.headers.get("location", "").strip()
                if not location:
                    raise HTMLSourceBundleError("remote_url_invalid", "remote redirect location is invalid")
                try:
                    current = self._validate_public_target(
                        urljoin(current.url, location),
                        policy,
                        required_origin=required_origin,
                    )
                except HTMLSourceBundleError as exc:
                    if required_origin is not None and exc.code in {
                        "html_dependency_origin_invalid",
                        "remote_origin_not_allowed",
                    }:
                        raise HTMLSourceBundleError(
                            "html_dependency_redirect_invalid",
                            _WARNING_MESSAGES["html_dependency_redirect_invalid"],
                        ) from exc
                    raise
                continue
            if not 200 <= response.status < 300:
                raise HTMLSourceBundleError("remote_fetch_failed", "remote response status is unsuccessful")
            if len(response.body) > maximum_bytes:
                raise HTMLSourceBundleError(
                    too_large_code, "remote response exceeds the configured byte limit"
                )
            return response, current.url
        raise AssertionError("redirect loop did not terminate")  # pragma: no cover

    def _validate_public_url(
        self,
        url: str,
        policy: HTMLAcquisitionPolicy,
        *,
        required_origin: str | None = None,
    ) -> str:
        return self._validate_public_target(
            url, policy, required_origin=required_origin
        ).url

    def _validate_public_target(
        self,
        url: str,
        policy: HTMLAcquisitionPolicy,
        *,
        required_origin: str | None = None,
    ) -> _ValidatedTarget:
        normalized = normalize_https_url(url, allowed_origins=policy.allowed_origins)
        if required_origin is not None and _origin(normalized) != required_origin:
            raise HTMLSourceBundleError(
                "html_dependency_origin_invalid", _WARNING_MESSAGES["html_dependency_origin_invalid"]
            )
        host = urlsplit(normalized).hostname
        if host is None:  # pragma: no cover - normalize_https_url guarantees it
            raise HTMLSourceBundleError("remote_url_invalid", _WARNING_MESSAGES["remote_url_invalid"])
        try:
            addresses = tuple(self.resolver(host))
        except (OSError, ValueError):
            raise HTMLSourceBundleError("remote_host_unresolved", "remote hostname cannot be resolved") from None
        if not addresses:
            raise HTMLSourceBundleError("remote_host_unresolved", "remote hostname cannot be resolved")
        for address in addresses:
            try:
                parsed = ipaddress.ip_address(str(address))
            except ValueError:
                raise HTMLSourceBundleError("remote_host_unresolved", "remote hostname resolved to an invalid address") from None
            if not parsed.is_global:
                raise HTMLSourceBundleError(
                    "remote_target_not_public", _WARNING_MESSAGES["remote_target_not_public"]
                )
        return _ValidatedTarget(normalized, tuple(str(item) for item in addresses))


def _unavailable(
    candidate: _Candidate,
    code: str,
    *,
    request_url: str = "",
    resolved_url: str = "",
) -> tuple[HTMLSourceDependency, HTMLSourceWarning]:
    message = _WARNING_MESSAGES.get(code, _WARNING_MESSAGES["html_dependency_fetch_failed"])
    dependency = HTMLSourceDependency(
        ordinal=candidate.ordinal,
        element=candidate.element,
        attribute=candidate.attribute,
        authored_target=candidate.authored_target,
        request_url=request_url,
        resolved_url=resolved_url,
        declared_media_type=candidate.declared_media_type,
        availability="unavailable",
        error_code=code,
        error_message=message,
    )
    return dependency, HTMLSourceWarning(
        code=code,
        message=message,
        dependency_ordinal=candidate.ordinal,
        element=candidate.element,
        attribute=candidate.attribute,
        authored_target=candidate.authored_target,
    )


def _response_media_type(response: WebResponse) -> str:
    value = response.headers.get("content-type", "").split(";", 1)[0].strip().casefold()
    if not value or "/" not in value:
        raise HTMLSourceBundleError("remote_content_type_invalid", "remote response media type is invalid")
    return value


def _pinned_https_connection(
    hostname: str, addresses: Sequence[str], timeout_seconds: float
) -> http.client.HTTPSConnection:
    """Connect to an already validated address while keeping TLS host validation."""

    context = ssl.create_default_context()
    failure: OSError | None = None
    for address in addresses:
        raw_socket = None
        try:
            raw_socket = socket.create_connection((address, 443), timeout_seconds)
            secure_socket = context.wrap_socket(raw_socket, server_hostname=hostname)
            connection = http.client.HTTPSConnection(
                hostname, 443, timeout=timeout_seconds, context=context
            )
            connection.sock = secure_socket
            return connection
        except OSError as exc:
            failure = exc
            if raw_socket is not None:
                raw_socket.close()
    raise HTMLSourceBundleError("remote_fetch_failed", "remote request failed") from failure


def _resolve_addresses(hostname: str) -> tuple[str, ...]:
    records = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    return tuple(sorted({str(record[4][0]) for record in records}))


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _digest(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "AddressResolver",
    "HTMLSourceAcquisitionService",
    "HTMLWebTransport",
    "StdlibHTTPSWebTransport",
    "WebResponse",
]
