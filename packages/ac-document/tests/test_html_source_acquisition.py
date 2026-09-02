from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import ac_document.registry as document_registry
from ac_document import (
    HTMLAcquisitionPolicy,
    HTMLSourceAcquisitionService,
    HTMLSourceBundle,
    HTMLSourceBundleCache,
    HTMLSourceBundleError,
    HTMLSourceDependency,
    HTMLSourceWarning,
    StdlibHTTPSWebTransport,
    StoredHTMLDependency,
    SourceFormat,
    SourceArtifact,
    SourceOrigin,
    SourceOriginKind,
    SourceRepository,
    SourceRepositoryHTMLSourceBundleStorage,
    WebResponse,
    html_source_bundle_export_from_document,
    html_source_bundle_export_to_document,
    html_source_bundle_from_document,
    html_source_bundle_to_document,
    materialize_html_source_bundle,
    verify_html_source_bundle_export,
)
from ac_document.cli import main
from ac_document.registry import OPERATION_REGISTRY


@dataclass
class _Transport:
    responses: dict[str, WebResponse]
    calls: list[str]
    validated_address_sets: list[tuple[str, ...]] = field(default_factory=list)

    def fetch(
        self,
        url: str,
        *,
        timeout_seconds: float,
        maximum_bytes: int,
        validated_addresses: tuple[str, ...],
    ) -> WebResponse:
        del timeout_seconds, maximum_bytes
        self.calls.append(url)
        self.validated_address_sets.append(validated_addresses)
        return self.responses[url]


def _resolver(*, private: set[str] = set()):
    def resolve(host: str) -> tuple[str, ...]:
        return ("127.0.0.1",) if host in private else ("1.1.1.1",)

    return resolve


def _response(url: str, body: bytes, media_type: str, status: int = 200, **headers: str) -> WebResponse:
    return WebResponse(
        url=url,
        status=status,
        headers={"content-type": media_type, **headers},
        body=body,
    )


def _service(tmp_path: Path, transport: _Transport, *, private: set[str] = set()):
    repository = SourceRepository(tmp_path / "cache")
    return (
        repository,
        HTMLSourceAcquisitionService(
            repository=repository,
            transport=transport,
            resolver=_resolver(private=private),
        ),
    )


def test_acquires_public_html_and_replays_verified_cache(tmp_path: Path) -> None:
    article = "https://public.example/article.html"
    image = "https://public.example/images/figure.png"
    transport = _Transport(
        {
            article: _response(
                article,
                b'<img src="images/figure.png"><img src="images/figure.png">',
                "text/html",
            ),
            image: _response(image, b"PNG", "image/png"),
        },
        [],
    )
    repository, service = _service(tmp_path, transport)

    first = service.acquire(article)
    assert first.requested_url == article
    assert first.final_url == article
    assert first.primary.source_format is SourceFormat.HTML
    assert [item.availability for item in first.dependencies] == ["available", "available"]
    assert {item.materialization_path for item in first.dependencies} == {"images/figure.png"}
    assert transport.calls == [article, image]
    assert transport.validated_address_sets == [("1.1.1.1",), ("1.1.1.1",)]

    replay = HTMLSourceAcquisitionService(
        repository=repository,
        transport=_Transport({}, []),
        resolver=_resolver(),
    ).acquire(article)
    assert replay.bundle_digest == first.bundle_digest


def test_bundle_codec_and_cache_detect_corrupt_repository_asset(tmp_path: Path) -> None:
    repository = SourceRepository(tmp_path / "cache")
    primary = repository.store_bytes(
        b"<p>primary</p>",
        source_format=SourceFormat.HTML,
        media_type="text/html",
        origin=SourceOrigin(SourceOriginKind.REMOTE_PROVIDER, provider="web", locator="https://public.example/a.html"),
    )
    asset = repository.store_asset_bytes(b"PNG", media_type="image/png")
    bundle = HTMLSourceBundle(
        primary=primary,
        requested_url="https://public.example/a.html",
        final_url="https://public.example/a.html",
        base_url="https://public.example/a.html",
        acquisition_policy=HTMLAcquisitionPolicy().to_document(),
        dependencies=(
            HTMLSourceDependency(
                ordinal=0,
                element="img",
                attribute="src",
                authored_target="figure.png",
                request_url="https://public.example/figure.png",
                resolved_url="https://public.example/figure.png",
                availability="available",
                materialization_path="figure.png",
                media_type=asset.media_type,
                artifact_digest=asset.artifact_digest,
                size=asset.size,
            ),
        ),
    )

    document = html_source_bundle_to_document(bundle)
    assert html_source_bundle_from_document(document).bundle_digest == bundle.bundle_digest
    storage = SourceRepositoryHTMLSourceBundleStorage(repository)
    cache = HTMLSourceBundleCache(repository.root, storage)
    cache.store(bundle, request_key=cache.request_key(bundle.requested_url, bundle.acquisition_policy))
    assert cache.load(bundle.bundle_digest).bundle_digest == bundle.bundle_digest

    asset_path = repository._asset_object_dir(asset.artifact_digest) / "asset"  # noqa: SLF001
    asset_path.write_bytes(b"corrupt")
    with pytest.raises(HTMLSourceBundleError, match="cache") as error:
        cache.load(bundle.bundle_digest)
    assert error.value.code == "html_bundle_cache_corrupt"


def test_rejects_initial_and_redirect_private_targets_before_request(tmp_path: Path) -> None:
    private = "https://private.example/article.html"
    transport = _Transport({}, [])
    _repository, service = _service(tmp_path, transport, private={"private.example"})

    with pytest.raises(HTMLSourceBundleError) as error:
        service.acquire(private)
    assert error.value.code == "remote_target_not_public"
    assert transport.calls == []

    article = "https://public.example/article.html"
    redirect = "https://private.example/next.html"
    transport = _Transport(
        {article: _response(article, b"", "text/html", 302, location=redirect)}, []
    )
    _repository, service = _service(tmp_path / "redirect", transport, private={"private.example"})
    with pytest.raises(HTMLSourceBundleError) as error:
        service.acquire(article)
    assert error.value.code == "remote_target_not_public"
    assert transport.calls == [article]


def test_dependency_boundary_limits_mime_and_path_collisions_are_structured_warnings(tmp_path: Path) -> None:
    article = "https://public.example/article.html"
    figure = "https://public.example/assets/figure.png?version=1"
    collision = "https://public.example/assets/figure.png?version=2"
    other_origin = "https://other.example/figure.png"
    transport = _Transport(
        {
            article: _response(
                article,
                (
                    b'<img src="https://public.example/assets/figure.png?version=1">'
                    b'<img src="https://public.example/assets/figure.png?version=2">'
                    b'<img src="https://other.example/figure.png">'
                    b'<img src="bad.jpg">'
                ),
                "text/html",
            ),
            figure: _response(figure, b"PNG", "image/png"),
            collision: _response(collision, b"PNG", "image/png"),
            "https://public.example/bad.jpg": _response(
                "https://public.example/bad.jpg", b"not jpeg", "image/png"
            ),
        },
        [],
    )
    _repository, service = _service(tmp_path, transport)

    bundle = service.acquire(article)
    assert [item.availability for item in bundle.dependencies] == [
        "available",
        "available",
        "unavailable",
        "unavailable",
    ]
    assert {warning.code for warning in bundle.warnings} == {
        "html_dependency_origin_invalid",
        "html_dependency_media_type_mismatch",
    }
    assert transport.calls == [article, figure, collision, "https://public.example/bad.jpg"]
    assert other_origin not in transport.calls


def test_dependency_size_limit_is_partial_and_never_stores_fabricated_bytes(tmp_path: Path) -> None:
    article = "https://public.example/article.html"
    image = "https://public.example/large.png"
    transport = _Transport(
        {
            article: _response(article, b'<img src="large.png">', "text/html"),
            image: _response(image, b"TOO-LARGE", "image/png"),
        },
        [],
    )
    repository, service = _service(tmp_path, transport)
    policy = HTMLAcquisitionPolicy(max_dependency_bytes=3, max_total_dependency_bytes=3)

    bundle = service.acquire(article, policy=policy)
    dependency = bundle.dependencies[0]
    assert dependency.availability == "unavailable"
    assert dependency.error_code == "html_dependency_too_large"
    assert list((repository.root / "source-repository" / "v1" / "asset").rglob("asset")) == []


def test_query_bearing_absolute_dependencies_materialize_to_distinct_paths(
    tmp_path: Path,
) -> None:
    article = "https://public.example/article.html"
    first = "https://public.example/plot.png?v=1"
    second = "https://public.example/plot.png?v=2"
    transport = _Transport(
        {
            article: _response(
                article,
                (
                    b'<img src="https://public.example/plot.png?v=1">'
                    b'<img src="https://public.example/plot.png?v=2">'
                ),
                "text/html",
            ),
            first: _response(first, b"first", "image/png"),
            second: _response(second, b"second", "image/png"),
        },
        [],
    )
    repository, service = _service(tmp_path, transport)
    bundle = service.acquire(article)
    paths = [item.materialization_path for item in bundle.dependencies]
    assert len(set(paths)) == 2
    assert all(path.startswith("assets/public.example/plot.") for path in paths)
    assert all(path.endswith(".png") for path in paths)

    output = tmp_path / "bundle"
    materialize_html_source_bundle(
        bundle, SourceRepositoryHTMLSourceBundleStorage(repository), output
    )
    assert [(output / path).read_bytes() for path in paths] == [b"first", b"second"]
    rendered = (output / "source.html").read_text(encoding="utf-8")
    assert all(path in rendered for path in paths)


def test_true_materialization_path_collisions_still_fail_closed(tmp_path: Path) -> None:
    repository = SourceRepository(tmp_path / "cache")
    primary = repository.store_bytes(
        b'<img src="one.png"><img src="two.png">',
        source_format=SourceFormat.HTML,
        media_type="text/html",
        origin=SourceOrigin(SourceOriginKind.REMOTE_PROVIDER, provider="web", locator="https://public.example/article.html"),
    )
    first = repository.store_asset_bytes(b"first", media_type="image/png")
    second = repository.store_asset_bytes(b"second", media_type="image/png")
    bundle = HTMLSourceBundle(
        primary=primary,
        requested_url="https://public.example/article.html",
        final_url="https://public.example/article.html",
        base_url="https://public.example/article.html",
        acquisition_policy=HTMLAcquisitionPolicy().to_document(),
        dependencies=(
            HTMLSourceDependency(
                ordinal=0,
                element="img",
                attribute="src",
                authored_target="one.png",
                request_url="https://public.example/one.png",
                resolved_url="https://public.example/one.png",
                availability="available",
                materialization_path="shared.png",
                media_type=first.media_type,
                artifact_digest=first.artifact_digest,
                size=first.size,
            ),
            HTMLSourceDependency(
                ordinal=1,
                element="img",
                attribute="src",
                authored_target="two.png",
                request_url="https://public.example/two.png",
                resolved_url="https://public.example/two.png",
                availability="available",
                materialization_path="shared.png",
                media_type=second.media_type,
                artifact_digest=second.artifact_digest,
                size=second.size,
            ),
        ),
    )
    output = tmp_path / "bundle"
    with pytest.raises(HTMLSourceBundleError) as error:
        materialize_html_source_bundle(
            bundle, SourceRepositoryHTMLSourceBundleStorage(repository), output
        )
    assert error.value.code == "html_bundle_path_collision"
    assert not output.exists()


def test_count_limit_warning_has_stable_values_and_affects_bundle_identity(
    tmp_path: Path,
) -> None:
    article = "https://public.example/article.html"
    image = "https://public.example/one.png"
    transport = _Transport(
        {
            article: _response(
                article,
                b'<img src="one.png"><img src="two.png"><img src="three.png">',
                "text/html",
            ),
            image: _response(image, b"PNG", "image/png"),
        },
        [],
    )
    _repository, service = _service(tmp_path, transport)
    bundle = service.acquire(article, policy=HTMLAcquisitionPolicy(max_dependency_count=1))
    warning = bundle.warnings[0]
    assert warning.code == "html_dependency_count_limit"
    assert warning.message == "ignored 2 dependency references beyond the limit of 1"
    assert html_source_bundle_from_document(
        html_source_bundle_to_document(bundle)
    ).bundle_digest == bundle.bundle_digest
    changed_warning = HTMLSourceBundle(
        primary=bundle.primary,
        requested_url=bundle.requested_url,
        final_url=bundle.final_url,
        base_url=bundle.base_url,
        acquisition_policy=bundle.acquisition_policy,
        dependencies=bundle.dependencies,
        warnings=(
            HTMLSourceWarning(
                "html_dependency_count_limit",
                "ignored 1 dependency references beyond the limit of 1",
            ),
        ),
    )
    assert changed_warning.bundle_digest != bundle.bundle_digest


def test_non_success_responses_outrank_body_limits_for_primary_and_dependencies(
    tmp_path: Path,
) -> None:
    article = "https://public.example/article.html"
    oversized = b"too-large"
    primary_transport = _Transport(
        {article: _response(article, oversized, "text/html", 404)}, []
    )
    _repository, service = _service(tmp_path / "primary", primary_transport)
    with pytest.raises(HTMLSourceBundleError) as error:
        service.acquire(article, policy=HTMLAcquisitionPolicy(max_primary_bytes=3))
    assert error.value.code == "remote_fetch_failed"

    image = "https://public.example/figure.png"
    dependency_transport = _Transport(
        {
            article: _response(article, b'<img src="figure.png">', "text/html"),
            image: _response(image, oversized, "image/png", 404),
        },
        [],
    )
    _repository, service = _service(tmp_path / "dependency", dependency_transport)
    bundle = service.acquire(
        article,
        policy=HTMLAcquisitionPolicy(
            max_dependency_bytes=3, max_total_dependency_bytes=3
        ),
    )
    assert bundle.dependencies[0].error_code == "html_dependency_fetch_failed"


def test_materialization_rewrites_available_dependencies_and_publishes_atomically(tmp_path: Path) -> None:
    article = "https://public.example/article.html"
    image = "https://public.example/figure.png"
    transport = _Transport(
        {
            article: _response(article, b'<img src="figure.png">', "text/html"),
            image: _response(image, b"PNG", "image/png"),
        },
        [],
    )
    repository, service = _service(tmp_path, transport)
    bundle = service.acquire(article)
    output = tmp_path / "bundle"

    storage = SourceRepositoryHTMLSourceBundleStorage(repository)
    materialized = materialize_html_source_bundle(bundle, storage, output)
    assert materialized.source_path == output / "source.html"
    assert (output / "figure.png").read_bytes() == b"PNG"
    assert materialized.source_path.read_bytes() == repository.read_bytes(bundle.primary)
    assert verify_html_source_bundle_export(output)["bundle"]["bundle_digest"] == bundle.bundle_digest

    protected = tmp_path / "protected"
    protected.mkdir()
    keep = protected / "keep.txt"
    keep.write_text("keep", encoding="utf-8")
    with pytest.raises(HTMLSourceBundleError) as error:
        materialize_html_source_bundle(bundle, storage, protected)
    assert error.value.code == "html_bundle_output_not_empty"
    assert keep.read_text(encoding="utf-8") == "keep"

    asset_path = repository._asset_object_dir(bundle.dependencies[0].artifact_digest) / "asset"  # noqa: SLF001
    asset_path.write_bytes(b"corrupt")
    unpublishable = tmp_path / "unpublishable"
    with pytest.raises(HTMLSourceBundleError) as error:
        materialize_html_source_bundle(bundle, storage, unpublishable)
    assert error.value.code == "html_bundle_materialization_failed"
    assert not unpublishable.exists()

    (output / "figure.png").write_bytes(b"tampered")
    with pytest.raises(HTMLSourceBundleError) as error:
        verify_html_source_bundle_export(output)
    assert error.value.code == "html_bundle_materialization_failed"


def test_registry_declares_explicit_network_and_cache_effects() -> None:
    spec = OPERATION_REGISTRY["acquire-html-bundle"]
    assert spec.operation_id == "ac-document.acquire-html-bundle.v1"
    assert {effect.value for effect in spec.effect_flags} == {
        "network",
        "cache_write",
        "arbitrary_local_path",
    }


def test_acquire_cli_materializes_a_local_bundle(tmp_path: Path, monkeypatch, capsys) -> None:
    repository = SourceRepository(tmp_path / "cache")
    storage = SourceRepositoryHTMLSourceBundleStorage(repository)
    article = "https://public.example/article.html"
    primary = repository.store_bytes(
        b'<img src="figure.png">',
        source_format=SourceFormat.HTML,
        media_type="text/html",
        origin=SourceOrigin(SourceOriginKind.REMOTE_PROVIDER, provider="web", locator=article),
    )
    asset = repository.store_asset_bytes(b"PNG", media_type="image/png")
    bundle = HTMLSourceBundle(
        primary=primary,
        requested_url=article,
        final_url=article,
        base_url=article,
        acquisition_policy=HTMLAcquisitionPolicy().to_document(),
        dependencies=(
            HTMLSourceDependency(
                ordinal=0,
                element="img",
                attribute="src",
                authored_target="figure.png",
                request_url="https://public.example/figure.png",
                resolved_url="https://public.example/figure.png",
                availability="available",
                materialization_path="figure.png",
                media_type=asset.media_type,
                artifact_digest=asset.artifact_digest,
                size=asset.size,
            ),
        ),
    )

    class _Service:
        def __init__(self, *, cache_root=None):
            assert cache_root == str(tmp_path / "cache")
            self.storage = storage

        def acquire(self, url: str, *, policy: HTMLAcquisitionPolicy) -> HTMLSourceBundle:
            assert url == article
            assert policy == HTMLAcquisitionPolicy()
            return bundle

    monkeypatch.setattr(document_registry, "HTMLSourceAcquisitionService", _Service)
    output = tmp_path / "output"
    assert main(
        [
            "acquire-html-bundle",
            article,
            "--output-dir",
            str(output),
            "--cache-root",
            str(tmp_path / "cache"),
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert result["data"]["bundle_digest"] == bundle.bundle_digest
    assert result["data"]["source"] == str(output / "source.html")
    assert result["data"]["manifest"] == str(output / "manifest.json")
    assert result["data"]["resources"] == [str(output / "figure.png")]
    assert (output / "source.html").read_bytes() == repository.read_bytes(primary)


def test_mixed_dns_answers_are_rejected_and_transport_receives_only_validated_addresses(
    tmp_path: Path,
) -> None:
    article = "https://public.example/article.html"
    transport = _Transport({article: _response(article, b"<p>ok</p>", "text/html")}, [])
    repository = SourceRepository(tmp_path / "cache")
    service = HTMLSourceAcquisitionService(
        repository=repository,
        transport=transport,
        resolver=lambda _host: ("1.1.1.1", "127.0.0.1"),
    )

    with pytest.raises(HTMLSourceBundleError) as error:
        service.acquire(article)
    assert error.value.code == "remote_target_not_public"
    assert transport.calls == []


def test_stdlib_transport_pins_the_service_validated_ip_set(monkeypatch) -> None:
    class _Response:
        status = 200

        def getheaders(self):
            return [("content-type", "text/html")]

        def read(self, _size: int) -> bytes:
            return b"" if getattr(self, "done", False) else setattr(self, "done", True) or b"<p>ok</p>"

    class _Connection:
        def request(self, *_args, **_kwargs) -> None:
            return None

        def getresponse(self):
            return _Response()

        def close(self) -> None:
            return None

    observed: list[tuple[str, tuple[str, ...], float]] = []

    def pinned(host: str, addresses: tuple[str, ...], timeout: float):
        observed.append((host, addresses, timeout))
        return _Connection()

    monkeypatch.setattr("ac_document.web_acquisition._pinned_https_connection", pinned)
    monkeypatch.setattr(
        "ac_document.web_acquisition.socket.getaddrinfo",
        lambda *_args, **_kwargs: pytest.fail("transport must not resolve the hostname again"),
    )

    response = StdlibHTTPSWebTransport().fetch(
        "https://public.example/article.html",
        timeout_seconds=2.0,
        maximum_bytes=1024,
        validated_addresses=("1.1.1.1",),
    )
    assert response.body == b"<p>ok</p>"
    assert observed == [("public.example", ("1.1.1.1",), 2.0)]


def test_stdlib_transport_does_not_read_non_success_bodies(monkeypatch) -> None:
    class _Response:
        status = 404

        def getheaders(self):
            return [("content-type", "text/html"), ("content-length", "999999")]

        def read(self, _size: int) -> bytes:
            pytest.fail("non-success response body must not be read")

    class _Connection:
        def request(self, *_args, **_kwargs) -> None:
            return None

        def getresponse(self):
            return _Response()

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "ac_document.web_acquisition._pinned_https_connection",
        lambda *_args: _Connection(),
    )
    response = StdlibHTTPSWebTransport().fetch(
        "https://public.example/missing.html",
        timeout_seconds=2.0,
        maximum_bytes=3,
        validated_addresses=("1.1.1.1",),
    )
    assert response.status == 404
    assert response.body == b""


def test_redirect_limit_and_invalid_base_fail_closed_without_dependency_requests(
    tmp_path: Path,
) -> None:
    first = "https://public.example/one.html"
    second = "https://public.example/two.html"
    loop = _Transport(
        {
            first: _response(first, b"", "text/html", 302, location="two.html"),
            second: _response(second, b"", "text/html", 302, location="one.html"),
        },
        [],
    )
    _repository, service = _service(tmp_path / "loop", loop)
    with pytest.raises(HTMLSourceBundleError) as error:
        service.acquire(first, policy=HTMLAcquisitionPolicy(max_redirects=1))
    assert error.value.code == "html_primary_redirect_limit"
    assert loop.calls == [first, second]

    article = "https://public.example/article.html"
    other = "https://other.example/"
    base = _Transport(
        {
            article: _response(
                article,
                b'<base href="https://other.example/"><img src="figure.png">',
                "text/html",
            )
        },
        [],
    )
    _repository, service = _service(tmp_path / "base", base)
    bundle = service.acquire(article)
    assert bundle.dependencies[0].availability == "unavailable"
    assert bundle.dependencies[0].error_code == "html_dependency_base_invalid"
    assert base.calls == [article]
    assert other not in base.calls


def test_dependency_redirect_that_leaves_origin_has_a_distinct_stable_warning(
    tmp_path: Path,
) -> None:
    article = "https://public.example/article.html"
    image = "https://public.example/figure.png"
    outside = "https://other.example/figure.png"
    transport = _Transport(
        {
            article: _response(article, b'<img src="figure.png">', "text/html"),
            image: _response(image, b"", "image/png", 302, location=outside),
        },
        [],
    )
    _repository, service = _service(tmp_path, transport)

    bundle = service.acquire(article)
    dependency = bundle.dependencies[0]
    assert dependency.availability == "unavailable"
    assert dependency.error_code == "html_dependency_redirect_invalid"
    assert bundle.warnings[0].code == "html_dependency_redirect_invalid"
    assert transport.calls == [article, image]


def test_article_scope_srcset_losslessness_and_export_codec(tmp_path: Path) -> None:
    article = "https://public.example/article.html"
    image = "https://public.example/inside.png"
    transport = _Transport(
        {
            article: _response(
                article,
                (
                    b'<header><img src="outside.png"></header>'
                    b'<article class="ltx_document"><img src="inside.png">'
                    b'<source srcset="one.png 1x, two.png 2x"></article>'
                ),
                "text/html",
            ),
            image: _response(image, b"PNG", "image/png"),
        },
        [],
    )
    repository, service = _service(tmp_path, transport)
    bundle = service.acquire(article)

    assert [(item.attribute, item.authored_target, item.availability) for item in bundle.dependencies] == [
        ("src", "inside.png", "available"),
        ("srcset", "one.png 1x, two.png 2x", "unavailable"),
    ]
    assert bundle.dependencies[1].error_code == "html_dependency_srcset_unsupported"
    assert transport.calls == [article, image]

    exported = html_source_bundle_export_to_document(
        bundle, materialized_source=repository.read_bytes(bundle.primary)
    )
    assert html_source_bundle_export_from_document(exported) == exported
    malformed = copy.deepcopy(exported)
    malformed["resources"] = []
    with pytest.raises(ValueError, match="does not match"):
        html_source_bundle_export_from_document(malformed)


def test_article_scoped_absolute_dependency_rewrites_the_matching_source_node(
    tmp_path: Path,
) -> None:
    article = "https://public.example/article.html"
    image = "https://public.example/inside.png"
    transport = _Transport(
        {
            article: _response(
                article,
                (
                    b'<header><img src="outside.png"></header>'
                    b'<article class="ltx_document"><img src="https://public.example/inside.png"></article>'
                ),
                "text/html",
            ),
            image: _response(image, b"PNG", "image/png"),
        },
        [],
    )
    repository, service = _service(tmp_path, transport)
    bundle = service.acquire(article)
    output = tmp_path / "bundle"

    materialize_html_source_bundle(
        bundle, SourceRepositoryHTMLSourceBundleStorage(repository), output
    )
    assert "assets/public.example/inside.png" in (output / "source.html").read_text(
        encoding="utf-8"
    )


def test_dependencies_only_uses_an_adapter_without_source_repository_assets(
    tmp_path: Path,
) -> None:
    primary_bytes = b'<img src="asset.png">'
    primary = SourceArtifact(
        source_format=SourceFormat.HTML,
        artifact_digest=hashlib.sha256(primary_bytes).hexdigest(),
        size=len(primary_bytes),
        media_type="text/html",
        origin=SourceOrigin(SourceOriginKind.REMOTE_PROVIDER, provider="fixture", locator="https://public.example/article.html"),
    )

    class _Adapter:
        def __init__(self):
            self.assets: dict[str, bytes] = {}

        def read_primary(self, value: SourceArtifact) -> bytes:
            assert value == primary
            return primary_bytes

        def store_dependency(self, payload: bytes, *, media_type: str) -> StoredHTMLDependency:
            digest = hashlib.sha256(payload).hexdigest()
            self.assets[digest] = payload
            return StoredHTMLDependency(digest, media_type, len(payload))

        def read_dependency(self, *, artifact_digest: str, media_type: str, size: int) -> bytes:
            payload = self.assets[artifact_digest]
            assert media_type == "image/png" and size == len(payload)
            return payload

    adapter = _Adapter()
    url = "https://public.example/article.html"
    asset = "https://public.example/asset.png"
    transport = _Transport(
        {asset: _response(asset, b"PNG", "image/png")}, []
    )
    service = HTMLSourceAcquisitionService(
        cache_root=tmp_path / "cache",
        storage=adapter,
        transport=transport,
        resolver=_resolver(),
    )

    bundle = service.acquire_dependencies(primary, document_url=url, storage=adapter)
    assert bundle.dependencies[0].artifact_digest in adapter.assets


def test_primary_identity_is_verified_for_cache_replay_and_materialization(
    tmp_path: Path,
) -> None:
    article = "https://public.example/article.html"
    transport = _Transport(
        {article: _response(article, b"<p>primary</p>", "text/html")}, []
    )
    repository, service = _service(tmp_path, transport)
    bundle = service.acquire(article)
    delegate = SourceRepositoryHTMLSourceBundleStorage(repository)

    class _TamperingStorage:
        def read_primary(self, primary: SourceArtifact) -> bytes:
            return b"x" * primary.size

        def store_dependency(self, payload: bytes, *, media_type: str) -> StoredHTMLDependency:
            return delegate.store_dependency(payload, media_type=media_type)

        def read_dependency(self, *, artifact_digest: str, media_type: str, size: int) -> bytes:
            return delegate.read_dependency(
                artifact_digest=artifact_digest, media_type=media_type, size=size
            )

    storage = _TamperingStorage()
    cache = HTMLSourceBundleCache(repository.root, storage)
    request_key = cache.request_key(bundle.requested_url, bundle.acquisition_policy)
    cache.store(bundle, request_key=request_key)
    with pytest.raises(HTMLSourceBundleError) as error:
        cache.lookup(request_key)
    assert error.value.code == "html_bundle_cache_corrupt"

    output = tmp_path / "output"
    with pytest.raises(HTMLSourceBundleError) as error:
        materialize_html_source_bundle(bundle, storage, output)
    assert error.value.code == "html_bundle_materialization_failed"
    assert not output.exists()


def test_cache_request_index_is_bound_to_its_bundle_identity(tmp_path: Path) -> None:
    first_url = "https://public.example/one.html"
    second_url = "https://public.example/two.html"
    transport = _Transport(
        {
            first_url: _response(first_url, b"<p>one</p>", "text/html"),
            second_url: _response(second_url, b"<p>two</p>", "text/html"),
        },
        [],
    )
    _repository, service = _service(tmp_path, transport)
    first = service.acquire(first_url)
    second = service.acquire(second_url)
    cache = service.cache
    first_key = cache.request_key(first.requested_url, first.acquisition_policy)
    second_key = cache.request_key(second.requested_url, second.acquisition_policy)

    with pytest.raises(HTMLSourceBundleError) as error:
        cache.store(first, request_key=second_key)
    assert error.value.code == "html_bundle_cache_corrupt"

    index_path = cache._request_path(first_key)  # noqa: SLF001 - tamper fixture
    index_path.write_text(
        json.dumps(
            {
                "schema_version": "ac.document.html_source_bundle_cache.v1",
                "bundle_digest": second.bundle_digest,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(HTMLSourceBundleError) as error:
        cache.lookup(first_key)
    assert error.value.code == "html_bundle_cache_corrupt"


def test_cross_origin_dependencies_require_explicit_policy_opt_in(tmp_path: Path) -> None:
    article = "https://public.example/article.html"
    image = "https://cdn.example/figure.png"
    transport = _Transport(
        {
            article: _response(article, b'<img src="https://cdn.example/figure.png">', "text/html"),
            image: _response(image, b"PNG", "image/png"),
        },
        [],
    )
    _repository, service = _service(tmp_path, transport)
    policy = HTMLAcquisitionPolicy(
        same_origin_dependencies=False,
        allowed_origins=("https://public.example", "https://cdn.example"),
    )

    bundle = service.acquire(article, policy=policy)
    assert bundle.dependencies[0].availability == "available"
    assert bundle.dependencies[0].resolved_url == image
    assert transport.calls == [article, image]


def test_policy_is_deeply_immutable_and_rejects_non_finite_timeouts(tmp_path: Path) -> None:
    article = "https://public.example/article.html"
    transport = _Transport(
        {article: _response(article, b"<p>primary</p>", "text/html")}, []
    )
    _repository, service = _service(tmp_path, transport)
    bundle = service.acquire(
        article,
        policy=HTMLAcquisitionPolicy(allowed_origins=("https://public.example",)),
    )
    digest = bundle.bundle_digest
    assert bundle.acquisition_policy["allowed_origins"] == ("https://public.example",)
    with pytest.raises(AttributeError):
        bundle.acquisition_policy["allowed_origins"].append("https://cdn.example")
    with pytest.raises(TypeError):
        bundle.acquisition_policy["allowed_origins"] = ()  # type: ignore[index]
    assert bundle.bundle_digest == digest
    assert html_source_bundle_to_document(bundle)["acquisition_policy"]["allowed_origins"] == [
        "https://public.example"
    ]

    for timeout in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="timeout"):
            HTMLAcquisitionPolicy(timeout_seconds=timeout)

    with pytest.raises(ValueError, match="allowed origins"):
        HTMLAcquisitionPolicy(same_origin_dependencies=False)
    invalid_policy = HTMLAcquisitionPolicy().to_document()
    invalid_policy["same_origin_dependencies"] = False
    with pytest.raises(ValueError, match="allowed origins"):
        HTMLAcquisitionPolicy.from_document(invalid_policy)


def test_dependency_failures_preserve_url_provenance_and_storage_classification(
    tmp_path: Path, monkeypatch
) -> None:
    article = "https://public.example/article.html"
    image = "https://public.example/figure.png"
    status_transport = _Transport(
        {
            article: _response(article, b'<img src="figure.png">', "text/html"),
            image: _response(image, b"missing", "image/png", 404),
        },
        [],
    )
    _repository, service = _service(tmp_path / "status", status_transport)
    status_bundle = service.acquire(article)
    status_dependency = status_bundle.dependencies[0]
    assert status_dependency.error_code == "html_dependency_fetch_failed"
    assert status_dependency.request_url == image
    assert status_dependency.resolved_url == image

    repository = SourceRepository(tmp_path / "storage" / "cache")
    storage = SourceRepositoryHTMLSourceBundleStorage(repository)

    def fail_store(*_args, **_kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(storage, "store_dependency", fail_store)
    storage_transport = _Transport(
        {
            article: _response(article, b'<img src="figure.png">', "text/html"),
            image: _response(image, b"PNG", "image/png"),
        },
        [],
    )
    storage_service = HTMLSourceAcquisitionService(
        repository=repository,
        storage=storage,
        transport=storage_transport,
        resolver=_resolver(),
    )
    storage_bundle = storage_service.acquire(article)
    storage_dependency = storage_bundle.dependencies[0]
    assert storage_bundle.primary.artifact_digest
    assert storage_dependency.error_code == "html_dependency_cache_write_failed"
    assert storage_dependency.request_url == image
    assert storage_dependency.resolved_url == image


def test_transport_size_error_uses_the_operation_specific_classification(
    tmp_path: Path,
) -> None:
    article = "https://public.example/article.html"
    image = "https://public.example/figure.png"

    class _TooLargeTransport:
        def __init__(self, response: WebResponse | None = None):
            self.response = response

        def fetch(self, _url: str, **_kwargs) -> WebResponse:
            if self.response is not None:
                response = self.response
                self.response = None
                return response
            raise HTMLSourceBundleError("remote_response_too_large", "too large")

    primary_service = HTMLSourceAcquisitionService(
        repository=SourceRepository(tmp_path / "primary" / "cache"),
        transport=_TooLargeTransport(),
        resolver=_resolver(),
    )
    with pytest.raises(HTMLSourceBundleError) as error:
        primary_service.acquire(article)
    assert error.value.code == "html_primary_too_large"

    dependency_service = HTMLSourceAcquisitionService(
        repository=SourceRepository(tmp_path / "dependency" / "cache"),
        transport=_TooLargeTransport(
            _response(article, b'<img src="figure.png">', "text/html")
        ),
        resolver=_resolver(),
    )
    bundle = dependency_service.acquire(article)
    dependency = bundle.dependencies[0]
    assert dependency.error_code == "html_dependency_too_large"
    assert dependency.request_url == image
