from __future__ import annotations

import hashlib
import json
import multiprocessing
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from ac_document import (
    ParseOutcome,
    ReconciliationReport,
    SourceBundle,
    SourceFormat,
    SourceOrigin,
    SourceOriginKind,
    SourceRepository,
    SourceRepositoryError,
    ValidationPolicy,
)


def _store_from_process(cache_root, ready, results):
    ready.wait(timeout=10)
    repository = SourceRepository(cache_root)
    artifact = repository.store_bytes(
        b"same process-safe source",
        source_format=SourceFormat.MARKDOWN,
        origin=SourceOrigin(
            SourceOriginKind.REMOTE_PROVIDER,
            provider="process-fixture",
        ),
    )
    results.put(artifact.content_identity)


@pytest.mark.parametrize(
    ("name", "source_format", "payload", "media_type"),
    [
        ("paper.html", SourceFormat.HTML, b"<p>paper</p>", "text/html"),
        ("paper.md", SourceFormat.MARKDOWN, b"# Paper\n", "text/markdown"),
        ("paper.tex", SourceFormat.TEX, b"\\section{Paper}\n", "text/x-tex"),
        ("paper.pdf", SourceFormat.PDF, b"%PDF-1.7\n", "application/pdf"),
    ],
)
def test_imports_supported_local_sources(
    tmp_path, name, source_format, payload, media_type
):
    source = tmp_path / name
    source.write_bytes(payload)
    repository = SourceRepository(tmp_path / "cache")

    artifact = repository.import_path(source)

    assert artifact.source_format is source_format
    assert artifact.artifact_digest == hashlib.sha256(payload).hexdigest()
    assert artifact.size == len(payload)
    assert artifact.media_type == media_type
    assert artifact.origin.kind is SourceOriginKind.LOCAL_IMPORT
    assert repository.read_bytes(artifact) == payload


def test_same_bytes_at_different_paths_have_same_content_identity(tmp_path):
    left = tmp_path / "left.md"
    right = tmp_path / "nested" / "right.md"
    right.parent.mkdir()
    left.write_text("same", encoding="utf-8")
    right.write_text("same", encoding="utf-8")
    repository = SourceRepository(tmp_path / "cache")

    first = repository.import_path(left)
    second = repository.import_path(right)

    assert first.content_identity == second.content_identity
    assert first.origin.locator != second.origin.locator


def test_bundle_normalizes_validators_and_rejects_duplicates(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    origin = SourceOrigin(SourceOriginKind.REMOTE_PROVIDER, provider="fixture")
    primary = repository.store_bytes(
        b"# primary", source_format=SourceFormat.MARKDOWN, origin=origin
    )
    html = repository.store_bytes(
        b"<p>validator</p>", source_format=SourceFormat.HTML, origin=origin
    )
    pdf = repository.store_bytes(
        b"%PDF validator", source_format=SourceFormat.PDF, origin=origin
    )

    bundle = SourceBundle(primary=primary, validators=(pdf, html))

    assert bundle.validators == tuple(
        sorted((html, pdf), key=lambda item: item.content_identity)
    )
    with pytest.raises(ValueError, match="duplicate validator"):
        SourceBundle(primary=primary, validators=(html, html))
    with pytest.raises(ValueError, match="primary"):
        SourceBundle(primary=primary, validators=(primary,))


def test_parse_outcome_rejects_untyped_legacy_document(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    artifact = repository.store_bytes(
        b"# paper",
        source_format=SourceFormat.MARKDOWN,
        origin=SourceOrigin(SourceOriginKind.LOCAL_IMPORT, locator="paper.md"),
    )
    report = ReconciliationReport(
        primary=artifact,
        policy=ValidationPolicy.VISUAL_ALL_PAGES,
    )

    with pytest.raises(TypeError, match="must be a ParsedDocument"):
        ParseOutcome(
            document={"equations": []},  # type: ignore[arg-type]
            report=report,
            warnings=("PDF page 2 was unreviewed",),
        )


def test_manifest_is_strict_and_payload_corruption_is_detected(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    artifact = repository.store_bytes(
        b"source",
        source_format=SourceFormat.TEX,
        origin=SourceOrigin(SourceOriginKind.LOCAL_IMPORT),
    )
    object_dir = repository._object_dir(  # noqa: SLF001 - corruption fixture
        artifact.source_format, artifact.artifact_digest
    )
    manifest_path = object_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unknown"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SourceRepositoryError) as error:
        repository.get(artifact.source_format, artifact.artifact_digest)
    assert error.value.code == "source_manifest_invalid"

    manifest.pop("unknown")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (object_dir / "source").write_bytes(b"broken")
    with pytest.raises(SourceRepositoryError) as error:
        repository.read_bytes(artifact)
    assert error.value.code == "source_corrupt"


def test_interrupted_payload_without_manifest_is_completed(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    payload = b"# recovered"
    digest = hashlib.sha256(payload).hexdigest()
    object_dir = repository._object_dir(  # noqa: SLF001 - interrupted-write fixture
        SourceFormat.MARKDOWN, digest
    )
    object_dir.mkdir(parents=True)
    (object_dir / "source").write_bytes(payload)

    artifact = repository.store_bytes(
        payload,
        source_format=SourceFormat.MARKDOWN,
        origin=SourceOrigin(SourceOriginKind.LOCAL_IMPORT),
    )

    assert artifact.artifact_digest == digest
    assert (object_dir / "manifest.json").is_file()


def test_concurrent_same_key_imports_publish_one_valid_object(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    payload = b"concurrent source"
    origin = SourceOrigin(SourceOriginKind.REMOTE_PROVIDER, provider="fixture")

    def store():
        return repository.store_bytes(
            payload, source_format=SourceFormat.HTML, origin=origin
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        artifacts = list(pool.map(lambda _: store(), range(24)))

    assert len({item.content_identity for item in artifacts}) == 1
    assert repository.read_bytes(artifacts[0]) == payload


def test_read_bytes_holds_content_lock_until_payload_is_returned(
    tmp_path, monkeypatch
):
    repository = SourceRepository(tmp_path / "cache")
    payload = b"serialized source"
    artifact = repository.store_bytes(
        payload,
        source_format=SourceFormat.MARKDOWN,
        origin=SourceOrigin(SourceOriginKind.LOCAL_IMPORT),
    )
    payload_path = (
        repository._object_dir(  # noqa: SLF001 - concurrency fixture
            artifact.source_format, artifact.artifact_digest
        )
        / "source"
    )
    reader_at_payload = threading.Barrier(2)
    allow_reader = threading.Event()
    remover_at_lock = threading.Barrier(2)
    remove_finished = threading.Event()
    original_read_bytes = Path.read_bytes
    original_content_lock = repository._content_lock  # noqa: SLF001

    def gated_read_bytes(path):
        if (
            path == payload_path
            and threading.current_thread().name == "source-reader"
        ):
            reader_at_payload.wait(timeout=5)
            assert allow_reader.wait(timeout=5)
        return original_read_bytes(path)

    @contextmanager
    def observed_content_lock(source_format, digest):
        if threading.current_thread().name == "source-remover":
            remover_at_lock.wait(timeout=5)
        with original_content_lock(source_format, digest):
            yield

    monkeypatch.setattr(Path, "read_bytes", gated_read_bytes)
    monkeypatch.setattr(repository, "_content_lock", observed_content_lock)
    results = {}

    def read():
        try:
            results["payload"] = repository.read_bytes(artifact)
        except Exception as exc:  # pragma: no cover - asserted below
            results["read_error"] = exc

    def remove():
        try:
            results["removed"] = repository.remove(
                artifact.source_format, artifact.artifact_digest
            )
        except Exception as exc:  # pragma: no cover - asserted below
            results["remove_error"] = exc
        finally:
            remove_finished.set()

    reader = threading.Thread(target=read, name="source-reader")
    remover = threading.Thread(target=remove, name="source-remover")
    reader.start()
    reader_at_payload.wait(timeout=5)
    remover.start()
    remover_at_lock.wait(timeout=5)
    assert not remove_finished.wait(timeout=0.2)
    allow_reader.set()
    reader.join(timeout=5)
    remover.join(timeout=5)

    assert not reader.is_alive()
    assert not remover.is_alive()
    assert results == {"payload": payload, "removed": True}


def test_read_bytes_checks_the_exact_payload_read_during_tamper(
    tmp_path, monkeypatch
):
    repository = SourceRepository(tmp_path / "cache")
    artifact = repository.store_bytes(
        b"original",
        source_format=SourceFormat.MARKDOWN,
        origin=SourceOrigin(SourceOriginKind.LOCAL_IMPORT),
    )
    payload_path = (
        repository._object_dir(  # noqa: SLF001 - corruption fixture
            artifact.source_format, artifact.artifact_digest
        )
        / "source"
    )
    reader_at_payload = threading.Barrier(2)
    tamper_complete = threading.Barrier(2)
    original_read_bytes = Path.read_bytes

    def gated_read_bytes(path):
        if (
            path == payload_path
            and threading.current_thread().name == "source-reader"
        ):
            reader_at_payload.wait(timeout=5)
            tamper_complete.wait(timeout=5)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", gated_read_bytes)
    results = {}

    def read():
        try:
            results["payload"] = repository.read_bytes(artifact)
        except Exception as exc:
            results["error"] = exc

    reader = threading.Thread(target=read, name="source-reader")
    reader.start()
    reader_at_payload.wait(timeout=5)
    payload_path.write_bytes(b"tampered")
    tamper_complete.wait(timeout=5)
    reader.join(timeout=5)

    assert not reader.is_alive()
    assert "payload" not in results
    assert isinstance(results.get("error"), SourceRepositoryError)
    assert results["error"].code == "source_corrupt"


def test_two_processes_publish_same_content_with_one_valid_manifest(tmp_path):
    cache_root = tmp_path / "cache"
    context = multiprocessing.get_context("spawn")
    ready = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_store_from_process,
            args=(cache_root, ready, results),
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)

        assert not any(process.is_alive() for process in processes)
        assert [process.exitcode for process in processes] == [0, 0]
        identities = [results.get(timeout=1) for _ in processes]
        assert identities[0] == identities[1]

        source_format, media_type, digest, size = identities[0]
        repository = SourceRepository(cache_root)
        artifact = repository.get(SourceFormat(source_format), digest)
        assert repository.read_bytes(artifact) == b"same process-safe source"
        manifest_path = (
            repository._object_dir(artifact.source_format, artifact.artifact_digest)
            / "manifest.json"
        )
        assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
            "artifact_digest": artifact.artifact_digest,
            "media_type": media_type,
            "schema_version": "ac.document.source_repository.v1",
            "size": size,
            "source_format": "markdown",
        }
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
        results.close()
        results.join_thread()


def test_unknown_local_suffix_is_typed_unsupported_source(tmp_path):
    source = tmp_path / "paper.rst"
    source.write_text("paper", encoding="utf-8")

    with pytest.raises(SourceRepositoryError) as error:
        SourceRepository(tmp_path / "cache").import_path(source)

    assert error.value.code == "unsupported_source"
