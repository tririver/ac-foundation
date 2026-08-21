from __future__ import annotations

import json
import re
import subprocess
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from ac_jobs import (
    ImmutableArtifactStore,
    ResumeReason,
    RunContext,
    RunRepository,
    RunSpec,
    RunStatus,
    StoppedError,
)
from ac_llm import (
    FailureCategory,
    HostAuthority,
    IsolationMode,
    LLMCompleted,
    LLMFailed,
    LLMPaused,
    LLMStopped,
    LLMTaskService,
    LLMExecutionOptions,
    ModelSelection,
    ProviderCapabilities,
    ProviderDiagnostic,
    ProviderExecution,
    ProviderFailure,
    ProviderRegistry,
    ProviderTerminalKind,
    StructuredOutputMode,
    UsageAvailability,
)
from ac_llm.output import CandidateMaterial
from ac_document import (
    PDFTextLayer,
    PageMathVerdict,
    DocumentParserService,
    PdftoppmFullPageRenderer,
    ReconciliationStatus,
    RenderedPDFPage,
    SourceBundle,
    SourceFormat,
    SourceOrigin,
    SourceOriginKind,
    SourceRepository,
    SourceRepositoryError,
    ValidationPolicy,
    VisualReviewService,
    MarkdownPDFVisualParseRunner,
    decode_visual_page_review,
    visual_page_review_schema,
)
from ac_document.parse.visual import PDFRenderError


def test_visual_review_provider_const_declares_string_type() -> None:
    schema = visual_page_review_schema()
    assert schema["properties"]["schema_version"]["type"] == "string"


def _png(width: int = 100, height: int = 200) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
    )


class FakePDFTextExtractor:
    contract_id = "ac.document.tests.fake_pdf_text.v1"

    def __init__(self, pages: tuple[str, ...]):
        self.pages = pages

    def extract(self, payload: bytes) -> PDFTextLayer:
        del payload
        return PDFTextLayer(self.pages)


class FakeRenderer:
    def __init__(self, count: int):
        self.count = count
        self.calls = 0

    def render(self, pdf_bytes: bytes) -> tuple[RenderedPDFPage, ...]:
        assert pdf_bytes.startswith(b"%PDF")
        self.calls += 1
        return tuple(
            RenderedPDFPage(number, _png(100 + number, 200), 100 + number, 200)
            for number in range(1, self.count + 1)
        )


class FailingRenderer:
    def render(self, pdf_bytes: bytes):
        del pdf_bytes
        raise PDFRenderError("pdf_renderer_unavailable", "renderer missing")


class ManifestAwareAdapter:
    name = "codex"
    compatibility_version = "visual-test-v1"

    def __init__(self) -> None:
        self.start_calls = 0
        self.requests: list[Any] = []

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            native_resume=True,
            structured_output=StructuredOutputMode.NATIVE,
            usage=UsageAvailability.UNAVAILABLE,
            config_isolation=IsolationMode.ISOLATED,
            tool_isolation=IsolationMode.ISOLATED,
            cooperative_stop=True,
            provider_persistence=True,
        )

    def doctor(self) -> ProviderDiagnostic:
        return ProviderDiagnostic(self.name, True, "fake-codex")

    def start(self, request, observer, stop) -> ProviderExecution:
        del stop
        self.start_calls += 1
        self.requests.append(request)
        import json

        control = json.loads(
            (request.workspace / "host" / "control.json").read_text(encoding="utf-8")
        )
        page_number = int(re.search(r"page (\d+)", control["prompt"]).group(1))
        input_paths = {item["input_id"]: item["path"] for item in control["inputs"]}
        manifest = json.loads(
            (request.workspace / input_paths["math-manifest"]).read_text(encoding="utf-8")
        )
        span_ids = [item["span_id"] for item in manifest["spans"]]
        reviews = (
            [
                {
                    "span_id": span_ids[0],
                    "verdict": "exact",
                    "observed_math": "x+y",
                    "notes": "",
                }
            ]
            if page_number == 1 and span_ids
            else []
        )
        value = {
            "schema_version": "ac.document.visual_page_review.v1",
            "page_number": page_number,
            "reviewed_span_ids": [item["span_id"] for item in reviews],
            "reviews": reviews,
            "unexpected_math": [],
            "notes": "",
        }
        return ProviderExecution(
            ProviderTerminalKind.COMPLETED,
            (CandidateMaterial(value=value, terminal=True),),
        )

    def resume(self, handle, request, observer, stop) -> ProviderExecution:
        del handle
        return self.start(request, observer, stop)


class ScriptedLLM:
    def __init__(self, values: list[Any]):
        self.values = deque(values)
        self.requests: list[Any] = []

    def execute_or_resume(self, context, request, *, options=None):
        del context
        self.requests.append(request)
        return self.values.popleft()


class NeverLLM:
    def __init__(self) -> None:
        self.calls = 0

    def execute_or_resume(self, context, request, *, options=None):
        del context, request
        self.calls += 1
        raise AssertionError("durable page terminal should suppress LLM replay")


def _store(repository, payload, source_format):
    return repository.store_bytes(
        payload,
        source_format=source_format,
        origin=SourceOrigin(SourceOriginKind.LOCAL_IMPORT),
    )


def _context(tmp_path: Path) -> RunContext:
    repository = RunRepository(tmp_path / "jobs")
    snapshot = repository.create(
        RunSpec("visual-parent", "test.visual", {"case": "pages"})
    )
    return RunContext(
        repository, snapshot, resume_input=None
    )


def test_markdown_pdf_default_reviews_every_full_page_and_replays_without_calls(
    tmp_path: Path,
) -> None:
    sources = SourceRepository(tmp_path / "sources")
    markdown = _store(
        sources, b"# Notes\nInline $x+y$.\n", SourceFormat.MARKDOWN
    )
    pdf = _store(sources, b"%PDF visual fixture", SourceFormat.PDF)
    renderer = FakeRenderer(2)
    adapter = ManifestAwareAdapter()
    registry = ProviderRegistry()
    registry.register("codex", lambda: adapter)
    jobs = RunRepository(tmp_path / "jobs")
    runner = MarkdownPDFVisualParseRunner(
        jobs,
        sources,
        renderer=renderer,
        pdf_text_extractor=FakePDFTextExtractor(("page one", "page two")),
        llm=LLMTaskService(registry=registry),
    )

    first = runner.execute(
        "visual-default",
        markdown,
        pdf,
        model=ModelSelection("codex"),
        options=LLMExecutionOptions(host_authority=HostAuthority.UNRESTRICTED),
    )
    second = runner.execute(
        "visual-default",
        markdown,
        pdf,
        model=ModelSelection("codex"),
        options=LLMExecutionOptions(host_authority=HostAuthority.UNRESTRICTED),
    )

    assert first.status is second.status is RunStatus.SUCCEEDED
    assert adapter.start_calls == 2
    assert len(adapter.requests) == 2
    controls = [
        json.loads((request.workspace / "host" / "control.json").read_text())
        for request in adapter.requests
    ]
    assert all(
        [item["input_id"] for item in control["inputs"]]
        == ["page", "markdown", "math-manifest"]
        for control in controls
    )
    assert all(
        (request.workspace / control["inputs"][0]["path"]).read_bytes().startswith(
            b"\x89PNG\r\n\x1a\n"
        )
        for request, control in zip(adapter.requests, controls, strict=True)
    )
    assert first.result_ref is not None
    result = json.loads(
        ImmutableArtifactStore(
            jobs.run_directory("visual-default"),
            repository_root=jobs.root,
        )
        .read_bytes(first.result_ref)
        .decode("utf-8")
    )
    assert result["report"]["policy"] == ValidationPolicy.VISUAL_ALL_PAGES.value
    page_entries = [
        entry
        for entry in result["report"]["entries"]
        if entry["subject_id"].startswith("visual-page:")
        and ":unexpected:" not in entry["subject_id"]
    ]
    assert [entry["provenance"]["page_number"] for entry in page_entries] == [1, 2]
    visual_span_entries = [
        entry
        for entry in result["report"]["entries"]
        if entry["provenance"].get("review_method") == "visual_all_pages"
    ]
    assert len(visual_span_entries) == 1
    assert visual_span_entries[0]["status"] == ReconciliationStatus.VERIFIED.value
    span_id = result["document"]["math_spans"][0]["span_id"]
    assert [
        entry["subject_id"] for entry in result["report"]["entries"]
    ].count(span_id) == 1
    deterministic = next(
        entry
        for entry in result["report"]["entries"]
        if entry["subject_id"] == f"deterministic:{span_id}"
    )
    assert deterministic["provenance"]["primary_span_id"] == span_id
    manifests = tuple(
        (jobs.run_directory("visual-default") / "artifacts" / "manifests").rglob(
            "*.json"
        )
    )
    assert not any("crop" in path.as_posix() for path in manifests)


def test_visual_aggregation_marks_mismatch_duplicate_missing_and_unexpected(
    tmp_path: Path,
) -> None:
    sources = SourceRepository(tmp_path / "sources")
    markdown = _store(
        sources,
        b"# Notes\n$a$ then $b$ then $c$.\n",
        SourceFormat.MARKDOWN,
    )
    pdf = _store(sources, b"%PDF aggregate fixture", SourceFormat.PDF)
    parser = DocumentParserService(
        sources, pdf_text_extractor=FakePDFTextExtractor(("one", "two"))
    )
    primary = parser.parse_source(markdown)
    parsed_pdf = parser.parse_source(pdf)
    a, b, c = [item.span_id for item in primary.math_spans]
    first = {
        "schema_version": "ac.document.visual_page_review.v1",
        "page_number": 1,
        "reviewed_span_ids": [a, b],
        "reviews": [
            {
                "span_id": a,
                "verdict": "mismatch",
                "observed_math": "a'",
                "notes": "",
            },
            {
                "span_id": b,
                "verdict": "exact",
                "observed_math": "b",
                "notes": "",
            },
        ],
        "unexpected_math": [{"observed_math": "z", "notes": "extra"}],
        "notes": "",
    }
    second = {
        "schema_version": "ac.document.visual_page_review.v1",
        "page_number": 2,
        "reviewed_span_ids": [b],
        "reviews": [
            {
                "span_id": b,
                "verdict": "equivalent",
                "observed_math": r"\mathrm{b}",
                "notes": "",
            }
        ],
        "unexpected_math": [],
        "notes": "",
    }
    llm = ScriptedLLM(
        [
            LLMCompleted(first, "codex", "fake", None, None),
            LLMCompleted(second, "codex", "fake", None, None),
        ]
    )
    outcome = VisualReviewService(
        FakeRenderer(2), llm=llm, model=ModelSelection("codex")
    ).review(
        _context(tmp_path),
        primary,
        parsed_pdf,
        markdown_bytes=sources.read_bytes(markdown),
        pdf_bytes=sources.read_bytes(pdf),
    )
    by_span = {
        entry.subject_id: entry.status
        for entry in outcome.entries
        if entry.provenance.get("review_method") == "visual_all_pages"
    }

    assert by_span == {
        a: ReconciliationStatus.MISMATCH,
        b: ReconciliationStatus.AMBIGUOUS,
        c: ReconciliationStatus.MISSING,
    }
    assert any(":unexpected:" in entry.subject_id for entry in outcome.entries)
    assert len(llm.requests) == 2


def test_paused_page_is_unreviewed_and_later_pages_continue(tmp_path: Path) -> None:
    sources = SourceRepository(tmp_path / "sources")
    markdown = _store(sources, b"# Notes\n$x$.\n", SourceFormat.MARKDOWN)
    pdf = _store(sources, b"%PDF pause fixture", SourceFormat.PDF)
    parser = DocumentParserService(
        sources, pdf_text_extractor=FakePDFTextExtractor(("one", "two"))
    )
    primary = parser.parse_source(markdown)
    parsed_pdf = parser.parse_source(pdf)
    llm = ScriptedLLM(
        [
            LLMPaused(
                ResumeReason.EXTERNAL_CONDITION,
                "page-pause",
            ),
            LLMCompleted(
                {
                    "schema_version": "ac.document.visual_page_review.v1",
                    "page_number": 2,
                    "reviewed_span_ids": [],
                    "reviews": [],
                    "unexpected_math": [],
                    "notes": "",
                },
                "codex",
                "fake",
                None,
                None,
            ),
        ]
    )

    outcome = VisualReviewService(FakeRenderer(2), llm=llm).review(
        _context(tmp_path),
        primary,
        parsed_pdf,
        markdown_bytes=sources.read_bytes(markdown),
        pdf_bytes=sources.read_bytes(pdf),
    )

    assert len(llm.requests) == 2
    assert outcome.page_reviews[0] is None
    span_entry = next(
        entry
        for entry in outcome.entries
        if entry.provenance.get("review_method") == "visual_all_pages"
    )
    assert span_entry.status is ReconciliationStatus.UNREVIEWED
    assert any("paused" in warning for warning in outcome.warnings)


def test_stopped_page_propagates_to_the_outer_durable_run(tmp_path: Path) -> None:
    sources = SourceRepository(tmp_path / "sources")
    markdown = _store(sources, b"# Notes\n$x$.\n", SourceFormat.MARKDOWN)
    pdf = _store(sources, b"%PDF stop fixture", SourceFormat.PDF)
    parser = DocumentParserService(
        sources, pdf_text_extractor=FakePDFTextExtractor(("one",))
    )
    primary = parser.parse_source(markdown)
    parsed_pdf = parser.parse_source(pdf)
    llm = ScriptedLLM([LLMStopped()])

    with pytest.raises(StoppedError, match="visual review LLM task stopped"):
        VisualReviewService(FakeRenderer(1), llm=llm).review(
            _context(tmp_path),
            primary,
            parsed_pdf,
            markdown_bytes=sources.read_bytes(markdown),
            pdf_bytes=sources.read_bytes(pdf),
        )

    assert len(llm.requests) == 1


def test_unreviewed_page_terminal_survives_crash_before_parse_commit_without_calls(
    tmp_path: Path,
) -> None:
    sources = SourceRepository(tmp_path / "sources")
    markdown = _store(sources, b"# Notes\n$x$.\n", SourceFormat.MARKDOWN)
    pdf = _store(sources, b"%PDF terminal fixture", SourceFormat.PDF)
    parser = DocumentParserService(
        sources, pdf_text_extractor=FakePDFTextExtractor(("one",))
    )
    primary = parser.parse_source(markdown)
    parsed_pdf = parser.parse_source(pdf)
    jobs = RunRepository(tmp_path / "jobs")
    snapshot = jobs.create(RunSpec("visual-restart", "test.visual", {"case": "restart"}))
    first_context = RunContext(
        jobs, snapshot, resume_input=None
    )
    first_llm = ScriptedLLM(
        [LLMPaused(ResumeReason.EXTERNAL_CONDITION, "provider-pause")]
    )

    first = VisualReviewService(FakeRenderer(1), llm=first_llm).review(
        first_context,
        primary,
        parsed_pdf,
        markdown_bytes=sources.read_bytes(markdown),
        pdf_bytes=sources.read_bytes(pdf),
    )

    restarted_context = RunContext(
        jobs,
        jobs.inspect("visual-restart").snapshot,
        resume_input=None,
    )
    never = NeverLLM()
    replayed = VisualReviewService(FakeRenderer(1), llm=never).review(
        restarted_context,
        primary,
        parsed_pdf,
        markdown_bytes=sources.read_bytes(markdown),
        pdf_bytes=sources.read_bytes(pdf),
    )

    assert first.page_reviews == replayed.page_reviews == (None,)
    assert first.warnings == replayed.warnings
    assert never.calls == 0


def test_provider_failure_and_invalid_output_page_terminals_replay(
    tmp_path: Path,
) -> None:
    sources = SourceRepository(tmp_path / "sources")
    markdown = _store(sources, b"# Notes\n$x$.\n", SourceFormat.MARKDOWN)
    pdf = _store(sources, b"%PDF terminal variants", SourceFormat.PDF)
    parser = DocumentParserService(
        sources, pdf_text_extractor=FakePDFTextExtractor(("one",))
    )
    primary = parser.parse_source(markdown)
    parsed_pdf = parser.parse_source(pdf)
    cases = (
        LLMFailed(
            ProviderFailure(
                "provider unavailable",
                category=FailureCategory.UNAVAILABLE,
            )
        ),
        LLMCompleted({"not": "the visual schema"}, "codex", "fake", None, None),
    )
    for index, terminal in enumerate(cases, 1):
        jobs = RunRepository(tmp_path / f"jobs-{index}")
        snapshot = jobs.create(
            RunSpec(f"visual-case-{index}", "test.visual", {"case": index})
        )
        context = RunContext(jobs, snapshot, resume_input=None)
        VisualReviewService(
            FakeRenderer(1), llm=ScriptedLLM([terminal])
        ).review(
            context,
            primary,
            parsed_pdf,
            markdown_bytes=sources.read_bytes(markdown),
            pdf_bytes=sources.read_bytes(pdf),
        )
        never = NeverLLM()
        replayed = VisualReviewService(FakeRenderer(1), llm=never).review(
            RunContext(
                jobs,
                jobs.inspect(f"visual-case-{index}").snapshot,
                resume_input=None,
            ),
            primary,
            parsed_pdf,
            markdown_bytes=sources.read_bytes(markdown),
            pdf_bytes=sources.read_bytes(pdf),
        )
        assert replayed.page_reviews == (None,)
        assert replayed.warnings
        assert never.calls == 0


@pytest.mark.parametrize("failure", ["read", "json", "schema"])
def test_corrupt_existing_page_terminal_is_unreviewed_without_rerun(
    tmp_path: Path, monkeypatch, failure: str
) -> None:
    sources = SourceRepository(tmp_path / "sources")
    markdown = _store(sources, b"# Notes\n$x$.\n", SourceFormat.MARKDOWN)
    pdf = _store(sources, b"%PDF corrupt terminal", SourceFormat.PDF)
    parser = DocumentParserService(
        sources, pdf_text_extractor=FakePDFTextExtractor(("one",))
    )
    primary = parser.parse_source(markdown)
    parsed_pdf = parser.parse_source(pdf)
    jobs = RunRepository(tmp_path / "jobs")
    snapshot = jobs.create(
        RunSpec("visual-corrupt-terminal", "test.visual", {"case": failure})
    )
    completed = LLMCompleted(
        {
            "schema_version": "ac.document.visual_page_review.v1",
            "page_number": 1,
            "reviewed_span_ids": [],
            "reviews": [],
            "unexpected_math": [],
            "notes": "",
        },
        "codex",
        "fake",
        None,
        None,
    )
    VisualReviewService(
        FakeRenderer(1), llm=ScriptedLLM([completed])
    ).review(
        RunContext(jobs, snapshot, resume_input=None),
        primary,
        parsed_pdf,
        markdown_bytes=sources.read_bytes(markdown),
        pdf_bytes=sources.read_bytes(pdf),
    )

    restarted = RunContext(
        jobs,
        jobs.inspect("visual-corrupt-terminal").snapshot,
        resume_input=None,
    )
    original_read = restarted.artifacts.read_bytes

    def corrupt_terminal(ref):
        if not ref.artifact_id.startswith("document-visual/terminal/"):
            return original_read(ref)
        if failure == "read":
            raise OSError("terminal object unreadable")
        if failure == "json":
            return b"{not-json"
        return json.dumps(
            {
                "schema_version": "ac.document.visual_page_terminal.v999",
                "page_number": 1,
                "status": "unreviewed",
                "review": None,
                "warning": "old warning",
            }
        ).encode("utf-8")

    monkeypatch.setattr(restarted.artifacts, "read_bytes", corrupt_terminal)
    never = NeverLLM()

    outcome = VisualReviewService(FakeRenderer(1), llm=never).review(
        restarted,
        primary,
        parsed_pdf,
        markdown_bytes=sources.read_bytes(markdown),
        pdf_bytes=sources.read_bytes(pdf),
    )

    assert outcome.page_reviews == (None,)
    assert never.calls == 0
    assert any("corrupt_visual_terminal" in warning for warning in outcome.warnings)
    assert all(
        entry.status is ReconciliationStatus.UNREVIEWED
        for entry in outcome.entries
        if entry.subject_id == "visual-page:1"
        or entry.provenance.get("review_method") == "visual_all_pages"
    )


def test_deterministic_parser_never_invokes_visual_reviewer(
    tmp_path: Path,
) -> None:
    sources = SourceRepository(tmp_path / "sources")
    markdown = _store(sources, b"# Notes\n$x$.\n", SourceFormat.MARKDOWN)
    pdf = _store(sources, b"%PDF review boundary", SourceFormat.PDF)
    parser = DocumentParserService(
        sources,
        pdf_text_extractor=FakePDFTextExtractor(("one",)),
    )

    outcome = parser.parse(
        SourceBundle(primary=markdown, validators=(pdf,)),
    )

    assert outcome.document.source == markdown
    assert any("durable Markdown+PDF visual workflow" in warning for warning in outcome.warnings)
    visual_entries = [
        entry
        for entry in outcome.report.entries
        if entry.subject_id == "visual-page:1"
        or entry.provenance.get("review_method") == "visual_all_pages"
    ]
    assert len(visual_entries) == 2
    assert all(
        entry.status is ReconciliationStatus.UNREVIEWED
        for entry in visual_entries
    )


def test_primary_source_read_failure_is_not_downgraded_to_visual_warning(
    tmp_path: Path, monkeypatch
) -> None:
    sources = SourceRepository(tmp_path / "sources")
    markdown = _store(sources, b"# Notes\n$x$.\n", SourceFormat.MARKDOWN)
    pdf = _store(sources, b"%PDF primary corruption", SourceFormat.PDF)
    original_read = sources.read_bytes
    primary_reads = 0

    def fail_second_primary_read(artifact):
        nonlocal primary_reads
        if artifact.content_identity == markdown.content_identity:
            primary_reads += 1
            if primary_reads == 2:
                raise SourceRepositoryError(
                    "source_corrupt", "authoritative primary became corrupt"
                )
        return original_read(artifact)

    monkeypatch.setattr(sources, "read_bytes", fail_second_primary_read)
    runner = MarkdownPDFVisualParseRunner(
        RunRepository(tmp_path / "jobs"),
        sources,
        renderer=FakeRenderer(1),
        pdf_text_extractor=FakePDFTextExtractor(("one",)),
        llm=NeverLLM(),
    )
    outcome = runner.execute("primary-corrupt", markdown, pdf)

    assert outcome.status is RunStatus.FAILED
    assert outcome.error is not None
    assert outcome.error.code == "source_corrupt"
    assert primary_reads == 2


def test_public_runner_reaches_default_markdown_pdf_full_page_review(
    tmp_path: Path,
) -> None:
    sources = SourceRepository(tmp_path / "sources")
    markdown = _store(sources, b"# Notes\n$x$.\n", SourceFormat.MARKDOWN)
    pdf = _store(sources, b"%PDF runner fixture", SourceFormat.PDF)
    jobs = RunRepository(tmp_path / "jobs")
    llm = ScriptedLLM(
        [
            LLMCompleted(
                {
                    "schema_version": "ac.document.visual_page_review.v1",
                    "page_number": 1,
                    "reviewed_span_ids": [],
                    "reviews": [],
                    "unexpected_math": [],
                    "notes": "",
                },
                "codex",
                "fake",
                None,
                None,
            )
        ]
    )
    runner = MarkdownPDFVisualParseRunner(
        jobs,
        sources,
        renderer=FakeRenderer(1),
        pdf_text_extractor=FakePDFTextExtractor(("one",)),
        llm=llm,
    )

    snapshot = runner.execute(
        "visual-runner",
        markdown,
        pdf,
        model=ModelSelection("codex"),
    )

    assert snapshot.status is RunStatus.SUCCEEDED
    assert len(llm.requests) == 1
    assert snapshot.result_ref is not None
    result = json.loads(
        ImmutableArtifactStore(
            jobs.run_directory("visual-runner"),
            repository_root=jobs.root,
        )
        .read_bytes(snapshot.result_ref)
        .decode("utf-8")
    )
    assert result["schema_version"] == "ac.document.parse_outcome.v1"
    assert result["report"]["policy"] == "visual_all_pages"


def test_public_runner_pauses_when_visual_llm_is_stopped(tmp_path: Path) -> None:
    sources = SourceRepository(tmp_path / "sources")
    markdown = _store(sources, b"# Notes\n$x$.\n", SourceFormat.MARKDOWN)
    pdf = _store(sources, b"%PDF stopped runner fixture", SourceFormat.PDF)
    jobs = RunRepository(tmp_path / "jobs")
    llm = ScriptedLLM([LLMStopped()])
    runner = MarkdownPDFVisualParseRunner(
        jobs,
        sources,
        renderer=FakeRenderer(1),
        pdf_text_extractor=FakePDFTextExtractor(("one",)),
        llm=llm,
    )

    snapshot = runner.execute(
        "visual-runner-stopped",
        markdown,
        pdf,
        model=ModelSelection("codex"),
    )

    assert snapshot.status is RunStatus.PAUSED
    assert snapshot.awaiting is not None
    assert snapshot.awaiting.reason is ResumeReason.EXECUTION_STOPPED
    assert snapshot.result_ref is None
    assert len(llm.requests) == 1


def test_renderer_failure_with_zero_text_pages_marks_spans_unreviewed(
    tmp_path: Path,
) -> None:
    sources = SourceRepository(tmp_path / "sources")
    markdown = _store(sources, b"# Notes\n$x$.\n", SourceFormat.MARKDOWN)
    pdf = _store(sources, b"%PDF no text pages", SourceFormat.PDF)
    parser = DocumentParserService(
        sources, pdf_text_extractor=FakePDFTextExtractor(())
    )
    primary = parser.parse_source(markdown)
    parsed_pdf = parser.parse_source(pdf)

    outcome = VisualReviewService(FailingRenderer()).review(
        _context(tmp_path),
        primary,
        parsed_pdf,
        markdown_bytes=sources.read_bytes(markdown),
        pdf_bytes=sources.read_bytes(pdf),
    )

    span_entry = next(
        entry
        for entry in outcome.entries
        if entry.provenance.get("review_method") == "visual_all_pages"
    )
    assert span_entry.status is ReconciliationStatus.UNREVIEWED
    assert span_entry.provenance["global_unreviewed"] is True


def test_explicit_deterministic_and_tex_pdf_defaults_do_not_call_visual_service(
    tmp_path: Path,
) -> None:
    sources = SourceRepository(tmp_path / "sources")
    markdown = _store(sources, b"# Notes\n$x$.\n", SourceFormat.MARKDOWN)
    tex = _store(sources, br"\section{Notes}" b"\n$x$\n", SourceFormat.TEX)
    pdf = _store(sources, b"%PDF deterministic", SourceFormat.PDF)
    renderer = FakeRenderer(1)
    parser = DocumentParserService(
        sources,
        pdf_text_extractor=FakePDFTextExtractor(("Notes x",)),
    )

    markdown_outcome = parser.parse(
        SourceBundle(primary=markdown, validators=(pdf,)),
        policy=ValidationPolicy.DETERMINISTIC_ONLY,
    )
    tex_outcome = parser.parse(
        SourceBundle(primary=tex, validators=(pdf,)),
    )

    assert markdown_outcome.report.policy is ValidationPolicy.DETERMINISTIC_ONLY
    assert tex_outcome.report.policy is ValidationPolicy.DETERMINISTIC_ONLY
    assert renderer.calls == 0


def test_visual_output_codec_rejects_unknown_and_duplicate_span_ids() -> None:
    value = {
        "schema_version": "ac.document.visual_page_review.v1",
        "page_number": 1,
        "reviewed_span_ids": ["known", "known"],
        "reviews": [
            {
                "span_id": "known",
                "verdict": PageMathVerdict.EXACT.value,
                "observed_math": "x",
                "notes": "",
            },
            {
                "span_id": "known",
                "verdict": PageMathVerdict.EXACT.value,
                "observed_math": "x",
                "notes": "",
            },
        ],
        "unexpected_math": [],
        "notes": "",
    }
    import pytest

    with pytest.raises(ValueError, match="duplicate"):
        decode_visual_page_review(value, expected_page=1, known_span_ids={"known"})
    value["reviewed_span_ids"] = ["unknown"]
    value["reviews"] = [
        {
            "span_id": "unknown",
            "verdict": "exact",
            "observed_math": "x",
            "notes": "",
        }
    ]
    with pytest.raises(ValueError, match="unknown"):
        decode_visual_page_review(value, expected_page=1, known_span_ids={"known"})
    value["reviewed_span_ids"] = ["known"]
    value["reviews"] = [
        {
            "span_id": "known",
            "verdict": "exact",
            "observed_math": "",
            "notes": "",
        }
    ]
    with pytest.raises(ValueError, match="observed math"):
        decode_visual_page_review(value, expected_page=1, known_span_ids={"known"})


def test_pdftoppm_adapter_uses_full_page_scale_and_timeout(
    monkeypatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["timeout"] = kwargs["timeout"]
        Path(f"{command[-1]}-1.png").write_bytes(_png(1000, 2000))
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    renderer = PdftoppmFullPageRenderer(timeout_seconds=3, longest_edge=2000)

    pages = renderer.render(b"%PDF fixture")

    assert len(pages) == 1
    assert observed["timeout"] == 3
    assert observed["command"][1:4] == ("-png", "-scale-to", "2000")
    assert not any("crop" in argument.casefold() for argument in observed["command"])


def test_pdftoppm_adapter_can_render_one_page_without_full_document(
    monkeypatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        Path(f"{command[-1]}.png").write_bytes(_png(1000, 2000))
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    page = PdftoppmFullPageRenderer().render_page(b"%PDF fixture", 7)

    assert page.page_number == 7
    assert observed["command"][1:4] == ("-png", "-scale-to", "2000")
    assert observed["command"][4:10] == ("-f", "7", "-l", "7", "-singlefile", observed["command"][-2])
