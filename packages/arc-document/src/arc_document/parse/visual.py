from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol

from arc_jobs import ArtifactSourceRef, RunContext, StoppedError
from arc_llm import (
    JsonOutput,
    LLMStopped,
    LLMCompleted,
    LLMExecutionOptions,
    LLMFailed,
    LLMInputArtifact,
    LLMPaused,
    LLMRequest,
    LLMTaskService,
    ModelSelection,
)

from ..sources import (
    ReconciliationEntry,
    ReconciliationStatus,
    SourceFormat,
)
from .models import MathSpan, ParsedDocument


VISUAL_PAGE_REVIEW_SCHEMA = "arc.document.visual_page_review.v1"
VISUAL_PAGE_TERMINAL_SCHEMA = "arc.document.visual_page_terminal.v1"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PAGE_NAME_RE = re.compile(r"page-(\d+)\.png")


class PDFRenderError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RenderedPDFPage:
    """One complete rendered PDF page.

    The contract intentionally has no bounding-box or crop fields.
    """

    page_number: int
    png_bytes: bytes
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("rendered page number must be positive")
        if self.width < 1 or self.height < 1:
            raise ValueError("rendered page dimensions must be positive")
        actual_width, actual_height = _png_dimensions(self.png_bytes)
        if (actual_width, actual_height) != (self.width, self.height):
            raise ValueError("rendered page dimensions do not match PNG bytes")


class PDFPageRenderer(Protocol):
    def render(self, pdf_bytes: bytes) -> tuple[RenderedPDFPage, ...]: ...


class PdftoppmFullPageRenderer:
    """Narrow subprocess adapter that renders complete pages and nothing else."""

    def __init__(
        self,
        *,
        executable: str = "pdftoppm",
        timeout_seconds: float = 120.0,
        longest_edge: int = 2000,
    ):
        if not executable:
            raise ValueError("renderer executable cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("renderer timeout must be positive")
        if longest_edge < 1:
            raise ValueError("renderer longest_edge must be positive")
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.longest_edge = longest_edge

    def render(self, pdf_bytes: bytes) -> tuple[RenderedPDFPage, ...]:
        if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
            raise PDFRenderError("pdf_render_invalid_input", "PDF bytes are empty")
        with tempfile.TemporaryDirectory(prefix="arc-document-pdf-render-") as directory:
            root = Path(directory)
            source = root / "source.pdf"
            source.write_bytes(pdf_bytes)
            command = (
                self.executable,
                "-png",
                "-scale-to",
                str(self.longest_edge),
                str(source),
                str(root / "page"),
            )
            try:
                completed = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=self.timeout_seconds,
                )
            except FileNotFoundError as exc:
                raise PDFRenderError(
                    "pdf_renderer_unavailable",
                    f"PDF renderer is unavailable: {self.executable}",
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise PDFRenderError(
                    "pdf_render_timeout",
                    f"PDF rendering exceeded {self.timeout_seconds:g} seconds",
                ) from exc
            if completed.returncode != 0:
                detail = completed.stderr.decode("utf-8", errors="replace").strip()
                raise PDFRenderError(
                    "pdf_render_failed",
                    f"PDF renderer failed: {detail[:500] or 'unknown error'}",
                )

            numbered: list[tuple[int, Path]] = []
            for path in root.glob("page-*.png"):
                match = _PAGE_NAME_RE.fullmatch(path.name)
                if match is not None:
                    numbered.append((int(match.group(1)), path))
            numbered.sort()
            if not numbered:
                raise PDFRenderError(
                    "pdf_render_empty", "PDF renderer produced no complete pages"
                )
            if [number for number, _ in numbered] != list(
                range(1, len(numbered) + 1)
            ):
                raise PDFRenderError(
                    "pdf_render_invalid_pages",
                    "PDF renderer produced non-contiguous page numbers",
                )

            pages: list[RenderedPDFPage] = []
            for page_number, path in numbered:
                payload = path.read_bytes()
                width, height = _png_dimensions(payload)
                if max(width, height) > self.longest_edge:
                    raise PDFRenderError(
                        "pdf_render_oversize",
                        "PDF renderer exceeded the configured longest edge",
                    )
                pages.append(RenderedPDFPage(page_number, payload, width, height))
            return tuple(pages)

    def render_page(self, pdf_bytes: bytes, page_number: int) -> RenderedPDFPage:
        """Render one complete page without materializing the whole document."""

        if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
            raise PDFRenderError("pdf_render_invalid_input", "PDF bytes are empty")
        if type(page_number) is not int or page_number < 1:
            raise PDFRenderError("pdf_page_invalid", "PDF page number must be positive")
        with tempfile.TemporaryDirectory(prefix="arc-document-pdf-page-") as directory:
            root = Path(directory)
            source = root / "source.pdf"
            target = root / "page"
            source.write_bytes(pdf_bytes)
            command = (
                self.executable,
                "-png",
                "-scale-to",
                str(self.longest_edge),
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-singlefile",
                str(source),
                str(target),
            )
            try:
                completed = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=self.timeout_seconds,
                )
            except FileNotFoundError as exc:
                raise PDFRenderError(
                    "pdf_renderer_unavailable",
                    f"PDF renderer is unavailable: {self.executable}",
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise PDFRenderError(
                    "pdf_render_timeout",
                    f"PDF rendering exceeded {self.timeout_seconds:g} seconds",
                ) from exc
            if completed.returncode != 0:
                detail = completed.stderr.decode("utf-8", errors="replace").strip()
                raise PDFRenderError(
                    "pdf_render_failed",
                    f"PDF renderer failed: {detail[:500] or 'unknown error'}",
                )
            output = target.with_suffix(".png")
            if not output.is_file():
                raise PDFRenderError("pdf_render_empty", "PDF renderer produced no page")
            payload = output.read_bytes()
            width, height = _png_dimensions(payload)
            if max(width, height) > self.longest_edge:
                raise PDFRenderError(
                    "pdf_render_oversize",
                    "PDF renderer exceeded the configured longest edge",
                )
            return RenderedPDFPage(page_number, payload, width, height)


class PageMathVerdict(str, Enum):
    EXACT = "exact"
    EQUIVALENT = "equivalent"
    MISMATCH = "mismatch"


@dataclass(frozen=True)
class VisualSpanReview:
    span_id: str
    verdict: PageMathVerdict
    observed_math: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.span_id:
            raise ValueError("visual span review requires a span ID")
        if not self.observed_math.strip():
            raise ValueError("visual span review requires observed math")
        object.__setattr__(self, "verdict", PageMathVerdict(self.verdict))


@dataclass(frozen=True)
class UnexpectedPageMath:
    observed_math: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.observed_math.strip():
            raise ValueError("unexpected page math cannot be empty")


@dataclass(frozen=True)
class VisualPageReview:
    page_number: int
    reviewed_span_ids: tuple[str, ...]
    reviews: tuple[VisualSpanReview, ...]
    unexpected_math: tuple[UnexpectedPageMath, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        reviewed = tuple(self.reviewed_span_ids)
        reviews = tuple(self.reviews)
        unexpected = tuple(self.unexpected_math)
        if self.page_number < 1:
            raise ValueError("visual review page number must be positive")
        if len(reviewed) != len(set(reviewed)):
            raise ValueError("visual review contains duplicate reviewed span IDs")
        if tuple(item.span_id for item in reviews) != reviewed:
            raise ValueError(
                "reviewed_span_ids must exactly match review items in output order"
            )
        object.__setattr__(self, "reviewed_span_ids", reviewed)
        object.__setattr__(self, "reviews", reviews)
        object.__setattr__(self, "unexpected_math", unexpected)


@dataclass(frozen=True)
class VisualReviewOutcome:
    entries: tuple[ReconciliationEntry, ...]
    warnings: tuple[str, ...]
    page_reviews: tuple[VisualPageReview | None, ...]


class VisualReviewService:
    """Execute one recoverable LLM task for every complete rendered PDF page."""

    def __init__(
        self,
        renderer: PDFPageRenderer,
        *,
        llm: LLMTaskService | None = None,
        model: ModelSelection = ModelSelection(),
    ):
        self.renderer = renderer
        self.llm = llm or LLMTaskService()
        self.model = model

    def review(
        self,
        context: RunContext,
        primary: ParsedDocument,
        pdf_validator: ParsedDocument,
        *,
        markdown_bytes: bytes,
        pdf_bytes: bytes,
        options: LLMExecutionOptions = LLMExecutionOptions(),
    ) -> VisualReviewOutcome:
        if primary.source.source_format is not SourceFormat.MARKDOWN:
            raise ValueError("visual review requires a Markdown primary")
        if pdf_validator.source.source_format is not SourceFormat.PDF:
            raise ValueError("visual review requires a PDF validator")

        try:
            rendered_pages = self.renderer.render(pdf_bytes)
        except Exception as exc:
            code = getattr(exc, "code", "pdf_render_failed")
            warning = f"PDF visual review unavailable ({code}): {exc}"
            entries = tuple(
                _page_unreviewed_entry(pdf_validator, page.page_number, warning)
                for page in pdf_validator.pages
            )
            entries += _aggregate_span_entries(
                primary.math_spans,
                pdf_validator,
                tuple(None for _ in pdf_validator.pages),
                global_unreviewed=warning,
            )[0]
            return VisualReviewOutcome(
                entries, (warning,), tuple(None for _ in pdf_validator.pages)
            )

        if not rendered_pages:
            warning = (
                "PDF visual review unavailable (pdf_render_empty): "
                "renderer produced no complete pages"
            )
            entries = tuple(
                _page_unreviewed_entry(pdf_validator, page.page_number, warning)
                for page in pdf_validator.pages
            )
            entries += _aggregate_span_entries(
                primary.math_spans,
                pdf_validator,
                tuple(None for _ in pdf_validator.pages),
                global_unreviewed=warning,
            )[0]
            return VisualReviewOutcome(
                entries, (warning,), tuple(None for _ in pdf_validator.pages)
            )

        if pdf_validator.pages and len(rendered_pages) != len(pdf_validator.pages):
            warning = (
                "PDF visual review unavailable (pdf_page_count_mismatch): "
                f"renderer produced {len(rendered_pages)} pages but parser reported "
                f"{len(pdf_validator.pages)}"
            )
            entries = tuple(
                _page_unreviewed_entry(pdf_validator, page.page_number, warning)
                for page in pdf_validator.pages
            )
            entries += _aggregate_span_entries(
                primary.math_spans,
                pdf_validator,
                tuple(None for _ in pdf_validator.pages),
                global_unreviewed=warning,
            )[0]
            return VisualReviewOutcome(
                entries, (warning,), tuple(None for _ in pdf_validator.pages)
            )

        markdown_ref = _publish_input(
            context,
            f"document-visual/markdown/{primary.source.artifact_digest}",
            markdown_bytes,
            "text/markdown",
        )
        manifest_document = _math_manifest(primary.math_spans)
        manifest_ref = _publish_json_input(
            context,
            f"document-visual/manifests/{primary.document_digest}",
            manifest_document,
        )

        reviews: list[VisualPageReview | None] = []
        entries: list[ReconciliationEntry] = []
        warnings: list[str] = []
        known_ids = {span.span_id for span in primary.math_spans}
        for page in rendered_pages:
            page_digest = hashlib.sha256(page.png_bytes).hexdigest()
            page_ref = _publish_input(
                context,
                (
                    f"document-visual/pages/{pdf_validator.source.artifact_digest}/"
                    f"{page.page_number:06d}-{page_digest}"
                ),
                page.png_bytes,
                "image/png",
            )
            request = _page_request(
                page,
                primary,
                pdf_validator,
                model=self.model,
                page_ref=page_ref,
                markdown_ref=markdown_ref,
                manifest_ref=manifest_ref,
            )
            terminal_id = f"document-visual/terminal/{request.task_id}"
            try:
                terminal_ref = context.artifacts.find(terminal_id)
                if terminal_ref is None:
                    terminal_error = None
                else:
                    review, warning = _decode_page_terminal(
                        json.loads(
                            context.artifacts.read_bytes(terminal_ref).decode("utf-8")
                        ),
                        expected_page=page.page_number,
                        known_span_ids=known_ids,
                    )
                    terminal_error = None
            except Exception as exc:
                terminal_ref = None
                terminal_error = (
                    f"PDF page {page.page_number} visual review was unreviewed "
                    f"(corrupt_visual_terminal): {exc}"
                )
                review = None
                warning = terminal_error

            if terminal_error is None and terminal_ref is None:
                try:
                    outcome = self.llm.execute_or_resume(
                        context, request, options=options
                    )
                except Exception as exc:  # provider boundaries must not fail the parse
                    review = None
                    warning = (
                        f"PDF page {page.page_number} visual review was unreviewed "
                        f"(visual_review_exception): {exc}"
                    )
                else:
                    if isinstance(outcome, LLMCompleted):
                        try:
                            review = decode_visual_page_review(
                                outcome.value,
                                expected_page=page.page_number,
                                known_span_ids=known_ids,
                            )
                            warning = ""
                        except (TypeError, ValueError) as exc:
                            review = None
                            warning = (
                                f"PDF page {page.page_number} visual review was unreviewed "
                                f"(invalid_visual_output): {exc}"
                            )
                    elif isinstance(outcome, LLMFailed):
                        review = None
                        warning = (
                            f"PDF page {page.page_number} visual review was unreviewed "
                            f"({outcome.error.code.value}): {outcome.error}"
                        )
                    elif isinstance(outcome, LLMPaused):
                        review = None
                        warning = (
                            f"PDF page {page.page_number} visual review was unreviewed "
                            f"(paused:{outcome.reason.value})"
                        )
                    elif isinstance(outcome, LLMStopped):
                        raise StoppedError("visual review LLM task stopped")
                    else:  # pragma: no cover - closed typed outcome union
                        review = None
                        warning = (
                            f"PDF page {page.page_number} visual review was unreviewed "
                            "(unknown_outcome)"
                        )
                try:
                    context.artifacts.publish_json(
                        terminal_id,
                        _page_terminal_document(
                            page_number=page.page_number,
                            review=review,
                            warning=warning,
                        ),
                    )
                except Exception as exc:
                    review = None
                    warning = (
                        f"PDF page {page.page_number} visual review was unreviewed "
                        f"(visual_terminal_persist_failed): {exc}"
                    )
            reviews.append(review)
            if review is None:
                warnings.append(warning)
                entries.append(
                    _page_unreviewed_entry(
                        pdf_validator, page.page_number, warning
                    )
                )
            else:
                entries.append(
                    ReconciliationEntry(
                        validator=pdf_validator.source,
                        status=ReconciliationStatus.VERIFIED,
                        subject_id=f"visual-page:{page.page_number}",
                        message="complete PDF page received one structured visual review",
                        provenance={
                            "page_number": page.page_number,
                            "reviewed_span_ids": list(review.reviewed_span_ids),
                            "notes": review.notes,
                        },
                    )
                )
                for index, unexpected in enumerate(review.unexpected_math, 1):
                    entries.append(
                        ReconciliationEntry(
                            validator=pdf_validator.source,
                            status=ReconciliationStatus.MISMATCH,
                            subject_id=(
                                f"visual-page:{page.page_number}:unexpected:{index}"
                            ),
                            message="PDF page contains math not matched to the primary",
                            provenance={
                                "page_number": page.page_number,
                                "observed_math": unexpected.observed_math,
                                "notes": unexpected.notes,
                            },
                        )
                    )
                    warnings.append(
                        f"PDF page {page.page_number} contains unexpected math"
                    )

        span_entries, span_warnings = _aggregate_span_entries(
            primary.math_spans, pdf_validator, tuple(reviews)
        )
        entries.extend(span_entries)
        warnings.extend(span_warnings)
        return VisualReviewOutcome(
            tuple(entries), tuple(_dedupe(warnings)), tuple(reviews)
        )


def decode_visual_page_review(
    value: Any,
    *,
    expected_page: int,
    known_span_ids: set[str],
) -> VisualPageReview:
    document = _mapping(value, "visual page review")
    _require_fields(
        document,
        {
            "schema_version",
            "page_number",
            "reviewed_span_ids",
            "reviews",
            "unexpected_math",
            "notes",
        },
        "visual page review",
    )
    if document["schema_version"] != VISUAL_PAGE_REVIEW_SCHEMA:
        raise ValueError("unsupported visual page review schema")
    page_number = _integer(document["page_number"], "page_number")
    if page_number != expected_page:
        raise ValueError("visual output page number does not match the task")
    reviewed_raw = _list(document["reviewed_span_ids"], "reviewed_span_ids")
    reviewed_ids = tuple(_string(item, "reviewed span ID") for item in reviewed_raw)
    unknown = set(reviewed_ids) - known_span_ids
    if unknown:
        raise ValueError(f"visual output references unknown span IDs: {sorted(unknown)}")
    reviews: list[VisualSpanReview] = []
    for raw in _list(document["reviews"], "reviews"):
        item = _mapping(raw, "visual span review")
        _require_fields(
            item, {"span_id", "verdict", "observed_math", "notes"}, "visual span review"
        )
        reviews.append(
            VisualSpanReview(
                span_id=_string(item["span_id"], "span_id"),
                verdict=PageMathVerdict(_string(item["verdict"], "verdict")),
                observed_math=_string(item["observed_math"], "observed_math"),
                notes=_string(item["notes"], "notes"),
            )
        )
    unexpected: list[UnexpectedPageMath] = []
    for raw in _list(document["unexpected_math"], "unexpected_math"):
        item = _mapping(raw, "unexpected page math")
        _require_fields(item, {"observed_math", "notes"}, "unexpected page math")
        unexpected.append(
            UnexpectedPageMath(
                observed_math=_string(item["observed_math"], "observed_math"),
                notes=_string(item["notes"], "notes"),
            )
        )
    return VisualPageReview(
        page_number=page_number,
        reviewed_span_ids=reviewed_ids,
        reviews=tuple(reviews),
        unexpected_math=tuple(unexpected),
        notes=_string(document["notes"], "notes"),
    )


def visual_page_review_schema() -> Mapping[str, Any]:
    span_review = {
        "type": "object",
        "additionalProperties": False,
        "required": ["span_id", "verdict", "observed_math", "notes"],
        "properties": {
            "span_id": {"type": "string", "minLength": 1},
            "verdict": {
                "type": "string",
                "enum": [item.value for item in PageMathVerdict],
            },
            "observed_math": {"type": "string", "minLength": 1},
            "notes": {"type": "string"},
        },
    }
    unexpected = {
        "type": "object",
        "additionalProperties": False,
        "required": ["observed_math", "notes"],
        "properties": {
            "observed_math": {"type": "string", "minLength": 1},
            "notes": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "page_number",
            "reviewed_span_ids",
            "reviews",
            "unexpected_math",
            "notes",
        ],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": VISUAL_PAGE_REVIEW_SCHEMA,
            },
            "page_number": {"type": "integer", "minimum": 1},
            "reviewed_span_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
            },
            "reviews": {"type": "array", "items": span_review},
            "unexpected_math": {"type": "array", "items": unexpected},
            "notes": {"type": "string"},
        },
    }


def _visual_page_review_document(review: VisualPageReview) -> dict[str, Any]:
    return {
        "schema_version": VISUAL_PAGE_REVIEW_SCHEMA,
        "page_number": review.page_number,
        "reviewed_span_ids": list(review.reviewed_span_ids),
        "reviews": [
            {
                "span_id": item.span_id,
                "verdict": item.verdict.value,
                "observed_math": item.observed_math,
                "notes": item.notes,
            }
            for item in review.reviews
        ],
        "unexpected_math": [
            {"observed_math": item.observed_math, "notes": item.notes}
            for item in review.unexpected_math
        ],
        "notes": review.notes,
    }


def _page_terminal_document(
    *,
    page_number: int,
    review: VisualPageReview | None,
    warning: str,
) -> dict[str, Any]:
    return {
        "schema_version": VISUAL_PAGE_TERMINAL_SCHEMA,
        "page_number": page_number,
        "status": "reviewed" if review is not None else "unreviewed",
        "review": (
            _visual_page_review_document(review) if review is not None else None
        ),
        "warning": warning,
    }


def _decode_page_terminal(
    value: Any,
    *,
    expected_page: int,
    known_span_ids: set[str],
) -> tuple[VisualPageReview | None, str]:
    document = _mapping(value, "visual page terminal")
    _require_fields(
        document,
        {"schema_version", "page_number", "status", "review", "warning"},
        "visual page terminal",
    )
    if document["schema_version"] != VISUAL_PAGE_TERMINAL_SCHEMA:
        raise ValueError("unsupported visual page terminal schema")
    if _integer(document["page_number"], "page_number") != expected_page:
        raise ValueError("visual page terminal page number does not match the task")
    status = _string(document["status"], "status")
    warning = _string(document["warning"], "warning")
    if status == "reviewed":
        if warning:
            raise ValueError("reviewed visual page terminal cannot contain a warning")
        return (
            decode_visual_page_review(
                document["review"],
                expected_page=expected_page,
                known_span_ids=known_span_ids,
            ),
            "",
        )
    if status == "unreviewed":
        if document["review"] is not None or not warning:
            raise ValueError("unreviewed visual page terminal requires only a warning")
        return None, warning
    raise ValueError("visual page terminal has an invalid status")


def _page_request(
    page: RenderedPDFPage,
    primary: ParsedDocument,
    pdf_validator: ParsedDocument,
    *,
    model: ModelSelection,
    page_ref: ArtifactSourceRef,
    markdown_ref: ArtifactSourceRef,
    manifest_ref: ArtifactSourceRef,
) -> LLMRequest:
    semantic = {
        "contract": VISUAL_PAGE_REVIEW_SCHEMA,
        "markdown_digest": primary.source.artifact_digest,
        "pdf_digest": pdf_validator.source.artifact_digest,
        "page_number": page.page_number,
        "page_png_digest": page_ref.expected_digest.value,
        "manifest_digest": manifest_ref.expected_digest.value,
        "model": {
            "provider": model.provider,
            "model": model.model,
            "tier": model.tier,
        },
    }
    task_digest = hashlib.sha256(
        json.dumps(
            semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    return LLMRequest(
        task_id=f"page-review-{task_digest}",
        prompt=(
            f"Review complete PDF page {page.page_number}. Read the full-page PNG "
            "input `page`, the original Markdown input `markdown`, and the complete "
            "MathSpan manifest input `math-manifest`. Identify only math visibly on "
            "this page. For each matched manifest span, return one exact, equivalent, "
            "or mismatch review. Put visible math with no manifest match in "
            "unexpected_math. A page with no math must still return a valid record "
            "with empty reviewed_span_ids, reviews, and unexpected_math. Do not "
            "suggest or apply source corrections."
        ),
        output=JsonOutput(visual_page_review_schema(), repair="strict"),
        model=model,
        inputs=(
            LLMInputArtifact("page", page_ref, "image/png"),
            LLMInputArtifact("markdown", markdown_ref, "text/markdown"),
            LLMInputArtifact("math-manifest", manifest_ref, "application/json"),
        ),
    )


def _math_manifest(spans: tuple[MathSpan, ...]) -> dict[str, Any]:
    return {
        "schema_version": "arc.document.math_span_manifest.v1",
        "spans": [
            {
                "span_id": span.span_id,
                "kind": span.kind.value,
                "source_line_start": span.source_line_start,
                "source_column_start": span.source_column_start,
                "source_line_end": span.source_line_end,
                "source_column_end": span.source_column_end,
                "normalized_tex": span.normalized_tex,
                "context_before": span.context_before,
                "context_after": span.context_after,
                "source_label": span.source_label,
            }
            for span in spans
        ],
    }


def _publish_input(
    context: RunContext,
    artifact_id: str,
    content: bytes,
    media_type: str,
) -> ArtifactSourceRef:
    ref = context.artifacts.publish_bytes(artifact_id, content, media_type=media_type)
    return ArtifactSourceRef(context.run_id, ref.artifact_id, ref.digest)


def _publish_json_input(
    context: RunContext, artifact_id: str, value: Mapping[str, Any]
) -> ArtifactSourceRef:
    ref = context.artifacts.publish_json(artifact_id, dict(value))
    return ArtifactSourceRef(context.run_id, ref.artifact_id, ref.digest)


def _aggregate_span_entries(
    spans: tuple[MathSpan, ...],
    pdf_validator: ParsedDocument,
    page_reviews: tuple[VisualPageReview | None, ...],
    *,
    global_unreviewed: str = "",
) -> tuple[tuple[ReconciliationEntry, ...], tuple[str, ...]]:
    failed_pages = {
        index for index, review in enumerate(page_reviews, 1) if review is None
    }
    occurrences: dict[str, list[tuple[int, VisualSpanReview]]] = {
        span.span_id: [] for span in spans
    }
    for page_number, review in enumerate(page_reviews, 1):
        if review is None:
            continue
        for item in review.reviews:
            occurrences[item.span_id].append((page_number, item))

    entries: list[ReconciliationEntry] = []
    warnings: list[str] = []
    for span in spans:
        matches = occurrences[span.span_id]
        provenance: dict[str, Any] = {"review_method": "visual_all_pages"}
        if not matches and global_unreviewed:
            status = ReconciliationStatus.UNREVIEWED
            message = global_unreviewed
            provenance["global_unreviewed"] = True
        elif not matches and failed_pages:
            status = ReconciliationStatus.UNREVIEWED
            message = "span could not be conclusively covered because pages were unreviewed"
            provenance["unreviewed_pages"] = sorted(failed_pages)
        elif not matches:
            status = ReconciliationStatus.MISSING
            message = "span was not found on any visually reviewed PDF page"
        elif len(matches) > 1:
            status = ReconciliationStatus.AMBIGUOUS
            message = "span was reported on more than one PDF page"
            provenance["page_candidates"] = [page for page, _ in matches]
            provenance["verdicts"] = [item.verdict.value for _, item in matches]
        else:
            page_number, item = matches[0]
            status = (
                ReconciliationStatus.MISMATCH
                if item.verdict is PageMathVerdict.MISMATCH
                else ReconciliationStatus.VERIFIED
            )
            message = (
                "visual PDF math differs from the authoritative primary span"
                if status is ReconciliationStatus.MISMATCH
                else "visual PDF math agrees with the authoritative primary span"
            )
            provenance.update(
                {
                    "page_number": page_number,
                    "verdict": item.verdict.value,
                    "observed_math": item.observed_math,
                    "notes": item.notes,
                }
            )
        entries.append(
            ReconciliationEntry(
                validator=pdf_validator.source,
                status=status,
                subject_id=span.span_id,
                message=message,
                provenance=provenance,
            )
        )
        if status is not ReconciliationStatus.VERIFIED:
            warnings.append(
                f"PDF visual math evidence {status.value} for {span.span_id}"
            )
    return tuple(entries), tuple(warnings)


def _page_unreviewed_entry(
    pdf_validator: ParsedDocument, page_number: int, message: str
) -> ReconciliationEntry:
    return ReconciliationEntry(
        validator=pdf_validator.source,
        status=ReconciliationStatus.UNREVIEWED,
        subject_id=f"visual-page:{page_number}",
        message=message,
        provenance={"page_number": page_number},
    )


def _png_dimensions(payload: bytes) -> tuple[int, int]:
    if len(payload) < 24 or not payload.startswith(_PNG_SIGNATURE):
        raise PDFRenderError("pdf_render_invalid_png", "renderer output is not a PNG")
    width = int.from_bytes(payload[16:20], "big")
    height = int.from_bytes(payload[20:24], "big")
    if width < 1 or height < 1:
        raise PDFRenderError(
            "pdf_render_invalid_png", "renderer output has invalid PNG dimensions"
        )
    return width, height


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    return value


def _require_fields(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields do not match the v1 contract")


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


__all__ = [
    "PDFPageRenderer",
    "PDFRenderError",
    "PageMathVerdict",
    "PdftoppmFullPageRenderer",
    "RenderedPDFPage",
    "UnexpectedPageMath",
    "VISUAL_PAGE_REVIEW_SCHEMA",
    "VisualPageReview",
    "VisualReviewOutcome",
    "VisualReviewService",
    "VisualSpanReview",
    "decode_visual_page_review",
    "visual_page_review_schema",
]
