"""In-run Markdown plus PDF parsing with default full-page visual review."""

from __future__ import annotations

from typing import Mapping

from ac_jobs import (
    Failed,
    JsonValue,
    RunContext,
    RunEngine,
    RunError,
    RunRepository,
    RunSnapshot,
    RunSpec,
    StoppedError,
    Succeeded,
)
from ac_llm import LLMExecutionOptions, LLMTaskService, ModelSelection

from ..parse import (
    PDFPageRenderer,
    PDFTextExtractor,
    DocumentParserService,
    ParseError,
    PdftoppmFullPageRenderer,
    PdftotextExtractor,
    VisualReviewService,
    parsed_document_to_document,
)
from ..parse.reconcile import reconcile_validator
from ..parse.service import _dedupe, _unreviewed_visual_entries
from ..source_repository import SourceRepository, SourceRepositoryError
from ..sources import (
    ParseOutcome,
    ReconciliationEntry,
    ReconciliationReport,
    ReconciliationStatus,
    SourceArtifact,
    SourceFormat,
    ValidationPolicy,
)


MARKDOWN_PDF_VISUAL_HANDLER = "ac.document.markdown_pdf_visual_parse.v1"
PARSE_OUTCOME_SCHEMA = "ac.document.parse_outcome.v1"


class MarkdownPDFVisualParseHandler:
    """Portable RunHandler for the default Markdown+PDF visual contract."""

    name = MARKDOWN_PDF_VISUAL_HANDLER

    def __init__(
        self,
        sources: SourceRepository,
        primary: SourceArtifact,
        pdf_validator: SourceArtifact,
        *,
        renderer: PDFPageRenderer | None = None,
        pdf_text_extractor: PDFTextExtractor | None = None,
        llm: LLMTaskService | None = None,
        model: ModelSelection = ModelSelection(),
        options: LLMExecutionOptions = LLMExecutionOptions(),
    ) -> None:
        if primary.source_format is not SourceFormat.MARKDOWN:
            raise ValueError("Markdown+PDF visual parse requires a Markdown primary")
        if pdf_validator.source_format is not SourceFormat.PDF:
            raise ValueError("Markdown+PDF visual parse requires a PDF validator")
        self.primary = primary
        self.pdf_validator = pdf_validator
        self.model = model
        self.options = options
        self.sources = sources
        self.reviewer = VisualReviewService(
            renderer or PdftoppmFullPageRenderer(),
            llm=llm,
            model=model,
        )
        self.service = DocumentParserService(
            sources,
            pdf_text_extractor=pdf_text_extractor or PdftotextExtractor(),
        )

    def semantic_input(self) -> dict[str, JsonValue]:
        return {
            "schema_version": "ac.document.markdown_pdf_visual_parse_request.v1",
            "primary": _artifact_semantic_document(self.primary),
            "pdf_validator": _artifact_semantic_document(self.pdf_validator),
            "validation_policy": ValidationPolicy.VISUAL_ALL_PAGES.value,
            "model_requirement": {
                "provider": self.model.provider,
                "model": self.model.model,
                "tier": self.model.tier,
            },
        }

    def execute(self, context: RunContext):
        if dict(context.semantic_input) != self.semantic_input():
            return Failed(
                RunError(
                    "parse_binding_mismatch",
                    "Handler bindings differ from the durable parse semantic input.",
                )
            )
        try:
            primary, primary_cache_warnings = self.service.materialize_source(
                self.primary
            )
        except (ParseError, SourceRepositoryError) as exc:
            return Failed(
                RunError(
                    getattr(exc, "code", "primary_parse_failed"),
                    str(exc),
                )
            )
        warnings = list(primary.warnings + primary_cache_warnings)
        try:
            parsed_pdf, validator_cache_warnings = (
                self.service.materialize_source(self.pdf_validator)
            )
        except (ParseError, SourceRepositoryError) as exc:
            code = getattr(exc, "code", "validator_parse_failed")
            message = f"validator could not be parsed ({code}): {exc}"
            outcome = ParseOutcome(
                document=primary,
                report=ReconciliationReport(
                    primary=self.primary,
                    policy=ValidationPolicy.VISUAL_ALL_PAGES,
                    entries=(
                        ReconciliationEntry(
                            validator=self.pdf_validator,
                            status=ReconciliationStatus.UNREVIEWED,
                            subject_id="validator",
                            message=message,
                            provenance={"error_code": code},
                        ),
                    ),
                ),
                warnings=tuple(_dedupe(warnings + [message])),
            )
        else:
            deterministic, deterministic_warnings = reconcile_validator(
                primary, parsed_pdf
            )
            primary_span_ids = {span.span_id for span in primary.math_spans}
            entries = [
                ReconciliationEntry(
                    validator=entry.validator,
                    status=entry.status,
                    subject_id=f"deterministic:{entry.subject_id}",
                    message=entry.message,
                    provenance={
                        **dict(entry.provenance),
                        "evidence_method": "deterministic_pdf",
                        "primary_span_id": entry.subject_id,
                    },
                )
                if entry.subject_id in primary_span_ids
                else entry
                for entry in deterministic
            ]
            warnings.extend(parsed_pdf.warnings)
            warnings.extend(validator_cache_warnings)
            warnings.extend(deterministic_warnings)
            try:
                markdown_bytes = self.sources.read_bytes(self.primary)
            except SourceRepositoryError as exc:
                return Failed(RunError(exc.code, str(exc)))
            try:
                visual = self.reviewer.review(
                    context,
                    primary,
                    parsed_pdf,
                    markdown_bytes=markdown_bytes,
                    pdf_bytes=self.sources.read_bytes(self.pdf_validator),
                    options=self.options,
                )
            except StoppedError:
                raise
            except Exception as exc:
                message = (
                    "PDF visual review was unavailable "
                    f"(visual_review_service_error): {exc}"
                )
                entries.extend(
                    _unreviewed_visual_entries(primary, parsed_pdf, message)
                )
                warnings.append(message)
            else:
                entries.extend(visual.entries)
                warnings.extend(visual.warnings)
            outcome = ParseOutcome(
                document=primary,
                report=ReconciliationReport(
                    primary=self.primary,
                    policy=ValidationPolicy.VISUAL_ALL_PAGES,
                    entries=tuple(entries),
                ),
                warnings=tuple(_dedupe(warnings)),
            )
        return Succeeded(
            context.artifacts.publish_json(
                "document-parse/result", parse_outcome_to_document(outcome)
            )
        )


class MarkdownPDFVisualParseRunner:
    """Thin run wrapper that installs the renderer, reviewer, and RunContext."""

    def __init__(
        self,
        jobs: RunRepository,
        sources: SourceRepository,
        *,
        renderer: PDFPageRenderer | None = None,
        pdf_text_extractor: PDFTextExtractor | None = None,
        llm: LLMTaskService | None = None,
    ) -> None:
        self.engine = RunEngine(jobs)
        self.sources = sources
        self.renderer = renderer
        self.pdf_text_extractor = pdf_text_extractor
        self.llm = llm

    def execute(
        self,
        run_id: str,
        primary: SourceArtifact,
        pdf_validator: SourceArtifact,
        *,
        model: ModelSelection = ModelSelection(),
        options: LLMExecutionOptions = LLMExecutionOptions(),
    ) -> RunSnapshot:
        handler = self._handler(primary, pdf_validator, model=model, options=options)
        return self.engine.execute(
            RunSpec(run_id, handler.name, handler.semantic_input()), handler
        )

    def resume(
        self,
        run_id: str,
        primary: SourceArtifact,
        pdf_validator: SourceArtifact,
        *,
        input: Mapping[str, JsonValue] | None = None,
        model: ModelSelection = ModelSelection(),
        options: LLMExecutionOptions = LLMExecutionOptions(),
    ) -> RunSnapshot:
        handler = self._handler(primary, pdf_validator, model=model, options=options)
        return self.engine.resume(run_id, handler, input=input)

    def _handler(
        self,
        primary: SourceArtifact,
        pdf_validator: SourceArtifact,
        *,
        model: ModelSelection,
        options: LLMExecutionOptions,
    ) -> MarkdownPDFVisualParseHandler:
        return MarkdownPDFVisualParseHandler(
            self.sources,
            primary,
            pdf_validator,
            renderer=self.renderer,
            pdf_text_extractor=self.pdf_text_extractor,
            llm=self.llm,
            model=model,
            options=options,
        )


def parse_outcome_to_document(outcome: ParseOutcome) -> dict[str, JsonValue]:
    """Encode a path-free parse result for publication by ac-jobs."""

    return {
        "schema_version": PARSE_OUTCOME_SCHEMA,
        "document": parsed_document_to_document(outcome.document),
        "report": {
            "primary": _artifact_semantic_document(outcome.report.primary),
            "policy": outcome.report.policy.value,
            "entries": [_entry_document(item) for item in outcome.report.entries],
        },
        "warnings": list(outcome.warnings),
    }


def _entry_document(entry: ReconciliationEntry) -> dict[str, JsonValue]:
    return {
        "validator": _artifact_semantic_document(entry.validator),
        "status": entry.status.value,
        "subject_id": entry.subject_id,
        "message": entry.message,
        "provenance": dict(entry.provenance),
    }


def _artifact_semantic_document(
    artifact: SourceArtifact,
) -> dict[str, JsonValue]:
    return {
        "source_format": artifact.source_format.value,
        "media_type": artifact.media_type,
        "artifact_digest": artifact.artifact_digest,
        "size": artifact.size,
    }


__all__ = [
    "MARKDOWN_PDF_VISUAL_HANDLER",
    "PARSE_OUTCOME_SCHEMA",
    "MarkdownPDFVisualParseHandler",
    "MarkdownPDFVisualParseRunner",
    "parse_outcome_to_document",
]
