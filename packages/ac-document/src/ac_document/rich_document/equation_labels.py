"""Visual, PDF-authoritative equation-label reconciliation for rich documents.

This module deliberately has a much narrower contract than the general PDF
math visual review.  It never guesses from extracted PDF text or page layout:
the model sees each complete rendered page and either establishes one complete
bijection or the caller retains the source labels unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, TypeAlias

from ac_jobs import ArtifactSourceRef, JsonValue, RunContext, StoppedError
from ac_llm import (
    JsonOutput,
    LLMCompleted,
    LLMExecutionOptions,
    LLMFailed,
    LLMInputArtifact,
    LLMPaused,
    LLMRequest,
    LLMStopped,
    LLMTaskService,
    ModelSelection,
    ResumeInput,
    resume_input_matches,
)

from ..parse.visual import PDFPageRenderer, RenderedPDFPage
from ..sources import SourceFormat
from .models import RichBlock, RichBlockKind, RichDocument


EQUATION_LABEL_VISUAL_PROMPT_VERSION = "ac.document.equation_label_visual_prompt.v1"
EQUATION_LABEL_PAGE_REVIEW_SCHEMA = "ac.document.equation_label_page_review.v1"
EQUATION_LABEL_VISUAL_DIAGNOSTIC_SCHEMA = "ac.document.equation_label_visual_diagnostic.v1"

_SIMPLE_POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*\Z")


class _TaskService(Protocol):
    def execute_or_resume(
        self,
        context: RunContext,
        request: LLMRequest,
        *,
        input: ResumeInput | None = None,
        options: LLMExecutionOptions = ...,
    ) -> Any: ...


@dataclass(frozen=True)
class EquationLabelPageMapping:
    """One numbered equation visibly matched on a particular PDF page."""

    block_id: str
    pdf_label: str
    observed_math: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("equation-label mapping requires a block ID")
        if not self.pdf_label.strip():
            raise ValueError("equation-label mapping requires a PDF label")
        if not self.observed_math.strip():
            raise ValueError("equation-label mapping requires observed math")


@dataclass(frozen=True)
class EquationLabelPageReview:
    """A strict, complete structured response for one rendered PDF page."""

    page_number: int
    mappings: tuple[EquationLabelPageMapping, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("equation-label review page must be positive")
        mappings = tuple(self.mappings)
        if len({item.block_id for item in mappings}) != len(mappings):
            raise ValueError("equation-label review has duplicate block IDs")
        if len({item.pdf_label for item in mappings}) != len(mappings):
            raise ValueError("equation-label review has duplicate PDF labels")
        object.__setattr__(self, "mappings", mappings)


@dataclass(frozen=True)
class EquationLabelMapping:
    """A complete mapping ready to overlay onto a ``RichDocument``."""

    block_id: str
    source_label: str
    pdf_label: str
    effective_label: str
    page_number: int
    observed_math: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.block_id or not self.source_label.strip():
            raise ValueError("equation-label mapping has invalid source identity")
        if not self.pdf_label.strip() or not self.effective_label.strip():
            raise ValueError("equation-label mapping has an empty effective label")
        if self.page_number < 1 or not self.observed_math.strip():
            raise ValueError("equation-label mapping has invalid PDF evidence")


@dataclass(frozen=True)
class EquationLabelReviewOutcome:
    """Aggregate review state.  ``mapping`` is populated only when complete."""

    complete: bool
    mapping: tuple[EquationLabelMapping, ...]
    page_reviews: tuple[EquationLabelPageReview | None, ...]
    warnings: tuple[str, ...]
    diagnostics_document: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        mapping = tuple(self.mapping)
        pages = tuple(self.page_reviews)
        warnings = tuple(self.warnings)
        if self.complete != bool(mapping):
            raise ValueError("complete equation-label outcome must have matching state")
        if len({item.block_id for item in mapping}) != len(mapping):
            raise ValueError("equation-label outcome has duplicate block mappings")
        object.__setattr__(self, "mapping", mapping)
        object.__setattr__(self, "page_reviews", pages)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "diagnostics_document", dict(self.diagnostics_document))

    @property
    def applicable(self) -> bool:
        """Whether callers may safely apply the all-or-nothing overlay."""

        return self.complete


EquationLabelReviewResult: TypeAlias = EquationLabelReviewOutcome | LLMPaused


def detect_suspicious_equation_labels(document: RichDocument) -> tuple[str, ...]:
    """Return stable reasons only for unequivocally suspicious simple sequences.

    A visual review is intentionally *not* triggered for documents containing
    unlabelled, appendix, subequation, or otherwise mixed labels.  Those forms
    have legitimate numbering conventions that cannot be inferred from a
    simple global sequence.
    """

    if document.source.source_format is not SourceFormat.HTML:
        return ()
    equations = _display_equations(document)
    if len(equations) < 2:
        return ()
    labels = [str(block.payload["label"]).strip() for block in equations]
    if not all(_SIMPLE_POSITIVE_INTEGER.fullmatch(label) for label in labels):
        return ()
    values = [int(label) for label in labels]
    reasons: list[str] = []
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        reasons.append(
            "duplicate simple-integer equation labels: "
            + ", ".join(str(value) for value in duplicates)
        )
    if any(current < previous for previous, current in zip(values, values[1:])):
        reasons.append("simple-integer equation labels regress in document order")
    if values[0] == 1 and any(value != index for index, value in enumerate(values, 1)):
        reasons.append("simple-integer equation labels have gaps after 1")
    return tuple(reasons)


class EquationLabelReviewService:
    """Review complete PDF pages and accept only an all-or-nothing label map."""

    def __init__(
        self,
        renderer: PDFPageRenderer,
        *,
        llm: _TaskService | None = None,
    ) -> None:
        self.renderer = renderer
        self.llm = llm or LLMTaskService()

    def review(
        self,
        context: RunContext,
        document: RichDocument,
        *,
        pdf_digest: str,
        pdf_bytes: bytes,
        model: ModelSelection = ModelSelection(),
        resume_input: ResumeInput | None = None,
        options: LLMExecutionOptions = LLMExecutionOptions(),
    ) -> EquationLabelReviewResult:
        """Return a complete map, diagnostics, or the exact paused child state.

        Completed child tasks replay through ``LLMTaskService``.  A paused task
        is deliberately not collapsed into a warning or a terminal artifact:
        the outer durable handler must surface the pause and later route its
        resume input back to this method.
        """

        _validate_pdf(pdf_digest, pdf_bytes)
        candidates = _candidate_equations(document)
        if not candidates:
            return _outcome(
                document,
                pdf_digest,
                (),
                (),
                (
                    "PDF visual equation-label review was not run: the document "
                    "has no labelled display equations.",
                ),
            )
        try:
            pages = tuple(self.renderer.render(pdf_bytes))
            _validate_rendered_pages(pages)
        except Exception as exc:
            code = getattr(exc, "code", "pdf_render_failed")
            return _outcome(
                document,
                pdf_digest,
                (),
                (),
                (f"PDF visual equation-label review unavailable ({code}): {exc}",),
            )

        manifest_ref = _publish_json(
            context,
            f"equation-label-visual/manifests/{document.document_digest}",
            _manifest_document(document, candidates),
        )
        reviews: list[EquationLabelPageReview | None] = []
        warnings: list[str] = []
        known_ids = {block.block_id for block in candidates}
        for page in pages:
            page_ref = _publish_bytes(
                context,
                _page_artifact_id(pdf_digest, page),
                page.png_bytes,
                "image/png",
            )
            request = _page_request(
                document=document,
                pdf_digest=pdf_digest,
                page=page,
                page_ref=page_ref,
                manifest_ref=manifest_ref,
                model=model,
            )
            try:
                raw_outcome = _execute_page(
                    self.llm,
                    context,
                    request,
                    resume_input=resume_input,
                    options=options,
                )
            except Exception as exc:
                reviews.append(None)
                warnings.append(
                    f"PDF page {page.page_number} equation-label review failed "
                    f"(visual_review_exception): {exc}"
                )
                continue
            if isinstance(raw_outcome, LLMPaused):
                return raw_outcome
            if isinstance(raw_outcome, LLMStopped):
                raise StoppedError("equation-label visual review LLM task stopped")
            if isinstance(raw_outcome, LLMFailed):
                reviews.append(None)
                warnings.append(
                    f"PDF page {page.page_number} equation-label review failed "
                    f"({raw_outcome.error.code.value}): {raw_outcome.error}"
                )
                continue
            if not isinstance(raw_outcome, LLMCompleted):  # pragma: no cover
                reviews.append(None)
                warnings.append(
                    f"PDF page {page.page_number} equation-label review failed "
                    "(unknown_outcome)"
                )
                continue
            try:
                review = decode_equation_label_page_review(
                    raw_outcome.value,
                    expected_page=page.page_number,
                    known_block_ids=known_ids,
                )
            except (TypeError, ValueError) as exc:
                reviews.append(None)
                warnings.append(
                    f"PDF page {page.page_number} equation-label review failed "
                    f"(invalid_visual_output): {exc}"
                )
                continue
            reviews.append(review)

        mapping, aggregate_warnings = _complete_mapping(candidates, tuple(reviews))
        warnings.extend(aggregate_warnings)
        return _outcome(
            document,
            pdf_digest,
            mapping,
            tuple(reviews),
            tuple(_dedupe(warnings)),
        )


def apply_visual_equation_labels(
    document: RichDocument, outcome: EquationLabelReviewOutcome
) -> RichDocument:
    """Return a new document with a complete visual-PDF reconciliation overlay."""

    if not outcome.complete:
        raise ValueError("incomplete visual equation-label outcome cannot be applied")
    expected = {block.block_id: block for block in _candidate_equations(document)}
    mappings = {item.block_id: item for item in outcome.mapping}
    if set(mappings) != set(expected):
        raise ValueError("visual equation-label outcome does not cover this document")
    provenance: dict[str, dict[str, JsonValue]] = {}
    for block_id, block in expected.items():
        item = mappings[block_id]
        source_label = str(block.payload["label"])
        if item.source_label != source_label:
            raise ValueError("visual equation-label source labels do not match document")
        provenance[block_id] = {
            "source_label": item.source_label,
            "pdf_label": item.pdf_label,
            "effective_label": item.effective_label,
            "page_number": item.page_number,
            "matching_method": "visual_pdf_page",
        }
    metadata = dict(document.metadata)
    metadata["equation_label_reconciliation"] = provenance
    return RichDocument(
        source=document.source,
        blocks=document.blocks,
        sections=document.sections,
        assets=document.assets,
        page_map=document.page_map,
        metadata=metadata,
    )


def decode_equation_label_page_review(
    value: Any,
    *,
    expected_page: int,
    known_block_ids: set[str],
) -> EquationLabelPageReview:
    """Decode one response, rejecting all non-bijective evidence immediately."""

    document = _mapping(value, "equation-label page review")
    _require_fields(
        document,
        {
            "schema_version",
            "page_number",
            "mappings",
            "unmatched_numbered_equations",
            "ambiguities",
            "notes",
        },
        "equation-label page review",
    )
    if document["schema_version"] != EQUATION_LABEL_PAGE_REVIEW_SCHEMA:
        raise ValueError("unsupported equation-label page review schema")
    page_number = _integer(document["page_number"], "page_number")
    if page_number != expected_page:
        raise ValueError("visual output page number does not match the task")
    unmatched = _list(
        document["unmatched_numbered_equations"], "unmatched_numbered_equations"
    )
    if unmatched:
        raise ValueError("visual output contains unmatched numbered equations")
    ambiguities = _list(document["ambiguities"], "ambiguities")
    if ambiguities:
        raise ValueError("visual output contains ambiguous equation matches")
    mappings: list[EquationLabelPageMapping] = []
    for raw in _list(document["mappings"], "mappings"):
        item = _mapping(raw, "equation-label page mapping")
        _require_fields(
            item,
            {"block_id", "pdf_label", "observed_math", "notes"},
            "equation-label page mapping",
        )
        block_id = _string(item["block_id"], "block_id")
        if block_id not in known_block_ids:
            raise ValueError(f"visual output references unknown block ID: {block_id}")
        mappings.append(
            EquationLabelPageMapping(
                block_id=block_id,
                pdf_label=_string(item["pdf_label"], "pdf_label"),
                observed_math=_string(item["observed_math"], "observed_math"),
                notes=_string(item["notes"], "notes"),
            )
        )
    return EquationLabelPageReview(
        page_number=page_number,
        mappings=tuple(mappings),
        notes=_string(document["notes"], "notes"),
    )


def equation_label_page_review_schema() -> Mapping[str, Any]:
    mapping = {
        "type": "object",
        "additionalProperties": False,
        "required": ["block_id", "pdf_label", "observed_math", "notes"],
        "properties": {
            "block_id": {"type": "string", "minLength": 1},
            "pdf_label": {"type": "string", "minLength": 1},
            "observed_math": {"type": "string", "minLength": 1},
            "notes": {"type": "string"},
        },
    }
    unmatched = {
        "type": "object",
        "additionalProperties": False,
        "required": ["pdf_label", "observed_math", "notes"],
        "properties": {
            "pdf_label": {"type": "string", "minLength": 1},
            "observed_math": {"type": "string", "minLength": 1},
            "notes": {"type": "string"},
        },
    }
    ambiguity = {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidate_block_ids", "pdf_label", "notes"],
        "properties": {
            "candidate_block_ids": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "pdf_label": {"type": "string", "minLength": 1},
            "notes": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "page_number",
            "mappings",
            "unmatched_numbered_equations",
            "ambiguities",
            "notes",
        ],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": EQUATION_LABEL_PAGE_REVIEW_SCHEMA,
            },
            "page_number": {"type": "integer", "minimum": 1},
            "mappings": {"type": "array", "items": mapping},
            "unmatched_numbered_equations": {"type": "array", "items": unmatched},
            "ambiguities": {"type": "array", "items": ambiguity},
            "notes": {"type": "string"},
        },
    }


def _display_equations(document: RichDocument) -> tuple[RichBlock, ...]:
    return tuple(
        block
        for block in document.blocks
        if block.kind is RichBlockKind.EQUATION and bool(block.payload["display"])
    )


def _candidate_equations(document: RichDocument) -> tuple[RichBlock, ...]:
    return tuple(
        block
        for block in _display_equations(document)
        if isinstance(block.payload["label"], str) and block.payload["label"].strip()
    )


def _validate_pdf(pdf_digest: str, pdf_bytes: bytes) -> None:
    if not isinstance(pdf_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", pdf_digest):
        raise ValueError("pdf_digest must be a SHA-256 digest")
    if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
        raise ValueError("pdf_bytes must be non-empty bytes")
    if hashlib.sha256(pdf_bytes).hexdigest() != pdf_digest:
        raise ValueError("pdf_digest does not match pdf_bytes")


def _validate_rendered_pages(pages: tuple[RenderedPDFPage, ...]) -> None:
    if not pages:
        raise ValueError("renderer produced no complete pages")
    if tuple(page.page_number for page in pages) != tuple(range(1, len(pages) + 1)):
        raise ValueError("renderer produced non-contiguous page numbers")


def _manifest_document(
    document: RichDocument, candidates: tuple[RichBlock, ...]
) -> dict[str, JsonValue]:
    by_ordinal = {block.ordinal: block for block in document.blocks}
    return {
        "schema_version": "ac.document.equation_label_manifest.v1",
        "document_digest": document.document_digest,
        "equations": [
            {
                "block_id": block.block_id,
                "tex": str(block.payload["tex"]),
                "current_label": str(block.payload["label"]),
                "section_path": list(block.section_path),
                "context_before": _block_context(by_ordinal.get(block.ordinal - 1)),
                "context_after": _block_context(by_ordinal.get(block.ordinal + 1)),
            }
            for block in candidates
        ],
    }


def _block_context(block: RichBlock | None) -> str:
    if block is None:
        return ""
    if block.kind in {RichBlockKind.HEADING, RichBlockKind.PARAGRAPH, RichBlockKind.CODE}:
        return str(block.payload["text"])[:600]
    if block.kind is RichBlockKind.EQUATION:
        return str(block.payload["tex"])[:600]
    if block.kind is RichBlockKind.LIST:
        return " ".join(str(item["text"]) for item in block.payload["items"])[:600]
    if block.kind is RichBlockKind.TABLE:
        return str(block.payload["caption"])[:600]
    return str(block.payload["caption"])[:600]


def _page_artifact_id(pdf_digest: str, page: RenderedPDFPage) -> str:
    page_digest = hashlib.sha256(page.png_bytes).hexdigest()
    return (
        f"equation-label-visual/pages/{pdf_digest}/"
        f"{page.page_number:06d}-{page_digest}"
    )


def _page_request(
    *,
    document: RichDocument,
    pdf_digest: str,
    page: RenderedPDFPage,
    page_ref: ArtifactSourceRef,
    manifest_ref: ArtifactSourceRef,
    model: ModelSelection,
) -> LLMRequest:
    model_document: dict[str, JsonValue] = {
        "provider": model.provider,
        "model": model.model,
        "tier": model.tier,
    }
    if model.reasoning_effort is not None:
        model_document["reasoning_effort"] = model.reasoning_effort
    semantic = {
        "prompt_contract": EQUATION_LABEL_VISUAL_PROMPT_VERSION,
        "output_contract": EQUATION_LABEL_PAGE_REVIEW_SCHEMA,
        "document_digest": document.document_digest,
        "pdf_digest": pdf_digest,
        "page_number": page.page_number,
        "page_png_digest": page_ref.expected_digest.value,
        "manifest_digest": manifest_ref.expected_digest.value,
        "model": model_document,
    }
    task_digest = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return LLMRequest(
        task_id=f"equation-label-page-{task_digest}",
        prompt=(
            f"Inspect complete PDF page {page.page_number} in the `page` PNG. "
            "Use the `equation-manifest` only to identify its numbered display "
            "equations; it is not page-local. Return every visibly numbered "
            "equation on this page once, matched to exactly one manifest block ID. "
            "Use the PDF's printed label as `pdf_label`, without surrounding "
            "parentheses. Set `unmatched_numbered_equations` or `ambiguities` "
            "instead of guessing. A page with no numbered equations returns empty "
            "arrays. Do not modify or suggest source text."
        ),
        output=JsonOutput(equation_label_page_review_schema(), repair="format"),
        model=model,
        inputs=(
            LLMInputArtifact("page", page_ref, "image/png"),
            LLMInputArtifact("equation-manifest", manifest_ref, "application/json"),
        ),
    )


def _execute_page(
    llm: _TaskService,
    context: RunContext,
    request: LLMRequest,
    *,
    resume_input: ResumeInput | None,
    options: LLMExecutionOptions,
) -> Any:
    if resume_input is not None and resume_input_matches(request, resume_input):
        return llm.execute_or_resume(context, request, input=resume_input, options=options)
    return llm.execute_or_resume(context, request, options=options)


def _complete_mapping(
    candidates: tuple[RichBlock, ...],
    reviews: tuple[EquationLabelPageReview | None, ...],
) -> tuple[tuple[EquationLabelMapping, ...], tuple[str, ...]]:
    warnings: list[str] = []
    if any(review is None for review in reviews):
        warnings.append(
            "PDF visual equation-label review is incomplete because one or more "
            "pages did not produce valid terminal evidence."
        )
        return (), tuple(warnings)
    all_mappings = [
        (review.page_number, mapping)
        for review in reviews
        if review is not None
        for mapping in review.mappings
    ]
    occurrences: dict[str, list[tuple[int, EquationLabelPageMapping]]] = {
        block.block_id: [] for block in candidates
    }
    pdf_occurrences: dict[str, list[str]] = {}
    for page_number, mapping in all_mappings:
        occurrences[mapping.block_id].append((page_number, mapping))
        pdf_occurrences.setdefault(mapping.pdf_label, []).append(mapping.block_id)
    duplicate_blocks = sorted(
        block_id for block_id, items in occurrences.items() if len(items) > 1
    )
    missing_blocks = sorted(
        block_id for block_id, items in occurrences.items() if not items
    )
    duplicate_pdf_labels = sorted(
        label for label, block_ids in pdf_occurrences.items() if len(block_ids) > 1
    )
    if duplicate_blocks:
        warnings.append(
            "PDF visual equation-label review is ambiguous: block IDs matched on "
            "multiple pages: " + ", ".join(duplicate_blocks)
        )
    if duplicate_pdf_labels:
        warnings.append(
            "PDF visual equation-label review is ambiguous: duplicate PDF labels: "
            + ", ".join(duplicate_pdf_labels)
        )
    if missing_blocks:
        warnings.append(
            "PDF visual equation-label review is incomplete: labelled display "
            "equations were not matched: " + ", ".join(missing_blocks)
        )
    if warnings:
        return (), tuple(warnings)
    source_by_id = {block.block_id: str(block.payload["label"]) for block in candidates}
    mapping = tuple(
        EquationLabelMapping(
            block_id=item.block_id,
            source_label=source_by_id[item.block_id],
            pdf_label=item.pdf_label,
            effective_label=item.pdf_label,
            page_number=page_number,
            observed_math=item.observed_math,
            notes=item.notes,
        )
        for page_number, item in all_mappings
    )
    return mapping, ()


def _outcome(
    document: RichDocument,
    pdf_digest: str,
    mapping: tuple[EquationLabelMapping, ...],
    page_reviews: tuple[EquationLabelPageReview | None, ...],
    warnings: tuple[str, ...],
) -> EquationLabelReviewOutcome:
    complete = bool(mapping)
    diagnostics: dict[str, JsonValue] = {
        "schema_version": EQUATION_LABEL_VISUAL_DIAGNOSTIC_SCHEMA,
        "status": "complete" if complete else "incomplete",
        "document_digest": document.document_digest,
        "pdf_digest": pdf_digest,
        "mapping": [
            {
                "block_id": item.block_id,
                "source_label": item.source_label,
                "pdf_label": item.pdf_label,
                "effective_label": item.effective_label,
                "page_number": item.page_number,
                "matching_method": "visual_pdf_page",
            }
            for item in mapping
        ],
        "pages": [
            None
            if review is None
            else {
                "page_number": review.page_number,
                "mapping_block_ids": [item.block_id for item in review.mappings],
                "notes": review.notes,
            }
            for review in page_reviews
        ],
        "warnings": list(warnings),
    }
    return EquationLabelReviewOutcome(
        complete=complete,
        mapping=mapping,
        page_reviews=page_reviews,
        warnings=warnings,
        diagnostics_document=diagnostics,
    )


def _publish_bytes(
    context: RunContext, artifact_id: str, content: bytes, media_type: str
) -> ArtifactSourceRef:
    ref = context.artifacts.publish_bytes(artifact_id, content, media_type=media_type)
    return ArtifactSourceRef(context.run_id, ref.artifact_id, ref.digest)


def _publish_json(
    context: RunContext, artifact_id: str, value: Mapping[str, JsonValue]
) -> ArtifactSourceRef:
    ref = context.artifacts.publish_json(artifact_id, dict(value))
    return ArtifactSourceRef(context.run_id, ref.artifact_id, ref.digest)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _require_fields(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} has invalid fields")


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


__all__ = [
    "EQUATION_LABEL_PAGE_REVIEW_SCHEMA",
    "EQUATION_LABEL_VISUAL_DIAGNOSTIC_SCHEMA",
    "EQUATION_LABEL_VISUAL_PROMPT_VERSION",
    "EquationLabelMapping",
    "EquationLabelPageMapping",
    "EquationLabelPageReview",
    "EquationLabelReviewOutcome",
    "EquationLabelReviewResult",
    "EquationLabelReviewService",
    "apply_visual_equation_labels",
    "decode_equation_label_page_review",
    "detect_suspicious_equation_labels",
    "equation_label_page_review_schema",
]
