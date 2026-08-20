from __future__ import annotations

import mimetypes
import re
import unicodedata
from dataclasses import dataclass

from .._parsing import ParseError
from ..parse.parser import PDFTextExtractor
from ..parse.reconcile import reconcile_validator
from ..parse.service import DocumentParserService
from ..source_repository import SourceRepository, SourceRepositoryError
from ..sources import (
    ReconciliationEntry,
    ReconciliationReport,
    ReconciliationStatus,
    SourceBundle,
    SourceFormat,
    ValidationPolicy,
)
from .models import RichAsset, RichBlockKind, RichDocument, RichPageMapEntry
from .parser import AssetImporter, parse_rich_artifact_bytes, resolve_local_asset_path


PDF_VALIDATOR_MISSING_WARNING = (
    "no PDF validator was supplied; rich source structure remains authoritative"
)


class RichDocumentValidationError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: tuple[dict[str, object], ...] = (),
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = tuple(dict(item) for item in details)


@dataclass(frozen=True)
class RichParseOutcome:
    document: RichDocument
    report: ReconciliationReport
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.document.source.content_identity
            != self.report.primary.content_identity
        ):
            raise ValueError("rich document source does not match report primary")
        object.__setattr__(self, "warnings", tuple(self.warnings))


class RichDocumentParserService:
    """Repository-backed public facade for rich source parsing and PDF checks."""

    def __init__(
        self,
        repository: SourceRepository,
        *,
        pdf_text_extractor: PDFTextExtractor | None = None,
        asset_importer: AssetImporter | None = None,
    ):
        self.repository = repository
        self.pdf_text_extractor = pdf_text_extractor
        self.asset_importer = asset_importer
        self.standard_parser = DocumentParserService(
            repository,
            pdf_text_extractor=pdf_text_extractor,
        )

    def parse(self, bundle: SourceBundle) -> RichParseOutcome:
        if bundle.primary.source_format not in {
            SourceFormat.MARKDOWN,
            SourceFormat.HTML,
            SourceFormat.TEX,
        }:
            raise RichDocumentValidationError(
                "rich_source_required",
                "the primary must be Markdown, HTML, or flattened single-file TeX",
            )
        if any(
            validator.source_format is not SourceFormat.PDF
            for validator in bundle.validators
        ):
            raise RichDocumentValidationError(
                "invalid_rich_validator",
                "rich document validation accepts only an optional PDF validator",
            )
        if len(bundle.validators) > 1:
            raise RichDocumentValidationError(
                "multiple_pdf_validators",
                "rich document parsing accepts at most one PDF validator",
            )
        payload = self.repository.read_bytes(bundle.primary)
        parsed = parse_rich_artifact_bytes(
            bundle.primary,
            payload,
            asset_importer=self._asset_importer(bundle.primary.origin.locator),
        )
        if not bundle.validators:
            return RichParseOutcome(
                document=parsed.document,
                report=ReconciliationReport(
                    primary=bundle.primary,
                    policy=ValidationPolicy.DETERMINISTIC_ONLY,
                ),
                warnings=parsed.warnings + (PDF_VALIDATOR_MISSING_WARNING,),
            )

        standard_primary = self.standard_parser.parse_source(bundle.primary)
        validator = bundle.validators[0]
        try:
            parsed_validator = self.standard_parser.parse_source(validator)
        except (ParseError, SourceRepositoryError) as exc:
            code = getattr(exc, "code", "pdf_validator_invalid")
            raise RichDocumentValidationError(
                "pdf_validator_invalid",
                f"PDF validator could not be parsed ({code}): {exc}",
            ) from exc
        if not bool(parsed_validator.metadata.get("text_layer")):
            raise RichDocumentValidationError(
                "pdf_validator_unverifiable",
                "PDF validator has no extractable text layer",
            )
        entries, reconciliation_warnings = reconcile_validator(
            standard_primary, parsed_validator
        )
        conflicts = [
            entry for entry in entries if _is_fatal_pdf_reconciliation_entry(entry)
        ]
        if conflicts:
            status = (
                "ambiguous"
                if any(
                    entry.status is ReconciliationStatus.AMBIGUOUS
                    for entry in conflicts
                )
                else "mismatch"
            )
            raise RichDocumentValidationError(
                f"pdf_validator_{status}",
                f"PDF validator {status} for {len(conflicts)} source subjects",
                details=tuple(_conflict_detail(entry) for entry in conflicts),
            )
        equation_provenance, equation_warning = _equation_label_provenance(
            parsed.document, entries
        )
        page_map = parsed.document.page_map
        if page_map:
            maximum_page = max(item.page_number for item in page_map)
            if maximum_page > len(parsed_validator.pages):
                raise RichDocumentValidationError(
                    "rich_page_marker_out_of_range",
                    "Markdown source page markers exceed the PDF page count",
                )
        else:
            page_map = _build_page_map(
                parsed.document,
                entries,
                standard_primary.sections,
                parsed_validator.pages,
            )
        page_map = _with_equation_pages(page_map, equation_provenance)
        metadata = dict(parsed.document.metadata)
        if equation_provenance:
            metadata["equation_label_reconciliation"] = equation_provenance
        document = RichDocument(
            source=parsed.document.source,
            blocks=parsed.document.blocks,
            sections=parsed.document.sections,
            assets=parsed.document.assets,
            page_map=page_map,
            metadata=metadata,
        )
        return RichParseOutcome(
            document=document,
            report=ReconciliationReport(
                primary=bundle.primary,
                policy=ValidationPolicy.DETERMINISTIC_ONLY,
                entries=entries,
            ),
            warnings=tuple(
                dict.fromkeys(
                    parsed.warnings
                    + reconciliation_warnings
                    + ((equation_warning,) if equation_warning else ())
                )
            ),
        )

    def parse_source(self, artifact) -> RichDocument:
        """Parse a primary rich artifact without accepting a validator."""

        return self.parse(SourceBundle(primary=artifact)).document

    def _asset_importer(self, source_locator: str):
        def import_asset(target: str) -> RichAsset | None:
            if self.asset_importer is not None:
                imported = self.asset_importer(target)
                if imported is not None:
                    return imported
            if not source_locator:
                return None
            path = resolve_local_asset_path(source_locator, target)
            if path is None:
                return None
            if not path.is_file() and not path.suffix:
                path = next(
                    (
                        candidate
                        for suffix in (
                            ".pdf",
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".svg",
                            ".eps",
                        )
                        if (candidate := path.with_suffix(suffix)).is_file()
                    ),
                    path,
                )
            if not path.is_file():
                return None
            media_type = (
                mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            )
            stored = self.repository.import_asset_path(path, media_type=media_type)
            return RichAsset(
                artifact_digest=stored.artifact_digest,
                media_type=stored.media_type,
                logical_name=target,
                size=stored.size,
            )

        return import_asset


def _build_page_map(
    document: RichDocument,
    entries: tuple[ReconciliationEntry, ...],
    standard_sections,
    pages,
) -> tuple[RichPageMapEntry, ...]:
    entries_by_subject = {entry.subject_id: entry for entry in entries}
    unmatched_standard = list(standard_sections)
    heading_page_by_section: dict[str, int] = {}
    for section in document.sections:
        starts_with_heading = (
            section.block_start < len(document.blocks)
            and document.blocks[section.block_start].kind
            is RichBlockKind.HEADING
        )
        if not starts_with_heading and not (
            len(document.sections) == 1 and section.title == "Document"
        ):
            continue
        match_index = next(
            (
                index
                for index, standard in enumerate(unmatched_standard)
                if standard.title == section.title
                and standard.level == section.level
            ),
            None,
        )
        if match_index is None:
            continue
        standard = unmatched_standard.pop(match_index)
        entry = entries_by_subject.get(f"section:{standard.section_id}")
        if entry is None or entry.status is not ReconciliationStatus.VERIFIED:
            continue
        evidence_pages = entry.provenance.get("page_candidates")
        if (
            isinstance(evidence_pages, list)
            and len(evidence_pages) == 1
            and isinstance(evidence_pages[0], int)
        ):
            heading_page_by_section[section.section_id] = evidence_pages[0]

    page_fingerprints = tuple(
        (page.page_number, _text_fingerprint(page.text))
        for page in pages
    )
    mapped: list[RichPageMapEntry] = []
    for block in document.blocks:
        if (
            block.kind is RichBlockKind.HEADING
            and block.section_path
            and block.section_path[-1] in heading_page_by_section
        ):
            mapped.append(
                RichPageMapEntry(
                    block_id=block.block_id,
                    page_number=heading_page_by_section[
                        block.section_path[-1]
                    ],
                )
            )
            continue
        phrase = _text_fingerprint(_block_text(block))
        if not _has_sufficient_page_evidence(phrase):
            continue
        candidates = [
            page_number
            for page_number, page_phrase in page_fingerprints
            if _phrase_occurs_on_page(phrase, page_phrase)
        ]
        if len(candidates) == 1:
            mapped.append(
                RichPageMapEntry(
                    block_id=block.block_id,
                    page_number=candidates[0],
                )
            )
    return tuple(mapped)


def _has_sufficient_page_evidence(phrase: str) -> bool:
    tokens = phrase.split()
    return len(tokens) >= 2 and len("".join(tokens)) >= 8


def _phrase_occurs_on_page(phrase: str, page_phrase: str) -> bool:
    return f" {phrase} " in f" {page_phrase} "


def _conflict_detail(entry: ReconciliationEntry) -> dict[str, object]:
    return {
        "subject_id": entry.subject_id,
        "status": entry.status.value,
        "message": entry.message,
        **dict(entry.provenance),
    }


def _is_fatal_pdf_reconciliation_entry(entry: ReconciliationEntry) -> bool:
    """Keep structure strict while treating weak PDF math evidence as diagnostic.

    PDF text extraction cannot distinguish a short formula from each repeated
    occurrence of that formula.  Missing or ambiguous math evidence therefore
    is not a contradiction with the rich source.  A confirmed mathematical
    mismatch remains fatal, as do all missing, ambiguous, and mismatched
    section anchors.
    """

    if entry.subject_id.startswith("section:"):
        return entry.status in {
            ReconciliationStatus.MISSING,
            ReconciliationStatus.MISMATCH,
            ReconciliationStatus.AMBIGUOUS,
        }
    return entry.status is ReconciliationStatus.MISMATCH


def _equation_label_provenance(
    document: RichDocument,
    entries: tuple[ReconciliationEntry, ...],
) -> tuple[dict[str, dict[str, object]], str]:
    """Project a complete reconciled PDF label sequence onto rich equation blocks."""

    entry = next(
        (
            item
            for item in entries
            if item.subject_id == "equation-labels"
            and item.status is ReconciliationStatus.VERIFIED
        ),
        None,
    )
    if entry is None:
        return {}, ""
    raw_mappings = entry.provenance.get("mappings")
    if not isinstance(raw_mappings, list):
        return {}, "PDF equation labels were not projected: reconciliation metadata is invalid"
    blocks = [
        block
        for block in document.blocks
        if block.kind is RichBlockKind.EQUATION
        and isinstance(block.payload.get("label"), str)
        and block.payload["label"]
    ]
    by_label: dict[str, object] = {}
    for block in blocks:
        label = str(block.payload["label"])
        if label in by_label:
            return {}, "PDF equation labels were not projected: rich labels are not unique"
        by_label[label] = block
    output: dict[str, dict[str, object]] = {}
    for raw in raw_mappings:
        if not isinstance(raw, dict):
            return {}, "PDF equation labels were not projected: reconciliation metadata is invalid"
        source_label = raw.get("source_label")
        block = by_label.get(source_label) if isinstance(source_label, str) else None
        if block is None:
            return {}, "PDF equation labels were not projected: rich and parsed labels disagree"
        block_id = getattr(block, "block_id")
        output[block_id] = {
            "source_label": source_label,
            "pdf_label": raw["pdf_label"],
            "effective_label": raw["effective_label"],
            "page_number": raw["page_number"],
            "matching_method": raw["matching_method"],
        }
    if len(output) != len(blocks):
        return {}, "PDF equation labels were not projected: rich equation coverage is incomplete"
    return output, ""


def _with_equation_pages(
    page_map: tuple[RichPageMapEntry, ...],
    provenance: dict[str, dict[str, object]],
) -> tuple[RichPageMapEntry, ...]:
    known = {item.block_id for item in page_map}
    extra = [
        RichPageMapEntry(block_id=block_id, page_number=page_number)
        for block_id, details in provenance.items()
        if block_id not in known
        and isinstance((page_number := details.get("page_number")), int)
    ]
    return page_map + tuple(extra)


def _block_text(block) -> str:
    payload = block.payload
    if block.kind in {
        RichBlockKind.HEADING,
        RichBlockKind.PARAGRAPH,
        RichBlockKind.CODE,
    }:
        return str(payload["text"])
    if block.kind is RichBlockKind.EQUATION:
        return str(payload["tex"])
    if block.kind is RichBlockKind.LIST:
        return " ".join(str(item["text"]) for item in payload["items"])
    if block.kind is RichBlockKind.TABLE:
        cells = tuple(payload["headers"]) + tuple(
            cell for row in payload["rows"] for cell in row
        )
        return " ".join(str(cell) for cell in cells)
    return f"{payload['alt_text']} {payload['caption']}"


def _text_fingerprint(value: str) -> str:
    return " ".join(
        re.findall(
            r"[^\W_]+",
            unicodedata.normalize("NFKC", value).casefold(),
            flags=re.UNICODE,
        )
    )


__all__ = [
    "PDF_VALIDATOR_MISSING_WARNING",
    "RichDocumentParserService",
    "RichDocumentValidationError",
    "RichParseOutcome",
]
