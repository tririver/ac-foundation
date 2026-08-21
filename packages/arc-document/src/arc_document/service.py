"""Typed facade for provider-neutral document infrastructure."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import re

from arc_jobs import JsonValue, RunStatus
from arc_llm import LLMExecutionOptions, ModelSelection

from ._cache_root import resolve_cache_root
from .cached_document import (
    CachedDocumentError,
    CachedDocumentRef,
    CachedSection,
    CachedSourceRange,
    CachedTableOfContents,
)
from .document_search import (
    EquationSearchResult,
    FullTextSearchResult,
    ParsedSection,
    TableOfContentsEntry,
    search_equations as _search_equations,
    search_full_text as _search_full_text,
    select_section as _select_section,
    table_of_contents as _table_of_contents,
)
from .document_structure import (
    CachedDocumentStructureRef,
    DocumentStructureError,
    DocumentStructureCache,
    DocumentStructureOverlay,
    reconstruct_document_structure,
)
from .parse import DocumentParserService, PDFTextExtractor, ParsedDocument
from .source_repository import SourceRepository, SourceRepositoryError
from .sources import (
    ParseOutcome,
    SourceArtifact,
    SourceBundle,
    SourceFormat,
    ValidationPolicy,
)


_STANDALONE_MARKDOWN_IMAGE_RE = re.compile(
    r'\s*!\[[^\]]*\]\(\S+?(?:\s+["\'].*?["\'])?\)\s*'
)


class DocumentInputError(ValueError):
    code = "invalid_request"

    def __init__(self, message: str, *, code: str = "invalid_request"):
        super().__init__(message)
        self.code = code
        self.message = message


def default_cache_root() -> Path:
    return resolve_cache_root()


class ArcDocumentService:
    """Injectable facade over local document storage and derived data."""

    def __init__(
        self,
        *,
        cache_root: str | Path | None = None,
        repository: SourceRepository | None = None,
        pdf_text_extractor: PDFTextExtractor | None = None,
        keyword_task_service: object | None = None,
    ) -> None:
        try:
            root = resolve_cache_root(cache_root, repository=repository)
        except ValueError as exc:
            raise DocumentInputError(
                "cache_root must match the injected SourceRepository root"
            ) from exc
        self.repository = repository or SourceRepository(root)
        self.cache_root = root
        self.parser = DocumentParserService(
            self.repository, pdf_text_extractor=pdf_text_extractor
        )
        self._document_structure_cache = DocumentStructureCache(root)
        from .cache_admin import DocumentCacheAdministrator

        self.cache_administrator = DocumentCacheAdministrator(root)
        self._keyword_task_service = keyword_task_service
        self._term_inventory_store: object | None = None

    def import_source(
        self,
        path: str | Path,
        *,
        source_format: SourceFormat | str | None = None,
    ) -> SourceArtifact:
        source = self.repository.import_path(path, source_format=source_format)
        self.parser.parse_source(source)
        return source

    def resolve_local_source(
        self,
        source: str | Path,
        *,
        source_format: SourceFormat | str | None = None,
    ) -> SourceArtifact:
        path = Path(source)
        if not path.is_file():
            raise SourceRepositoryError(
                "source_not_found", f"source is not an existing local file: {path}"
            )
        return self.repository.import_path(path, source_format=source_format)

    def parse_bundle(
        self,
        bundle: SourceBundle,
        *,
        policy: ValidationPolicy | str | None = None,
    ) -> ParseOutcome:
        resolved = ValidationPolicy(policy) if policy is not None else None
        return self.parser.parse(bundle, policy=resolved)

    def parse_local(
        self,
        primary_path: str | Path,
        *,
        validator_paths: Sequence[str | Path] = (),
        primary_format: SourceFormat | str | None = None,
        validator_formats: Sequence[SourceFormat | str | None] = (),
        policy: ValidationPolicy | str | None = None,
    ) -> ParseOutcome:
        if validator_formats and len(validator_formats) != len(validator_paths):
            raise self._input_error(
                "validator_formats must be empty or match validator_paths"
            )
        primary = self.repository.import_path(
            primary_path, source_format=primary_format
        )
        formats = tuple(validator_formats) if validator_formats else (None,) * len(validator_paths)
        validators = tuple(
            self.repository.import_path(path, source_format=source_format)
            for path, source_format in zip(validator_paths, formats, strict=True)
        )
        return self.parse_bundle(
            SourceBundle(primary=primary, validators=validators), policy=policy
        )

    def export_rich_document(
        self,
        source: str | Path,
        *,
        output_dir: str | Path,
        validator: str | Path | None = None,
        source_format: SourceFormat | str | None = None,
    ) -> dict[str, object]:
        from .rich_document import export_rich_document_workspace

        return export_rich_document_workspace(
            self.repository,
            source,
            output_dir=output_dir,
            validator=validator,
            source_format=source_format,
        )

    @property
    def term_inventory_store(self):
        from .terms import TermInventoryStore

        if self._term_inventory_store is None:
            self._term_inventory_store = TermInventoryStore(self.cache_root)
        return self._term_inventory_store

    def cache_document(
        self, source: SourceArtifact | ParsedDocument
    ) -> CachedDocumentRef:
        expected: str | None = None
        if isinstance(source, ParsedDocument):
            artifact = source.source
            expected = source.document_digest
        elif isinstance(source, SourceArtifact):
            artifact = source
        else:
            raise self._input_error(
                "cached document source must be a SourceArtifact or ParsedDocument"
            )
        document, _ = self.parser.materialize_source(artifact)
        if expected is not None and expected != document.document_digest:
            raise CachedDocumentError(
                "cached_document_digest_mismatch",
                "parsed document does not match the verified cached projection",
            )
        return self._cached_document_ref(artifact, document)

    def reconstruct_cached_structure(
        self,
        document: CachedDocumentRef,
        outline_document: CachedDocumentRef,
    ) -> CachedDocumentStructureRef:
        parsed, _ = self._resolve_cached_document(document)
        outline, _ = self._resolve_cached_document(outline_document)
        cached = self._document_structure_cache.lookup(document, outline_document)
        if cached is not None:
            return cached.reference
        overlay = reconstruct_document_structure(
            document,
            outline_document,
            markdown_payload=self.repository.read_bytes(parsed.source),
            pdf_payload=self.repository.read_bytes(outline.source),
            pdf_pages=tuple(page.text for page in outline.pages),
        )
        return self._document_structure_cache.store(overlay)

    def get_cached_table_of_contents(
        self,
        document: CachedDocumentRef,
        *,
        structure: CachedDocumentStructureRef | None = None,
    ) -> CachedTableOfContents:
        parsed, warnings = self._resolve_cached_document(document)
        if structure is None:
            return CachedTableOfContents(
                document, _table_of_contents(parsed), warnings
            )
        overlay = self._resolve_cached_structure(document, structure)
        return CachedTableOfContents(
            document,
            tuple(
                TableOfContentsEntry(
                    item.section_id,
                    item.title,
                    item.level,
                    item.ordinal,
                    item.pdf_page_start,
                    item.pdf_page_end,
                )
                for item in overlay.entries
            ),
            (*warnings, *overlay.warnings),
        )

    def get_cached_section(
        self,
        document: CachedDocumentRef,
        selector: str | int,
        *,
        structure: CachedDocumentStructureRef | None = None,
    ) -> CachedSection:
        parsed, warnings = self._resolve_cached_document(document)
        if structure is None:
            section = _select_section(parsed, selector)
            return CachedSection(
                document,
                section.section_id,
                section.title,
                section.text,
                section.level,
                section.ordinal,
                section.page_start,
                section.page_end,
                warnings,
            )
        overlay = self._resolve_cached_structure(document, structure)
        entry = _select_structure_entry(
            overlay.entries, selector, error_factory=self._input_error
        )
        source_range = self.read_cached_source_range(
            document, entry.source_line_start, entry.source_line_end
        )
        return CachedSection(
            document,
            entry.section_id,
            entry.title,
            source_range.text,
            entry.level,
            entry.ordinal,
            entry.pdf_page_start,
            entry.pdf_page_end,
            (*warnings, *overlay.warnings),
        )

    def read_cached_source_range(
        self,
        document: CachedDocumentRef,
        start_line: int,
        end_line: int,
        *,
        text_only: bool = False,
    ) -> CachedSourceRange:
        parsed, _ = self._resolve_cached_document(document)
        if (
            isinstance(start_line, bool)
            or not isinstance(start_line, int)
            or isinstance(end_line, bool)
            or not isinstance(end_line, int)
            or start_line < 1
            or end_line < start_line
        ):
            raise CachedDocumentError(
                "invalid_source_range",
                "source range requires one-based start_line <= end_line",
            )
        if not isinstance(text_only, bool):
            raise CachedDocumentError(
                "invalid_text_only",
                "text_only must be a boolean",
            )
        if parsed.source.source_format is SourceFormat.PDF:
            raise CachedDocumentError(
                "cached_source_not_text",
                "raw source ranges are unavailable for PDF sources",
            )
        payload = self.repository.read_bytes(parsed.source)
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CachedDocumentError(
                "cached_source_not_utf8",
                "cached source is not valid UTF-8 text",
            ) from exc
        lines = text.splitlines()
        total_lines = len(lines)
        if end_line > total_lines:
            raise CachedDocumentError(
                "source_range_out_of_bounds",
                f"source range ends at {end_line}, but source has {total_lines} lines",
            )
        selected_lines = lines[start_line - 1 : end_line]
        if text_only and parsed.source.source_format is SourceFormat.MARKDOWN:
            selected_lines = self._markdown_text_only_range(
                parsed.source,
                lines,
                start_line=start_line,
                end_line=end_line,
            )
        return CachedSourceRange(
            document=document,
            start_line=start_line,
            end_line=end_line,
            total_lines=total_lines,
            text="\n".join(selected_lines),
        )

    def _markdown_text_only_range(
        self,
        source: SourceArtifact,
        lines: Sequence[str],
        *,
        start_line: int,
        end_line: int,
    ) -> list[str]:
        """Project a Markdown range without standalone figure source lines."""

        from .rich_document import RichBlockKind, RichDocumentParserService

        rich = RichDocumentParserService(self.repository).parse_source(source)
        excluded: set[int] = set()
        replacements: dict[int, str] = {}
        for block in rich.blocks:
            if block.kind is not RichBlockKind.FIGURE:
                continue
            line_start = block.locator.line_start
            line_end = block.locator.line_end
            if (
                line_start is None
                or line_end is None
                or line_start < 1
                or line_start > len(lines)
                or _STANDALONE_MARKDOWN_IMAGE_RE.fullmatch(
                    lines[line_start - 1]
                )
                is None
            ):
                continue
            excluded.update(range(line_start, line_end + 1))
            caption = str(block.payload.get("caption", "")).strip()
            if caption:
                replacements[line_start] = caption
        projected: list[str] = []
        for line_number in range(start_line, end_line + 1):
            if line_number not in excluded:
                projected.append(lines[line_number - 1])
            elif line_number in replacements:
                projected.append(replacements[line_number])
        return projected

    def search_full_text(
        self,
        documents: ParsedDocument | Iterable[ParsedDocument],
        query: str,
        *,
        limit: int = 20,
        context_lines: int = 1,
        case_sensitive: bool = False,
    ) -> FullTextSearchResult:
        return _search_full_text(
            documents,
            query,
            limit=limit,
            context_lines=context_lines,
            case_sensitive=case_sensitive,
        )

    def search_equations(
        self,
        documents: ParsedDocument | Iterable[ParsedDocument],
        query: str,
        *,
        limit: int = 20,
        case_sensitive: bool = False,
    ) -> EquationSearchResult:
        return _search_equations(
            documents, query, limit=limit, case_sensitive=case_sensitive
        )

    def table_of_contents(
        self, document: ParsedDocument
    ) -> tuple[TableOfContentsEntry, ...]:
        return _table_of_contents(document)

    def select_section(
        self, document: ParsedDocument, selector: str | int
    ) -> ParsedSection:
        return _select_section(document, selector)

    def list_cache(
        self,
        *,
        document_ids: Sequence[str] = (),
        entry_ids: Sequence[str] = (),
        since_seconds: int | None = None,
    ):
        return self.cache_administrator.list(
            document_ids=document_ids,
            entry_ids=entry_ids,
            since_seconds=since_seconds,
        )

    def remove_cache(
        self,
        *,
        document_ids: Sequence[str] = (),
        entry_ids: Sequence[str] = (),
        dry_run: bool = True,
    ):
        try:
            return self.cache_administrator.remove(
                document_ids=document_ids,
                entry_ids=entry_ids,
                dry_run=dry_run,
            )
        except ValueError as exc:
            raise self._input_error(str(exc)) from exc

    def extract_keywords(
        self,
        source: SourceArtifact | ParsedDocument | object,
        *,
        project_dir: str | Path,
        structure: (
            CachedDocumentStructureRef | DocumentStructureOverlay | None
        ) = None,
        section_ids: Sequence[str] | None = None,
        approx_count: int = 50,
        model: ModelSelection = ModelSelection(tier="medium"),
        run_id: str | None = None,
        resume_input: Mapping[str, JsonValue] | None = None,
        options: LLMExecutionOptions = LLMExecutionOptions(),
    ):
        """Extract a cache-aware approximate keyword view."""

        from .rich_document import RichDocument, RichDocumentParserService
        from .terms import KeywordResult
        from .workflows.keywords import (
            KeywordExtractionError,
            KeywordExtractionPaused,
            KeywordExtractionRunner,
        )

        if isinstance(source, SourceArtifact):
            document: ParsedDocument | RichDocument
            if structure is None:
                document = self.parser.parse_source(source)
            else:
                document = RichDocumentParserService(self.repository).parse(
                    SourceBundle(primary=source)
                ).document
        elif isinstance(source, (ParsedDocument, RichDocument)):
            document = source
        else:
            raise self._input_error(
                "keyword source must be a repository SourceArtifact, "
                "ParsedDocument, or RichDocument"
            )
        overlay: DocumentStructureOverlay | None
        if isinstance(structure, CachedDocumentStructureRef):
            overlay = self._resolve_cached_structure(
                structure.document, structure
            )
        elif isinstance(structure, DocumentStructureOverlay):
            overlay = structure
        elif structure is None:
            overlay = None
        else:
            raise self._input_error(
                "structure must be a cached structure reference or overlay"
            )
        if overlay is not None and isinstance(document, ParsedDocument):
            raise self._input_error(
                "structured keyword extraction requires a rich text document"
            )
        runner = KeywordExtractionRunner(
            project_dir,
            store=self.term_inventory_store,
            task_service=self._keyword_task_service,
        )
        snapshot = runner.execute(
            document,
            structure=overlay,
            section_ids=section_ids,
            approx_count=approx_count,
            model=model,
            run_id=run_id,
            resume_input=resume_input,
            options=options,
        )
        if snapshot.status is RunStatus.SUCCEEDED:
            result: KeywordResult = runner.read_result(snapshot)
            return result
        if snapshot.status is RunStatus.PAUSED:
            raise KeywordExtractionPaused(snapshot)
        if snapshot.status is RunStatus.FAILED and snapshot.error is not None:
            raise KeywordExtractionError(
                snapshot.error.code, snapshot.error.message
            )
        raise KeywordExtractionError(
            "keyword_extraction_incomplete",
            "keyword extraction ended without a terminal result",
        )

    def resolve_cached_document(
        self, reference: CachedDocumentRef
    ) -> tuple[ParsedDocument, tuple[str, ...]]:
        """Reopen and verify one immutable cached document reference."""

        return self._resolve_cached_document(reference)

    def _resolve_cached_document(
        self, reference: CachedDocumentRef
    ) -> tuple[ParsedDocument, tuple[str, ...]]:
        if not isinstance(reference, CachedDocumentRef):
            raise CachedDocumentError(
                "invalid_cached_document_ref",
                "document must be a CachedDocumentRef",
            )
        source = self.repository.get(
            reference.source_format, reference.source_sha256
        )
        if (
            source.size != reference.source_size
            or source.media_type != reference.media_type
        ):
            raise CachedDocumentError(
                "cached_document_source_mismatch",
                "cached source metadata does not match the document reference",
            )
        if self.parser.parser_contract_for(source) != reference.parser_contract:
            raise CachedDocumentError(
                "cached_document_parser_contract_mismatch",
                "current parser contract does not match the document reference",
            )
        parsed, warnings = self.parser.materialize_source(source)
        if parsed.document_digest != reference.parsed_document_sha256:
            raise CachedDocumentError(
                "cached_document_digest_mismatch",
                "cached parsed document does not match the document reference",
            )
        return parsed, warnings

    def _resolve_cached_structure(
        self,
        document: CachedDocumentRef,
        structure: CachedDocumentStructureRef,
    ):
        if not isinstance(structure, CachedDocumentStructureRef):
            raise self._input_error(
                "structure must be a CachedDocumentStructureRef"
            )
        if structure.document != document:
            raise DocumentStructureError(
                "document_structure_source_mismatch",
                "document structure overlay belongs to a different source",
            )
        return self._document_structure_cache.read(structure)

    def _input_error(
        self, message: str, *, code: str = "invalid_request"
    ) -> DocumentInputError:
        """Construct the package-specific invalid-input error for subclasses."""

        return DocumentInputError(message, code=code)

    def _cached_document_ref(
        self, source: SourceArtifact, document: ParsedDocument
    ) -> CachedDocumentRef:
        return CachedDocumentRef(
            source_format=source.source_format,
            source_sha256=source.artifact_digest,
            source_size=source.size,
            media_type=source.media_type,
            parser_contract=self.parser.parser_contract_for(source),
            parsed_document_sha256=document.document_digest,
        )


__all__ = ["ArcDocumentService", "DocumentInputError", "default_cache_root"]


def _select_structure_entry(
    entries,
    selector: str | int,
    *,
    error_factory=DocumentInputError,
):
    if isinstance(selector, bool):
        raise error_factory("section selector cannot be boolean")
    if isinstance(selector, int):
        if selector < 0 or selector >= len(entries):
            raise error_factory(
                "section ordinal is outside the document structure"
            )
        return entries[selector]
    normalized = " ".join(str(selector).split()).casefold()
    if not normalized:
        raise error_factory("section selector is empty")
    exact_ids = [item for item in entries if item.section_id == selector]
    if exact_ids:
        return exact_ids[0]
    matches = [
        item
        for item in entries
        if " ".join(item.title.split()).casefold() == normalized
    ]
    if not matches:
        raise error_factory(
            f"document structure section not found: {selector}"
        )
    if len(matches) > 1:
        raise error_factory(
            f"document structure section title is ambiguous: {selector}"
        )
    return matches[0]
