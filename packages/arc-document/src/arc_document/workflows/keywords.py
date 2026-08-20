"""Durable approximate keyword extraction over a parsed or rich document."""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arc_jobs import (
    Awaiting,
    Failed,
    ImmutableArtifactStore,
    JsonValue,
    Paused,
    ResumeReason,
    RunContext,
    RunEngine,
    RunError,
    RunRepository,
    RunSnapshot,
    RunSpec,
    RunStatus,
    StoppedError,
    Succeeded,
    canonical_json_bytes,
)
from arc_llm import (
    JsonOutput,
    LLMCompleted,
    LLMFailed,
    LLMExecutionOptions,
    LLMPaused,
    LLMRequest,
    LLMStopped,
    LLMTaskService,
    ModelSelection,
    RESUME_SCHEMA_VERSION,
    ResumeInput,
    decode_resume_input,
)

from ..document_structure import DocumentStructureOverlay
from ..parse import ParsedDocument
from ..terms import (
    KeywordDocument,
    KeywordResult,
    KeywordTextUnit,
    TermCandidate,
    TermInventoryLineage,
    TermInventoryStore,
    TermInventoryStoreError,
    keyword_chapters,
    keyword_result_from_document,
    normalize_term,
    result_from_inventory,
    validate_approx_count,
)
from ._llm import (
    TaskService,
    awaiting_from_pause,
    execute_routed,
    model_document,
    run_error_from_failure,
    semantic_retry_request,
)


KEYWORD_EXTRACTION_HANDLER = "arc.document.keyword_extraction.v1"
KEYWORD_REVIEW_PROMPT_CONTRACT = "arc.document.keyword_review_prompt.v1"
KEYWORD_CHAPTER_PROMPT_CONTRACT = "arc.document.keyword_chapter_prompt.v3"
EXPLICIT_TERM_SUPERVISION_SCHEMA = (
    "arc.document.explicit_term_supervision_response.v1"
)
KEYWORD_NORMALIZATION_CONTRACT = "arc.document.keyword_normalization.v1"
KEYWORD_OCCURRENCE_CONTRACT = "arc.document.keyword_occurrence_literal.v2"
KEYWORD_REVIEW_SEMANTIC_VALIDATOR = "arc.document.keyword_review_semantics.v1"
_EXPLICIT_WINDOW_SIZE = 80
_EXPLICIT_REVIEW_INVALID_ARTIFACT = (
    "document-keywords/explicit-review-semantic-invalid"
)
_EXPLICIT_REVIEW_INVALID_WARNING = (
    "Explicit term review remained machine-invalid after one fresh retry; "
    "the explicit field was discarded and chapter extraction continued."
)

_REVIEW_ITEMS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["term", "source_entry_ids"],
        "properties": {
            "term": {"type": "string", "minLength": 1, "maxLength": 300},
            "source_entry_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "uniqueItems": True,
            },
        },
    },
}
_REVIEW_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "arc.document.explicit_term_review.v1",
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "reason", "entries", "discarded_source_entry_ids"],
    "properties": {
        "status": {"enum": ["usable", "unusable"]},
        "reason": {"type": "string"},
        "entries": _REVIEW_ITEMS_SCHEMA,
        "discarded_source_entry_ids": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
    },
}
_CHAPTER_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "arc.document.chapter_term_extraction.v1",
    "type": "object",
    "additionalProperties": False,
    "required": ["entries"],
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["term"],
                "properties": {
                    "term": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 300,
                    }
                },
            },
        }
    },
}
_SUPERVISION_RESPONSE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": EXPLICIT_TERM_SUPERVISION_SCHEMA,
    "type": "object",
    "additionalProperties": False,
    "required": ["resume_key", "action"],
    "properties": {
        "resume_key": {"type": "string", "minLength": 1},
        "action": {
            "enum": ["discard_index_and_continue", "abort"],
        },
    },
}


@dataclass(frozen=True)
class KeywordExtractionCompleted:
    result: KeywordResult


@dataclass(frozen=True)
class _ExplicitField:
    field_id: str
    kind: str
    label: str
    entries: tuple["_ExplicitEntry", ...]


@dataclass(frozen=True)
class _ExplicitEntry:
    entry_id: str
    text: str


class KeywordExtractionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class KeywordExtractionPaused(KeywordExtractionError):
    def __init__(self, snapshot: RunSnapshot) -> None:
        super().__init__(
            "keyword_extraction_paused",
            "keyword extraction requires supervision or provider input",
        )
        self.snapshot = snapshot


class KeywordInventoryService:
    """Execute one cache-aware keyword extraction inside an existing run."""

    def __init__(
        self,
        store: TermInventoryStore,
        *,
        task_service: TaskService | None = None,
    ) -> None:
        self.store = store
        self.task_service = task_service or LLMTaskService()

    def extract_keywords(
        self,
        context: RunContext,
        document: KeywordDocument,
        *,
        structure: DocumentStructureOverlay | None = None,
        section_ids: Sequence[str] | None = None,
        approx_count: int = 50,
        model: ModelSelection = ModelSelection(tier="medium"),
        resume_input: Mapping[str, JsonValue] | None = None,
        options: LLMExecutionOptions = LLMExecutionOptions(),
    ) -> KeywordResult | Paused | Failed:
        approx_count = validate_approx_count(approx_count)
        planned_count = math.ceil(1.5 * approx_count)
        effective_resume = (
            context.resume_input if resume_input is None else resume_input
        )
        sections = keyword_chapters(
            document,
            structure=structure,
            section_ids=section_ids,
        )
        lineage = _lineage(
            document,
            model,
            structure=structure,
            section_ids=section_ids,
        )
        try:
            cached, cache_warnings = self.store.load(document, lineage)
        except TermInventoryStoreError as exc:
            return Failed(RunError(exc.code, exc.message))
        semantic_warnings = _semantic_retry_warnings(context)
        if cached is not None and cached.high_water >= planned_count:
            return result_from_inventory(
                cached,
                approx_count=approx_count,
                planned_count=planned_count,
                warnings=(*cache_warnings, *semantic_warnings),
            )

        existing_candidates = (
            [
                TermCandidate(item.term, item.aliases, item.source_refs)
                for item in cached.terms
            ]
            if cached is not None
            else []
        )
        disposition = (
            cached.explicit_disposition if cached is not None else "absent"
        )
        explicit_candidates: list[TermCandidate] = []
        fields = _explicit_fields(
            document,
            structure=structure,
            section_ids=section_ids,
        )
        if cached is None and fields:
            explicit_outcome = self._review_explicit_fields(
                context,
                document,
                fields,
                model=model,
                resume_input=effective_resume,
                options=options,
            )
            if isinstance(explicit_outcome, (Paused, Failed)):
                return explicit_outcome
            (
                explicit_candidates,
                disposition,
                explicit_warnings,
            ) = explicit_outcome
            semantic_warnings.extend(explicit_warnings)

        current_unique = {
            normalize_term(item.term)
            for item in (*existing_candidates, *explicit_candidates)
            if normalize_term(item.term)
        }
        chapter_candidates: list[TermCandidate] = []
        if len(current_unique) < planned_count:
            chapter_outcome = self._extract_chapters(
                context,
                document,
                sections=sections,
                requested_total=max(1, planned_count - len(current_unique)),
                model=model,
                resume_input=effective_resume,
                existing_terms=tuple(sorted(current_unique)),
                options=options,
            )
            if isinstance(chapter_outcome, (Paused, Failed)):
                return chapter_outcome
            chapter_candidates = chapter_outcome

        try:
            stored = self.store.merge(
                document,
                lineage,
                high_water=planned_count,
                explicit_disposition=disposition,
                candidates=(
                    *existing_candidates,
                    *explicit_candidates,
                    *chapter_candidates,
                ),
            )
        except TermInventoryStoreError as exc:
            return Failed(RunError(exc.code, exc.message))
        return result_from_inventory(
            stored,
            approx_count=approx_count,
            planned_count=planned_count,
            warnings=(*cache_warnings, *semantic_warnings),
        )

    def _review_explicit_fields(
        self,
        context: RunContext,
        document: KeywordDocument,
        fields: Sequence[_ExplicitField],
        *,
        model: ModelSelection,
        resume_input: Mapping[str, JsonValue] | None,
        options: LLMExecutionOptions,
    ) -> tuple[list[TermCandidate], str, tuple[str, ...]] | Paused | Failed:
        resume = _supervision_resume(resume_input, document)
        if resume == "abort":
            return Failed(
                RunError(
                    "explicit_term_list_unusable",
                    "The explicit keyword or index field was rejected by supervision.",
                )
            )
        if resume == "discard_index_and_continue":
            return [], "discarded", ()

        reviewed: list[TermCandidate] = []
        for field in fields:
            for window_index, entries in enumerate(
                _windows(field.entries, _EXPLICIT_WINDOW_SIZE)
            ):
                request = _explicit_review_request(
                    document,
                    field,
                    entries,
                    window_index=window_index,
                    model=model,
                )
                outcome = execute_routed(
                    self.task_service,
                    context,
                    request,
                    resume_input=_llm_resume_input(resume_input),
                    options=options,
                )
                if isinstance(outcome, LLMPaused):
                    return Paused(awaiting_from_pause(outcome))
                if isinstance(outcome, LLMFailed):
                    return Failed(run_error_from_failure(outcome))
                if isinstance(outcome, LLMStopped):
                    raise StoppedError("explicit term review stopped")
                if not isinstance(outcome, LLMCompleted):
                    raise RuntimeError("unknown explicit term review outcome")
                try:
                    status, reason, candidates = _decode_explicit_review(
                        outcome.value,
                        field=field,
                        window=entries,
                    )
                except ValueError as exc:
                    retry_request = semantic_retry_request(
                        request,
                        validator_contract=KEYWORD_REVIEW_SEMANTIC_VALIDATOR,
                        feedback=str(exc),
                    )
                    retry_outcome = execute_routed(
                        self.task_service,
                        context,
                        retry_request,
                        resume_input=_llm_resume_input(resume_input),
                        options=options,
                    )
                    if isinstance(retry_outcome, LLMPaused):
                        return Paused(awaiting_from_pause(retry_outcome))
                    if isinstance(retry_outcome, LLMFailed):
                        return Failed(run_error_from_failure(retry_outcome))
                    if isinstance(retry_outcome, LLMStopped):
                        raise StoppedError("explicit term review retry stopped")
                    if not isinstance(retry_outcome, LLMCompleted):
                        raise RuntimeError(
                            "unknown explicit term review retry outcome"
                        )
                    try:
                        (
                            status,
                            reason,
                            candidates,
                        ) = _decode_explicit_review(
                            retry_outcome.value,
                            field=field,
                            window=entries,
                        )
                    except ValueError as retry_exc:
                        _publish_explicit_review_invalid(
                            context,
                            request=request,
                            retry_request=retry_request,
                            initial_value=outcome.value,
                            retry_value=retry_outcome.value,
                            initial_error=str(exc),
                            retry_error=str(retry_exc),
                        )
                        return (
                            [],
                            "discarded",
                            (_EXPLICIT_REVIEW_INVALID_WARNING,),
                        )
                if status == "unusable":
                    return _explicit_supervision_pause(
                        context,
                        document,
                        field,
                        reason=reason,
                    )
                if status != "usable":
                    return Failed(
                        RunError(
                            "keyword_output_invalid",
                            "explicit term review returned an unknown status",
                        )
                    )
                reviewed.extend(candidates)
        return reviewed, "reviewed", ()

    def _extract_chapters(
        self,
        context: RunContext,
        document: KeywordDocument,
        *,
        sections: Sequence[KeywordTextUnit],
        requested_total: int,
        model: ModelSelection,
        resume_input: Mapping[str, JsonValue] | None,
        existing_terms: Sequence[str],
        options: LLMExecutionOptions,
    ) -> list[TermCandidate] | Paused | Failed:
        allocations = _chapter_allocations(sections, requested_total)
        if not allocations:
            return []
        output: list[TermCandidate] = []
        known = set(existing_terms)
        for section, requested_count in allocations:
            request = _chapter_request(
                document,
                section,
                requested_count=requested_count,
                model=model,
                existing_terms=tuple(sorted(known)),
            )
            outcome = execute_routed(
                self.task_service,
                context,
                request,
                resume_input=_llm_resume_input(resume_input),
                options=options,
            )
            if isinstance(outcome, LLMPaused):
                return Paused(awaiting_from_pause(outcome))
            if isinstance(outcome, LLMFailed):
                return Failed(run_error_from_failure(outcome))
            if isinstance(outcome, LLMStopped):
                raise StoppedError("chapter term extraction stopped")
            if not isinstance(outcome, LLMCompleted):
                raise RuntimeError("unknown chapter term extraction outcome")
            try:
                value = _object(outcome.value, "chapter term extraction")
                candidates = _chapter_candidates(
                    value.get("entries"),
                    source_ref=f"section:{section.section_id}",
                )
                output.extend(candidates)
                known.update(
                    normalize_term(item.term)
                    for item in candidates
                    if normalize_term(item.term)
                )
            except ValueError as exc:
                return Failed(RunError("keyword_output_invalid", str(exc)))
        return output


def _chapter_allocations(
    sections: Sequence[KeywordTextUnit],
    requested_total: int,
) -> tuple[tuple[KeywordTextUnit, int], ...]:
    """Allocate a deterministic, length-weighted quota to non-empty chapters."""

    nonempty = tuple(section for section in sections if section.text.strip())
    if not nonempty:
        return ()
    chapter_count = len(nonempty)
    target = max(requested_total, chapter_count)
    remaining = target - chapter_count
    lengths = tuple(len(section.text) for section in nonempty)
    total_length = sum(lengths)
    numerators = tuple(remaining * length for length in lengths)
    extras = [numerator // total_length for numerator in numerators]
    unassigned = remaining - sum(extras)
    remainder_order = sorted(
        range(chapter_count),
        key=lambda index: (-(numerators[index] % total_length), index),
    )
    for index in remainder_order[:unassigned]:
        extras[index] += 1
    return tuple(
        (section, 1 + extras[index])
        for index, section in enumerate(nonempty)
    )


KeywordExtractionService = KeywordInventoryService


class KeywordExtractionHandler:
    name = KEYWORD_EXTRACTION_HANDLER

    def __init__(
        self,
        document: KeywordDocument,
        *,
        structure: DocumentStructureOverlay | None = None,
        section_ids: Sequence[str] | None = None,
        store: TermInventoryStore,
        approx_count: int = 50,
        model: ModelSelection = ModelSelection(tier="medium"),
        task_service: TaskService | None = None,
        options: LLMExecutionOptions = LLMExecutionOptions(),
    ) -> None:
        self.document = document
        self.structure = structure
        self.section_ids = (
            tuple(str(item) for item in section_ids)
            if section_ids is not None
            else None
        )
        self.approx_count = validate_approx_count(approx_count)
        self.model = model
        self.options = options
        self.service = KeywordInventoryService(
            store, task_service=task_service
        )

    def semantic_input(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "schema_version": "arc.document.keyword_extraction_request.v1",
            "document_digest": self.document.document_digest,
            "source_digest": self.document.source.artifact_digest,
            "source_size": self.document.source.size,
            "source_media_type": self.document.source.media_type,
            "approx_count": self.approx_count,
            "planned_count": math.ceil(1.5 * self.approx_count),
            "model_requirement": model_document(self.model),
        }
        if self.structure is not None:
            value["schema_version"] = "arc.document.keyword_extraction_request.v2"
            value["source_structure"] = _source_structure_identity(
                self.structure,
                self.section_ids,
            )
        return value

    def execute(self, context: RunContext):
        if dict(context.semantic_input) != self.semantic_input():
            return Failed(
                RunError(
                    "keyword_extraction_binding_mismatch",
                    "Handler bindings differ from the durable keyword request.",
                )
            )
        try:
            outcome = self.service.extract_keywords(
                context,
                self.document,
                structure=self.structure,
                section_ids=self.section_ids,
                approx_count=self.approx_count,
                model=self.model,
                options=self.options,
            )
        except KeywordExtractionError as exc:
            return Failed(RunError(exc.code, exc.message))
        if isinstance(outcome, (Paused, Failed)):
            return outcome
        return Succeeded(
            context.artifacts.publish_json(
                "document-keywords/result", outcome.to_document()
            )
        )


class KeywordExtractionRunner:
    def __init__(
        self,
        project_dir: str | Path,
        *,
        store: TermInventoryStore,
        task_service: TaskService | None = None,
    ) -> None:
        self.repository = RunRepository(project_dir)
        self.engine = RunEngine(self.repository)
        self.store = store
        self.task_service = task_service

    def execute(
        self,
        document: KeywordDocument,
        *,
        structure: DocumentStructureOverlay | None = None,
        section_ids: Sequence[str] | None = None,
        approx_count: int = 50,
        model: ModelSelection = ModelSelection(tier="medium"),
        run_id: str | None = None,
        resume_input: Mapping[str, JsonValue] | None = None,
        options: LLMExecutionOptions = LLMExecutionOptions(),
    ) -> RunSnapshot:
        handler = KeywordExtractionHandler(
            document,
            structure=structure,
            section_ids=section_ids,
            store=self.store,
            approx_count=approx_count,
            model=model,
            task_service=self.task_service,
            options=options,
        )
        resolved_run_id = run_id or _run_id(handler.semantic_input())
        run_dir = self.repository.run_directory(resolved_run_id)
        if (run_dir / "spec.json").exists():
            snapshot = self.repository.inspect(resolved_run_id).snapshot
            if snapshot.status is RunStatus.PAUSED:
                if (
                    snapshot.awaiting is not None
                    and snapshot.awaiting.input_required
                    and resume_input is None
                ):
                    return snapshot
                return self.engine.resume(
                    resolved_run_id,
                    handler,
                    input=resume_input,
                )
        return self.engine.execute(
            RunSpec(
                resolved_run_id,
                handler.name,
                handler.semantic_input(),
            ),
            handler,
        )

    def read_result(self, snapshot: RunSnapshot) -> KeywordResult:
        if snapshot.status is not RunStatus.SUCCEEDED or snapshot.result_ref is None:
            raise ValueError("keyword run has no completed result")
        store = ImmutableArtifactStore(
            self.repository.run_directory(snapshot.run_id),
            repository_root=self.repository.root,
        )
        value = json.loads(store.read_bytes(snapshot.result_ref))
        if not isinstance(value, Mapping):
            raise ValueError("keyword result artifact must be an object")
        return keyword_result_from_document(value)


def _explicit_fields(
    document: KeywordDocument,
    *,
    structure: DocumentStructureOverlay | None = None,
    section_ids: Sequence[str] | None = None,
) -> tuple[_ExplicitField, ...]:
    raw_fields: list[tuple[str, str, tuple[str, ...]]] = []
    metadata_fields = document.metadata.get("explicit_term_fields")
    if isinstance(metadata_fields, Sequence) and not isinstance(
        metadata_fields, (str, bytes)
    ):
        for raw in metadata_fields:
            if not isinstance(raw, Mapping):
                continue
            kind = str(raw.get("kind") or "keywords")
            label = str(raw.get("label") or kind)
            entries = _deterministic_entries(raw.get("entries"))
            if entries:
                raw_fields.append((kind, label, entries))
    for key, raw in document.metadata.items():
        normalized_key = re.sub(r"[^a-z]", "", str(key).casefold())
        if normalized_key not in {
            "keyword",
            "keywords",
            "keyterms",
            "subjectindex",
            "indexterms",
        }:
            continue
        entries = _deterministic_entries(raw)
        if entries:
            raw_fields.append(("index" if "index" in normalized_key else "keywords", str(key), entries))
    for section in keyword_chapters(
        document,
        structure=structure,
        section_ids=section_ids,
    ):
        title = re.sub(r"[^a-z]", "", section.title.casefold())
        if title not in {"keyword", "keywords", "keyterms", "index", "subjectindex"}:
            continue
        entries = _entries_from_text(section.text, label=section.title)
        if entries:
            raw_fields.append(
                (
                    "index" if "index" in title else "keywords",
                    section.title,
                    entries,
                )
            )
    output: list[_ExplicitField] = []
    for ordinal, (kind, label, entries) in enumerate(raw_fields):
        material = canonical_json_bytes(
            {"kind": kind, "label": label, "entries": list(entries)}
        )
        field_id = (
            f"field-{ordinal:04d}-"
            f"{hashlib.sha256(material).hexdigest()[:16]}"
        )
        output.append(
            _ExplicitField(
                field_id,
                kind,
                label,
                tuple(
                    _ExplicitEntry(
                        f"{field_id}-entry-{entry_ordinal:06d}",
                        text,
                    )
                    for entry_ordinal, text in enumerate(entries)
                ),
            )
        )
    return tuple(output)


def _deterministic_entries(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return _entries_from_text(value)
    if isinstance(value, Mapping):
        output: list[str] = []
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            output.append(str(key))
            if isinstance(item, str) and item.strip():
                output.append(item)
        return _stable_nonempty(output)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        output = []
        for item in value:
            if isinstance(item, str):
                output.append(item)
            elif isinstance(item, Mapping):
                term = item.get("term") or item.get("name") or item.get("label")
                if isinstance(term, str):
                    output.append(term)
        return _stable_nonempty(output)
    return ()


def _entries_from_text(value: str, *, label: str = "") -> tuple[str, ...]:
    lines = []
    for raw in value.splitlines():
        item = re.sub(r"^\s*(?:[-*+]|\d+[.)]|\\item)\s*", "", raw).strip()
        if label and normalize_term(item) == normalize_term(label):
            continue
        if item:
            lines.extend(part.strip() for part in re.split(r"[;,]", item))
    return _stable_nonempty(lines)


def _stable_nonempty(values: Sequence[str]) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = re.sub(r"\s+", " ", value).strip()
        normalized = normalize_term(item)
        if normalized and normalized not in seen:
            output.append(item)
            seen.add(normalized)
    return tuple(output)


def _windows(
    values: Sequence[_ExplicitEntry], size: int
) -> Sequence[tuple[_ExplicitEntry, ...]]:
    return tuple(
        tuple(values[offset : offset + size])
        for offset in range(0, len(values), size)
    )


def _explicit_review_request(
    document: KeywordDocument,
    field: _ExplicitField,
    entries: Sequence[_ExplicitEntry],
    *,
    window_index: int,
    model: ModelSelection,
) -> LLMRequest:
    prompt = "\n".join(
        (
            f"Contract: {KEYWORD_REVIEW_PROMPT_CONTRACT}",
            "Review one deterministic window from an explicit document keyword or index field.",
            "Correct only extraction artifacts such as OCR noise, broken lines, and page numbers.",
            "Do not semantically merge abbreviations, synonyms, or related concepts.",
            "Do not add concepts absent from this window.",
            "Every corrected entry must cite its source_entry_ids. Every source entry ID",
            "must appear exactly once, either in an output entry or discarded_source_entry_ids.",
            "If the field is not a usable term list, return status unusable and explain why.",
            f"Document digest: {document.document_digest}",
            f"Field kind: {field.kind}",
            f"Field label: {field.label}",
            f"Window index: {window_index}",
            "Entries:",
            json.dumps(
                [
                    {"source_entry_id": item.entry_id, "text": item.text}
                    for item in entries
                ],
                ensure_ascii=False,
            ),
        )
    )
    identity = {
        "contract": KEYWORD_REVIEW_PROMPT_CONTRACT,
        "document_digest": document.document_digest,
        "field_id": field.field_id,
        "window_index": window_index,
        "entries": [
            {"source_entry_id": item.entry_id, "text": item.text}
            for item in entries
        ],
        "model": model_document(model),
    }
    return LLMRequest(
        _task_id("keyword-review", identity),
        prompt,
        JsonOutput(_explicit_review_schema(entries), repair="format"),
        model,
    )


def _explicit_review_schema(
    entries: Sequence[_ExplicitEntry],
) -> dict[str, Any]:
    """Bind source identifiers that are known only for this review window."""

    schema = deepcopy(_REVIEW_SCHEMA)
    allowed = [item.entry_id for item in entries]
    schema["properties"]["entries"]["items"]["properties"][
        "source_entry_ids"
    ]["items"]["enum"] = allowed
    schema["properties"]["discarded_source_entry_ids"]["items"][
        "enum"
    ] = allowed
    return schema


def _chapter_request(
    document: KeywordDocument,
    section: Any,
    *,
    requested_count: int,
    model: ModelSelection,
    existing_terms: Sequence[str],
) -> LLMRequest:
    prompt = "\n".join(
        (
            f"Contract: {KEYWORD_CHAPTER_PROMPT_CONTRACT}",
            "Read this chapter text directly and extract scientifically relevant terms.",
            "Do not summarize it. Do not rank against other chapters. Do not invent definitions.",
            "Do not select terms by occurrence frequency.",
            "Return distinct concepts actually supported by this text.",
            "Keep a complete title or proper name when an isolated word would have a different meaning.",
            "Keep abbreviations, synonyms, and related concepts as separate terms.",
            "Prefer relevant terms not already present in the normalized inventory below.",
            f"Approximate requested terms for this chapter: {requested_count}",
            f"Document digest: {document.document_digest}",
            f"Section ID: {section.section_id}",
            f"Section title: {section.title}",
            "Existing normalized inventory:",
            json.dumps(list(existing_terms), ensure_ascii=False),
            "Chapter text:",
            section.text,
        )
    )
    identity = {
        "contract": KEYWORD_CHAPTER_PROMPT_CONTRACT,
        "document_digest": document.document_digest,
        "section_id": section.section_id,
        "section_text_digest": hashlib.sha256(
            section.text.encode("utf-8")
        ).hexdigest(),
        "requested_count": requested_count,
        "existing_terms": list(existing_terms),
        "model": model_document(model),
    }
    return LLMRequest(
        _task_id("keyword-chapter", identity),
        prompt,
        JsonOutput(_CHAPTER_SCHEMA, repair="format"),
        model,
    )


def _explicit_supervision_pause(
    context: RunContext,
    document: KeywordDocument,
    field: _ExplicitField,
    *,
    reason: str,
) -> Paused:
    resume_key = _supervision_key(document)
    request_ref = context.artifacts.publish_json(
        "document-keywords/explicit-term-supervision",
        {
            "schema_version": "arc.document.explicit_term_supervision_request.v1",
            "document_digest": document.document_digest,
            "field_id": field.field_id,
            "field_kind": field.kind,
            "field_label": field.label,
            "reason": reason,
            "allowed_actions": ["discard_index_and_continue", "abort"],
            "response_schema": _SUPERVISION_RESPONSE_SCHEMA,
        },
    )
    return Paused(
        Awaiting(
            ResumeReason.SUPERVISION_REQUIRED,
            resume_key,
            True,
            request_ref,
            EXPLICIT_TERM_SUPERVISION_SCHEMA,
            {
                "code": "explicit_term_list_unusable",
                "field_id": field.field_id,
                "reason": reason,
            },
        )
    )


def _supervision_resume(
    resume_input: Mapping[str, JsonValue] | None,
    document: KeywordDocument,
) -> str | None:
    value = resume_input
    if value is None:
        return None
    action = value.get("action")
    if action not in {"discard_index_and_continue", "abort"}:
        return None
    if value.get("resume_key") != _supervision_key(document):
        raise KeywordExtractionError(
            "keyword_resume_input_invalid",
            "explicit term supervision resume key does not match",
        )
    if set(value) != {"resume_key", "action"}:
        raise KeywordExtractionError(
            "keyword_resume_input_invalid",
            "explicit term supervision input has unexpected fields",
        )
    return str(action)


def _llm_resume_input(
    resume_input: Mapping[str, JsonValue] | None,
) -> ResumeInput | None:
    if resume_input is None:
        return None
    if resume_input.get("action") in {
        "discard_index_and_continue",
        "abort",
    }:
        return None
    # Keyword extraction may run inside another durable workflow.  Parent
    # interaction responses (for example a Companion evidence response) are
    # owned by that workflow, not by arc-llm.  Only a payload that explicitly
    # claims the arc-llm resume schema belongs on this recovery path.
    if resume_input.get("schema_version") != RESUME_SCHEMA_VERSION:
        return None
    try:
        return decode_resume_input(resume_input)
    except Exception as exc:
        raise KeywordExtractionError(
            "keyword_resume_input_invalid", f"Invalid LLM resume input: {exc}"
        ) from exc


def _supervision_key(document: KeywordDocument) -> str:
    return f"explicit-terms-{document.document_digest[:24]}"


def _review_candidates(
    value: Mapping[str, Any],
    *,
    field: _ExplicitField,
    window: Sequence[_ExplicitEntry],
) -> list[TermCandidate]:
    entries = value.get("entries")
    discarded = value.get("discarded_source_entry_ids")
    if not isinstance(entries, list) or not isinstance(discarded, list) or any(
        not isinstance(item, str) for item in discarded
    ):
        raise ValueError(
            "explicit review entries and discarded_source_entry_ids must be arrays"
        )
    allowed = {item.entry_id for item in window}
    accounted: list[str] = list(discarded)
    output: list[TermCandidate] = []
    for raw in entries:
        item = _object(raw, "explicit review entry")
        term = _required_string(item, "term")
        source_ids = item.get("source_entry_ids")
        if (
            not isinstance(source_ids, list)
            or not source_ids
            or any(not isinstance(source_id, str) for source_id in source_ids)
        ):
            raise ValueError(
                "explicit review source_entry_ids must be a non-empty string array"
            )
        accounted.extend(source_ids)
        output.append(
            TermCandidate(
                term,
                (),
                tuple(
                    f"explicit:{field.field_id}:{source_id}"
                    for source_id in source_ids
                ),
            )
        )
    if set(accounted) != allowed or len(accounted) != len(set(accounted)):
        raise ValueError(
            "explicit review must account exactly once for every current-window source entry ID"
        )
    return output


def _decode_explicit_review(
    value: Any,
    *,
    field: _ExplicitField,
    window: Sequence[_ExplicitEntry],
) -> tuple[str, str, list[TermCandidate]]:
    document = _object(value, "explicit term review")
    status = _required_string(document, "status")
    reason = _required_string(document, "reason", allow_empty=True)
    candidates = _review_candidates(
        document,
        field=field,
        window=window,
    )
    return status, reason, candidates


def _publish_explicit_review_invalid(
    context: RunContext,
    *,
    request: LLMRequest,
    retry_request: LLMRequest,
    initial_value: Any,
    retry_value: Any,
    initial_error: str,
    retry_error: str,
) -> None:
    context.artifacts.publish_json(
        _EXPLICIT_REVIEW_INVALID_ARTIFACT,
        {
            "schema_version": "arc.document.keyword_review_semantic_invalid.v1",
            "validator_contract": KEYWORD_REVIEW_SEMANTIC_VALIDATOR,
            "initial_task_id": request.task_id,
            "retry_task_id": retry_request.task_id,
            "initial_error": initial_error[:4000],
            "retry_error": retry_error[:4000],
            "initial_candidate": initial_value,
            "retry_candidate": retry_value,
            "disposition": "discarded",
        },
    )


def _semantic_retry_warnings(context: RunContext) -> list[str]:
    return (
        [_EXPLICIT_REVIEW_INVALID_WARNING]
        if context.artifacts.find(_EXPLICIT_REVIEW_INVALID_ARTIFACT)
        is not None
        else []
    )


def _chapter_candidates(value: Any, *, source_ref: str) -> list[TermCandidate]:
    if not isinstance(value, list):
        raise ValueError("keyword entries must be an array")
    output: list[TermCandidate] = []
    for raw in value:
        item = _object(raw, "keyword entry")
        term = _required_string(item, "term")
        output.append(
            TermCandidate(
                term.strip(),
                (),
                (source_ref,),
            )
        )
    return output


def _object(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return dict(value)


def _required_string(
    value: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    item = value.get(key)
    if not isinstance(item, str) or (not allow_empty and not item.strip()):
        raise ValueError(f"{key} must be a non-empty string")
    return item.strip()


def _task_id(prefix: str, value: Mapping[str, Any]) -> str:
    return (
        f"{prefix}-"
        f"{hashlib.sha256(canonical_json_bytes(value)).hexdigest()[:24]}"
    )


def _run_id(value: Mapping[str, JsonValue]) -> str:
    return (
        "document-keywords-"
        f"{hashlib.sha256(canonical_json_bytes(value)).hexdigest()[:24]}"
    )


def _source_structure_identity(
    structure: DocumentStructureOverlay,
    section_ids: Sequence[str] | None,
) -> dict[str, JsonValue]:
    return {
        "schema_version": structure.schema_version,
        "structure_contract": structure.structure_contract,
        "structure_sha256": structure.structure_sha256,
        "section_ids": (
            list(section_ids) if section_ids is not None else None
        ),
    }


def _lineage(
    document: KeywordDocument,
    model: ModelSelection,
    *,
    structure: DocumentStructureOverlay | None = None,
    section_ids: Sequence[str] | None = None,
) -> TermInventoryLineage:
    return TermInventoryLineage(
        document_digest=document.document_digest,
        source_digest=document.source.artifact_digest,
        source_format=document.source.source_format.value,
        source_media_type=document.source.media_type,
        source_size=document.source.size,
        parser_contract=document.schema_version,
        discovery_contract=KEYWORD_CHAPTER_PROMPT_CONTRACT,
        review_contract=KEYWORD_REVIEW_PROMPT_CONTRACT,
        normalization_contract=KEYWORD_NORMALIZATION_CONTRACT,
        count_contract=KEYWORD_OCCURRENCE_CONTRACT,
        model_requirement=model_document(model),
        source_structure=(
            _source_structure_identity(structure, section_ids)
            if structure is not None
            else {}
        ),
    )


__all__ = [
    "EXPLICIT_TERM_SUPERVISION_SCHEMA",
    "KEYWORD_CHAPTER_PROMPT_CONTRACT",
    "KEYWORD_EXTRACTION_HANDLER",
    "KEYWORD_NORMALIZATION_CONTRACT",
    "KEYWORD_OCCURRENCE_CONTRACT",
    "KEYWORD_REVIEW_PROMPT_CONTRACT",
    "KeywordExtractionCompleted",
    "KeywordExtractionError",
    "KeywordExtractionHandler",
    "KeywordExtractionPaused",
    "KeywordExtractionRunner",
    "KeywordExtractionService",
    "KeywordInventoryService",
]
