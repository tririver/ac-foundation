"""Typed registry for provider-neutral document operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ac_llm import HostAuthority, LLMExecutionOptions, ModelSelection

from .cached_document import cached_document_ref_from_document
from .document_structure import cached_document_structure_ref_from_document
from .operation_registry import (
    DEFAULT_EXCLUDED_EFFECTS,
    JsonCodec,
    JsonOutputCodec,
    OperationEffect,
    OperationRequestError,
    OperationSpec,
    object_schema,
    operation_registry_document,
    operation_spec,
    registry_mapping,
    resolve_operation_specs,
    to_json_value,
)
from .service import AcDocumentService


REGISTRY_SCHEMA_VERSION = "ac.document.operation_registry.v1"
_STRING = {"type": "string"}
_NONEMPTY_STRING = {"type": "string", "minLength": 1}
_NULLABLE_STRING = {"type": ["string", "null"]}
_FORMAT = {"enum": ["html", "markdown", "tex", "pdf", None]}
_SHA256 = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
_DOCUMENT_REF = object_schema(
    {
        "source_format": {"enum": ["html", "markdown", "tex", "pdf"]},
        "source_sha256": _SHA256,
        "source_size": {"type": "integer", "minimum": 0},
        "media_type": _NONEMPTY_STRING,
        "parser_contract": _NONEMPTY_STRING,
        "parsed_document_sha256": _SHA256,
    },
    required=(
        "source_format",
        "source_sha256",
        "source_size",
        "media_type",
        "parser_contract",
        "parsed_document_sha256",
    ),
)
_STRUCTURE_REF = {"type": ["object", "null"]}
_STRING_ARRAY = {"type": "array", "items": _STRING}
_OBJECT_RESULT = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
}


def _service(cache_root: str | None) -> AcDocumentService:
    return AcDocumentService(cache_root=cache_root)


def _decode_document(value: Mapping[str, Any]) -> dict[str, Any]:
    decoded = dict(value)
    decoded["document_ref"] = cached_document_ref_from_document(
        decoded.pop("document_ref")
    )
    raw_structure = decoded.pop("structure_ref", None)
    decoded["structure_ref"] = (
        cached_document_structure_ref_from_document(raw_structure)
        if raw_structure is not None
        else None
    )
    return decoded


def _decode_full_text_search(value: Mapping[str, Any]) -> dict[str, Any]:
    decoded = _decode_document(value)
    decoded["equations"] = False
    return decoded


def _decode_equation_search(value: Mapping[str, Any]) -> dict[str, Any]:
    decoded = _decode_document(value)
    decoded["equations"] = True
    return decoded


def _import_source(
    path: str,
    source_format: str | None = None,
    cache_root: str | None = None,
) -> Any:
    return _service(cache_root).import_source(path, source_format=source_format)


def _parse_local(
    primary_path: str,
    validator_paths: Sequence[str] = (),
    validator_formats: Sequence[str | None] = (),
    primary_format: str | None = None,
    policy: str | None = None,
    cache_root: str | None = None,
) -> Any:
    return _service(cache_root).parse_local(
        primary_path,
        validator_paths=validator_paths,
        validator_formats=validator_formats,
        primary_format=primary_format,
        policy=policy,
    )


def _export_rich_document(
    source: str,
    output_dir: str,
    validator: str | None = None,
    source_format: str | None = None,
    cache_root: str | None = None,
) -> Any:
    return _service(cache_root).export_rich_document(
        source,
        output_dir=output_dir,
        validator=validator,
        source_format=source_format,
    )


def _extract_keywords(
    source: str,
    project_dir: str,
    structure_ref: Mapping[str, Any] | None = None,
    section_ids: Sequence[str] | None = None,
    approx_count: int = 50,
    cache_root: str | None = None,
    llm_provider: str = "auto",
    model: str | None = None,
    model_tier: str = "medium",
    run_id: str | None = None,
    resume_input: Mapping[str, Any] | None = None,
    host_authority: str = HostAuthority.UNKNOWN.value,
) -> Any:
    document_service = _service(cache_root)
    artifact = document_service.resolve_local_source(source)
    return document_service.extract_keywords(
        artifact,
        project_dir=project_dir,
        structure=(
            cached_document_structure_ref_from_document(structure_ref)
            if structure_ref is not None
            else None
        ),
        section_ids=section_ids,
        approx_count=approx_count,
        model=ModelSelection(
            provider=llm_provider,
            model=model,
            tier=model_tier,
        ),
        run_id=run_id,
        resume_input=resume_input,
        options=LLMExecutionOptions(
            host_authority=HostAuthority(host_authority)
        ),
    )


def _reconstruct_cached_structure(
    document_ref: Any,
    outline_document_ref: Mapping[str, Any],
    structure_ref: Any = None,
    cache_root: str | None = None,
) -> Any:
    del structure_ref
    return _service(cache_root).reconstruct_cached_structure(
        document_ref,
        cached_document_ref_from_document(outline_document_ref),
    )


def _get_table_of_contents(
    document_ref: Any,
    structure_ref: Any = None,
    cache_root: str | None = None,
) -> Any:
    return _service(cache_root).get_cached_table_of_contents(
        document_ref, structure=structure_ref
    )


def _get_section(
    document_ref: Any,
    selector: str | int,
    structure_ref: Any = None,
    cache_root: str | None = None,
) -> Any:
    return _service(cache_root).get_cached_section(
        document_ref, selector, structure=structure_ref
    )


def _read_cached_source_range(
    document_ref: Any,
    start_line: int,
    end_line: int,
    text_only: bool = False,
    structure_ref: Any = None,
    cache_root: str | None = None,
) -> Any:
    del structure_ref
    return _service(cache_root).read_cached_source_range(
        document_ref, start_line, end_line, text_only=text_only
    )


def _search_cached(
    document_ref: Any,
    terms: Sequence[str],
    limit: int = 100,
    context_lines: int = 0,
    case_sensitive: bool = False,
    equations: bool = False,
    structure_ref: Any = None,
    cache_root: str | None = None,
) -> dict[str, Any]:
    del structure_ref
    document_service = _service(cache_root)
    parsed, warnings = document_service.resolve_cached_document(document_ref)
    search = (
        document_service.search_equations
        if equations
        else document_service.search_full_text
    )
    results = [
        search(
            parsed,
            term,
            limit=limit,
            **(
                {}
                if equations
                else {"context_lines": context_lines}
            ),
            case_sensitive=case_sensitive,
        )
        for term in terms
    ]
    return {"terms": list(terms), "results": results, "warnings": warnings}


def _cache_list(
    document_ids: Sequence[str] = (),
    entry_ids: Sequence[str] = (),
    since_seconds: int | None = None,
    cache_root: str | None = None,
) -> Any:
    return _service(cache_root).list_cache(
        document_ids=document_ids,
        entry_ids=entry_ids,
        since_seconds=since_seconds,
    )


def _cache_remove(
    document_ids: Sequence[str] = (),
    entry_ids: Sequence[str] = (),
    dry_run: bool = True,
    cache_root: str | None = None,
) -> Any:
    return _service(cache_root).remove_cache(
        document_ids=document_ids,
        entry_ids=entry_ids,
        dry_run=dry_run,
    )


def _spec(
    name: str,
    schema: Mapping[str, Any],
    callable: Any,
    *,
    effects: frozenset[OperationEffect] = frozenset(),
    decoder: Any = None,
) -> OperationSpec[Any]:
    return operation_spec(
        package_name="ac-document",
        schema_namespace="ac.document",
        name=name,
        schema=schema,
        callable=callable,
        output_schema=_OBJECT_RESULT,
        effects=effects,
        decoder=decoder,
    )


def _document_parameters(
    extra: Mapping[str, Any] | None = None,
    *,
    required: Sequence[str] = (),
) -> dict[str, Any]:
    return object_schema(
        {
            "document_ref": _DOCUMENT_REF,
            "structure_ref": _STRUCTURE_REF,
            "cache_root": _NULLABLE_STRING,
            **(extra or {}),
        },
        required=("document_ref", *required),
    )


_OPERATIONS = (
    _spec(
        "import-source",
        object_schema(
            {
                "path": _NONEMPTY_STRING,
                "source_format": _FORMAT,
                "cache_root": _NULLABLE_STRING,
            },
            required=("path",),
        ),
        _import_source,
        effects=frozenset(
            {OperationEffect.CACHE_WRITE, OperationEffect.ARBITRARY_LOCAL_PATH}
        ),
    ),
    _spec(
        "parse-local",
        object_schema(
            {
                "primary_path": _NONEMPTY_STRING,
                "validator_paths": _STRING_ARRAY,
                "validator_formats": {"type": "array", "items": _FORMAT},
                "primary_format": _FORMAT,
                "policy": {
                    "enum": [
                        "none",
                        "deterministic_only",
                        "visual_all_pages",
                        None,
                    ]
                },
                "cache_root": _NULLABLE_STRING,
            },
            required=("primary_path",),
        ),
        _parse_local,
        effects=frozenset(
            {OperationEffect.CACHE_WRITE, OperationEffect.ARBITRARY_LOCAL_PATH}
        ),
    ),
    _spec(
        "export-rich-document",
        object_schema(
            {
                "source": _NONEMPTY_STRING,
                "output_dir": _NONEMPTY_STRING,
                "validator": _NULLABLE_STRING,
                "source_format": _FORMAT,
                "cache_root": _NULLABLE_STRING,
            },
            required=("source", "output_dir"),
        ),
        _export_rich_document,
        effects=frozenset(
            {OperationEffect.CACHE_WRITE, OperationEffect.ARBITRARY_LOCAL_PATH}
        ),
    ),
    _spec(
        "extract-keywords",
        object_schema(
            {
                "source": _NONEMPTY_STRING,
                "project_dir": _NONEMPTY_STRING,
                "structure_ref": {"type": ["object", "null"]},
                "section_ids": {
                    "type": ["array", "null"],
                    "items": _NONEMPTY_STRING,
                    "minItems": 1,
                },
                "approx_count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                },
                "cache_root": _NULLABLE_STRING,
                "llm_provider": _NONEMPTY_STRING,
                "model": _NULLABLE_STRING,
                "model_tier": {"enum": ["low", "medium", "high", "xhigh"]},
                "run_id": _NULLABLE_STRING,
                "resume_input": {"type": ["object", "null"]},
                "host_authority": {
                    "enum": ["unknown", "restricted", "unrestricted"]
                },
            },
            required=("source", "project_dir"),
        ),
        _extract_keywords,
        effects=frozenset(
            {
                OperationEffect.NETWORK,
                OperationEffect.CACHE_WRITE,
                OperationEffect.ARBITRARY_LOCAL_PATH,
                OperationEffect.RECURSIVE_LLM,
            }
        ),
    ),
    _spec(
        "reconstruct-cached-structure",
        _document_parameters(
            {"outline_document_ref": _DOCUMENT_REF},
            required=("outline_document_ref",),
        ),
        _reconstruct_cached_structure,
        effects=frozenset({OperationEffect.CACHE_WRITE}),
        decoder=_decode_document,
    ),
    _spec(
        "get-table-of-contents",
        _document_parameters(),
        _get_table_of_contents,
        decoder=_decode_document,
    ),
    _spec(
        "get-section",
        _document_parameters(
            {"selector": {"type": ["string", "integer"]}},
            required=("selector",),
        ),
        _get_section,
        decoder=_decode_document,
    ),
    _spec(
        "read-cached-source-range",
        _document_parameters(
            {
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
                "text_only": {"type": "boolean"},
            },
            required=("start_line", "end_line"),
        ),
        _read_cached_source_range,
        decoder=_decode_document,
    ),
    *(
        _spec(
            name,
            _document_parameters(
                {
                    "terms": {
                        "type": "array",
                        "items": _NONEMPTY_STRING,
                        "minItems": 1,
                    },
                    "limit": {"type": "integer", "minimum": 1},
                    "context_lines": {"type": "integer", "minimum": 0},
                    "case_sensitive": {"type": "boolean"},
                },
                required=("terms",),
            ),
            _search_cached,
            decoder=(
                _decode_equation_search
                if is_equations
                else _decode_full_text_search
            ),
        )
        for name, is_equations in (
            ("search-full-text", False),
            ("search-equations", True),
        )
    ),
    _spec(
        "cache-list",
        object_schema(
            {
                "document_ids": _STRING_ARRAY,
                "entry_ids": _STRING_ARRAY,
                "since_seconds": {"type": ["integer", "null"], "minimum": 1},
                "cache_root": _NULLABLE_STRING,
            }
        ),
        _cache_list,
        effects=frozenset({OperationEffect.CACHE_ADMIN}),
    ),
    _spec(
        "cache-remove",
        object_schema(
            {
                "document_ids": _STRING_ARRAY,
                "entry_ids": _STRING_ARRAY,
                "dry_run": {"type": "boolean"},
                "cache_root": _NULLABLE_STRING,
            }
        ),
        _cache_remove,
        effects=frozenset(
            {OperationEffect.CACHE_ADMIN, OperationEffect.DESTRUCTIVE}
        ),
    ),
)


OPERATION_REGISTRY = registry_mapping(_OPERATIONS)


def get_operation(operation: str) -> OperationSpec[Any] | None:
    return OPERATION_REGISTRY.get(operation)


def resolve_operations(
    *,
    excluded_effects: frozenset[OperationEffect] = DEFAULT_EXCLUDED_EFFECTS,
) -> tuple[OperationSpec[Any], ...]:
    return resolve_operation_specs(
        _OPERATIONS, excluded_effects=excluded_effects
    )


def registry_document(
    *,
    excluded_effects: frozenset[OperationEffect] = DEFAULT_EXCLUDED_EFFECTS,
) -> dict[str, Any]:
    return operation_registry_document(
        _OPERATIONS,
        schema_version=REGISTRY_SCHEMA_VERSION,
        excluded_effects=excluded_effects,
    )


def dispatch_operation(operation: str, parameters: Mapping[str, Any]) -> Any:
    spec = get_operation(operation)
    if spec is None:
        raise OperationRequestError(
            "operation_not_found",
            f"unknown ac-document operation: {operation}",
        )
    return spec.invoke(parameters)


__all__ = [
    "DEFAULT_EXCLUDED_EFFECTS",
    "JsonCodec",
    "JsonOutputCodec",
    "OPERATION_REGISTRY",
    "OperationEffect",
    "OperationRequestError",
    "OperationSpec",
    "REGISTRY_SCHEMA_VERSION",
    "dispatch_operation",
    "get_operation",
    "registry_document",
    "resolve_operations",
    "to_json_value",
]
