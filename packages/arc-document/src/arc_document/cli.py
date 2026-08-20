"""Small JSON CLI for local document import and rich-document export."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from collections.abc import Sequence
from pathlib import Path

from .cached_document import cached_document_ref_from_document
from .document_structure import cached_document_structure_ref_from_document
from .service import ArcDocumentService


def _json_object(value: str):
    document = json.loads(value)
    if not isinstance(document, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return document


def _json_default(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _cached_document_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--document-ref", required=True, type=_json_object
    )
    parser.add_argument("--cache-root", type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arc-document")
    commands = parser.add_subparsers(dest="command", required=True)
    imported = commands.add_parser("import-source")
    imported.add_argument("source", type=Path)
    imported.add_argument("--cache-root", type=Path)
    exported = commands.add_parser("export-rich-document")
    exported.add_argument("source", type=Path)
    exported.add_argument("--output-dir", type=Path, required=True)
    exported.add_argument("--validator", type=Path)
    exported.add_argument("--cache-root", type=Path)
    source_range = commands.add_parser("read-cached-source-range")
    _cached_document_arguments(source_range)
    source_range.add_argument("--text-only", action="store_true")
    source_range.add_argument("start_line", type=int)
    source_range.add_argument("end_line", type=int)
    toc = commands.add_parser("get-table-of-contents")
    _cached_document_arguments(toc)
    toc.add_argument("--structure-ref", type=_json_object)
    section = commands.add_parser("get-section")
    _cached_document_arguments(section)
    section.add_argument("selector")
    section.add_argument("--structure-ref", type=_json_object)
    full_text = commands.add_parser("search-full-text")
    _cached_document_arguments(full_text)
    full_text.add_argument("--term", action="append", required=True)
    full_text.add_argument("--limit", type=int, default=100)
    full_text.add_argument("--context-lines", type=int, default=0)
    full_text.add_argument("--case-sensitive", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    service = ArcDocumentService(cache_root=args.cache_root)
    if args.command == "import-source":
        source = service.import_source(args.source)
        result = {
            "source_format": source.source_format.value,
            "artifact_digest": source.artifact_digest,
            "size": source.size,
            "media_type": source.media_type,
        }
    elif args.command == "export-rich-document":
        result = service.export_rich_document(
            args.source,
            output_dir=args.output_dir,
            validator=args.validator,
        )
    else:
        reference = cached_document_ref_from_document(args.document_ref)
        if args.command == "read-cached-source-range":
            result = service.read_cached_source_range(
                reference,
                args.start_line,
                args.end_line,
                text_only=args.text_only,
            )
        elif args.command == "get-table-of-contents":
            structure = (
                cached_document_structure_ref_from_document(
                    args.structure_ref
                )
                if args.structure_ref is not None
                else None
            )
            result = service.get_cached_table_of_contents(
                reference, structure=structure
            )
        elif args.command == "get-section":
            structure = (
                cached_document_structure_ref_from_document(
                    args.structure_ref
                )
                if args.structure_ref is not None
                else None
            )
            selector = (
                int(args.selector)
                if args.selector.isdecimal()
                else args.selector
            )
            result = service.get_cached_section(
                reference, selector, structure=structure
            )
        else:
            parsed, _warnings = service._resolve_cached_document(reference)
            result = {
                "terms": args.term,
                "results": [
                    service.search_full_text(
                        parsed,
                        term,
                        limit=args.limit,
                        context_lines=args.context_lines,
                        case_sensitive=args.case_sensitive,
                    )
                    for term in args.term
                ],
            }
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
