"""Typed JSON command protocol for provider-neutral document operations."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from ac_jobs import (
    CommandError,
    CommandResult,
    CommandStatus,
    CommandWarning,
    command_result_json,
    command_result_from_snapshot,
    run_control_main,
)

from .registry import dispatch_operation, to_json_value
from .workflows.keywords import KeywordExtractionPaused


class _UsageError(ValueError):
    pass


class _HelpRequested(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if status == 0:
            raise _HelpRequested
        super().exit(status, message)


def _json_object(value: str) -> dict[str, Any]:
    try:
        document = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"value must be a JSON object: {exc.msg}"
        ) from exc
    if not isinstance(document, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return document


def _cache_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-root")


def _cached_document(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--document-ref", required=True, type=_json_object)
    parser.add_argument("--structure-ref", type=_json_object)
    _cache_root(parser)


def _parser() -> _Parser:
    parser = _Parser(
        prog="ac-document",
        description="Import, parse, search, and administer local documents.",
    )
    commands = parser.add_subparsers(dest="command")

    imported = commands.add_parser("import-source")
    imported.add_argument("path")
    imported.add_argument("--source-format", choices=("html", "markdown", "tex", "pdf"))
    _cache_root(imported)

    parsed = commands.add_parser("parse-local")
    parsed.add_argument("primary_path")
    parsed.add_argument("--validator", action="append", default=[])
    parsed.add_argument("--validator-format", action="append", default=[])
    parsed.add_argument("--primary-format", choices=("html", "markdown", "tex", "pdf"))
    parsed.add_argument(
        "--policy",
        choices=("none", "deterministic_only", "visual_all_pages"),
    )
    _cache_root(parsed)

    exported = commands.add_parser("export-rich-document")
    exported.add_argument("source")
    exported.add_argument("--output-dir", required=True)
    exported.add_argument("--validator")
    exported.add_argument("--source-format", choices=("html", "markdown", "tex", "pdf"))
    _cache_root(exported)

    acquired = commands.add_parser("acquire-html-bundle")
    acquired.add_argument("url")
    acquired.add_argument("--output-dir", required=True)
    acquired.add_argument("--allowed-origin", action="append", dest="allowed_origins", default=[])
    _cache_root(acquired)

    keywords = commands.add_parser("extract-keywords")
    keywords.add_argument("source")
    keywords.add_argument("--project-dir", required=True)
    keywords.add_argument("--approx-count", type=int, default=50)
    keywords.add_argument("--llm-provider", default="auto")
    keywords.add_argument("--model")
    keywords.add_argument(
        "--model-tier",
        choices=("low", "medium", "high", "xhigh"),
        default="medium",
    )
    keywords.add_argument("--run-id")
    keywords.add_argument("--resume-input", type=_json_object)
    keywords.add_argument("--structure-ref", type=_json_object)
    keywords.add_argument("--section-id", action="append", dest="section_ids")
    keywords.add_argument(
        "--host-authority",
        choices=("unknown", "restricted", "unrestricted"),
        default="unknown",
    )
    _cache_root(keywords)

    reconstructed = commands.add_parser("reconstruct-cached-structure")
    _cached_document(reconstructed)
    reconstructed.add_argument(
        "--outline-document-ref", required=True, type=_json_object
    )

    source_range = commands.add_parser("read-cached-source-range")
    _cached_document(source_range)
    source_range.add_argument("start_line", type=int)
    source_range.add_argument("end_line", type=int)
    source_range.add_argument("--text-only", action="store_true")

    toc = commands.add_parser("get-table-of-contents")
    _cached_document(toc)

    section = commands.add_parser("get-section")
    _cached_document(section)
    section_selector = section.add_mutually_exclusive_group(required=True)
    section_selector.add_argument("selector", nargs="?")
    section_selector.add_argument("--ordinal", type=int)

    for name in ("search-full-text", "search-equations"):
        search = commands.add_parser(name)
        _cached_document(search)
        search.add_argument("--term", action="append", required=True)
        search.add_argument("--limit", type=int, default=100)
        search.add_argument("--case-sensitive", action="store_true")
        if name == "search-full-text":
            search.add_argument("--context-lines", type=int, default=0)

    cache = commands.add_parser("cache")
    cache_commands = cache.add_subparsers(dest="cache_command", required=True)
    listed = cache_commands.add_parser("list")
    listed.add_argument("--document-id", action="append", default=[])
    listed.add_argument("--entry-id", action="append", default=[])
    listed.add_argument("--since-seconds", type=int)
    _cache_root(listed)
    removed = cache_commands.add_parser("remove")
    removed.add_argument("--document-id", action="append", default=[])
    removed.add_argument("--entry-id", action="append", default=[])
    removed.add_argument("--yes", action="store_true")
    _cache_root(removed)
    return parser


def _parameters(args: argparse.Namespace) -> dict[str, Any]:
    values = vars(args).copy()
    command = values.pop("command")
    cache_command = values.pop("cache_command", None)
    if command == "cache":
        values["document_ids"] = values.pop("document_id")
        values["entry_ids"] = values.pop("entry_id")
        if cache_command == "remove":
            values["dry_run"] = not values.pop("yes")
    if command == "parse-local":
        values["validator_paths"] = values.pop("validator")
        values["validator_formats"] = values.pop("validator_format")
    if command == "get-section":
        ordinal = values.pop("ordinal")
        if ordinal is not None:
            values["selector"] = ordinal
        elif values["selector"].isdecimal():
            values["selector"] = int(values["selector"])
    if command.startswith("search-"):
        values["terms"] = values.pop("term")
        values.setdefault("context_lines", 0)
    return values


def _result_data(value: Any) -> Mapping[str, Any]:
    encoded = to_json_value(value)
    if isinstance(encoded, Mapping):
        return dict(encoded)
    return {"result": encoded}


def _emit(result: CommandResult, *, exit_code: int) -> int:
    sys.stdout.write(command_result_json(result) + "\n")
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"status", "stop", "validate"}:
        return run_control_main(arguments, prog="ac-document")
    parser = _parser()
    try:
        args = parser.parse_args(arguments)
        if args.command is None:
            parser.print_help()
            return 0
        operation = (
            f"cache-{args.cache_command}"
            if args.command == "cache"
            else args.command
        )
        value = dispatch_operation(operation, _parameters(args))
        raw_warnings = (
            value.get("warnings", ())
            if isinstance(value, Mapping)
            else getattr(value, "warnings", ())
        )
        warnings = tuple(
            CommandWarning("document_warning", str(item))
            for item in raw_warnings
        )
        return _emit(
            CommandResult(
                CommandStatus.COMPLETED,
                data=_result_data(value),
                warnings=warnings,
            ),
            exit_code=0,
        )
    except KeywordExtractionPaused as exc:
        return _emit(command_result_from_snapshot(exc.snapshot), exit_code=0)
    except _HelpRequested:
        return 0
    except _UsageError as exc:
        return _emit(
            CommandResult(
                CommandStatus.FAILED,
                error=CommandError("invalid_request", str(exc)),
            ),
            exit_code=2,
        )
    except OSError as exc:
        return _emit(
            CommandResult(
                CommandStatus.FAILED,
                error=CommandError("local_io_error", str(exc)),
            ),
            exit_code=1,
        )
    except Exception as exc:
        code = getattr(exc, "code", None) or "internal_error"
        message = str(getattr(exc, "message", str(exc)))
        return _emit(
            CommandResult(
                CommandStatus.FAILED,
                error=CommandError(str(code), message),
            ),
            exit_code=1,
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
