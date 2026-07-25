from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, cast

from arc_jobs import (
    ArcJobsError,
    CommandError,
    CommandResult,
    CommandRun,
    CommandStatus,
    CommandWarning,
    JsonValue,
    command_result_from_snapshot,
    command_result_json,
)
from arc_llm import ArcLLMError

from .protocol import decode_batch_request
from .projection import (
    BatchProjectionIntegrityError,
    CommittedRoundNotFoundError,
)
from .runner import BatchRunner


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="arc-proposer-reviewer",
        description="Run typed proposer-reviewer batches",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--request", required=True)
    run = commands.add_parser("run")
    run.add_argument("--request", required=True)
    run.add_argument("--run-root", required=True)
    run.add_argument("--run-id")
    resume = commands.add_parser("resume")
    resume.add_argument("--run-root", required=True)
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--input")
    inspect = commands.add_parser("inspect")
    _query_arguments(inspect)
    inspect.add_argument("--include-trace", action="store_true")
    trace = commands.add_parser("trace")
    _query_arguments(trace)
    show_round = commands.add_parser("show-round")
    _query_arguments(show_round)
    show_round.add_argument("--loop-id", required=True)
    show_round.add_argument("--round", required=True, type=int, dest="round_number")
    return parser


def _query_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id", required=True)


def _load_object(path: str) -> Mapping[str, JsonValue]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON document must be an object")
    return cast(Mapping[str, JsonValue], value)


def _emit(result: CommandResult, *, exit_code: int) -> int:
    sys.stdout.write(command_result_json(result) + "\n")
    return exit_code


def _runner() -> BatchRunner:
    return BatchRunner()


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
        if args.command == "validate":
            request = decode_batch_request(_load_object(args.request))
            return _emit(
                CommandResult(
                    CommandStatus.COMPLETED,
                    data={
                        "schema_version": request.schema_version,
                        "batch_id": request.batch_id,
                        "loop_count": len(request.loops),
                        "valid": True,
                    },
                ),
                exit_code=0,
            )

        runner = _runner()
        if args.command in {"inspect", "trace", "show-round"}:
            projection = runner.projection(args.run_root, args.run_id)
            if args.command == "inspect":
                inspection = projection.inspect()
                data: dict[str, Any] = {
                    "inspection": _json_value(inspection),
                }
                warnings = ()
                if args.include_trace:
                    try:
                        data["trace"] = _json_value(projection.trace())
                    except BatchProjectionIntegrityError:
                        data["trace"] = None
                        warnings = (
                            CommandWarning(
                                "trace_integrity_error",
                                "committed trace could not be verified",
                            ),
                        )
                result = CommandResult(
                    CommandStatus.COMPLETED,
                    CommandRun(args.run_id, inspection.run_revision),
                    data,
                    warnings=warnings,
                )
            elif args.command == "trace":
                trace = projection.trace()
                result = CommandResult(
                    CommandStatus.COMPLETED,
                    CommandRun(args.run_id, trace.run_revision),
                    {"trace": _json_value(trace)},
                )
            else:
                inspection = projection.inspect()
                expanded = projection.read_round(
                    args.loop_id, args.round_number
                )
                result = CommandResult(
                    CommandStatus.COMPLETED,
                    CommandRun(args.run_id, inspection.run_revision),
                    {"round": _json_value(expanded)},
                )
            return _emit(result, exit_code=0)

        if args.command == "run":
            request = decode_batch_request(_load_object(args.request))
            snapshot = runner.run(
                request,
                args.run_root,
                args.run_id,
            )
        elif args.command == "resume":
            resume_input = (
                None if args.input is None else _load_object(args.input)
            )
            snapshot = runner.resume(
                args.run_root,
                args.run_id,
                resume_input,
            )
        else:
            raise AssertionError(args.command)
        result = command_result_from_snapshot(snapshot)
        return _emit(
            result,
            exit_code=1 if result.status is CommandStatus.FAILED else 0,
        )
    except _UsageError as exc:
        return _emit(
            CommandResult(
                CommandStatus.FAILED,
                error=CommandError("invalid_request", str(exc)),
            ),
            exit_code=2,
        )
    except (ArcJobsError, ArcLLMError, OSError, ValueError) as exc:
        code = {
            "IdempotencyConflictError": "idempotency_conflict",
            "ResumeMismatchError": "resume_mismatch",
            "ResumeInputConflictError": "resume_input_conflict",
            "RunNotFoundError": "run_not_found",
            "RunBusyError": "run_busy",
            "BatchProjectionIntegrityError": "trace_integrity_error",
            "CommittedRoundNotFoundError": "committed_round_not_found",
        }.get(type(exc).__name__, "invalid_request")
        return _emit(
            CommandResult(
                CommandStatus.FAILED,
                error=CommandError(code, str(exc)),
            ),
            exit_code=1,
        )
    except Exception:
        return _emit(
            CommandResult(
                CommandStatus.FAILED,
                error=CommandError(
                    "internal_error",
                    "unexpected internal error",
                ),
            ),
            exit_code=1,
        )


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return _json_value(value.value)
    if is_dataclass(value):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"unsupported projection value: {type(value).__name__}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
