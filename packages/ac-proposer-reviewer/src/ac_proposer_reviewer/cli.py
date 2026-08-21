from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, cast

from ac_jobs import (
    AcJobsError,
    CommandError,
    CommandResult,
    CommandRun,
    CommandStatus,
    CommandWarning,
    JsonValue,
    command_result_from_snapshot,
    command_result_json,
)
from ac_llm import AcLLMError

from .protocol import decode_batch_request
from .projection import (
    BatchProjectionIntegrityError,
    CommittedRoundNotFoundError,
)
from .runner import BatchRunner


class _UsageError(Exception):
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


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="ac-proposer-reviewer",
        description=(
            "Run and inspect typed, durable proposer-reviewer batches with "
            "committed round artifacts."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate",
        help="validate a batch request without running it",
        description="Validate a proposer-reviewer batch request JSON document.",
    )
    validate.add_argument("--request", required=True, help="batch request JSON path")
    run = commands.add_parser(
        "run",
        help="run a proposer-reviewer batch",
        description="Run a typed proposer-reviewer batch from a JSON request.",
    )
    run.add_argument("--request", required=True, help="batch request JSON path")
    run.add_argument("--run-root", required=True, help="durable run repository root")
    run.add_argument("--run-id", help="explicit durable run identifier")
    resume = commands.add_parser(
        "resume",
        help="resume a paused or interrupted batch",
        description="Resume a paused or interrupted proposer-reviewer batch.",
    )
    resume.add_argument("--run-root", required=True, help="durable run repository root")
    resume.add_argument("--run-id", required=True, help="durable run identifier")
    resume.add_argument("--input", help="ResumeInput JSON path")
    stop = commands.add_parser(
        "stop",
        help="request cooperative stop of a batch",
        description="Request a cooperative stop of a running proposer-reviewer batch.",
    )
    _query_arguments(stop)
    stop.add_argument(
        "--reason",
        help="short operator reason recorded with the stop request",
    )
    inspect = commands.add_parser(
        "inspect",
        help="inspect batch and loop state",
        description="Inspect durable batch and loop state.",
    )
    _query_arguments(inspect)
    inspect.add_argument(
        "--include-trace", action="store_true", help="include the committed trace projection"
    )
    trace = commands.add_parser(
        "trace",
        help="read the committed batch trace",
        description="Read the verified committed trace for a batch.",
    )
    _query_arguments(trace)
    show_round = commands.add_parser(
        "show-round",
        help="read one committed loop round",
        description="Read proposals, review, and transcript references for one committed round.",
    )
    _query_arguments(show_round)
    show_round.add_argument("--loop-id", required=True, help="loop identifier")
    show_round.add_argument(
        "--round", required=True, type=int, dest="round_number", help="one-based round number"
    )
    return parser


def _query_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", required=True, help="durable run repository root")
    parser.add_argument("--run-id", required=True, help="durable run identifier")


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


def _help_command(arguments: list[str]) -> str:
    command = (
        arguments[0]
        if arguments
        and arguments[0]
        in {"validate", "run", "resume", "stop", "inspect", "trace", "show-round"}
        else None
    )
    return " ".join(
        part
        for part in ("ac-proposer-reviewer", command, "--help")
        if part is not None
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _parser().parse_args(arguments)
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
        if args.command == "stop":
            view = runner.stop(args.run_root, args.run_id, args.reason)
            return _emit(
                CommandResult(
                    CommandStatus.COMPLETED,
                    CommandRun(args.run_id, view.snapshot.revision),
                    {
                        "stop_requested": view.stop_request is not None,
                        "durable_lifecycle": view.snapshot.status.value,
                    },
                ),
                exit_code=0,
            )
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
    except _HelpRequested:
        return 0
    except _UsageError as exc:
        return _emit(
            CommandResult(
                CommandStatus.FAILED,
                error=CommandError(
                    "invalid_request",
                    str(exc),
                    {"help_command": _help_command(arguments)},
                ),
            ),
            exit_code=2,
        )
    except (AcJobsError, AcLLMError, OSError, ValueError) as exc:
        code = {
            "IdempotencyConflictError": "idempotency_conflict",
            "ResumeMismatchError": "resume_mismatch",
            "ResumeInputConflictError": "resume_input_conflict",
            "RunNotFoundError": "run_not_found",
            "RunBusyError": "run_busy",
            "BatchProjectionIntegrityError": "trace_integrity_error",
            "CommittedRoundNotFoundError": "committed_round_not_found",
        }.get(type(exc).__name__, "invalid_request")
        details = (
            {"help_command": _help_command(arguments)}
            if code == "invalid_request"
            else {}
        )
        return _emit(
            CommandResult(
                CommandStatus.FAILED,
                error=CommandError(code, str(exc), details),
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
