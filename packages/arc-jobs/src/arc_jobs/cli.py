from __future__ import annotations

import argparse
import sys

from .errors import ArcJobsError
from .protocol import (
    CommandError,
    CommandResult,
    CommandRun,
    CommandStatus,
    command_result_from_snapshot,
    command_result_json,
)
from .engine import RunRepository


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
        prog="arc-jobs",
        description=(
            "Inspect and control durable ARC runs created by higher-level "
            "ARC commands."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    summaries = {
        "status": "inspect the current durable run state",
        "validate": "validate stored run state and artifacts",
    }
    for name, summary in summaries.items():
        command = commands.add_parser(name, help=summary, description=summary.capitalize() + ".")
        command.add_argument("--run-root", required=True, help="durable run repository root")
        command.add_argument("--run-id", required=True, help="durable run identifier")
    stop = commands.add_parser(
        "stop",
        help="request a cooperative stop",
        description="Request a cooperative stop for a durable ARC run.",
    )
    stop.add_argument("--run-root", required=True, help="durable run repository root")
    stop.add_argument("--run-id", required=True, help="durable run identifier")
    stop.add_argument("--reason", help="human-readable stop reason")
    return parser


def _emit(result: CommandResult, *, exit_code: int) -> int:
    sys.stdout.write(command_result_json(result) + "\n")
    return exit_code


def _help_command(arguments: list[str]) -> str:
    command = (
        arguments[0]
        if arguments and arguments[0] in {"status", "stop", "validate"}
        else None
    )
    return " ".join(
        part for part in ("arc-jobs", command, "--help") if part is not None
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _parser().parse_args(arguments)
        repository = RunRepository(args.run_root)
        if args.command == "status":
            result = command_result_from_snapshot(
                repository.inspect(args.run_id).snapshot, query=True
            )
            return _emit(result, exit_code=0)
        if args.command == "stop":
            view = repository.request_stop(args.run_id, reason=args.reason)
            result = CommandResult(
                CommandStatus.COMPLETED,
                CommandRun(view.snapshot.run_id, view.snapshot.revision),
                {
                    "run": {
                        "status": view.snapshot.status.value,
                        "stop_requested": view.stop_request is not None,
                    }
                },
            )
            return _emit(result, exit_code=0)
        if args.command == "validate":
            report = repository.validate(args.run_id)
            result = CommandResult(
                CommandStatus.COMPLETED,
                data={
                    "valid": report.ok,
                    "issues": [
                        {
                            "code": issue.code,
                            "message": issue.message,
                            "path": list(issue.path),
                        }
                        for issue in report.issues
                    ],
                },
            )
            return _emit(result, exit_code=0)
        raise AssertionError(args.command)
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
    except (ArcJobsError, OSError, ValueError) as exc:
        code = {
            "RunNotFoundError": "run_not_found",
            "RunBusyError": "run_busy",
            "IdempotencyConflictError": "idempotency_conflict",
            "ResumeInputConflictError": "resume_input_conflict",
            "UnsupportedSchemaError": "unsupported_state_version",
        }.get(type(exc).__name__, "arc_jobs_error")
        return _emit(
            CommandResult(
                CommandStatus.FAILED,
                error=CommandError(code, str(exc)),
            ),
            exit_code=1,
        )
    except Exception as exc:
        return _emit(
            CommandResult(
                CommandStatus.FAILED,
                error=CommandError(
                    "internal_error",
                    f"{type(exc).__name__}: {str(exc)[:300]}",
                ),
            ),
            exit_code=1,
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
