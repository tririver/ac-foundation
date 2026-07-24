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


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="arc-jobs", description="Inspect durable ARC runs")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--run-root", required=True)
        command.add_argument("--run-id", required=True)
    cancel = commands.add_parser("cancel")
    cancel.add_argument("--run-root", required=True)
    cancel.add_argument("--run-id", required=True)
    cancel.add_argument("--reason")
    return parser


def _emit(result: CommandResult, *, exit_code: int) -> int:
    sys.stdout.write(command_result_json(result) + "\n")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
        repository = RunRepository(args.run_root)
        if args.command == "status":
            result = command_result_from_snapshot(
                repository.inspect(args.run_id).snapshot, query=True
            )
            return _emit(result, exit_code=0)
        if args.command == "cancel":
            view = repository.request_cancel(args.run_id, reason=args.reason)
            result = CommandResult(
                CommandStatus.COMPLETED,
                CommandRun(view.snapshot.run_id, view.snapshot.revision),
                {
                    "run": {
                        "status": view.snapshot.status.value,
                        "cancel_requested": view.cancel_request is not None,
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
    except _UsageError as exc:
        return _emit(
            CommandResult(
                CommandStatus.FAILED,
                error=CommandError("invalid_request", str(exc)),
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
