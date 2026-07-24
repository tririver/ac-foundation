from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, cast

from arc_jobs import (
    ArcJobsError,
    CommandError,
    CommandResult,
    CommandStatus,
    JsonValue,
    RunEngine,
    RunRepository,
    RunSpec,
    command_result_from_snapshot,
    command_result_json,
)
from arc_llm import ArcLLMError, LLMTaskService

from .handler import ProposerReviewerHandler
from .identity import derive_batch_run_id
from .protocol import decode_batch_request, encode_batch_request
from .service import ProposerReviewerService


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
    return parser


def _load_object(path: str) -> Mapping[str, JsonValue]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("JSON document must be an object")
    return cast(Mapping[str, JsonValue], value)


def _emit(result: CommandResult, *, exit_code: int) -> int:
    sys.stdout.write(command_result_json(result) + "\n")
    return exit_code


def _handler() -> ProposerReviewerHandler:
    return ProposerReviewerHandler(ProposerReviewerService(LLMTaskService()))


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

        repository = RunRepository(args.run_root)
        handler = _handler()
        if args.command == "run":
            request = decode_batch_request(_load_object(args.request))
            run_id = args.run_id or derive_batch_run_id(request.batch_id)
            snapshot = RunEngine(repository).execute(
                RunSpec(run_id, handler.name, encode_batch_request(request)),
                handler,
            )
        elif args.command == "resume":
            run_id = args.run_id
            resume_input = (
                None if args.input is None else _load_object(args.input)
            )
            snapshot = RunEngine(repository).resume(
                run_id,
                handler,
                input=resume_input,
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
