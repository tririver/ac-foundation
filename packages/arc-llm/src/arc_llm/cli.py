"""Machine-oriented CLI using the shared arc-jobs command protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from arc_jobs import (
    CommandError,
    CommandResult,
    CommandRun,
    CommandStatus,
    EventWriter,
    ProgressEvent,
    RunRepository,
    command_result_from_snapshot,
    command_result_json,
    encode_progress_event,
)

from .api import LLMClient
from .config import resolve_model_selection
from .errors import ArcLLMError, ErrorCode
from .outcome import LLMFailed
from .providers import ProviderRegistry, default_registry
from .request import ModelSelection, decode_request, decode_resume_input


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _UsageError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="arc-llm", description="Durable ARC LLM task runner")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate")
    generate.add_argument("--request", required=True, type=Path)
    generate.add_argument("--run-root", required=True, type=Path)
    generate.add_argument("--run-id")

    resume = commands.add_parser("resume")
    resume.add_argument("--run-root", required=True, type=Path)
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--input", type=Path)

    for name in ("status", "cancel"):
        command = commands.add_parser(name)
        command.add_argument("--run-root", required=True, type=Path)
        command.add_argument("--run-id", required=True)
    commands.choices["cancel"].add_argument("--reason")

    doctor = commands.add_parser("doctor")
    doctor.add_argument(
        "--provider",
        choices=("auto", "codex", "claude", "kimi"),
        default="auto",
    )
    return parser


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _UsageError(f"cannot read JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise _UsageError(f"expected a JSON object in {path}")
    return value


def _dispatch(
    args: argparse.Namespace,
    *,
    client: LLMClient,
    registry: ProviderRegistry,
) -> CommandResult:
    if args.command == "generate":
        request = decode_request(_read_object(args.request))
        result = client.generate(
            request,
            run_root=args.run_root,
            run_id=args.run_id,
        )
        if isinstance(result.outcome, LLMFailed):
            return CommandResult(
                CommandStatus.FAILED,
                CommandRun(result.snapshot.run_id, result.snapshot.revision),
                error=CommandError(
                    result.outcome.error.code.value,
                    str(result.outcome.error),
                    result.outcome.error.details,
                ),
            )
        return command_result_from_snapshot(result.snapshot)
    if args.command == "resume":
        resume_input = (
            None
            if args.input is None
            else decode_resume_input(_read_object(args.input))
        )
        result = client.resume(
            run_root=args.run_root,
            run_id=args.run_id,
            input=resume_input,
        )
        if isinstance(result.outcome, LLMFailed):
            return CommandResult(
                CommandStatus.FAILED,
                CommandRun(result.snapshot.run_id, result.snapshot.revision),
                error=CommandError(
                    result.outcome.error.code.value,
                    str(result.outcome.error),
                    result.outcome.error.details,
                ),
            )
        return command_result_from_snapshot(result.snapshot)
    if args.command == "status":
        view = client.inspect(run_root=args.run_root, run_id=args.run_id)
        return command_result_from_snapshot(view.run.snapshot, query=True)
    if args.command == "cancel":
        view = RunRepository(args.run_root).request_cancel(
            args.run_id,
            reason=args.reason,
        )
        return CommandResult(
            CommandStatus.COMPLETED,
            CommandRun(view.snapshot.run_id, view.snapshot.revision),
            data={
                "run": command_result_from_snapshot(
                    view.snapshot,
                    query=True,
                ).data["run"]
            },
        )
    selection = resolve_model_selection(
        ModelSelection(provider=args.provider),
        available=registry.names(),
    )
    diagnostic = registry.create(selection.provider).doctor()
    return CommandResult(
        CommandStatus.COMPLETED,
        data={
            "provider": diagnostic.provider,
            "available": diagnostic.available,
            "executable": diagnostic.executable,
            "details": dict(diagnostic.details),
        },
    )


def _failure(exc: Exception) -> CommandResult:
    if isinstance(exc, ArcLLMError):
        code = exc.code.value
        details = exc.details
    else:
        code = ErrorCode.INVALID_REQUEST.value
        details = {}
    return CommandResult(
        CommandStatus.FAILED,
        error=CommandError(code, str(exc) or type(exc).__name__, details),
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    client: LLMClient | None = None,
    registry: ProviderRegistry | None = None,
) -> int:
    try:
        args = _build_parser().parse_args(argv)
        providers = registry or default_registry()
        result = _dispatch(
            args,
            client=client or LLMClient(registry=providers),
            registry=providers,
        )
        if args.command in {"generate", "resume"} and result.run is not None:
            repository = RunRepository(args.run_root)
            events = EventWriter(
                repository.run_directory(result.run.id) / "events.jsonl",
                run_id=result.run.id,
            ).tail()
            for event in events:
                progress = ProgressEvent(
                    result.run.id,
                    int(event["sequence"]),
                    str(event["event"]),
                    dict(event["data"]),
                    str(event["emitted_at"]),
                )
                sys.stderr.write(
                    json.dumps(
                        encode_progress_event(progress),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        exit_code = 1 if result.status is CommandStatus.FAILED else 0
    except _UsageError as exc:
        result = _failure(exc)
        exit_code = 2
    except Exception as exc:
        result = _failure(exc)
        exit_code = 1
    sys.stdout.write(command_result_json(result) + "\n")
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
