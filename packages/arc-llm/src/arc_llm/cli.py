"""Machine-oriented CLI using the shared arc-jobs command protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from arc_jobs import (
    ArcJobsError,
    CommandError,
    CommandResult,
    CommandWarning,
    CommandRun,
    CommandStatus,
    EventSink,
    InvalidRunIdError,
    ProgressEvent,
    RunRepository,
    command_result_from_snapshot,
    command_result_json,
    encode_progress_event,
    validate_simple_id,
)

from .api import LLMClient
from .config import resolve_model_selection
from .errors import ArcLLMError, ErrorCode
from .host import HostAuthority
from .outcome import LLMCompleted, LLMFailed
from .providers import ProviderRegistry, default_registry
from .request import (
    LLMExecutionOptions,
    ModelSelection,
    ProviderGateOptions,
    decode_request,
    decode_resume_input,
)


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


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="arc-llm",
        description=(
            "Run typed LLM requests with durable state, provider selection, "
            "and resumable execution."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser(
        "generate",
        help="execute a typed LLM request",
        description="Execute a typed LLM request from a JSON document.",
    )
    generate.add_argument("--request", required=True, type=Path, help="request JSON path")
    generate.add_argument(
        "--run-root", required=True, type=Path, help="durable run repository root"
    )
    generate.add_argument("--run-id", help="explicit durable run identifier")
    generate.add_argument(
        "--host-authority",
        choices=tuple(item.value for item in HostAuthority),
        default=HostAuthority.UNKNOWN.value,
        help="host authority attestation (default: unknown, so ARC uses host turns)",
    )
    _add_memory_gate_arguments(generate)

    resume = commands.add_parser(
        "resume",
        help="resume a paused or interrupted request",
        description="Resume a paused or interrupted typed LLM request.",
    )
    resume.add_argument(
        "--run-root", required=True, type=Path, help="durable run repository root"
    )
    resume.add_argument("--run-id", required=True, help="durable run identifier")
    resume.add_argument("--input", type=Path, help="ResumeInput JSON path")
    resume.add_argument(
        "--host-authority",
        choices=tuple(item.value for item in HostAuthority),
        default=HostAuthority.UNKNOWN.value,
        help="host authority attestation (default: unknown, so ARC uses host turns)",
    )
    _add_memory_gate_arguments(resume)

    status = commands.add_parser(
        "status",
        help="inspect a durable LLM request",
        description="Inspect the current state of a durable LLM request.",
    )
    status.add_argument("--run-root", required=True, type=Path, help="durable run repository root")
    status.add_argument("--run-id", required=True, help="durable run identifier")

    stop = commands.add_parser(
        "stop",
        help="request a cooperative stop",
        description="Request a cooperative stop for a durable LLM request.",
    )
    stop.add_argument("--run-root", required=True, type=Path, help="durable run repository root")
    stop.add_argument("--run-id", required=True, help="durable run identifier")
    stop.add_argument("--reason", help="human-readable stop reason")

    doctor = commands.add_parser(
        "doctor",
        help="check provider availability",
        description="Check whether an ARC LLM provider is configured and executable.",
    )
    doctor.add_argument(
        "--provider",
        choices=("auto", "codex", "claude", "kimi"),
        default="auto",
        help="provider to diagnose (default: auto)",
    )
    return parser


def _add_memory_gate_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--minimum-available-memory-percent",
        type=_memory_percent,
        metavar="PERCENT",
        help=(
            "pause new provider calls below this available-memory percentage "
            "(default: 10)"
        ),
    )
    group.add_argument(
        "--disable-memory-guard",
        action="store_true",
        help="launch provider calls without available-memory admission checks",
    )


def _memory_percent(value: str) -> float:
    try:
        percent = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not 0 < percent <= 100:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 100")
    return percent


def _execution_options(args: argparse.Namespace) -> LLMExecutionOptions:
    threshold: float | None = 0.10
    if args.disable_memory_guard:
        threshold = None
    elif args.minimum_available_memory_percent is not None:
        threshold = args.minimum_available_memory_percent / 100.0
    return LLMExecutionOptions(
        gate=ProviderGateOptions(
            minimum_available_memory_fraction=threshold,
        ),
        host_authority=HostAuthority(args.host_authority),
    )


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
    event_sink: EventSink | None,
) -> CommandResult:
    if args.command == "generate":
        request = decode_request(_read_object(args.request))
        options = _execution_options(args)
        result = client.generate(
            request,
            run_root=args.run_root,
            run_id=args.run_id,
            options=options,
            event_sink=event_sink,
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
        return _command_with_runtime_warnings(
            command_result_from_snapshot(result.snapshot), result.outcome
        )
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
            options=_execution_options(args),
            event_sink=event_sink,
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
        return _command_with_runtime_warnings(
            command_result_from_snapshot(result.snapshot), result.outcome
        )
    if args.command == "status":
        view = client.inspect(run_root=args.run_root, run_id=args.run_id)
        return command_result_from_snapshot(view.run.snapshot, query=True)
    if args.command == "stop":
        view = RunRepository(args.run_root).request_stop(
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


def _command_with_runtime_warnings(
    command: CommandResult,
    outcome: object,
) -> CommandResult:
    if not isinstance(outcome, LLMCompleted) or not outcome.warnings:
        return command
    warnings = tuple(
        CommandWarning(
            warning["code"],
            warning["message"],
            {
                key: value
                for key, value in warning.items()
                if key not in {"code", "message"}
            },
        )
        for warning in outcome.warnings
    )
    return CommandResult(
        command.status,
        command.run,
        data=command.data,
        artifacts=command.artifacts,
        warnings=command.warnings + warnings,
        error=command.error,
        resume=command.resume,
    )


def _failure(
    exc: Exception,
    *,
    help_command: str | None = None,
) -> CommandResult:
    if isinstance(exc, ArcLLMError):
        code = exc.code.value
        details = exc.details
    elif isinstance(exc, _UsageError):
        code = ErrorCode.INVALID_REQUEST.value
        details = {}
    elif isinstance(exc, OSError):
        code = "local_io_error"
        details = {}
    elif isinstance(exc, ArcJobsError):
        code = {
            "RunNotFoundError": "run_not_found",
            "RunBusyError": "run_busy",
            "IdempotencyConflictError": "idempotency_conflict",
            "ResumeInputConflictError": "resume_input_conflict",
            "UnsupportedSchemaError": "unsupported_state_version",
        }.get(type(exc).__name__, "arc_jobs_error")
        details = {}
    else:
        code = "internal_error"
        details = {}
    if code == ErrorCode.INVALID_REQUEST.value and help_command is not None:
        details = {**details, "help_command": help_command}
    return CommandResult(
        CommandStatus.FAILED,
        error=CommandError(code, str(exc) or type(exc).__name__, details),
    )


def _help_command(arguments: list[str]) -> str:
    command = (
        arguments[0]
        if arguments
        and arguments[0] in {"generate", "resume", "status", "stop", "doctor"}
        else None
    )
    return " ".join(
        part for part in ("arc-llm", command, "--help") if part is not None
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    client: LLMClient | None = None,
    registry: ProviderRegistry | None = None,
) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _build_parser().parse_args(arguments)
        run_id = getattr(args, "run_id", None)
        if run_id is not None:
            try:
                validate_simple_id(run_id, label="run id")
            except InvalidRunIdError as exc:
                raise _UsageError(str(exc)) from exc
        providers = registry or default_registry()
        result = _dispatch(
            args,
            client=client or LLMClient(registry=providers),
            registry=providers,
            event_sink=_stderr_event_sink,
        )
        exit_code = 1 if result.status is CommandStatus.FAILED else 0
    except _HelpRequested:
        return 0
    except _UsageError as exc:
        result = _failure(exc, help_command=_help_command(arguments))
        exit_code = 2
    except Exception as exc:
        result = _failure(exc, help_command=_help_command(arguments))
        exit_code = 1
    sys.stdout.write(command_result_json(result) + "\n")
    return exit_code


def _stderr_event_sink(document: Mapping[str, Any]) -> None:
    progress = ProgressEvent(
        str(document["run_id"]),
        int(document["sequence"]),
        str(document["event"]),
        dict(document["data"]),
        str(document["emitted_at"]),
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
    sys.stderr.flush()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
