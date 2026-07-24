from __future__ import annotations

import json
import sys
from pathlib import Path

from arc_llm import (
    LLMRequest,
    ModelSelection,
    NativeResumeHandle,
    ProviderExecution,
    ProviderTerminalKind,
    TextOutput,
    request_to_document,
)
from arc_llm.cli import main
from arc_llm.output import CandidateMaterial
from arc_llm.providers.process import ProcessRunner


def test_cli_emits_one_shared_envelope_and_query_status(
    tmp_path: Path, adapter, registry, capsys
) -> None:
    adapter.steps.append(
        ProviderExecution(
            ProviderTerminalKind.COMPLETED,
            (CandidateMaterial(text="done", terminal=True),),
            NativeResumeHandle("codex", "thread"),
        )
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            request_to_document(
                LLMRequest(
                    "cli-task",
                    "Do it.",
                    TextOutput(),
                    ModelSelection("codex"),
                )
            )
        )
    )
    code = main(
        [
            "generate",
            "--request",
            str(request_path),
            "--run-root",
            str(tmp_path / "runs"),
        ],
        registry=registry,
    )
    generated = json.loads(capsys.readouterr().out)
    assert code == 0
    assert generated["schema_version"] == "arc.command_result.v1"
    assert generated["status"] == "completed"
    run_id = generated["run"]["id"]

    code = main(
        ["status", "--run-root", str(tmp_path / "runs"), "--run-id", run_id],
        registry=registry,
    )
    queried = json.loads(capsys.readouterr().out)
    assert code == 0
    assert queried["status"] == "completed"
    assert queried["data"]["run"]["status"] == "succeeded"


def test_cli_usage_error_is_same_envelope_with_exit_two(capsys) -> None:
    assert main(["generate"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["schema_version"] == "arc.command_result.v1"
    assert result["status"] == "failed"
    assert result["error"]["code"] == "invalid_request"


def test_cli_semantic_conflict_is_failed_even_when_existing_run_succeeded(
    tmp_path: Path, adapter, registry, capsys
) -> None:
    adapter.steps.append(
        ProviderExecution(
            ProviderTerminalKind.COMPLETED,
            (CandidateMaterial(text="done", terminal=True),),
            NativeResumeHandle("codex", "thread"),
        )
    )
    first = LLMRequest("same", "first", TextOutput(), ModelSelection("codex"))
    second = LLMRequest("same", "second", TextOutput(), ModelSelection("codex"))
    paths = []
    for index, request in enumerate((first, second)):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(request_to_document(request)))
        paths.append(path)
    assert (
        main(
            [
                "generate",
                "--request",
                str(paths[0]),
                "--run-root",
                str(tmp_path / "runs"),
            ],
            registry=registry,
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "generate",
                "--request",
                str(paths[1]),
                "--run-root",
                str(tmp_path / "runs"),
            ],
            registry=registry,
        )
        == 1
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "idempotency_conflict"


def test_process_runner_drains_both_streams_and_calls_delivery_barrier() -> None:
    barrier: list[str] = []
    result = ProcessRunner().run(
        [
            sys.executable,
            "-c",
            "import sys; data=sys.stdin.buffer.read(); "
            "sys.stdout.buffer.write(data); sys.stderr.write('diagnostic')",
        ],
        stdin=b"payload",
        env=None,
        idle_timeout_seconds=5,
        before_stdin=lambda: barrier.append("before"),
        cancel_check=lambda: None,
    )
    assert result.returncode == 0
    assert result.stdout == b"payload"
    assert result.stderr == b"diagnostic"
    assert barrier == ["before"]
