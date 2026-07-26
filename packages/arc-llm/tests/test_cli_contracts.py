from __future__ import annotations

import json

from arc_jobs import CommandResult, CommandStatus, RunRepository, RunSpec, RunStatus, command_result_json

from arc_llm import LLMCompleted, ProviderRegistry
from arc_llm.cli import _command_with_runtime_warnings, main


def test_cli_places_runtime_warnings_in_the_command_warning_channel() -> None:
    command = CommandResult(CommandStatus.COMPLETED, data={"kept": True})
    result = _command_with_runtime_warnings(
        command,
        LLMCompleted(
            {"answer": 1},
            "codex",
            "model",
            None,
            None,
            (
                {
                    "code": "internet_best_effort",
                    "message": "Internet access is best effort.",
                },
            ),
        ),
    )

    document = json.loads(command_result_json(result))
    assert document["data"] == {"kept": True}
    assert document["warnings"] == [
        {
            "code": "internet_best_effort",
            "message": "Internet access is best effort.",
            "details": {},
        }
    ]


def test_cli_unexpected_dispatch_error_is_one_failed_envelope(
    monkeypatch, capsys
) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr("arc_llm.cli._dispatch", fail)
    code = main(["doctor"], registry=ProviderRegistry())
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result["schema_version"] == "arc.command_result.v2"
    assert result["status"] == "failed"
    assert result["error"]["code"] == "internal_error"


def test_cli_local_io_error_is_not_misreported_as_invalid_request(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "arc_llm.cli._dispatch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("disk unavailable")
        ),
    )

    assert main(["doctor"], registry=ProviderRegistry()) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "local_io_error"


def test_cli_rejects_invalid_explicit_run_id_before_dispatch(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "arc_llm.cli._dispatch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid run ID must not reach dispatch")
        ),
    )

    assert (
        main(
            [
                "status",
                "--run-root",
                "runs",
                "--run-id",
                "../outside",
            ],
            registry=ProviderRegistry(),
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "invalid_request"


def test_cli_preserves_typed_jobs_failure_code(tmp_path, capsys) -> None:
    assert (
        main(
            [
                "status",
                "--run-root",
                str(tmp_path / "runs"),
                "--run-id",
                "missing-run",
            ],
            registry=ProviderRegistry(),
        )
        == 1
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "run_not_found"


def test_cli_rejects_removed_cancel_command(capsys) -> None:
    code = main(["cancel", "--run-root", "runs", "--run-id", "task"])
    captured = capsys.readouterr()

    assert code == 2
    result = json.loads(captured.out)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "invalid_request"


def test_cli_stop_uses_the_durable_stop_control(tmp_path, capsys) -> None:
    repository = RunRepository(tmp_path / "runs")
    repository.create(RunSpec("task", "test.handler", {}))

    code = main(
        [
            "stop",
            "--run-root",
            str(tmp_path / "runs"),
            "--run-id",
            "task",
            "--reason",
            "pause for review",
        ]
    )

    assert code == 0
    assert repository.inspect("task").snapshot.status is RunStatus.PAUSED
    result = json.loads(capsys.readouterr().out)
    assert result["schema_version"] == "arc.command_result.v2"
    assert result["status"] == "completed"
