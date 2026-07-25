from __future__ import annotations

import json

from arc_jobs import RunRepository, RunSpec, RunStatus

from arc_llm import ProviderRegistry
from arc_llm.cli import main


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
    assert result["error"]["code"] == "invalid_request"


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
