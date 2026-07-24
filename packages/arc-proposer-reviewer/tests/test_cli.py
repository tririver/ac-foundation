from __future__ import annotations

import json
from pathlib import Path

from arc_proposer_reviewer import cli
from arc_proposer_reviewer.models import (
    BATCH_SCHEMA_VERSION,
    BatchRequest,
    LoopSpec,
    WorkerSpec,
)
from arc_proposer_reviewer.protocol import encode_batch_request


def request() -> BatchRequest:
    schema = {"type": "object", "additionalProperties": True}
    return BatchRequest(
        BATCH_SCHEMA_VERSION,
        "batch-cli",
        (
            LoopSpec(
                "loop-cli",
                {"question": "Q"},
                (WorkerSpec("p", "Propose.", schema),),
                WorkerSpec("r", "Review.", schema),
                1,
            ),
        ),
    )


def test_validate_emits_one_command_envelope_and_constructs_no_llm(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(encode_batch_request(request())),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "_handler",
        lambda: (_ for _ in ()).throw(AssertionError("LLM must not be constructed")),
    )
    assert cli.main(["validate", "--request", str(request_path)]) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1
    envelope = json.loads(lines[0])
    assert envelope["schema_version"] == "arc.command_result.v1"
    assert envelope["status"] == "completed"
    assert envelope["run"] is None
    assert envelope["data"]["valid"] is True
    assert captured.err == ""


def test_cli_surface_rejects_retired_dry_run_and_consensus_commands(capsys) -> None:
    for argv in (["run", "--dry-run"], ["consensus"]):
        assert cli.main(argv) == 2
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["status"] == "failed"
        assert envelope["error"]["code"] == "invalid_request"


def test_run_emits_no_synthetic_progress_events(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(encode_batch_request(request())),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "_handler",
        lambda: type("_Handler", (), {"name": "test.handler"})(),
    )
    monkeypatch.setattr(
        cli.RunEngine,
        "execute",
        lambda self, spec, handler: object(),
    )
    monkeypatch.setattr(
        cli,
        "command_result_from_snapshot",
        lambda snapshot: cli.CommandResult(cli.CommandStatus.COMPLETED),
    )

    assert (
        cli.main(
            [
                "run",
                "--request",
                str(request_path),
                "--run-root",
                str(tmp_path / "runs"),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(captured.out.splitlines()) == 1


def test_unexpected_exception_uses_internal_error_envelope(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(encode_batch_request(request())),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "_handler",
        lambda: (_ for _ in ()).throw(RuntimeError("sensitive detail")),
    )

    assert (
        cli.main(
            [
                "run",
                "--request",
                str(request_path),
                "--run-root",
                str(tmp_path / "runs"),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.err == ""
    envelope = json.loads(captured.out)
    assert envelope["status"] == "failed"
    assert envelope["error"]["code"] == "internal_error"
    assert "sensitive detail" not in captured.out
