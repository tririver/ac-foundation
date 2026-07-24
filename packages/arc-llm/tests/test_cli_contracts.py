from __future__ import annotations

import json

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
    assert result["schema_version"] == "arc.command_result.v1"
    assert result["status"] == "failed"
    assert result["error"]["code"] == "invalid_request"
