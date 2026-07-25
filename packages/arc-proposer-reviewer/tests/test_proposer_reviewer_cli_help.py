from __future__ import annotations

import json

import pytest

from arc_proposer_reviewer.cli import main


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["validate", "--help"],
        ["run", "--help"],
        ["resume", "--help"],
        ["inspect", "--help"],
        ["trace", "--help"],
        ["show-round", "--help"],
    ],
)
def test_root_and_subcommand_help_is_human_readable(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.startswith("usage: arc-proposer-reviewer")
    assert "arc.command_result.v2" not in captured.out


def test_usage_error_points_to_contextual_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["show-round"]) == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result["error"]["code"] == "invalid_request"
    assert result["error"]["details"] == {
        "help_command": "arc-proposer-reviewer show-round --help"
    }
