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
from arc_proposer_reviewer.projection import (
    BatchInspection,
    BatchProjectionIntegrityError,
    BatchTrace,
    BestEffortActivity,
    CommittedRound,
    CommittedRoundRef,
    LoopInspection,
    LoopTrace,
    SafeArtifactRef,
)


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


def _safe_ref() -> SafeArtifactRef:
    return SafeArtifactRef(
        "proposer-reviewer/loops/loop-cli/rounds/001/proposals/p",
        "a" * 64,
        12,
        "application/json",
    )


def _inspection() -> BatchInspection:
    return BatchInspection(
        run_id="run-cli",
        run_lifecycle="running",
        run_revision=7,
        loop_revisions={"loop-cli": 3},
        loops=(
            LoopInspection(
                loop_id="loop-cli",
                lifecycle="running",
                phase="reviewer",
                current_round=2,
                rounds_completed=1,
                revision=3,
                pause=None,
                activity=BestEffortActivity(True, None, 0, 1, 0),
            ),
        ),
    )


def _trace() -> BatchTrace:
    reference = _safe_ref()
    round_ref = CommittedRoundRef(
        loop_id="loop-cli",
        round_number=1,
        proposal_refs={"p": reference},
        review_ref=reference,
        transcript_refs=(reference,),
    )
    return BatchTrace(
        run_id="run-cli",
        run_revision=7,
        loop_revisions={"loop-cli": 3},
        loops=(LoopTrace("loop-cli", 3, (round_ref,)),),
    )


def _round() -> CommittedRound:
    reference = _safe_ref()
    return CommittedRound(
        loop_id="loop-cli",
        round_number=1,
        proposals={"p": {"proposal": "committed"}},
        review={"review": "committed"},
        proposal_refs={"p": reference},
        review_ref=reference,
        transcript_refs=(reference,),
    )


class _Projection:
    def __init__(self, _repository, _run_id: str, *, corrupt_trace: bool = False) -> None:
        self.corrupt_trace = corrupt_trace

    def inspect(self) -> BatchInspection:
        return _inspection()

    def trace(self) -> BatchTrace:
        if self.corrupt_trace:
            raise BatchProjectionIntegrityError("loop_integrity_error")
        return _trace()

    def read_round(self, loop_id: str, round_number: int) -> CommittedRound:
        assert (loop_id, round_number) == ("loop-cli", 1)
        return _round()


class _Runner:
    def __init__(self, *, corrupt_trace: bool = False) -> None:
        self.corrupt_trace = corrupt_trace

    def projection(self, _run_root, run_id: str) -> _Projection:
        return _Projection(None, run_id, corrupt_trace=self.corrupt_trace)


def _query_envelope(capsys) -> dict:
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    envelope = json.loads(lines[0])
    assert envelope["schema_version"] == "arc.command_result.v2"
    assert envelope["status"] == "completed"
    assert envelope["run"] == {"id": "run-cli", "revision": 7}
    return envelope


def _assert_query_output_is_safe(envelope: dict) -> None:
    rendered = json.dumps(envelope, sort_keys=True)
    for forbidden in (
        "relative_path",
        "session",
        "task_id",
        "private-group",
        "resume_key",
        "pause_details",
    ):
        assert forbidden not in rendered


def test_inspect_query_emits_one_safe_command_envelope(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "BatchRunner", _Runner)

    assert (
        cli.main(["inspect", "--run-root", str(tmp_path), "--run-id", "run-cli"])
        == 0
    )

    envelope = _query_envelope(capsys)
    assert envelope["data"]["inspection"]["run_lifecycle"] == "running"
    assert "trace" not in envelope["data"]
    _assert_query_output_is_safe(envelope)


def test_trace_and_show_round_queries_emit_safe_command_envelopes(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "_runner", _Runner)

    assert cli.main(["trace", "--run-root", str(tmp_path), "--run-id", "run-cli"]) == 0
    trace_envelope = _query_envelope(capsys)
    assert trace_envelope["data"]["trace"]["loops"][0]["rounds"][0]["round_number"] == 1
    _assert_query_output_is_safe(trace_envelope)

    assert (
        cli.main(
            [
                "show-round",
                "--run-root",
                str(tmp_path),
                "--run-id",
                "run-cli",
                "--loop-id",
                "loop-cli",
                "--round",
                "1",
            ]
        )
        == 0
    )
    round_envelope = _query_envelope(capsys)
    assert round_envelope["data"]["round"]["proposals"] == {
        "p": {"proposal": "committed"}
    }
    _assert_query_output_is_safe(round_envelope)


def test_inspect_include_trace_warns_without_exposing_corruption_details(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(
        cli,
        "_runner",
        lambda: _Runner(corrupt_trace=True),
    )

    assert (
        cli.main(
            [
                "inspect",
                "--run-root",
                str(tmp_path),
                "--run-id",
                "run-cli",
                "--include-trace",
            ]
        )
        == 0
    )

    envelope = _query_envelope(capsys)
    assert envelope["data"]["trace"] is None
    assert envelope["warnings"] == [
        {
            "code": "trace_integrity_error",
            "message": "committed trace could not be verified",
            "details": {},
        }
    ]
    assert "loop_integrity_error" not in json.dumps(envelope)
    _assert_query_output_is_safe(envelope)


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
        "_runner",
        lambda: (_ for _ in ()).throw(AssertionError("runner must not be constructed")),
    )
    assert cli.main(["validate", "--request", str(request_path)]) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1
    envelope = json.loads(lines[0])
    assert envelope["schema_version"] == "arc.command_result.v2"
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
    calls = []

    class _RunRunner:
        def run(self, batch, run_root, run_id):
            calls.append((batch, run_root, run_id))
            return object()

    monkeypatch.setattr(cli, "_runner", _RunRunner)
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
    assert calls == [(request(), str(tmp_path / "runs"), None)]


def test_resume_delegates_to_shared_runner(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.json"
    resume_input = {"resume_key": "resume-a", "action": "continue"}
    input_path.write_text(json.dumps(resume_input), encoding="utf-8")
    calls = []

    class _ResumeRunner:
        def resume(self, run_root, run_id, input):
            calls.append((run_root, run_id, input))
            return object()

    monkeypatch.setattr(cli, "_runner", _ResumeRunner)
    monkeypatch.setattr(
        cli,
        "command_result_from_snapshot",
        lambda snapshot: cli.CommandResult(cli.CommandStatus.COMPLETED),
    )

    assert (
        cli.main(
            [
                "resume",
                "--run-root",
                str(tmp_path / "runs"),
                "--run-id",
                "run-a",
                "--input",
                str(input_path),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(captured.out.splitlines()) == 1
    assert calls == [(str(tmp_path / "runs"), "run-a", resume_input)]


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
        "_runner",
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
