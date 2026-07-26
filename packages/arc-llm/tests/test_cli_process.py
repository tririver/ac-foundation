from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from arc_jobs import StoppedError
from arc_llm import (
    FailureCategory,
    LLMRequest,
    ModelSelection,
    NativeResumeHandle,
    ProviderExecution,
    ProviderTerminalKind,
    TextOutput,
    request_to_document,
)
from arc_llm.errors import ProviderFailure
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
    assert generated["schema_version"] == "arc.command_result.v2"
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


def test_cli_host_authority_defaults_to_brokered_and_accepts_unrestricted(
    tmp_path: Path, adapter, registry, capsys
) -> None:
    request = LLMRequest(
        "cli-authority",
        "Do it.",
        TextOutput(),
        ModelSelection("codex"),
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request_to_document(request)))

    adapter.steps.append(
        ProviderExecution(
            ProviderTerminalKind.COMPLETED,
            (CandidateMaterial(text="brokered", terminal=True),),
            NativeResumeHandle("codex", "brokered"),
        )
    )
    assert (
        main(
            [
                "generate",
                "--request",
                str(request_path),
                "--run-root",
                str(tmp_path / "brokered-runs"),
            ],
            registry=registry,
        )
        == 0
    )
    capsys.readouterr()
    assert adapter.requests[-1].capabilities["host_authority"] == "unknown"
    assert adapter.requests[-1].capabilities["effective_host_mode"] == "brokered"

    adapter.steps.append(
        ProviderExecution(
            ProviderTerminalKind.COMPLETED,
            (CandidateMaterial(text="direct", terminal=True),),
            NativeResumeHandle("codex", "direct"),
        )
    )
    assert (
        main(
            [
                "generate",
                "--request",
                str(request_path),
                "--run-root",
                str(tmp_path / "direct-runs"),
                "--host-authority",
                "unrestricted",
            ],
            registry=registry,
        )
        == 0
    )
    capsys.readouterr()
    assert adapter.requests[-1].capabilities["host_authority"] == "unrestricted"
    assert adapter.requests[-1].capabilities["effective_host_mode"] == "direct"


def test_cli_stderr_progress_and_stdout_final_result(
    tmp_path: Path, adapter, registry, capsys, monkeypatch
) -> None:
    original = adapter.start

    def start_with_progress(request, observer, stop):
        observer.progress("provider_phase", {"phase": "requesting"})
        return original(request, observer, stop)

    monkeypatch.setattr(adapter, "start", start_with_progress)
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
                    "cli-progress",
                    "Do it.",
                    TextOutput(),
                    ModelSelection("codex"),
                )
            )
        )
    )
    assert (
        main(
            [
                "generate",
                "--request",
                str(request_path),
                "--run-root",
                str(tmp_path / "runs"),
            ],
            registry=registry,
        )
        == 0
    )
    captured = capsys.readouterr()
    final_lines = captured.out.splitlines()
    assert len(final_lines) == 1
    assert json.loads(final_lines[0])["schema_version"] == "arc.command_result.v2"
    progress = [json.loads(line) for line in captured.err.splitlines()]
    assert progress
    assert all(item["schema_version"] == "arc.progress_event.v1" for item in progress)
    assert any(item["event"] == "provider_phase" for item in progress)
    request_messages = [
        item
        for item in progress
        if item["event"] == "llm_message"
        and item["data"]["direction"] == "request"
    ]
    assert request_messages[-1]["data"]["preview"] == "Do it."
    assert request_messages[-1]["data"]["message_kind"] == "task_prompt"


def test_cli_usage_error_is_same_envelope_with_exit_two(capsys) -> None:
    assert main(["generate"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["schema_version"] == "arc.command_result.v2"
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


def test_process_runner_drains_both_streams() -> None:
    result = ProcessRunner().run(
        [
            sys.executable,
            "-c",
            "import sys; data=sys.stdin.buffer.read(); "
            "sys.stdout.buffer.write(data); sys.stderr.write('diagnostic')",
        ],
        stdin=b"payload",
        env=None,
        cwd=Path.cwd(),
        idle_timeout_seconds=5,
        stop_check=lambda: None,
    )
    assert result.returncode == 0
    assert result.stdout == b"payload"
    assert result.stderr == b"diagnostic"


def test_process_runner_starts_child_in_required_workspace(tmp_path: Path) -> None:
    result = ProcessRunner().run(
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        stdin=b"",
        env=None,
        cwd=tmp_path,
        idle_timeout_seconds=5,
        stop_check=lambda: None,
    )
    assert Path(result.stdout.decode().strip()) == tmp_path


def test_process_creation_failure_is_not_delivered_and_unavailable(
    monkeypatch,
) -> None:
    def fail_popen(*args, **kwargs):
        raise OSError("missing executable")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    try:
        ProcessRunner().run(
            ["missing-provider"],
            stdin=b"",
            env={},
            cwd=Path.cwd(),
            idle_timeout_seconds=1,
            stop_check=lambda: None,
        )
    except ProviderFailure as failure:
        assert failure.category is FailureCategory.UNAVAILABLE
    else:
        raise AssertionError("process creation failure was not normalized")


def test_process_idle_timeout_terminates_and_reaps_provider() -> None:
    try:
        ProcessRunner().run(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=b"",
            env=None,
            cwd=Path.cwd(),
            idle_timeout_seconds=0.05,
            stop_check=lambda: None,
        )
    except ProviderFailure as failure:
        assert failure.category is FailureCategory.TIMEOUT
    else:
        raise AssertionError("idle provider was not terminated")


def test_process_allows_long_runtime_when_small_chunks_stay_active() -> None:
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    result = ProcessRunner().run(
        [
            sys.executable,
            "-c",
            "import sys,time\n"
            "for i in range(6):\n"
            " sys.stdout.write(str(i)); sys.stdout.flush()\n"
            " sys.stderr.write('.'); sys.stderr.flush()\n"
            " time.sleep(0.03)\n",
        ],
        stdin=b"",
        env=None,
        cwd=Path.cwd(),
        idle_timeout_seconds=0.08,
        stop_check=lambda: None,
        on_stdout=stdout_chunks.append,
        on_stderr=stderr_chunks.append,
    )
    assert result.returncode == 0
    # A streaming stdout consumer is the sole owner of provider output.
    assert result.stdout == b""
    assert result.stdout_bytes == 6
    assert result.stderr == b"......"
    assert len(stdout_chunks) > 1
    assert len(stderr_chunks) > 1


def test_process_streams_large_stdout_and_bounds_diagnostic_retention() -> None:
    stdout_bytes = 0

    def consume(chunk: bytes) -> None:
        nonlocal stdout_bytes
        stdout_bytes += len(chunk)

    size = 5 * 1024 * 1024
    result = ProcessRunner().run(
        [
            sys.executable,
            "-c",
            (
                "import os,sys; size=int(sys.argv[1]); "
                "sys.stdout.buffer.write(b'x'*size); "
                "sys.stderr.buffer.write(b'y'*size)"
            ),
            str(size),
        ],
        stdin=b"",
        env=None,
        cwd=Path.cwd(),
        idle_timeout_seconds=5,
        stop_check=lambda: None,
        on_stdout=consume,
    )

    assert result.returncode == 0
    assert stdout_bytes == size
    assert result.stdout == b""
    assert result.stdout_bytes == size
    assert result.stdout_truncated
    assert result.stderr_bytes == size
    assert len(result.stderr) == 256 * 1024
    assert result.stderr_truncated


def test_process_idle_timeout_covers_blocked_stdin_delivery() -> None:
    started = time.monotonic()
    with pytest.raises(ProviderFailure) as caught:
        ProcessRunner().run(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=b"x" * (2 * 1024 * 1024),
            env=None,
            cwd=Path.cwd(),
            idle_timeout_seconds=0.05,
            stop_check=lambda: None,
        )
    assert caught.value.category is FailureCategory.TIMEOUT
    assert time.monotonic() - started < 3


def test_process_stop_remains_active_after_delivery() -> None:
    checks = 0

    def stop() -> None:
        nonlocal checks
        checks += 1
        if checks >= 2:
            raise ProviderFailure(
                "stopped",
                category=FailureCategory.STOPPED,
            )

    with pytest.raises(ProviderFailure) as caught:
        ProcessRunner().run(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=b"delivered",
            env=None,
            cwd=Path.cwd(),
            idle_timeout_seconds=5,
            stop_check=stop,
        )
    assert caught.value.category is FailureCategory.STOPPED
    assert checks >= 2


def test_stop_requested_during_final_drain_outranks_completed_process() -> None:
    stopped = False

    def receive(_chunk: bytes) -> None:
        nonlocal stopped
        stopped = True

    def stop() -> None:
        if stopped:
            raise StoppedError("user requested stop")

    with pytest.raises(StoppedError):
        ProcessRunner().run(
            [sys.executable, "-c", "print('complete', flush=True)"],
            stdin=b"",
            env=None,
            cwd=Path.cwd(),
            idle_timeout_seconds=5,
            stop_check=stop,
            on_stdout=receive,
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group contract")
def test_timeout_terminates_descendant_after_group_leader_exits(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "descendant"
    child = (
        "import signal,sys,time\n"
        "from pathlib import Path\n"
        "base=Path(sys.argv[1])\n"
        "def stop(*_args):\n"
        " base.with_suffix('.terminated').write_text('term')\n"
        " raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "base.with_suffix('.ready').write_text('ready')\n"
        "time.sleep(60)\n"
    )
    leader = (
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        "base=Path(sys.argv[1])\n"
        "process=subprocess.Popen([sys.executable,'-c',sys.argv[2],str(base)])\n"
        "deadline=time.monotonic()+2\n"
        "while not base.with_suffix('.ready').exists():\n"
        " assert time.monotonic()<deadline\n"
        " time.sleep(0.005)\n"
        "print(process.pid,flush=True)\n"
    )

    with pytest.raises(ProviderFailure) as caught:
        ProcessRunner().run(
            [sys.executable, "-c", leader, str(marker), child],
            stdin=b"",
            env=None,
            cwd=Path.cwd(),
            idle_timeout_seconds=0.05,
            stop_check=lambda: None,
        )
    assert caught.value.category is FailureCategory.TIMEOUT
    assert marker.with_suffix(".terminated").read_text() == "term"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group contract")
def test_timeout_force_kills_term_ignoring_descendant_after_leader_exit(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "stubborn-descendant"
    child = (
        "import signal,sys,time\n"
        "from pathlib import Path\n"
        "base=Path(sys.argv[1])\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "base.with_suffix('.ready').write_text('ready')\n"
        "time.sleep(3)\n"
    )
    leader = (
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        "base=Path(sys.argv[1])\n"
        "process=subprocess.Popen([sys.executable,'-c',sys.argv[2],str(base)])\n"
        "deadline=time.monotonic()+2\n"
        "while not base.with_suffix('.ready').exists():\n"
        " assert time.monotonic()<deadline\n"
        " time.sleep(0.005)\n"
        "print(process.pid,flush=True)\n"
    )

    started = time.monotonic()
    with pytest.raises(ProviderFailure) as caught:
        ProcessRunner().run(
            [sys.executable, "-c", leader, str(marker), child],
            stdin=b"",
            env=None,
            cwd=Path.cwd(),
            idle_timeout_seconds=0.05,
            stop_check=lambda: None,
        )
    assert caught.value.category is FailureCategory.TIMEOUT
    assert time.monotonic() - started < 1.5
