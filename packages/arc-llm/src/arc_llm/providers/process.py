"""Blocking child-process execution with bounded diagnostics and full cleanup."""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..errors import FailureCategory, ProviderFailure


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_bytes: int | None = None
    stderr_bytes: int | None = None
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    last_activity_at: float | None = None


_DIAGNOSTIC_TAIL_BYTES = 256 * 1024


class _BoundedTail:
    def __init__(self, limit: int = _DIAGNOSTIC_TAIL_BYTES) -> None:
        self.limit = limit
        self.value = b""
        self.total = 0
        self.truncated = False

    def append(self, data: bytes) -> None:
        self.total += len(data)
        if len(data) >= self.limit:
            self.value = data[-self.limit :]
            self.truncated = self.total > self.limit
            return
        overflow = len(self.value) + len(data) - self.limit
        if overflow > 0:
            self.value = self.value[overflow:]
            self.truncated = True
        self.value += data


class ProcessRunner:
    """Run one process and join every helper thread before returning."""

    def run(
        self,
        argv: Sequence[str],
        *,
        stdin: bytes,
        env: Mapping[str, str] | None,
        cwd: Path,
        idle_timeout_seconds: float | None,
        stop_check: Callable[[], None],
        on_stdout: Callable[[bytes], None] | None = None,
        on_stderr: Callable[[bytes], None] | None = None,
    ) -> ProcessResult:
        popen_kwargs: dict[str, object] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "env": None if env is None else dict(env),
            "cwd": str(cwd),
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        elif os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            process = subprocess.Popen(list(argv), **popen_kwargs)
        except OSError as exc:
            raise ProviderFailure(
                f"Unable to start provider process: {exc}",
                category=FailureCategory.UNAVAILABLE,
            ) from exc

        chunks: queue.Queue[tuple[str, bytes | None]] = queue.Queue()
        stdout_tail = _BoundedTail()
        stderr_tail = _BoundedTail()
        errors: list[BaseException] = []

        def read_stream(name: str, stream: object) -> None:
            try:
                read = getattr(stream, "read1", None)
                if read is None:
                    read = stream.read  # type: ignore[attr-defined]
                while True:
                    data = read(65536)
                    if not data:
                        break
                    chunks.put((name, data))
            except BaseException as exc:
                errors.append(exc)
            finally:
                chunks.put((name, None))

        def write_stdin() -> None:
            try:
                assert process.stdin is not None
                process.stdin.write(stdin)
                process.stdin.flush()
                process.stdin.close()
            except BaseException as exc:
                errors.append(exc)

        assert process.stdout is not None and process.stderr is not None
        readers = [
            threading.Thread(target=read_stream, args=("stdout", process.stdout)),
            threading.Thread(target=read_stream, args=("stderr", process.stderr)),
        ]
        writer = threading.Thread(target=write_stdin)
        for thread in (*readers, writer):
            thread.start()

        closed: set[str] = set()
        last_activity = time.monotonic()
        last_activity_at = time.time()
        failure: BaseException | None = None
        try:
            while len(closed) < 2 or process.poll() is None:
                try:
                    stop_check()
                except BaseException as exc:
                    failure = exc
                    break
                if errors:
                    failure = ProviderFailure(
                        f"Provider pipe failed: {errors[0]}",
                        category=FailureCategory.TRANSPORT,
                        details={"code": "provider_pipe_failed"},
                    )
                    break
                remaining = None
                if idle_timeout_seconds is not None:
                    remaining = idle_timeout_seconds - (
                        time.monotonic() - last_activity
                    )
                    if remaining <= 0:
                        failure = ProviderFailure(
                            "Provider produced no activity before the idle timeout.",
                            category=FailureCategory.TIMEOUT,
                        )
                        break
                try:
                    name, data = chunks.get(
                        timeout=(
                            0.1 if remaining is None else min(0.1, remaining)
                        )
                    )
                except queue.Empty:
                    continue
                if data is None:
                    closed.add(name)
                    continue
                last_activity = time.monotonic()
                last_activity_at = time.time()
                if name == "stdout":
                    if on_stdout is None:
                        stdout_tail.append(data)
                    else:
                        stdout_tail.total += len(data)
                        stdout_tail.truncated = True
                    if on_stdout is not None:
                        on_stdout(data)
                else:
                    stderr_tail.append(data)
                    if on_stderr is not None:
                        on_stderr(data)
            if failure is not None:
                self._terminate(process)
            returncode = process.wait()
        finally:
            if process.poll() is None:
                self._terminate(process)
                process.wait()
            for thread in (*readers, writer):
                thread.join()
        # Readers can enqueue their final chunks while the provider is being
        # reaped. Drain those chunks before classifying the execution so a
        # terminal event already written to the pipe is not lost.
        while True:
            try:
                name, data = chunks.get_nowait()
            except queue.Empty:
                break
            if data is None:
                closed.add(name)
                continue
            last_activity = time.monotonic()
            last_activity_at = time.time()
            if name == "stdout":
                if on_stdout is None:
                    stdout_tail.append(data)
                else:
                    stdout_tail.total += len(data)
                    stdout_tail.truncated = True
                if on_stdout is not None:
                    on_stdout(data)
            else:
                stderr_tail.append(data)
                if on_stderr is not None:
                    on_stderr(data)
        # A stop requested before the provider result is handed back remains
        # authoritative, including one observed during the final stream drain.
        try:
            stop_check()
        except BaseException as exc:
            failure = exc
        if failure is not None:
            if isinstance(failure, ProviderFailure):
                failure.details.update(
                    {
                        "returncode": returncode,
                        "stdout_bytes": stdout_tail.total,
                        "stderr_bytes": stderr_tail.total,
                        "stdout_truncated": stdout_tail.truncated,
                        "stderr_truncated": stderr_tail.truncated,
                        "stderr_tail": stderr_tail.value.decode(
                            "utf-8", "replace"
                        ),
                        "last_activity_at": last_activity_at,
                        "termination_reason": failure.category.value,
                    }
                )
            raise failure
        if errors:
            raise ProviderFailure(
                f"Provider pipe failed: {errors[0]}",
                category=FailureCategory.TRANSPORT,
            )
        return ProcessResult(
            returncode,
            stdout_tail.value,
            stderr_tail.value,
            stdout_tail.total,
            stderr_tail.total,
            stdout_tail.truncated,
            stderr_tail.truncated,
            last_activity_at,
        )

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        if os.name == "posix":
            ProcessRunner._terminate_posix_group(process)
            return
        try:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        except ProcessLookupError:
            return

    @staticmethod
    def _terminate_posix_group(process: subprocess.Popen[bytes]) -> None:
        group_id = process.pid
        try:
            os.killpg(group_id, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return

        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            try:
                os.killpg(group_id, 0)
            except ProcessLookupError:
                process.wait()
                return
            time.sleep(0.01)

        try:
            os.killpg(group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
