"""Blocking child-process execution with idle cancellation and full cleanup."""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from ..errors import (
    DeliveryState,
    FailureCategory,
    ProviderFailure,
)


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


class ProcessRunner:
    """Run one process and join every helper thread before returning."""

    def run(
        self,
        argv: Sequence[str],
        *,
        stdin: bytes,
        env: Mapping[str, str] | None,
        idle_timeout_seconds: float,
        before_stdin: Callable[[], None],
        cancel_check: Callable[[], None],
        on_stdout: Callable[[bytes], None] | None = None,
        on_stderr: Callable[[bytes], None] | None = None,
    ) -> ProcessResult:
        popen_kwargs: dict[str, object] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "env": None if env is None else dict(env),
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
                delivery=DeliveryState.NOT_DELIVERED,
            ) from exc

        chunks: queue.Queue[tuple[str, bytes | None]] = queue.Queue()
        stdout_parts: list[bytes] = []
        stderr_parts: list[bytes] = []
        errors: list[BaseException] = []

        def read_stream(name: str, stream: object) -> None:
            try:
                while True:
                    data = stream.read(65536)  # type: ignore[attr-defined]
                    if not data:
                        break
                    chunks.put((name, data))
            except BaseException as exc:
                errors.append(exc)
            finally:
                chunks.put((name, None))

        def write_stdin() -> None:
            try:
                before_stdin()
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
        failure: BaseException | None = None
        try:
            while len(closed) < 2 or process.poll() is None:
                if errors:
                    failure = errors[0]
                    break
                try:
                    cancel_check()
                except BaseException as exc:
                    failure = exc
                    break
                remaining = idle_timeout_seconds - (time.monotonic() - last_activity)
                if remaining <= 0:
                    failure = ProviderFailure(
                        "Provider produced no activity before the idle timeout.",
                        category=FailureCategory.TIMEOUT,
                        delivery=DeliveryState.MAY_HAVE_RUN,
                    )
                    break
                try:
                    name, data = chunks.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    continue
                if data is None:
                    closed.add(name)
                    continue
                last_activity = time.monotonic()
                if name == "stdout":
                    stdout_parts.append(data)
                    if on_stdout is not None:
                        on_stdout(data)
                else:
                    stderr_parts.append(data)
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
        if failure is not None:
            raise failure
        if errors:
            raise ProviderFailure(
                f"Provider pipe failed: {errors[0]}",
                category=FailureCategory.TRANSPORT,
                delivery=DeliveryState.MAY_HAVE_RUN,
            )
        return ProcessResult(returncode, b"".join(stdout_parts), b"".join(stderr_parts))

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
        except ProcessLookupError:
            return
