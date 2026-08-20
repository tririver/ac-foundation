"""Short-lived ripgrep adapter for cached full-text candidate selection."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path


class RipgrepError(RuntimeError):
    """Typed ripgrep failure with a stable package error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class RipgrepCandidateSelector:
    """Select only caller-supplied files through bounded ``rg`` processes."""

    def __init__(
        self,
        *,
        executable: str = "rg",
        timeout_seconds: float = 60.0,
        max_paths_per_call: int = 128,
    ) -> None:
        normalized = str(executable).strip()
        if not normalized:
            raise ValueError("ripgrep executable is required")
        if timeout_seconds <= 0:
            raise ValueError("ripgrep timeout must be positive")
        if (
            not isinstance(max_paths_per_call, int)
            or isinstance(max_paths_per_call, bool)
            or max_paths_per_call < 1
        ):
            raise ValueError("ripgrep path batch size must be positive")
        self.executable = normalized
        self.timeout_seconds = timeout_seconds
        self.max_paths_per_call = max_paths_per_call

    def ensure_available(self) -> None:
        """Fail before catalog inspection when the configured rg is missing."""

        if shutil.which(self.executable) is None:
            raise RipgrepError(
                "rg_unavailable",
                "ripgrep is unavailable; install the rg executable to search cached full text",
            )

    def files_with_matches(
        self,
        patterns: Sequence[str],
        paths: Sequence[Path],
        *,
        case_sensitive: bool,
    ) -> tuple[Path, ...]:
        normalized_patterns = tuple(str(item) for item in patterns)
        normalized_paths = tuple(
            Path(item).resolve(strict=False) for item in paths
        )
        if not normalized_patterns:
            raise ValueError("at least one ripgrep pattern is required")
        if not normalized_paths:
            return ()
        if len(set(normalized_paths)) != len(normalized_paths):
            raise ValueError("ripgrep candidate paths must be unique")

        matched: set[Path] = set()
        for start in range(0, len(normalized_paths), self.max_paths_per_call):
            matched.update(
                self._files_with_matches_once(
                    normalized_patterns,
                    normalized_paths[start : start + self.max_paths_per_call],
                    case_sensitive=case_sensitive,
                )
            )
        return tuple(item for item in normalized_paths if item in matched)

    def _files_with_matches_once(
        self,
        patterns: tuple[str, ...],
        paths: tuple[Path, ...],
        *,
        case_sensitive: bool,
    ) -> tuple[Path, ...]:
        command = [
            self.executable,
            "--files-with-matches",
            "--null",
            "--no-config",
            "--no-messages",
            "--color",
            "never",
        ]
        if not case_sensitive:
            command.append("--ignore-case")
        for pattern in patterns:
            command.extend(("--regexp", pattern))
        command.append("--")
        command.extend(str(path) for path in paths)

        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RipgrepError(
                "rg_unavailable",
                "ripgrep is unavailable; install the rg executable to search cached full text",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RipgrepError(
                "rg_failed", "ripgrep cached full-text candidate filtering timed out"
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise RipgrepError(
                "rg_failed", "ripgrep cached full-text candidate filtering failed"
            ) from exc

        if completed.returncode not in {0, 1}:
            raise RipgrepError(
                "rg_failed", "ripgrep cached full-text candidate filtering failed"
            )
        stdout = completed.stdout
        if not isinstance(stdout, bytes):
            raise RipgrepError("rg_failed", "ripgrep returned invalid candidate output")
        if completed.returncode == 1:
            if stdout:
                raise RipgrepError("rg_failed", "ripgrep returned invalid candidate output")
            return ()
        if not stdout:
            raise RipgrepError("rg_failed", "ripgrep returned invalid candidate output")
        if not stdout.endswith(b"\0"):
            raise RipgrepError("rg_failed", "ripgrep returned invalid candidate output")

        try:
            values = tuple(
                Path(os.fsdecode(item)).resolve(strict=False)
                for item in stdout[:-1].split(b"\0")
                if item
            )
        except (TypeError, UnicodeError) as exc:
            raise RipgrepError(
                "rg_failed", "ripgrep returned invalid candidate output"
            ) from exc
        allowed = set(paths)
        if len(set(values)) != len(values) or any(item not in allowed for item in values):
            raise RipgrepError("rg_failed", "ripgrep returned invalid candidate output")
        return tuple(item for item in paths if item in set(values))


__all__ = [
    "RipgrepCandidateSelector",
    "RipgrepError",
]
