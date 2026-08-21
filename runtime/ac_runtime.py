"""Private-runtime bootstrap shared by AC Foundation product launchers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

LAUNCHER_VERSION = 1
LOCK_SCHEMA = "ac.runtime_sources.v2"
COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
IDENTIFIER_RE = re.compile(r"[a-z][a-z0-9-]*")


class RuntimeConfigError(ValueError):
    """Raised when the checked-in source lock is unusable."""


@dataclass(frozen=True)
class Source:
    source_id: str
    repository: str
    commit: str
    packages: tuple[str, ...]
    tools: tuple[str, ...]
    local_root_env: str


@dataclass(frozen=True)
class RuntimeLock:
    profile: str
    sources: tuple[Source, ...]
    environment_defaults: dict[str, str]

    @property
    def tools(self) -> tuple[str, ...]:
        return tuple(tool for source in self.sources for tool in source.tools)


def _die(message: str, code: int = 78) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeConfigError(f"{label} must be an object")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeConfigError(f"{label} must be a non-empty string")
    return value


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeConfigError(f"{label} must be a non-empty array")
    result = tuple(_string(item, label) for item in value)
    if len(set(result)) != len(result):
        raise RuntimeConfigError(f"{label} must not contain duplicates")
    return result


def load_lock(path: Path) -> RuntimeLock:
    try:
        document = _object(json.loads(path.read_text(encoding="utf-8")), "source lock")
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeConfigError(f"cannot read source lock {path}: {exc}") from exc
    if document.get("schema_version") != LOCK_SCHEMA:
        raise RuntimeConfigError(f"source lock schema must be {LOCK_SCHEMA}")
    if set(document) != {
        "schema_version",
        "profile",
        "sources",
        "environment_defaults",
    }:
        raise RuntimeConfigError("source lock contains missing or unknown fields")
    profile = _string(document["profile"], "profile")
    if not IDENTIFIER_RE.fullmatch(profile):
        raise RuntimeConfigError("profile must be a lowercase identifier")
    raw_sources = document["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise RuntimeConfigError("sources must be a non-empty array")
    sources: list[Source] = []
    seen_packages: set[str] = set()
    seen_tools: set[str] = set()
    for index, raw in enumerate(raw_sources):
        item = _object(raw, f"sources[{index}]")
        expected = {
            "id",
            "repository",
            "commit",
            "packages",
            "tools",
            "local_root_env",
        }
        if set(item) != expected:
            raise RuntimeConfigError(f"sources[{index}] contains missing or unknown fields")
        source_id = _string(item["id"], f"sources[{index}].id")
        repository = _string(item["repository"], f"sources[{index}].repository")
        commit = _string(item["commit"], f"sources[{index}].commit").lower()
        packages = _string_list(item["packages"], f"sources[{index}].packages")
        tools = _string_list(item["tools"], f"sources[{index}].tools")
        local_root_env = _string(
            item["local_root_env"], f"sources[{index}].local_root_env"
        )
        if not IDENTIFIER_RE.fullmatch(source_id):
            raise RuntimeConfigError(f"invalid source id: {source_id}")
        parsed_repository = urlsplit(repository)
        if (
            parsed_repository.scheme != "https"
            or not parsed_repository.hostname
            or parsed_repository.username is not None
            or parsed_repository.password is not None
            or parsed_repository.query
            or parsed_repository.fragment
            or not parsed_repository.path.endswith(".git")
        ):
            raise RuntimeConfigError(f"source repository must be an HTTPS Git URL: {repository}")
        if not COMMIT_RE.fullmatch(commit):
            raise RuntimeConfigError(f"source commit must be a full Git SHA: {commit}")
        if not local_root_env.startswith("AC_"):
            raise RuntimeConfigError("local_root_env must be AC-owned")
        if seen_packages.intersection(packages) or seen_tools.intersection(tools):
            raise RuntimeConfigError("packages and tools must have one owning source")
        seen_packages.update(packages)
        seen_tools.update(tools)
        sources.append(
            Source(
                source_id,
                repository,
                commit,
                packages,
                tools,
                local_root_env,
            )
        )
    raw_defaults = _object(document["environment_defaults"], "environment_defaults")
    defaults: dict[str, str] = {}
    for key, value in raw_defaults.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise RuntimeConfigError(f"invalid environment key: {key!r}")
        defaults[key] = _string(value, f"environment_defaults.{key}")
    return RuntimeLock(profile, tuple(sources), defaults)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _content_hash(root: Path, packages: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for package in packages:
        package_root = root / "packages" / package
        if not (package_root / "pyproject.toml").is_file():
            raise RuntimeConfigError(f"local source lacks package {package}: {root}")
        for path in sorted(package_root.rglob("*")):
            if not path.is_file() or any(
                part in {".venv", "__pycache__", "build", "dist"}
                or part.endswith(".egg-info")
                for part in path.parts
            ):
                continue
            relative = path.relative_to(root).as_posix().encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            data = path.read_bytes()
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
    return digest.hexdigest()


def _local_roots(lock: RuntimeLock) -> dict[str, Path] | None:
    roots: dict[str, Path] = {}
    for source in lock.sources:
        value = os.environ.get(source.local_root_env)
        if not value:
            return None
        root = Path(value).expanduser().resolve()
        for package in source.packages:
            if not (root / "packages" / package / "pyproject.toml").is_file():
                return None
        roots[source.source_id] = root
    return roots


def _source_selection(lock: RuntimeLock) -> tuple[str, dict[str, Path] | None]:
    requested = os.environ.get("AC_INSTALL_SOURCE", "auto")
    if requested not in {"auto", "local", "git"}:
        raise RuntimeConfigError("AC_INSTALL_SOURCE must be auto, local, or git")
    roots = _local_roots(lock)
    if requested == "local" and roots is None:
        missing = ", ".join(source.local_root_env for source in lock.sources)
        raise RuntimeConfigError(f"local install requires complete roots: {missing}")
    if requested == "local" or (requested == "auto" and roots is not None):
        return "local", roots
    return "git", None


def _expand_default(value: str, *, cwd: Path, ac_home: Path) -> str:
    return value.replace("{cwd}", str(cwd)).replace("{ac_home}", str(ac_home))


def _runtime_environment(lock: RuntimeLock, roots: dict[str, Path] | None) -> dict[str, str]:
    cwd = Path.cwd().resolve()
    ac_home = Path(os.environ.get("AC_HOME", Path.home() / ".ac")).expanduser().resolve()
    os.environ["AC_HOME"] = str(ac_home)
    runtime_home = Path(
        os.environ.get("AC_RUNTIME_HOME", ac_home / "runtimes")
    ).expanduser().resolve()
    os.environ["AC_RUNTIME_HOME"] = str(runtime_home)
    if "AC_DOCUMENT_CACHE" not in os.environ:
        cache = cwd / ".ac" / "cache" / "ac-document"
        if roots is not None:
            for root in roots.values():
                if cwd == root or root in cwd.parents:
                    cache = root / "local" / "cache" / "ac-document"
                    break
        os.environ["AC_DOCUMENT_CACHE"] = str(cache)
    environment = {
        "AC_HOME": str(ac_home),
        "AC_RUNTIME_HOME": str(runtime_home),
        "AC_DOCUMENT_CACHE": os.environ["AC_DOCUMENT_CACHE"],
    }
    for key, value in lock.environment_defaults.items():
        os.environ.setdefault(key, _expand_default(value, cwd=cwd, ac_home=ac_home))
        environment[key] = os.environ[key]
    return environment


def _fingerprint(
    lock_path: Path,
    lock: RuntimeLock,
    mode: str,
    roots: dict[str, Path] | None,
    constraints_path: Path,
) -> tuple[str, dict[str, Any]]:
    identity: dict[str, Any] = {
        "launcher_version": LAUNCHER_VERSION,
        "lock_sha256": _sha256(lock_path.read_bytes()),
        "mode": mode,
        "python": [str(Path(sys.executable).resolve()), list(sys.version_info[:3])],
        "constraints_sha256": (
            _sha256(constraints_path.read_bytes()) if constraints_path.is_file() else None
        ),
        "sources": [],
    }
    for source in lock.sources:
        source_identity: dict[str, Any] = {
            "id": source.source_id,
            "repository": source.repository,
            "commit": source.commit,
            "packages": source.packages,
        }
        if roots is not None:
            root = roots[source.source_id]
            source_identity.update(
                root=str(root),
                revision=_git_revision(root),
                content_sha256=_content_hash(root, source.packages),
            )
        identity["sources"].append(source_identity)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return _sha256(encoded), identity


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "no-vcs-revision"


def _atomic_json(path: Path, document: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


class InstallLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.owner = f"{socket.gethostname()}:{os.getpid()}:{time.time_ns()}"

    def __enter__(self) -> "InstallLock":
        timeout = int(os.environ.get("AC_INSTALL_LOCK_TIMEOUT_SEC", "600"))
        stale_after = int(os.environ.get("AC_INSTALL_LOCK_STALE_SEC", "1800"))
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.path.mkdir(mode=0o700)
                _atomic_json(
                    self.path / "owner.json",
                    {
                        "owner": self.owner,
                        "host": socket.gethostname(),
                        "pid": os.getpid(),
                        "created": time.time(),
                    },
                )
                return self
            except FileExistsError:
                if self._stale(stale_after):
                    stale = self.path.with_name(f"{self.path.name}.stale.{os.getpid()}")
                    try:
                        os.replace(self.path, stale)
                    except OSError:
                        continue
                    shutil.rmtree(stale)
                    continue
                if time.monotonic() >= deadline:
                    _die(f"timed out waiting for runtime install lock: {self.path}", 75)
                time.sleep(0.2)

    def _stale(self, stale_after: int) -> bool:
        try:
            owner = json.loads((self.path / "owner.json").read_text(encoding="utf-8"))
            age = time.time() - float(owner["created"])
            if owner["host"] != socket.gethostname():
                return age >= stale_after
            pid = int(owner["pid"])
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                return age >= stale_after
            return False
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            try:
                return time.time() - self.path.stat().st_mtime >= 5
            except OSError:
                return False

    def __exit__(self, *_: object) -> None:
        try:
            owner = json.loads((self.path / "owner.json").read_text(encoding="utf-8"))
            if owner.get("owner") == self.owner:
                shutil.rmtree(self.path)
        except (OSError, json.JSONDecodeError):
            return


def _requirements(lock: RuntimeLock, mode: str, roots: dict[str, Path] | None) -> list[str]:
    requirements: list[str] = []
    for source in lock.sources:
        if mode == "local":
            assert roots is not None
            requirements.extend(
                str(roots[source.source_id] / "packages" / package)
                for package in source.packages
            )
        else:
            base = f"git+{source.repository}@{source.commit}"
            requirements.extend(
                f"{package} @ {base}#subdirectory=packages/{package}"
                for package in source.packages
            )
    return requirements


def _run_logged(command: list[str], log_path: Path) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    output = completed.stdout + completed.stderr
    output = re.sub(r"([a-z][a-z0-9+.-]*://)[^/@\s]+@", r"\1[REDACTED]@", output)
    log_path.write_text(output, encoding="utf-8")
    os.chmod(log_path, 0o600)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit status {completed.returncode}")


def _install(
    runtime_dir: Path,
    lock: RuntimeLock,
    mode: str,
    roots: dict[str, Path] | None,
    constraints_path: Path,
    fingerprint: str,
    identity: dict[str, Any],
) -> None:
    venv = runtime_dir / "venv"
    if venv.exists():
        shutil.rmtree(venv)
    requirements = _requirements(lock, mode, roots)
    uv_override = os.environ.get("AC_INSTALL_UV")
    uv = uv_override or shutil.which("uv")
    log_path = runtime_dir / "install.log"
    try:
        if uv:
            _run_logged([uv, "venv", str(venv), "--python", ">=3.11"], log_path)
            command = [uv, "pip", "install", "--python", str(venv / "bin/python")]
        else:
            _run_logged([sys.executable, "-m", "venv", str(venv)], log_path)
            command = [str(venv / "bin/python"), "-m", "pip", "install"]
        if constraints_path.is_file():
            command.extend(["--constraint", str(constraints_path)])
        command.extend(requirements)
        _run_logged(command, log_path)
        for tool in lock.tools:
            if not (venv / "bin" / tool).is_file():
                raise RuntimeError(f"installed runtime lacks command: {tool}")
        _atomic_json(
            runtime_dir / "install.ok",
            {
                "schema_version": "ac.runtime_install.v1",
                "fingerprint": fingerprint,
                "identity": identity,
                "installed_at": time.time(),
            },
        )
        failure = runtime_dir / "install.failed"
        if failure.exists():
            failure.unlink()
    except Exception as exc:
        if venv.exists():
            shutil.rmtree(venv)
        _atomic_json(
            runtime_dir / "install.failed",
            {
                "schema_version": "ac.runtime_failure.v1",
                "error": f"{type(exc).__name__}: {exc}",
                "log": str(log_path),
            },
        )
        raise


def _ready(runtime_dir: Path, fingerprint: str, tools: tuple[str, ...]) -> bool:
    try:
        marker = json.loads((runtime_dir / "install.ok").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return marker.get("fingerprint") == fingerprint and all(
        (runtime_dir / "venv" / "bin" / tool).is_file() for tool in tools
    )


def _ensure_runtime(
    runtime_dir: Path,
    lock: RuntimeLock,
    mode: str,
    roots: dict[str, Path] | None,
    constraints_path: Path,
    fingerprint: str,
    identity: dict[str, Any],
    *,
    retry: bool,
) -> None:
    if _ready(runtime_dir, fingerprint, lock.tools):
        return
    runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    failure = runtime_dir / "install.failed"
    if failure.exists() and not retry:
        _die(
            f"previous runtime install failed: {failure}; rerun setup --retry after fixing cause",
            1,
        )
    with InstallLock(runtime_dir / "install.lock"):
        if _ready(runtime_dir, fingerprint, lock.tools):
            return
        if failure.exists() and not retry:
            _die(f"previous runtime install failed: {failure}", 1)
        try:
            _install(
                runtime_dir,
                lock,
                mode,
                roots,
                constraints_path,
                fingerprint,
                identity,
            )
        except Exception as exc:
            _die(f"runtime install failed: {exc}; see {runtime_dir / 'install.log'}", 1)


def _parser(launcher: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=launcher)
    subparsers = parser.add_subparsers(dest="operation")
    setup = subparsers.add_parser("setup", help="install or verify private runtime")
    setup.add_argument("--retry", action="store_true")
    subparsers.add_parser("doctor", help="print runtime identity and readiness")
    run = subparsers.add_parser("run", help="run one locked command")
    run.add_argument("tool")
    run.add_argument("args", nargs=argparse.REMAINDER)
    script = subparsers.add_parser(
        "script", help="run one Python script inside the private runtime"
    )
    script.add_argument("path")
    script.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def _python_script_command(
    runtime_dir: Path, raw_path: str, args: list[str]
) -> tuple[Path, list[str]]:
    script = Path(raw_path).expanduser().resolve()
    if not script.is_file():
        raise RuntimeConfigError(f"Python script does not exist: {script}")
    python = runtime_dir / "venv" / "bin" / "python"
    return python, [str(python), str(script), *args]


def main(argv: list[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    lock_path = Path(
        os.environ.get("AC_RUNTIME_SOURCES_FILE", script_dir / "runtime-sources.json")
    ).expanduser().resolve()
    constraints_path = Path(
        os.environ.get("AC_RUNTIME_CONSTRAINTS_FILE", script_dir / "runtime-constraints.txt")
    ).expanduser().resolve()
    try:
        lock = load_lock(lock_path)
        mode, roots = _source_selection(lock)
        environment = _runtime_environment(lock, roots)
        fingerprint, identity = _fingerprint(
            lock_path, lock, mode, roots, constraints_path
        )
    except RuntimeConfigError as exc:
        _die(str(exc))
    runtime_dir = (
        Path(environment["AC_RUNTIME_HOME"])
        / f"v{LAUNCHER_VERSION}"
        / lock.profile
        / fingerprint
    )
    launcher = os.environ.get("AC_RUNTIME_LAUNCHER_NAME", "ac-runtime")
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in lock.tools:
        arguments = ["run", *arguments]
    parser = _parser(launcher)
    namespace = parser.parse_args(arguments)
    retry = bool(getattr(namespace, "retry", False)) or os.environ.get(
        "AC_INSTALL_RETRY", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if namespace.operation == "doctor":
        document = {
            "schema_version": "ac.runtime_doctor.v1",
            "profile": lock.profile,
            "source_mode": mode,
            "fingerprint": fingerprint,
            "runtime": str(runtime_dir),
            "ready": _ready(runtime_dir, fingerprint, lock.tools),
            "environment": environment,
            "sources": identity["sources"],
        }
        print(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if document["ready"] else 1
    if namespace.operation is None:
        parser.print_usage(sys.stderr)
        return 64
    _ensure_runtime(
        runtime_dir,
        lock,
        mode,
        roots,
        constraints_path,
        fingerprint,
        identity,
        retry=retry,
    )
    if namespace.operation == "setup":
        print(f"{lock.profile} runtime ready: {runtime_dir}")
        return 0
    if namespace.operation == "script":
        try:
            executable, command = _python_script_command(
                runtime_dir, namespace.path, namespace.args
            )
        except RuntimeConfigError as exc:
            _die(str(exc), 64)
        os.execv(executable, command)
        raise AssertionError("unreachable")
    if namespace.tool not in lock.tools:
        _die(f"command is not present in source lock: {namespace.tool}", 64)
    os.execv(
        runtime_dir / "venv" / "bin" / namespace.tool,
        [namespace.tool, *namespace.args],
    )
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
