from __future__ import annotations

import json
from dataclasses import dataclass, replace
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from arc_jobs import (
    AtomicStateStore,
    CommandError,
    CommandResult,
    CommandStatus,
    CorruptStateError,
    ProgressEvent,
    RunRepository,
    RunSpec,
    RevisionConflictError,
    command_result_json,
    decode_command_result,
    decode_progress_event,
    encode_command_result,
    encode_progress_event,
    validate_progress_data,
)
from arc_jobs.cli import main


@dataclass(frozen=True)
class State:
    revision: int
    payload: str


class StateContract:
    schema_version = "example.state.v1"

    def encode(self, value):
        return {"revision": value.revision, "payload": value.payload}

    def decode(self, value):
        if set(value) != {"revision", "payload"}:
            raise ValueError("unknown field")
        return State(value["revision"], value["payload"])

    def validate_transition(self, previous, next):
        return None


def test_state_has_no_generic_four_megabyte_limit(tmp_path):
    store = AtomicStateStore(tmp_path / "state.json", StateContract())
    value = State(0, "x" * (5 * 1024 * 1024))
    assert store.create(value) == value
    assert store.read() == value
    with pytest.raises(RevisionConflictError):
        store.compare_and_swap(1, replace(value, revision=2))


def test_compare_and_swap_has_one_winner_across_threads(tmp_path):
    store = AtomicStateStore(tmp_path / "state.json", StateContract())
    store.create(State(0, "initial"))
    barrier = Barrier(2)

    def update(payload):
        barrier.wait()
        try:
            store.compare_and_swap(0, State(1, payload))
            return "won"
        except RevisionConflictError:
            return "lost"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(update, ("a", "b")))

    assert sorted(outcomes) == ["lost", "won"]
    assert store.read().payload in {"a", "b"}


def test_command_codec_is_closed_and_enforces_invariants():
    result = CommandResult(
        CommandStatus.FAILED,
        error=CommandError("bad", "broken"),
    )
    document = encode_command_result(result)
    assert decode_command_result(document) == result
    assert json.loads(command_result_json(result)) == document

    with pytest.raises(CorruptStateError):
        decode_command_result({**document, "future_optional": None})


def test_progress_rejects_partial_output_and_runless_events():
    encoded = encode_progress_event(ProgressEvent("run-1", 1, "step"))
    assert set(encoded) == {
        "schema_version",
        "run_id",
        "sequence",
        "at",
        "event",
        "data",
    }
    assert decode_progress_event(encoded) == ProgressEvent(
        "run-1",
        1,
        "step",
        at=encoded["at"],
    )
    with pytest.raises(ValueError):
        encode_progress_event(ProgressEvent("", 1, "step"))
    with pytest.raises(ValueError):
        encode_progress_event(
            ProgressEvent("run-1", 1, "step", {"nested": {"delta": "secret"}})
        )


@pytest.mark.parametrize(
    "key",
    ("text", "token", "content", "output", "delta", "prompt", "candidate", "result"),
)
def test_progress_codec_and_public_validator_reject_all_body_keys(key):
    data = {"nested": [{key.swapcase(): "secret"}]}
    with pytest.raises(ValueError):
        validate_progress_data(data)
    with pytest.raises(ValueError):
        encode_progress_event(ProgressEvent("run-1", 1, "step", data))

    document = encode_progress_event(ProgressEvent("run-1", 1, "step"))
    document["data"] = data
    with pytest.raises(CorruptStateError):
        decode_progress_event(document)


def test_cli_status_and_usage_emit_one_shared_envelope(tmp_path, capsys):
    RunRepository(tmp_path).create(RunSpec("run-1", "example.v1", {}))

    exit_code = main(
        ["status", "--run-root", str(tmp_path), "--run-id", "run-1"]
    )
    lines = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert len(lines) == 1
    document = json.loads(lines[0])
    assert document["status"] == "completed"
    assert document["data"]["run"]["status"] == "pending"

    exit_code = main(["status"])
    lines = capsys.readouterr().out.splitlines()
    assert exit_code == 2
    assert len(lines) == 1
    assert json.loads(lines[0])["error"]["code"] == "invalid_request"
