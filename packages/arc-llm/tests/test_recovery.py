from __future__ import annotations

import pytest

from arc_jobs import (
    ArtifactDigest,
    ArtifactRef,
    CorruptStateError,
    EffectStage,
    ExecutionFingerprint,
    ResumeReason,
    SemanticKeyDigest,
)

from arc_llm.executor import LLMTaskExecutor
from arc_llm.recovery import (
    AcceptedSessionTurn,
    AcceptedOrigin,
    AcceptedRecord,
    GenerationRecord,
    LLMTaskState,
    LLMSessionState,
    RecoveryAction,
    TaskPause,
    TaskStateContract,
    SessionStateContract,
    decide_recovery,
    effect_id_for,
    replace_current,
)


def _ref(name: str = "x") -> ArtifactRef:
    return ArtifactRef(name, ArtifactDigest("sha256", "a" * 64, 1), "application/json", name)


def _state() -> LLMTaskState:
    fingerprint = ExecutionFingerprint("arc.llm.execution_recipe.v1", "b" * 64)
    return LLMTaskState(
        0,
        "task",
        SemanticKeyDigest("c" * 64),
        "codex",
        "model",
        1,
        (GenerationRecord(1, effect_id_for("task", 1), fingerprint),),
        _ref("request"),
    )


@pytest.mark.parametrize(
    "digest",
    (
        "A" * 64,
        "g" * 64,
        "a" * 63,
    ),
)
def test_task_state_contract_rejects_invalid_artifact_reference_digest(digest: str) -> None:
    contract = TaskStateContract()
    document = dict(contract.encode(_state()))
    request_ref = dict(document["request_ref"])
    request_ref["digest"] = {
        "algorithm": "sha256",
        "value": digest,
        "size_bytes": 1,
    }
    document["request_ref"] = request_ref

    with pytest.raises(CorruptStateError):
        contract.decode(document)


def test_recovery_decision_is_bounded_and_delivery_aware() -> None:
    state = _state()
    assert decide_recovery(
        state,
        EffectStage.PREPARED,
        execution=state.current.execution,
        supports_native_resume=True,
        safe_retry_limit=1,
        native_resume_limit=1,
        automatic_replacement_limit=1,
    ) is RecoveryAction.START
    with_handle = LLMTaskState(
        **{
            **state.__dict__,
            "generations": (
                GenerationRecord(
                    1,
                    state.current.effect_id,
                    state.current.execution,
                    native_handle="thread",
                ),
            ),
        }
    )
    assert decide_recovery(
        with_handle,
        EffectStage.MAY_HAVE_RUN,
        execution=state.current.execution,
        supports_native_resume=True,
        safe_retry_limit=1,
        native_resume_limit=1,
        automatic_replacement_limit=1,
    ) is RecoveryAction.NATIVE_RESUME
    replacement = replace_current(
        state,
        execution=state.current.execution,
        reason="uncertain",
        possible_duplicate=True,
    )
    assert replacement.current.replacement_of == 1
    assert replacement.current.possible_duplicate_execution


def test_replacement_generation_uses_base_effect_after_earlier_interaction() -> None:
    original = _state()
    state = replace_current(
        original,
        execution=original.current.execution,
        reason="uncertain",
        possible_duplicate=True,
    )
    state = LLMTaskState(
        **{
            **state.__dict__,
            "interaction_round": 1,
        }
    )

    assert state.current.effect_id == effect_id_for("task", 2)
    assert LLMTaskExecutor()._prepared_interaction_prompt(None, state) is None


@pytest.mark.parametrize(
    ("limit", "safe_retries", "expected"),
    (
        (0, 0, RecoveryAction.START),
        (1, 1, RecoveryAction.START),
        (1, 2, RecoveryAction.PAUSE_UNCERTAIN),
    ),
)
def test_prepared_recovery_includes_initial_call_and_reserved_safe_retry(
    limit: int,
    safe_retries: int,
    expected: RecoveryAction,
) -> None:
    state = _state()
    current = GenerationRecord(
        1,
        state.current.effect_id,
        state.current.execution,
        safe_retries=safe_retries,
    )
    state = LLMTaskState(
        **{
            **state.__dict__,
            "generations": (current,),
        }
    )

    assert (
        decide_recovery(
            state,
            EffectStage.PREPARED,
            execution=state.current.execution,
            supports_native_resume=True,
            safe_retry_limit=limit,
            native_resume_limit=1,
            automatic_replacement_limit=1,
        )
        is expected
    )


def test_interaction_effect_id_extends_the_initial_effect_id() -> None:
    initial = effect_id_for("task", 1)

    assert effect_id_for("task", 1, 0) == initial
    assert effect_id_for("task", 1, 2) == f"{initial}-i2"


@pytest.mark.parametrize(
    ("stage", "handle", "safe_retries", "native_resumes", "replacements", "expected"),
    [
        (EffectStage.OUTPUT_SAVED, None, 0, 0, 0, RecoveryAction.RECOVER_SAVED_OUTPUT),
        (EffectStage.COMMITTED, None, 0, 0, 0, RecoveryAction.RECOVER_SAVED_OUTPUT),
        (EffectStage.PREPARED, None, 0, 0, 0, RecoveryAction.START),
        (EffectStage.PREPARED, None, 2, 0, 0, RecoveryAction.PAUSE_UNCERTAIN),
        (EffectStage.MAY_HAVE_RUN, "thread", 0, 0, 0, RecoveryAction.NATIVE_RESUME),
        (EffectStage.MAY_HAVE_RUN, "thread", 0, 1, 0, RecoveryAction.PAUSE_UNCERTAIN),
        (EffectStage.MAY_HAVE_RUN, None, 0, 0, 0, RecoveryAction.PAUSE_UNCERTAIN),
        (EffectStage.MAY_HAVE_RUN, None, 0, 0, 1, RecoveryAction.PAUSE_UNCERTAIN),
    ],
)
def test_recovery_action_matrix(
    stage,
    handle,
    safe_retries,
    native_resumes,
    replacements,
    expected,
) -> None:
    state = _state()
    current = GenerationRecord(
        1,
        state.current.effect_id,
        state.current.execution,
        native_handle=handle,
        safe_retries=safe_retries,
        native_resumes=native_resumes,
    )
    generations = (current,)
    current_generation = 1
    if replacements:
        replacement = GenerationRecord(
            2,
            effect_id_for("task", 2),
            state.current.execution,
            replacement_of=1,
            replacement_reason="uncertain",
            possible_duplicate_execution=True,
        )
        generations += (replacement,)
        current_generation = 2
    state = LLMTaskState(
        **{
            **state.__dict__,
            "current_generation": current_generation,
            "generations": generations,
        }
    )
    assert (
        decide_recovery(
            state,
            stage,
            execution=state.current.execution,
            supports_native_resume=True,
            safe_retry_limit=1,
            native_resume_limit=1,
            automatic_replacement_limit=1,
        )
        is expected
    )


def test_task_state_contract_rejects_mutating_accepted_result() -> None:
    state = _state()
    accepted = LLMTaskState(
        **{
            **state.__dict__,
            "revision": 1,
            "accepted": AcceptedRecord(
                _ref("accepted"),
                AcceptedOrigin.PROVIDER,
                1,
                "codex",
                "model",
            ),
        }
    )
    TaskStateContract().validate_transition(state, accepted)
    changed = LLMTaskState(
        **{
            **accepted.__dict__,
            "revision": 2,
            "accepted": AcceptedRecord(
                _ref("other"),
                AcceptedOrigin.PROVIDER,
                1,
                "codex",
                "model",
            ),
        }
    )
    try:
        TaskStateContract().validate_transition(accepted, changed)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted result mutation was allowed")


def test_session_v2_history_is_append_only_unique_and_runtime_bound() -> None:
    fingerprint = ExecutionFingerprint("arc.llm.execution_recipe.v1", "b" * 64)
    empty_prefix = "0" * 64
    initial = LLMSessionState(
        0,
        "session",
        1,
        "codex",
        "model",
        fingerprint,
        None,
        0,
        empty_prefix,
        (),
    )
    contract = SessionStateContract()
    contract.validate_transition(None, initial)
    first_turn = AcceptedSessionTurn("1" * 64, "2" * 64, "3" * 64)
    first = LLMSessionState(
        1,
        "session",
        1,
        "codex",
        "model",
        fingerprint,
        "thread",
        1,
        first_turn.result_prefix_sha256,
        (first_turn,),
    )
    contract.validate_transition(initial, first)
    assert contract.decode(contract.encode(first)) == first

    second_turn = AcceptedSessionTurn("1" * 64, "4" * 64, "5" * 64)
    duplicate = LLMSessionState(
        2,
        "session",
        2,
        "codex",
        "model",
        fingerprint,
        "thread",
        2,
        second_turn.result_prefix_sha256,
        (first_turn, second_turn),
    )
    try:
        contract.validate_transition(first, duplicate)
    except Exception:
        pass
    else:
        raise AssertionError("one task semantic key was accepted twice")

    rebound = LLMSessionState(
        2,
        "session",
        2,
        "codex",
        "other-model",
        fingerprint,
        "thread",
        2,
        "6" * 64,
        (first_turn, AcceptedSessionTurn("7" * 64, "8" * 64, "6" * 64)),
    )
    try:
        contract.validate_transition(first, rebound)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted session runtime was rebound")
