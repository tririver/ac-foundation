from __future__ import annotations

from arc_jobs import (
    ArtifactDigest,
    ArtifactRef,
    EffectStage,
    ExecutionFingerprint,
    ResumeReason,
    SemanticKeyDigest,
)

from arc_llm.recovery import (
    AcceptedOrigin,
    AcceptedRecord,
    GenerationRecord,
    LLMTaskState,
    RecoveryAction,
    TaskPause,
    TaskStateContract,
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
