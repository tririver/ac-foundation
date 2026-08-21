from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ac_jobs import ResumeReason, RunContext, RunRepository, RunSpec, StoppedError
from ac_llm import (
    FailureCategory,
    LLMCompleted,
    LLMFailed,
    LLMPaused,
    LLMStopped,
    ProviderFailure,
    ResumeAction,
    ResumeInput,
)
from ac_llm.identity import semantic_key
from ac_document import (
    EquationLabelReviewService,
    RenderedPDFPage,
    RichBlock,
    RichBlockKind,
    RichDocument,
    SourceArtifact,
    SourceFormat,
    SourceLocator,
    SourceOrigin,
    SourceOriginKind,
    apply_visual_equation_labels,
    decode_equation_label_page_review,
    detect_suspicious_equation_labels,
)


def _png(width: int = 100, height: int = 200) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
    )


class FakeRenderer:
    def __init__(self, count: int) -> None:
        self.count = count
        self.calls = 0

    def render(self, pdf_bytes: bytes) -> tuple[RenderedPDFPage, ...]:
        assert pdf_bytes.startswith(b"%PDF")
        self.calls += 1
        return tuple(
            RenderedPDFPage(index, _png(100 + index), 100 + index, 200)
            for index in range(1, self.count + 1)
        )


@dataclass
class ScriptedLLM:
    outcomes: list[Any]

    def __post_init__(self) -> None:
        self.requests: list[tuple[Any, ResumeInput | None]] = []

    def execute_or_resume(self, context, request, *, input=None, options=None):
        del context, options
        self.requests.append((request, input))
        return self.outcomes.pop(0)


class PausingReplayLLM:
    def __init__(self) -> None:
        self.new_calls: list[int] = []
        self.replay_calls: list[int] = []
        self.resume_calls: list[int] = []
        self.pause_key = ""
        self._completed: set[str] = set()
        self._paused = False

    def execute_or_resume(self, context, request, *, input=None, options=None):
        del context, options
        page = int(re.search(r"page (\d+)", request.prompt).group(1))
        if request.task_id in self._completed:
            self.replay_calls.append(page)
            return _completed(_response(page, [(f"eq-{page - 1}", str(page))]))
        if page == 2 and not self._paused:
            self._paused = True
            self.pause_key = (
                f"resume-{semantic_key(request).sha256[:24]}-1"
            )
            self.new_calls.append(page)
            return LLMPaused(ResumeReason.EXTERNAL_CONDITION, self.pause_key)
        if page == 2:
            assert input is not None and input.resume_key == self.pause_key
            self.resume_calls.append(page)
        else:
            self.new_calls.append(page)
        self._completed.add(request.task_id)
        return _completed(_response(page, [(f"eq-{page - 1}", str(page))]))


def _context(tmp_path: Path) -> RunContext:
    repository = RunRepository(tmp_path / "jobs")
    snapshot = repository.create(
        RunSpec("equation-label-visual", "test.visual", {"case": "labels"})
    )
    return RunContext(repository, snapshot, resume_input=None)


def _document(
    labels: tuple[str, ...], *, source_format: SourceFormat = SourceFormat.HTML
) -> RichDocument:
    payload = b"<html>equation fixture</html>"
    source = SourceArtifact(
        source_format,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        "text/html" if source_format is SourceFormat.HTML else "text/markdown",
        SourceOrigin(SourceOriginKind.LOCAL_IMPORT),
    )
    blocks = tuple(
        RichBlock(
            block_id=f"eq-{index}",
            ordinal=index,
            kind=RichBlockKind.EQUATION,
            section_path=(),
            locator=SourceLocator(source_format, selector=f"eq-{index}"),
            payload={"tex": f"x_{index} = {index}", "display": True, "label": label},
        )
        for index, label in enumerate(labels)
    )
    return RichDocument(source=source, blocks=blocks)


def _response(page: int, mappings: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "schema_version": "ac.document.equation_label_page_review.v1",
        "page_number": page,
        "mappings": [
            {
                "block_id": block_id,
                "pdf_label": label,
                "observed_math": f"math for {block_id}",
                "notes": "",
            }
            for block_id, label in mappings
        ],
        "unmatched_numbered_equations": [],
        "ambiguities": [],
        "notes": "",
    }


def _completed(value: Any) -> LLMCompleted:
    return LLMCompleted(value, "fake", "fake-model", None, None)


def _review(service, context, document, *, resume_input=None):
    pdf = b"%PDF visual equation labels"
    return service.review(
        context,
        document,
        pdf_digest=hashlib.sha256(pdf).hexdigest(),
        pdf_bytes=pdf,
        resume_input=resume_input,
    )


def test_detect_suspicious_equation_labels_requires_uniform_simple_sequence() -> None:
    assert detect_suspicious_equation_labels(_document(("1", "3"))) == (
        "simple-integer equation labels have gaps after 1",
    )
    assert "duplicate simple-integer" in detect_suspicious_equation_labels(
        _document(("1", "1"))
    )[0]
    assert "regress" in detect_suspicious_equation_labels(_document(("2", "1")))[0]
    assert detect_suspicious_equation_labels(_document(("1", ""))) == ()
    assert detect_suspicious_equation_labels(_document(("1", "A.1"))) == ()
    assert (
        detect_suspicious_equation_labels(
            _document(("1", "3"), source_format=SourceFormat.MARKDOWN)
        )
        == ()
    )


def test_complete_visual_mapping_uses_format_repair_and_applies_overlay(
    tmp_path: Path,
) -> None:
    document = _document(("1", "3"))
    llm = ScriptedLLM(
        [_completed(_response(1, [("eq-0", "1")])), _completed(_response(2, [("eq-1", "2")]))]
    )

    outcome = _review(EquationLabelReviewService(FakeRenderer(2), llm=llm), _context(tmp_path), document)

    assert outcome.complete is outcome.applicable is True
    assert [item.pdf_label for item in outcome.mapping] == ["1", "2"]
    assert all(request.output.repair == "format" for request, _ in llm.requests)
    assert all(
        [item.input_id for item in request.inputs] == ["page", "equation-manifest"]
        for request, _ in llm.requests
    )
    effective = apply_visual_equation_labels(document, outcome)
    reconciliation = effective.metadata["equation_label_reconciliation"]
    assert reconciliation["eq-1"]["source_label"] == "3"
    assert reconciliation["eq-1"]["effective_label"] == "2"
    assert reconciliation["eq-1"]["matching_method"] == "visual_pdf_page"
    assert effective.document_digest != document.document_digest
    assert document.blocks[1].payload["label"] == "3"


@pytest.mark.parametrize(
    "value, error",
    [
        (_response(2, [("eq-0", "1")]), "page number"),
        (_response(1, [("unknown", "1")]), "unknown block"),
        (_response(1, [("eq-0", "1"), ("eq-0", "2")]), "duplicate block"),
        (
            {
                **_response(1, []),
                "unmatched_numbered_equations": [
                    {"pdf_label": "1", "observed_math": "x", "notes": ""}
                ],
            },
            "unmatched",
        ),
        (
            {
                **_response(1, []),
                "ambiguities": [
                    {
                        "candidate_block_ids": ["eq-0"],
                        "pdf_label": "1",
                        "notes": "",
                    }
                ],
            },
            "ambiguous",
        ),
    ],
)
def test_strict_page_decoder_rejects_non_bijective_evidence(value, error) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        decode_equation_label_page_review(
            value, expected_page=1, known_block_ids={"eq-0", "eq-1"}
        )


def test_cross_page_duplicate_and_missing_mapping_never_partially_applies(
    tmp_path: Path,
) -> None:
    document = _document(("1", "3"))
    llm = ScriptedLLM(
        [_completed(_response(1, [("eq-0", "1")])), _completed(_response(2, [("eq-0", "2")]))]
    )

    outcome = _review(EquationLabelReviewService(FakeRenderer(2), llm=llm), _context(tmp_path), document)

    assert not outcome.complete
    assert outcome.mapping == ()
    assert any("multiple pages" in warning for warning in outcome.warnings)
    assert any("not matched" in warning for warning in outcome.warnings)
    with pytest.raises(ValueError, match="incomplete"):
        apply_visual_equation_labels(document, outcome)


def test_terminal_failure_returns_warning_and_retains_no_partial_mapping(
    tmp_path: Path,
) -> None:
    document = _document(("1", "3"))
    llm = ScriptedLLM(
        [
            _completed(_response(1, [("eq-0", "1")])),
            LLMFailed(
                ProviderFailure(
                    "fake transport failure",
                    category=FailureCategory.TRANSPORT,
                )
            ),
        ]
    )

    outcome = _review(EquationLabelReviewService(FakeRenderer(2), llm=llm), _context(tmp_path), document)

    assert not outcome.complete
    assert outcome.mapping == ()
    assert any("transport" in warning for warning in outcome.warnings)
    assert outcome.diagnostics_document["status"] == "incomplete"


def test_pause_propagates_and_completed_page_replays_before_paused_child_resumes(
    tmp_path: Path,
) -> None:
    document = _document(("1", "3"))
    llm = PausingReplayLLM()
    service = EquationLabelReviewService(FakeRenderer(2), llm=llm)
    context = _context(tmp_path)

    paused = _review(service, context, document)

    assert isinstance(paused, LLMPaused)
    assert llm.new_calls == [1, 2]
    resumed = _review(
        service,
        context,
        document,
        resume_input=ResumeInput(paused.resume_key, ResumeAction.CONTINUE),
    )

    assert resumed.complete
    assert llm.new_calls == [1, 2]
    assert llm.replay_calls == [1]
    assert llm.resume_calls == [2]


def test_stopped_child_propagates_to_outer_run(tmp_path: Path) -> None:
    with pytest.raises(StoppedError, match="equation-label visual review LLM task stopped"):
        _review(
            EquationLabelReviewService(FakeRenderer(1), llm=ScriptedLLM([LLMStopped()])),
            _context(tmp_path),
            _document(("1", "3")),
        )
