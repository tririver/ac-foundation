from __future__ import annotations

from ac_document import (
    KeywordTerm,
    ParsedDocument,
    ParsedSection,
    SourceArtifact,
    SourceFormat,
    SourceOrigin,
    SourceOriginKind,
    StoredTermInventory,
    TermCandidate,
    build_keyword_terms,
    result_from_inventory,
)


def _document() -> ParsedDocument:
    text = "The curvature perturbation controls the observable signal."
    return ParsedDocument(
        SourceArtifact(
            SourceFormat.MARKDOWN,
            "a" * 64,
            len(text.encode("utf-8")),
            "text/markdown",
            SourceOrigin(SourceOriginKind.LOCAL_IMPORT, locator="paper.md"),
        ),
        sections=(ParsedSection("section-1", "Introduction", 1, text, 0),),
    )


def test_keyword_terms_omit_model_guesses_without_a_literal_source_hit() -> None:
    terms = build_keyword_terms(
        _document(),
        (
            TermCandidate("curvature perturbation"),
            TermCandidate("field-space trajectory"),
        ),
    )

    assert [item.term for item in terms] == ["curvature perturbation"]
    assert terms[0].occurrence_count == 1
    assert terms[0].matched_sentences


def test_keyword_result_filters_ungrounded_terms_from_legacy_cache() -> None:
    grounded = build_keyword_terms(
        _document(), (TermCandidate("curvature perturbation"),)
    )[0]
    ungrounded = KeywordTerm(
        "term-ungrounded",
        "field-space trajectory",
        (),
        0,
        ("section:section-1",),
        (),
    )
    stored = StoredTermInventory(
        "b" * 64,
        _document().document_digest,
        _document().source.artifact_digest,
        1,
        75,
        "absent",
        (grounded, ungrounded),
        {grounded.term_id: 4, ungrounded.term_id: None},
        "c" * 64,
        "2026-09-03T00:00:00Z",
    )

    result = result_from_inventory(stored, approx_count=50, planned_count=75)

    assert result.returned_count == 1
    assert result.terms == (grounded,)
