from __future__ import annotations

import importlib
import subprocess
from collections.abc import Mapping

import pytest

from ac_document import (
    AcDocumentService,
    MathSpanKind,
    ParseError,
    PDFTextLayer,
    DocumentInputError,
    DocumentParserService,
    PdftotextExtractor,
    ParsedDocument,
    ReconciliationStatus,
    RichDocumentParserService,
    SourceBundle,
    SourceFormat,
    SourceOrigin,
    SourceOriginKind,
    SourceRepository,
    ValidationPolicy,
    build_visual_page_review_inputs,
    parsed_document_from_document,
    parsed_document_to_document,
)


class FakePDFTextExtractor:
    contract_id = "ac.document.tests.fake_pdf_text.v1"

    def __init__(self, values: dict[bytes, PDFTextLayer]):
        self.values = values
        self.calls: list[bytes] = []

    def extract(self, payload: bytes) -> PDFTextLayer:
        self.calls.append(payload)
        return self.values[payload]


def _store(
    repository: SourceRepository,
    payload: bytes,
    source_format: SourceFormat,
    *,
    locator: str = "",
):
    return repository.store_bytes(
        payload,
        source_format=source_format,
        origin=SourceOrigin(SourceOriginKind.LOCAL_IMPORT, locator=locator),
    )


@pytest.mark.parametrize(
    ("source_format", "payload", "expected_tex"),
    [
        (
            SourceFormat.HTML,
            b"<article><h1>Intro</h1><p>Before</p>"
            b"<math alttext='x+y' display='block'></math><p>After</p></article>",
            "x+y",
        ),
        (
            SourceFormat.MARKDOWN,
            b"# Intro\nBefore\n\n$$\nx+y\n$$\n\nAfter\n",
            "x+y",
        ),
        (
            SourceFormat.TEX,
            br"\section{Intro}" b"\nBefore\n" br"\[x+y\]" b"\nAfter\n",
            "x+y",
        ),
        (SourceFormat.PDF, b"%PDF fixture", "x + y"),
    ],
)
def test_public_parser_service_reads_all_formats_from_repository(
    tmp_path, source_format, payload, expected_tex
):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, source_format)
    extractor = FakePDFTextExtractor(
        {
            b"%PDF fixture": PDFTextLayer(
                ("Intro\nBefore\nx + y (1.1)\nAfter",)
            )
        }
    )

    outcome = DocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=artifact))

    assert isinstance(outcome.document, ParsedDocument)
    assert not isinstance(outcome.document, Mapping)
    assert not hasattr(outcome.document, "equations")
    assert outcome.document.source.content_identity == artifact.content_identity
    assert outcome.document.sections[0].title == "Intro"
    assert outcome.document.math_spans[0].normalized_tex == expected_tex
    assert any(
        item.kind is MathSpanKind.DISPLAY
        for item in outcome.document.math_spans
    )
    assert extractor.calls == ([payload] if source_format is SourceFormat.PDF else [])


def test_standard_html_uses_ordered_top_level_articles_and_real_tag_positions(
    tmp_path,
):
    payload = b"\n".join(
        (
            b"<nav><h1>Navigation</h1><math alttext='outside'></math></nav>",
            b"<article id='first'>",
            b"<h1 data-kind='x' id='h1'>First</h1>",
            b"<math class='formula' alttext='x' display='block'></math>",
            b"<article><h2>Nested</h2><math alttext='y'></math></article>",
            b"</article>",
            b"<article id='second'>",
            b"<h1>Second</h1>",
            b"<math display='block' alttext='z'></math>",
            b"</article>",
        )
    )
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.HTML)

    document = DocumentParserService(repository).parse_source(artifact)

    assert [section.title for section in document.sections] == [
        "First",
        "Nested",
        "Second",
    ]
    assert [span.normalized_tex for span in document.math_spans] == ["x", "y", "z"]
    assert [
        (
            span.source_line_start,
            span.source_column_start,
            span.source_line_end,
            span.source_column_end,
        )
        for span in document.math_spans
    ] == [(4, 1, 4, 1), (5, 25, 5, 25), (9, 1, 9, 1)]


def test_standard_html_does_not_invent_unavailable_source_positions(
    tmp_path, monkeypatch
):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(
        repository,
        b"<article><math alttext='x'></math></article>",
        SourceFormat.HTML,
    )
    monkeypatch.setattr(
        importlib.import_module("ac_document.parse.parser"),
        "html_source_position",
        lambda node: (None, None, None, None),
    )

    span = DocumentParserService(repository).parse_source(artifact).math_spans[0]

    assert (
        span.source_line_start,
        span.source_column_start,
        span.source_line_end,
        span.source_column_end,
    ) == (None, None, None, None)


def test_standard_markdown_supports_setext_and_commonmark_atx_closing(
    tmp_path,
):
    payload = b"Setext\n======\n\n## Kept#\n\n### Trimmed ###\n"
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.MARKDOWN)

    document = DocumentParserService(repository).parse_source(artifact)

    assert [(item.level, item.title) for item in document.sections] == [
        (1, "Setext"),
        (2, "Kept#"),
        (3, "Trimmed"),
    ]


def test_standard_markdown_ignores_display_delimiters_in_inline_code(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(
        repository,
        b"# Code\nThe literal `$$` is not display math.\n",
        SourceFormat.MARKDOWN,
    )

    document = DocumentParserService(repository).parse_source(artifact)

    assert document.math_spans == ()


def test_standard_markdown_excludes_front_matter_from_all_body_projections(
    tmp_path,
):
    payload = (
        b"---\n"
        b"keywords: [alpha, beta]\n"
        b"hidden_heading: '# Hidden'\n"
        b"unclosed_math: '$$'\n"
        b"- yaml-list-value\n"
        b"---\n"
        b"Scientific body.\n"
    )
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.MARKDOWN)

    document = DocumentParserService(repository).parse_source(artifact)

    assert document.math_spans == ()
    assert [(item.title, item.text) for item in document.sections] == [
        ("Document", "Scientific body.")
    ]
    assert document.metadata["explicit_term_fields"][0]["entries"] == [
        "alpha",
        "beta",
    ]


def test_markdown_explicit_terms_accept_yaml_document_end(tmp_path):
    payload = b"---\nkeywords: [alpha, beta]\n...\nScientific body.\n"
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.MARKDOWN)

    standard = DocumentParserService(repository).parse_source(artifact)
    rich = RichDocumentParserService(repository).parse_source(artifact)

    assert standard.metadata["explicit_term_fields"][0]["entries"] == [
        "alpha",
        "beta",
    ]
    assert rich.metadata["explicit_term_fields"][0]["entries"] == (
        "alpha",
        "beta",
    )


def test_html_explicit_terms_ignore_dash_only_metadata_placeholder(tmp_path):
    payload = """
        <html><head><meta name="keywords" content=" — "></head>
        <body><article>
          <h1>Paper</h1>
          <div class="ltx_keywords">
            <h6 class="ltx_title ltx_title_keywords">Keywords:</h6>
            <a href="https://example.test/infrared">Infrared astronomy</a> —
            <a href="https://example.test/symbiotic">Symbiotic stars</a>
          </div>
        </article></body></html>
    """.encode()
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.HTML)

    standard = DocumentParserService(repository).parse_source(artifact)
    rich = RichDocumentParserService(repository).parse_source(artifact)

    assert "explicit_term_fields" not in standard.metadata
    assert "explicit_term_fields" not in rich.metadata
    assert any(
        section.title == "Keywords:"
        and "Infrared astronomy" in section.text
        and "Symbiotic stars" in section.text
        for section in standard.sections
    )


def test_standard_tex_balanced_headings_respect_body_comments_and_literals(
    tmp_path,
):
    payload = "\n".join(
        (
            r"\section{Preamble}",
            r"\begin{document}",
            r"\section*",
            r"[Short {Title}]",
            r"{Outer {Nested} \texorpdfstring{TeX}{PDF} \{brace\}}",
            r"\begin{verbatim}",
            r"\subsection{Literal}",
            r"\begin{equation} hidden \end{equation}",
            r"\end{verbatim}",
            r"% \subsection{Comment}",
            r"\begin{equation}x=y\end{equation}",
            r"\subsection{Visible}",
            r"\end{document}",
            r"\section{After}",
        )
    ).encode()
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.TEX)

    document = DocumentParserService(repository).parse_source(artifact)

    assert [(item.level, item.title) for item in document.sections] == [
        (1, "Outer Nested TeX {brace}"),
        (2, "Visible"),
    ]
    assert [span.normalized_tex for span in document.math_spans] == ["x=y"]
    assert document.math_spans[0].source_line_start == 11
    assert document.math_spans[0].source_column_start is None


def test_tex_parsers_preserve_multiple_headings_on_one_source_line(tmp_path):
    payload = (
        br"\begin{document}"
        br"\section{First}\subsection{Second}"
        br"\end{document}"
    )
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.TEX)

    standard = DocumentParserService(repository).parse_source(artifact)
    rich = RichDocumentParserService(repository).parse_source(artifact)

    assert [(item.level, item.title) for item in standard.sections] == [
        (1, "First"),
        (2, "Second"),
    ]
    assert [(item.level, item.title) for item in rich.sections] == [
        (1, "First"),
        (2, "Second"),
    ]


@pytest.mark.parametrize(
    ("source_format", "payload"),
    (
        (SourceFormat.MARKDOWN, b"# Heading\n$$\nunclosed\n"),
        (SourceFormat.MARKDOWN, b"# Heading\n\\[\nunclosed\n"),
        (SourceFormat.TEX, br"\section{Unclosed" b"\n"),
        (SourceFormat.TEX, br"\begin{equation}unclosed" b"\n"),
    ),
)
def test_standard_parser_rejects_unclosed_rich_blocks(
    tmp_path, source_format, payload
):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, source_format)

    with pytest.raises(ParseError) as error:
        DocumentParserService(repository).parse_source(artifact)

    assert error.value.code == "unclosed_rich_block"


def test_markdown_aligned_row_spacing_is_not_a_display_delimiter(tmp_path):
    payload = (
        b"# Relations\n\n"
        b"\\[\n"
        b"\\begin{aligned}\n"
        b"a&=1,\\\\[4pt]\n"
        b"b&=2.\n"
        b"\\end{aligned}\n"
        b"\\]\n"
    )
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.MARKDOWN)

    standard = DocumentParserService(repository).parse_source(artifact)
    rich = RichDocumentParserService(repository).parse_source(artifact)

    assert [item.kind for item in standard.math_spans] == [
        MathSpanKind.DISPLAY
    ]
    assert (
        standard.math_spans[0].source_line_start,
        standard.math_spans[0].source_line_end,
    ) == (3, 8)
    assert r"\\[4pt]" in standard.math_spans[0].normalized_tex
    assert any(
        block.kind.value == "equation" for block in rich.blocks
    )


def test_tex_delimiter_activity_uses_backslash_run_parity(tmp_path):
    payload = (
        b"# Parity\n"
        + br"Row spacing \\[4pt] is text; \\\[x\\\] and \\\(y\\\) are math."
        + b"\n"
    )
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.MARKDOWN)

    document = DocumentParserService(repository).parse_source(artifact)

    assert [item.kind for item in document.math_spans] == [
        MathSpanKind.DISPLAY,
        MathSpanKind.INLINE,
    ]
    assert [item.normalized_tex for item in document.math_spans] == [
        r"x\\",
        r"y\\",
    ]


def test_bracket_like_ocr_inside_dollar_math_is_not_an_outer_delimiter(
    tmp_path,
):
    payload = (
        b"# OCR\n"
        b"Inline $S\\[\\phi\\]$ text.\n\n"
        b"$$T\\[\\psi\\]$$\n"
    )
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.MARKDOWN)

    standard = DocumentParserService(repository).parse_source(artifact)
    rich = RichDocumentParserService(repository).parse_source(artifact)

    assert [item.kind for item in standard.math_spans] == [
        MathSpanKind.INLINE,
        MathSpanKind.DISPLAY,
    ]
    assert any(
        block.kind.value == "equation" for block in rich.blocks
    )


def test_standard_markdown_projection_has_canonical_encoded_output(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(
        repository,
        b"# Golden\nBefore $x+y$.\n\n$$\nz = 1\n$$\n",
        SourceFormat.MARKDOWN,
    )

    document = DocumentParserService(repository).parse_source(artifact)

    assert parsed_document_to_document(document) == {
        "schema_version": "ac.document.parsed_document.v2",
        "document_digest": (
                "1c35fb4b0dbe7ad548299e48e0a170466b1712e7f63e497c238c2d14f1097c60"
        ),
        "source": {
            "source_format": "markdown",
            "artifact_digest": (
                "394fca4064f41a2583b7d57646649dea65e04a358a13b888abb2306dada0c484"
            ),
            "size": 36,
            "media_type": "text/markdown",
        },
        "sections": [
            {
                "section_id": "sec-0a1663a039ea8af7f22d",
                "title": "Golden",
                "level": 1,
                "text": "# Golden\nBefore $x+y$.\n\n$$\nz = 1\n$$",
                "ordinal": 0,
                "page_start": None,
                "page_end": None,
            }
        ],
        "math_spans": [
            {
                "span_id": "math-0fdd2cfeec4932e0842059a7",
                "kind": "inline",
                "source_line_start": 2,
                "source_column_start": None,
                "source_line_end": 2,
                "source_column_end": None,
                "normalized_tex": "x+y",
                "context_before": "Before",
                "context_after": ".",
                "source_label": "",
            },
            {
                "span_id": "math-acdec12f8a953f76d2b05527",
                "kind": "display",
                "source_line_start": 4,
                "source_column_start": None,
                "source_line_end": 6,
                "source_column_end": None,
                "normalized_tex": "z = 1",
                "context_before": "Before $x+y$.",
                "context_after": "",
                "source_label": "",
            },
        ],
        "pages": [],
        "warnings": [],
        "metadata": {"format": "markdown"},
    }


def test_html_math_context_does_not_cross_explicit_section_boundaries(tmp_path):
    payload = b"""
    <html><body>
      <section id="S1"><h2>Previous</h2><p>Previous section text.</p></section>
      <section id="S2"><h2>Current</h2>
        <table class="ltx_equation" id="E1">
          <tr><td><math alttext="x = y"></math></td></tr>
        </table>
      </section>
      <section id="S3"><h2>Next</h2><p>Next section text.</p></section>
    </body></html>
    """
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.HTML)

    document = DocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    assert len(document.math_spans) == 1
    assert document.math_spans[0].context_before == ""
    assert document.math_spans[0].context_after == ""


def test_html_math_context_is_read_from_the_containing_section(tmp_path):
    payload = b"""
    <html><body>
      <section id="S1"><h2>Previous</h2><p>Other before.</p></section>
      <section id="S2"><h2>Model</h2>
        <p>Model text before equation.</p>
        <table class="ltx_equation" id="E1">
          <tr><td><math alttext="E = mc^2"></math></td></tr>
        </table>
        <p>Model text after equation.</p>
      </section>
      <section id="S3"><h2>Next</h2><p>Other after.</p></section>
    </body></html>
    """
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.HTML)

    document = DocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document
    span = document.math_spans[0]

    assert span.normalized_tex == "E = mc^2"
    assert span.context_before == "Model text before equation."
    assert span.context_after == "Model text after equation."


def test_html_nested_math_in_equation_container_is_not_duplicated(tmp_path):
    payload = b"""
    <html><body>
      <section id="S1"><h2>Model</h2>
        <table class="ltx_equation" id="E1">
          <tr><td><math alttext="x = y"><mi>x</mi><mo>=</mo><mi>y</mi></math></td></tr>
        </table>
      </section>
    </body></html>
    """
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.HTML)

    document = DocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    assert len(document.math_spans) == 1
    assert document.math_spans[0].source_label == ""
    assert document.math_spans[0].normalized_tex == "x = y"


def test_html_equation_table_groups_fragments_into_logical_display_spans(tmp_path):
    payload = b"""
    <html><body><section id="S1"><h2>Model</h2>
      <table class="ltx_equation">
        <tr><td><math alttext="x"></math><math alttext="= y"></math></td><td><span class="ltx_tag">(4)</span></td></tr>
        <tr><td><math alttext="+ z"></math></td></tr>
      </table>
      <table class="ltx_equation">
        <tr><td><math alttext="a"></math><math alttext="= b"></math></td><td><span class="ltx_tag">(5)</span></td></tr>
        <tr><td><math alttext="c"></math><math alttext="= d"></math></td><td><span class="ltx_tag">(6)</span></td></tr>
      </table>
    </section></body></html>
    """
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.HTML)

    document = DocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    spans = [span for span in document.math_spans if span.kind is MathSpanKind.DISPLAY]
    assert [span.normalized_tex for span in spans] == ["x = y + z", "a = b", "c = d"]
    assert [span.source_label for span in spans] == ["4", "5", "6"]


def test_html_ltx_math_wrapper_does_not_turn_inline_math_into_display_math(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(
        repository,
        b"<article><h1>Model</h1><p><span class='ltx_Math'><math alttext='x = y'></math></span></p></article>",
        SourceFormat.HTML,
    )

    document = DocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    assert document.math_spans[0].kind is MathSpanKind.INLINE


def test_tex_comment_environment_excludes_sections_and_math_but_keeps_lines(
    tmp_path,
):
    payload = "\n".join(
        [
            r"\section{Active}",
            "Visible text.",
            r"\begin{comment}",
            r"\section{Hidden}",
            r"\begin{equation}",
            r"x = y",
            r"\end{equation}",
            r"\end{comment}",
            "Still visible.",
            r"\begin{equation}",
            r"z = w",
            r"\end{equation}",
        ]
    ).encode()
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.TEX)

    document = DocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    assert [section.title for section in document.sections] == ["Active"]
    assert len(document.math_spans) == 1
    span = document.math_spans[0]
    assert span.normalized_tex == "z = w"
    assert span.context_before == "Still visible."
    assert span.source_line_start == 10


def test_tex_percent_comments_exclude_sections_and_math_but_keep_lines(
    tmp_path,
):
    payload = "\n".join(
        [
            r"\section{Active}",
            "Visible text.",
            r"% \section{Hidden}",
            r"% \begin{equation}",
            r"% x = y",
            r"% \end{equation}",
            r"\begin{equation}",
            r"z = w",
            r"\end{equation}",
        ]
    ).encode()
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.TEX)

    document = DocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    assert [section.title for section in document.sections] == ["Active"]
    assert len(document.math_spans) == 1
    span = document.math_spans[0]
    assert span.normalized_tex == "z = w"
    assert span.source_line_start == 7


def test_markdown_math_manifest_covers_inline_and_display_with_stable_positions(tmp_path):
    payload = (
        b"# Dynamics\n"
        b"The invariant $E = mc^2$ controls the system.\n"
        b"\n"
        b"Before display.\n"
        b"\\[\n"
        b"a^2 + b^2 = c^2\n"
        b"\\]\n"
        b"After display.\n"
        b"`$not_math$`\n"
    )
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.MARKDOWN)
    service = DocumentParserService(repository)

    first = service.parse(SourceBundle(primary=artifact)).document
    second = service.parse(SourceBundle(primary=artifact)).document

    assert [span.kind for span in first.math_spans] == [
        MathSpanKind.INLINE,
        MathSpanKind.DISPLAY,
    ]
    inline, display = first.math_spans
    assert (inline.source_line_start, inline.source_column_start) == (2, None)
    assert inline.normalized_tex == "E = mc^2"
    assert display.source_line_start == 5
    assert display.source_line_end == 7
    assert display.source_column_start is None
    assert display.source_column_end is None
    assert display.context_before == "Before display."
    assert display.context_after == "After display."
    assert [span.span_id for span in first.math_spans] == [
        span.span_id for span in second.math_spans
    ]
    assert [
        item.span_id
        for item in first.math_spans
        if item.kind is MathSpanKind.DISPLAY
    ] == [display.span_id]


@pytest.mark.parametrize(
    ("source_format", "payload"),
    (
        (
            SourceFormat.MARKDOWN,
            b"# Repeated math\nThe same value $x$ appears again as $x$.\n",
        ),
        (
            SourceFormat.TEX,
            br"\section{Repeated math}" b"\n"
            br"The same value $x$ appears again as $x$." b"\n",
        ),
    ),
)
def test_repeated_identical_math_on_one_line_has_unique_stable_ids(
    tmp_path: Path,
    source_format: SourceFormat,
    payload: bytes,
) -> None:
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, source_format)
    service = DocumentParserService(repository)

    first = service.parse_source(artifact)
    second = service.parse_source(artifact)

    assert [span.normalized_tex for span in first.math_spans] == ["x", "x"]
    assert all(span.source_column_start is None for span in first.math_spans)
    assert len({span.span_id for span in first.math_spans}) == 2
    assert [span.span_id for span in first.math_spans] == [
        span.span_id for span in second.math_spans
    ]


def test_markdown_indented_code_blocks_are_excluded_from_math_manifest(tmp_path):
    payload = (
        b"# Example\n"
        b"\n"
        b"    $not_inline_math$\n"
        b"    $$not_display_math$$\n"
        b"\t\\(also_not_math\\)\n"
        b"\n"
        b"- A list item\n"
        b"\n"
        b"    whose continuation contains $list_math$.\n"
        b"\n"
        b"$$\n"
        b"\n"
        b"    display_math\n"
        b"$$\n"
        b"\n"
        b"The real expression is $x+y$.\n"
    )
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.MARKDOWN)

    document = DocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    assert [span.normalized_tex for span in document.math_spans] == [
        "list_math",
        "display_math",
        "x+y",
    ]


def test_markdown_block_quotes_apply_indented_code_rules_per_container(tmp_path):
    payload = (
        b">     $quoted_code$\n"
        b">\n"
        b"> >     $$nested_quoted_code$$\n"
        b"> >\n"
        b"> > - A nested list item\n"
        b"> >\n"
        b"> >     whose continuation contains $nested_list_math$.\n"
        b"\n"
        b"> Quoted prose contains $quoted_math$.\n"
    )
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.MARKDOWN)

    document = DocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    assert [span.normalized_tex for span in document.math_spans] == [
        "nested_list_math",
        "quoted_math",
    ]


def test_markdown_block_quote_fences_use_container_relative_content(tmp_path):
    payload = (
        b"> ```text\n"
        b"> $quoted_fenced_code$\n"
        b"> ```\n"
        b">\n"
        b"> > ~~~\n"
        b"> > $$nested_quoted_fenced_code$$\n"
        b"> > ~~~\n"
        b"\n"
        b"Outside the fences is $real_math$.\n"
    )
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.MARKDOWN)

    document = DocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    assert [span.normalized_tex for span in document.math_spans] == ["real_math"]


def test_outer_quote_fence_contains_nested_quote_content(tmp_path):
    payload = (
        b"> ```text\n"
        b"> > Nested quote code contains $not_math$.\n"
        b"> ```\n"
        b"\n"
        b"Outside is $real_math$.\n"
    )
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.MARKDOWN)

    document = DocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    assert [span.normalized_tex for span in document.math_spans] == ["real_math"]


def test_unclosed_quote_fence_does_not_leak_into_later_quote(tmp_path):
    payload = (
        b"> ```text\n"
        b"> $not_math$\n"
        b"\n"
        b"> A new quote contains $real_math$.\n"
    )
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, payload, SourceFormat.MARKDOWN)

    document = DocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    assert [span.normalized_tex for span in document.math_spans] == ["real_math"]


def test_validators_are_independent_and_conflicts_never_overwrite_primary(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"# Dynamics\nThe equation is $x+y$.\n",
        SourceFormat.MARKDOWN,
    )
    agreeing = _store(
        repository,
        b"<article><h1>Dynamics</h1><p>The equation is "
        b"<math alttext='x+y'></math>.</p></article>",
        SourceFormat.HTML,
    )
    conflicting = _store(
        repository,
        br"\section{Dynamics}" b"\nThe equation is $x-y$.\n",
        SourceFormat.TEX,
    )

    outcome = DocumentParserService(repository).parse(
        SourceBundle(primary=primary, validators=(conflicting, agreeing))
    )

    assert outcome.document.math_spans[0].normalized_tex == "x+y"
    by_validator = {
        artifact.artifact_digest: [
            entry
            for entry in outcome.report.entries
            if entry.validator.artifact_digest == artifact.artifact_digest
            and entry.subject_id != "structure"
        ]
        for artifact in (agreeing, conflicting)
    }
    assert any(
        entry.status is ReconciliationStatus.VERIFIED
        for entry in by_validator[agreeing.artifact_digest]
    )
    assert any(
        entry.status is ReconciliationStatus.MISMATCH
        for entry in by_validator[conflicting.artifact_digest]
    )
    assert all(
        entry.provenance.get("observed_tex") != outcome.document.math_spans[0].normalized_tex
        for entry in by_validator[conflicting.artifact_digest]
        if entry.status is ReconciliationStatus.MISMATCH
    )


def test_scanned_pdf_validator_is_successful_partial_and_visual_hook_is_pagewise(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository, b"# Notes\nInline $x+y$.\n", SourceFormat.MARKDOWN
    )
    pdf = _store(repository, b"%PDF scanned", SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {
            b"%PDF scanned": PDFTextLayer(
                ("", ""), "PDF contains no extractable text layer; partial parse retained"
            )
        }
    )
    service = DocumentParserService(repository, pdf_text_extractor=extractor)

    outcome = service.parse(
        SourceBundle(primary=primary, validators=(pdf,)),
        policy=ValidationPolicy.VISUAL_ALL_PAGES,
    )
    parsed_pdf = service.parse_source(pdf)  # noqa: SLF001 - visual handoff fixture
    requests = build_visual_page_review_inputs(outcome.document, parsed_pdf)

    assert outcome.document.math_spans[0].normalized_tex == "x+y"
    assert any("no extractable text layer" in item for item in outcome.warnings)
    assert outcome.report.entries[0].status is ReconciliationStatus.UNREVIEWED
    assert [request.page_number for request in requests] == [1, 2]
    assert all(
        request.math_span_ids == (outcome.document.math_spans[0].span_id,)
        for request in requests
    )


def test_scanned_pdf_primary_returns_partial_document_instead_of_failing(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    pdf = _store(repository, b"%PDF scanned primary", SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {
            b"%PDF scanned primary": PDFTextLayer(
                ("",), "PDF contains no extractable text layer; partial parse retained"
            )
        }
    )

    outcome = DocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=pdf))

    assert outcome.document.source_format is SourceFormat.PDF
    assert len(outcome.document.pages) == 1
    assert outcome.document.math_spans == ()
    assert outcome.document.metadata["text_layer"] is False
    assert "partial parse retained" in outcome.warnings[0]


def test_pdf_repeated_math_at_same_page_position_has_unique_stable_ids(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    payload = b"%PDF repeated math position"
    pdf = _store(repository, payload, SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {payload: PDFTextLayer(("x = 1", "x = 1"))}
    )
    service = DocumentParserService(repository, pdf_text_extractor=extractor)

    first = service.parse_source(pdf)
    second = service.parse_source(pdf)

    assert [span.normalized_tex for span in first.math_spans] == ["x = 1", "x = 1"]
    assert [span.source_line_start for span in first.math_spans] == [1, 1]
    assert len({span.span_id for span in first.math_spans}) == 2
    assert [span.span_id for span in first.math_spans] == [
        span.span_id for span in second.math_spans
    ]


@pytest.mark.parametrize(
    ("source_format", "source_payload"),
    [
        (
            SourceFormat.MARKDOWN,
            b"# Dynamics\n\n$$x+y \\tag {2.1}$$\n",
        ),
        (
            SourceFormat.TEX,
            b"\\section{Dynamics}\n\\[x+y \\tag {2.1}\\]\n",
        ),
    ],
)
def test_pdf_validator_records_deterministic_page_and_printed_number_evidence(
    tmp_path,
    source_format,
    source_payload,
):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        source_payload,
        source_format,
    )
    pdf = _store(repository, b"%PDF deterministic", SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {
            b"%PDF deterministic": PDFTextLayer(
                ("Front matter", "Dynamics\nx + y (2.1)")
            )
        }
    )

    outcome = DocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(
        SourceBundle(primary=primary, validators=(pdf,)),
        policy=ValidationPolicy.DETERMINISTIC_ONLY,
    )
    span = outcome.document.math_spans[0]
    entry = next(item for item in outcome.report.entries if item.subject_id == span.span_id)

    assert span.normalized_tex == "x+y"
    assert span.source_label == "2.1"
    assert entry.status is ReconciliationStatus.VERIFIED
    assert entry.provenance["page_candidates"] == [2]
    assert entry.provenance["printed_equation_number"] == "2.1"


def test_pdf_section_title_prefers_exact_body_heading_over_toc_reference(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"# Introduction\nText.\n",
        SourceFormat.MARKDOWN,
    )
    pdf = _store(repository, b"%PDF toc and body", SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {
            b"%PDF toc and body": PDFTextLayer(
                (
                    "Contents\n1. Introduction ........ 3",
                    "1. Introduction\nText.",
                )
            )
        }
    )

    outcome = DocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=primary, validators=(pdf,)))
    section_id = outcome.document.sections[0].section_id
    entry = next(
        item
        for item in outcome.report.entries
        if item.subject_id == f"section:{section_id}"
    )

    assert entry.status is ReconciliationStatus.VERIFIED
    assert entry.provenance["page_candidates"] == [2]
    assert entry.provenance["matching_method"] == "normalized_exact_line"


def test_pdf_section_title_exact_duplicate_remains_ambiguous(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"# Introduction\nText.\n",
        SourceFormat.MARKDOWN,
    )
    pdf = _store(repository, b"%PDF repeated headings", SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {
            b"%PDF repeated headings": PDFTextLayer(
                ("Introduction\nFirst body.", "Introduction\nSecond body.")
            )
        }
    )

    outcome = DocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=primary, validators=(pdf,)))
    section_id = outcome.document.sections[0].section_id
    entry = next(
        item
        for item in outcome.report.entries
        if item.subject_id == f"section:{section_id}"
    )

    assert entry.status is ReconciliationStatus.AMBIGUOUS
    assert entry.provenance["page_candidates"] == [1, 2]
    assert entry.provenance["matching_method"] == "normalized_exact_line"


@pytest.mark.parametrize(
    ("source_title", "pdf_heading"),
    (
        ("Model", "1 Model"),
        ("Model", "12 Model"),
        ("Model", "1.2 Model"),
        ("Introduction", "1. Introduction"),
        ("Model", "I Model"),
        ("Model", "II Model"),
        ("Model", "II. Model"),
    ),
)
def test_pdf_section_title_accepts_conventional_pdf_only_section_prefix(
    tmp_path, source_title, pdf_heading
):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        f"# {source_title}\nText.\n".encode(),
        SourceFormat.MARKDOWN,
    )
    pdf = _store(repository, b"%PDF numbered heading", SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {b"%PDF numbered heading": PDFTextLayer((f"{pdf_heading}\nText.",))}
    )

    outcome = DocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=primary, validators=(pdf,)))
    section_id = outcome.document.sections[0].section_id
    entry = next(
        item
        for item in outcome.report.entries
        if item.subject_id == f"section:{section_id}"
    )

    assert entry.status is ReconciliationStatus.VERIFIED
    assert entry.provenance["page_candidates"] == [1]
    assert entry.provenance["matching_method"] == "normalized_exact_line"


def test_pdf_section_title_preserves_genuine_numeric_source_title(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"# 2024 Results\nText.\n",
        SourceFormat.MARKDOWN,
    )
    pdf = _store(repository, b"%PDF numeric title", SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {
            b"%PDF numeric title": PDFTextLayer(
                ("2024 Results\nText.", "Results\nDifferent text.")
            )
        }
    )

    outcome = DocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=primary, validators=(pdf,)))
    section_id = outcome.document.sections[0].section_id
    entry = next(
        item
        for item in outcome.report.entries
        if item.subject_id == f"section:{section_id}"
    )

    assert entry.status is ReconciliationStatus.VERIFIED
    assert entry.provenance["page_candidates"] == [1]
    assert entry.provenance["matching_method"] == "normalized_exact_line"


@pytest.mark.parametrize("pdf_line", ("2024 Results", "2024 Results .... 7"))
def test_pdf_section_title_rejects_ambiguous_numeric_title_prefix(
    tmp_path, pdf_line
):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"# Results\nText.\n",
        SourceFormat.MARKDOWN,
    )
    pdf = _store(repository, b"%PDF ambiguous numeric prefix", SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {b"%PDF ambiguous numeric prefix": PDFTextLayer((f"{pdf_line}\nText.",))}
    )

    outcome = DocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=primary, validators=(pdf,)))
    section_id = outcome.document.sections[0].section_id
    entry = next(
        item
        for item in outcome.report.entries
        if item.subject_id == f"section:{section_id}"
    )

    assert entry.status is ReconciliationStatus.MISSING
    assert entry.provenance["page_candidates"] == []
    assert entry.provenance["matching_method"] == "none"


def test_pdf_section_title_rejects_prefixed_toc_line(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(repository, b"# Model\nText.\n", SourceFormat.MARKDOWN)
    pdf = _store(repository, b"%PDF prefixed toc", SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {b"%PDF prefixed toc": PDFTextLayer(("II Model .... 5",))}
    )

    outcome = DocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=primary, validators=(pdf,)))
    section_id = outcome.document.sections[0].section_id
    entry = next(
        item
        for item in outcome.report.entries
        if item.subject_id == f"section:{section_id}"
    )

    assert entry.status is ReconciliationStatus.MISSING
    assert entry.provenance["page_candidates"] == []
    assert entry.provenance["matching_method"] == "none"


def test_pdf_section_title_uses_independent_prose_beside_ambiguous_line(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(repository, b"# Results\nText.\n", SourceFormat.MARKDOWN)
    pdf = _store(repository, b"%PDF ambiguous and prose", SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {
            b"%PDF ambiguous and prose": PDFTextLayer(
                ("2024 Results .... 7\nThe results are discussed in this paragraph.",)
            )
        }
    )

    outcome = DocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=primary, validators=(pdf,)))
    section_id = outcome.document.sections[0].section_id
    entry = next(
        item
        for item in outcome.report.entries
        if item.subject_id == f"section:{section_id}"
    )

    assert entry.status is ReconciliationStatus.VERIFIED
    assert entry.provenance["page_candidates"] == [1]
    assert entry.provenance["matching_method"] == "normalized_page_substring"


def test_pdf_section_title_falls_back_to_page_substring_or_reports_missing(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"# Introduction\nText.\n\n# Results\nMore text.\n",
        SourceFormat.MARKDOWN,
    )
    pdf = _store(repository, b"%PDF fallback and missing", SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {
            b"%PDF fallback and missing": PDFTextLayer(
                ("The introduction begins within this extracted line.",)
            )
        }
    )

    outcome = DocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=primary, validators=(pdf,)))
    entries = {
        item.subject_id: item
        for item in outcome.report.entries
        if item.subject_id.startswith("section:")
    }
    introduction, results = outcome.document.sections

    fallback = entries[f"section:{introduction.section_id}"]
    assert fallback.status is ReconciliationStatus.VERIFIED
    assert fallback.provenance["page_candidates"] == [1]
    assert fallback.provenance["matching_method"] == "normalized_page_substring"

    missing = entries[f"section:{results.section_id}"]
    assert missing.status is ReconciliationStatus.MISSING
    assert missing.provenance["page_candidates"] == []
    assert missing.provenance["matching_method"] == "none"


def test_non_pdf_primary_fails_before_text_extraction(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    payload = b"this is not a PDF"
    artifact = _store(repository, payload, SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {payload: PDFTextLayer(("",), "no extractable text layer")}
    )

    with pytest.raises(Exception) as error:
        DocumentParserService(
            repository, pdf_text_extractor=extractor
        ).parse(SourceBundle(primary=artifact))

    assert getattr(error.value, "code", "") == "pdf_invalid"
    assert extractor.calls == []


def test_pdftotext_rejection_is_a_parse_failure_not_a_missing_text_layer(
    tmp_path, monkeypatch
):
    repository = SourceRepository(tmp_path / "cache")
    payload = b"%PDF-1.7\nmalformed body"
    artifact = _store(repository, payload, SourceFormat.PDF)

    def reject(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="invalid PDF")

    monkeypatch.setattr(subprocess, "run", reject)

    with pytest.raises(Exception) as error:
        DocumentParserService(
            repository,
            pdf_text_extractor=PdftotextExtractor(),
        ).parse(SourceBundle(primary=artifact))

    assert getattr(error.value, "code", "") == "pdf_invalid"


def test_equal_count_rich_validator_uses_sequence_for_every_conflict(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"# Equations\n$x$\n$y$\n$z$\n",
        SourceFormat.MARKDOWN,
    )
    validator = _store(
        repository,
        b"# Equations\n$a$\n$b$\n$c$\n",
        SourceFormat.MARKDOWN,
    )

    outcome = DocumentParserService(repository).parse(
        SourceBundle(primary=primary, validators=(validator,))
    )
    math_entries = [
        entry
        for entry in outcome.report.entries
        if entry.subject_id != "structure"
    ]

    assert [entry.status for entry in math_entries] == [
        ReconciliationStatus.MISMATCH,
        ReconciliationStatus.MISMATCH,
        ReconciliationStatus.MISMATCH,
    ]
    assert [
        entry.provenance.get("matching_method") for entry in math_entries
    ] == ["sequence", "sequence", "sequence"]
    assert not any(entry.subject_id.startswith("validator:") for entry in math_entries)


def test_injected_repository_rejects_a_different_explicit_cache_root(tmp_path):
    repository = SourceRepository(tmp_path / "sources")

    with pytest.raises(DocumentInputError, match="must match"):
        AcDocumentService(
            repository=repository,
            cache_root=tmp_path / "request-cache",
        )


def test_same_bytes_different_paths_have_same_document_and_span_identity(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    first_path = tmp_path / "one.md"
    second_path = tmp_path / "nested" / "two.md"
    second_path.parent.mkdir()
    payload = b"# Identity\nInline $x+y$.\n"
    first_path.write_bytes(payload)
    second_path.write_bytes(payload)
    first_artifact = repository.import_path(first_path)
    second_artifact = repository.import_path(second_path)
    service = DocumentParserService(repository)

    first = service.parse(SourceBundle(primary=first_artifact)).document
    second = service.parse(SourceBundle(primary=second_artifact)).document

    assert first_artifact.origin.locator != second_artifact.origin.locator
    assert first_artifact.content_identity == second_artifact.content_identity
    assert first.document_digest == second.document_digest
    assert first.math_spans[0].span_id == second.math_spans[0].span_id


def test_parsed_document_codec_round_trips_and_rejects_unknown_fields(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(
        repository, b"# Codec\nInline $x+y$.\n", SourceFormat.MARKDOWN
    )
    parsed = DocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    encoded = parsed_document_to_document(parsed)
    decoded = parsed_document_from_document(encoded)

    assert encoded["schema_version"] == "ac.document.parsed_document.v2"
    assert encoded["math_spans"][0]["source_column_start"] is None
    assert decoded.math_spans[0].source_column_start is None
    assert decoded.document_digest == parsed.document_digest
    assert decoded.source.content_identity == parsed.source.content_identity
    invalid = {**encoded, "unknown": True}
    with pytest.raises(ValueError, match="invalid fields"):
        parsed_document_from_document(invalid)
    corrupt = {
        **encoded,
        "math_spans": [
            {**encoded["math_spans"][0], "normalized_tex": "changed"}
        ],
    }
    with pytest.raises(ValueError, match="digest"):
        parsed_document_from_document(corrupt)
    with pytest.raises(ValueError, match="unsupported parsed document schema"):
        parsed_document_from_document(
            {**encoded, "schema_version": "ac.document.parsed_document.v1"}
        )


def test_single_file_tex_rejects_input_include(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(
        repository,
        br"\section{Main}" b"\n" br"\input{chapter}",
        SourceFormat.TEX,
    )

    with pytest.raises(Exception) as error:
        DocumentParserService(repository).parse(SourceBundle(primary=artifact))

    assert getattr(error.value, "code", "") == "unsupported_tex_project"
