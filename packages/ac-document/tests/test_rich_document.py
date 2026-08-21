from __future__ import annotations

import hashlib

import pytest

from ac_document import (
    PDF_VALIDATOR_MISSING_WARNING,
    PDFTextLayer,
    RICH_DOCUMENT_SCHEMA,
    RichBlock,
    RichBlockKind,
    RichDocumentParserService,
    RichDocumentValidationError,
    RichPageMapEntry,
    RichSection,
    ReconciliationStatus,
    SourceBundle,
    SourceFormat,
    SourceOrigin,
    SourceOriginKind,
    SourceRepository,
    SourceLocator,
    rich_block_from_document,
    rich_block_to_document,
    rich_document_from_document,
    rich_document_to_document,
)


class FakePDFTextExtractor:
    contract_id = "ac.document.tests.fake_pdf_text.v1"

    def __init__(self, values: dict[bytes, PDFTextLayer]):
        self.values = values
        self.calls: list[bytes] = []

    def extract(self, payload: bytes) -> PDFTextLayer:
        self.calls.append(payload)
        return self.values[payload]


def _store(repository, payload, source_format, *, locator=""):
    return repository.store_bytes(
        payload,
        source_format=source_format,
        origin=SourceOrigin(
            SourceOriginKind.LOCAL_IMPORT,
            locator=locator,
        ),
    )


def test_markdown_rich_parse_preserves_blocks_links_math_and_assets(tmp_path):
    source = tmp_path / "paper.md"
    image = tmp_path / "diagram.png"
    image.write_bytes(b"\x89PNG\r\nfixture")
    source.write_text(
        "\n".join(
            [
                "# Dynamics",
                "Read the [notes](https://example.test/notes) with $E=mc^2$.",
                "",
                "- first",
                "- [second](appendix.html)",
                "",
                "$$",
                r"x^2 + y^2 = z^2",
                "$$",
                "",
                "| Name | Value |",
                "| --- | --- |",
                "| mass | m |",
                "",
                "```python",
                "print('example')",
                "```",
                "",
                '![phase portrait](diagram.png "Figure 1")',
            ]
        ),
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")
    artifact = repository.import_path(source)

    outcome = RichDocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    )
    document = outcome.document

    assert document.schema_version == RICH_DOCUMENT_SCHEMA
    assert [block.kind for block in document.blocks] == [
        RichBlockKind.HEADING,
        RichBlockKind.PARAGRAPH,
        RichBlockKind.LIST,
        RichBlockKind.EQUATION,
        RichBlockKind.TABLE,
        RichBlockKind.CODE,
        RichBlockKind.FIGURE,
    ]
    paragraph = document.blocks[1]
    link_span = next(
        item for item in paragraph.payload["inline_spans"]
        if item["kind"] == "link"
    )
    math_span = next(
        item for item in paragraph.payload["inline_spans"]
        if item["kind"] == "math"
    )
    assert link_span["target"] == "https://example.test/notes"
    assert math_span["tex"] == "E=mc^2"
    assert [item["kind"] for item in paragraph.payload["inline_spans"]] == [
        "text",
        "link",
        "text",
        "math",
        "text",
    ]
    assert "".join(
        item["text"] for item in paragraph.payload["inline_spans"]
    ) == paragraph.payload["text"]
    assert document.blocks[3].payload["tex"] == "x^2 + y^2 = z^2"
    assert document.blocks[4].payload["rows"] == (("mass", "m"),)
    assert document.blocks[5].payload["language"] == "python"
    assert len(document.assets) == 1
    asset = document.assets[0]
    assert asset.artifact_digest == hashlib.sha256(image.read_bytes()).hexdigest()
    assert repository.read_asset_bytes(
        repository.get_asset(asset.artifact_digest)
    ) == image.read_bytes()
    assert document.blocks[-1].payload["asset_digest"] == asset.artifact_digest
    assert document.blocks[-1].payload["media_type"] == "image/png"
    assert document.blocks[-1].payload["size"] == len(image.read_bytes())
    assert outcome.warnings == (PDF_VALIDATOR_MISSING_WARNING,)
    assert all(block.section_path == document.sections[0].path for block in document.blocks)


@pytest.mark.parametrize(
    "sidecar_kind",
    ["natural_image", "text_image", "flowchart", "chemical", "line"],
)
def test_markdown_figure_consumes_known_extraction_sidecar(
    tmp_path,
    sidecar_kind,
):
    source = tmp_path / "extracted.md"
    source.write_text(
        "\n".join(
            [
                "# Figures",
                "",
                "![authored alt](missing.png)",
                "",
                "<details>",
                f"<summary>{sidecar_kind}</summary>",
                "",
                "Extractor-only description and raw metadata.",
                "</details>",
                "",
                "Figure 1. This authored caption remains source prose.",
            ]
        ),
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")

    document = RichDocumentParserService(repository).parse_source(
        repository.import_path(source)
    )

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.HEADING,
        RichBlockKind.FIGURE,
        RichBlockKind.PARAGRAPH,
    ]
    figure = document.blocks[1]
    assert figure.locator.line_start == 3
    assert figure.locator.line_end == 9
    assert figure.payload["alt_text"] == "authored alt"
    assert "Extractor-only" not in str(figure.payload)
    assert document.blocks[2].payload["text"] == (
        "Figure 1. This authored caption remains source prose."
    )


def test_markdown_keeps_author_details_and_fenced_extraction_tags(tmp_path):
    source = tmp_path / "authored-details.md"
    source.write_text(
        "\n".join(
            [
                "![plot](missing.png)",
                "",
                "<details>",
                "<summary>Author note</summary>",
                "",
                "Authored explanation.",
                "</details>",
                "",
                "```markdown",
                "![example](inside-code.png)",
                "<details><summary>natural_image</summary>",
                "Literal example.",
                "</details>",
                "```",
            ]
        ),
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")

    document = RichDocumentParserService(repository).parse_source(
        repository.import_path(source)
    )

    figure = document.blocks[0]
    assert figure.kind is RichBlockKind.FIGURE
    assert figure.locator.line_end == 1
    authored_text = "\n".join(
        str(block.payload.get("text", ""))
        for block in document.blocks
        if block.kind is RichBlockKind.PARAGRAPH
    )
    assert "<summary>Author note</summary>" in authored_text
    assert "Authored explanation." in authored_text
    code = next(
        block for block in document.blocks if block.kind is RichBlockKind.CODE
    )
    assert "<summary>natural_image</summary>" in code.payload["text"]
    assert "Literal example." in code.payload["text"]


def test_markdown_rich_parse_supports_setext_headings(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(
        repository,
        b"Primary\n=======\n\nSecondary\n---------\n",
        SourceFormat.MARKDOWN,
    )

    document = RichDocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    assert [
        (block.payload["level"], block.payload["text"])
        for block in document.blocks
        if block.kind is RichBlockKind.HEADING
    ] == [(1, "Primary"), (2, "Secondary")]


def test_markdown_rich_parse_excludes_front_matter_from_body_blocks(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(
        repository,
        (
            b"---\n"
            b"keywords: [alpha, beta]\n"
            b"hidden_heading: '# Hidden'\n"
            b"unclosed_math: '$$'\n"
            b"- yaml-list-value\n"
            b"---\n"
            b"Scientific body.\n"
        ),
        SourceFormat.MARKDOWN,
    )

    document = RichDocumentParserService(repository).parse_source(artifact)

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.PARAGRAPH
    ]
    assert document.blocks[0].payload["text"] == "Scientific body."
    assert document.metadata["explicit_term_fields"][0]["entries"] == (
        "alpha",
        "beta",
    )


def test_markdown_rich_parse_checks_fence_before_setext(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(
        repository,
        b"```text\n---\n```\nAfter.\n",
        SourceFormat.MARKDOWN,
    )

    document = RichDocumentParserService(repository).parse_source(artifact)

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.CODE,
        RichBlockKind.PARAGRAPH,
    ]
    assert document.blocks[0].payload["text"] == "---"
    assert document.blocks[1].payload["text"] == "After."


def test_rich_markdown_projection_has_canonical_encoded_output(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(
        repository,
        b"# Golden\nBefore $x+y$.\n\n$$\nz = 1\n$$\n",
        SourceFormat.MARKDOWN,
    )

    document = RichDocumentParserService(repository).parse_source(artifact)

    assert rich_document_to_document(document) == {
        "schema_version": "ac.document.rich_document.v2",
        "document_digest": (
                "aad05574886cdf6e004efa9157f133ceeae8aae55a3c9e80ecc6fee6793dbd08"
        ),
        "source": {
            "source_format": "markdown",
            "artifact_digest": (
                "394fca4064f41a2583b7d57646649dea65e04a358a13b888abb2306dada0c484"
            ),
            "size": 36,
            "media_type": "text/markdown",
        },
        "blocks": [
            {
                "block_id": "block-ac7da08a86910854845ed54a",
                "ordinal": 0,
                "kind": "heading",
                "section_path": ["sec-0c239875f2148884c1ab"],
                "locator": {
                    "source_format": "markdown",
                    "line_start": 1,
                    "column_start": None,
                    "line_end": 1,
                    "column_end": None,
                    "selector": "",
                    "source_id": "",
                },
                "payload": {"text": "Golden", "level": 1},
            },
            {
                "block_id": "block-cda30770f0706802d2f48c51",
                "ordinal": 1,
                "kind": "paragraph",
                "section_path": ["sec-0c239875f2148884c1ab"],
                "locator": {
                    "source_format": "markdown",
                    "line_start": 2,
                    "column_start": None,
                    "line_end": 2,
                    "column_end": None,
                    "selector": "",
                    "source_id": "",
                },
                "payload": {
                    "text": "Before $x+y$.",
                    "inline_spans": [
                        {
                            "kind": "text",
                            "start": 0,
                            "end": 7,
                            "text": "Before ",
                        },
                        {
                            "kind": "math",
                            "start": 7,
                            "end": 12,
                            "text": "$x+y$",
                            "tex": "x+y",
                            "source": "$x+y$",
                        },
                        {
                            "kind": "text",
                            "start": 12,
                            "end": 13,
                            "text": ".",
                        },
                    ],
                },
            },
            {
                "block_id": "block-6579552946b1897b5a233497",
                "ordinal": 2,
                "kind": "equation",
                "section_path": ["sec-0c239875f2148884c1ab"],
                "locator": {
                    "source_format": "markdown",
                    "line_start": 4,
                    "column_start": None,
                    "line_end": 6,
                    "column_end": None,
                    "selector": "",
                    "source_id": "",
                },
                "payload": {"tex": "z = 1", "display": True, "label": ""},
            },
        ],
        "sections": [
            {
                "section_id": "sec-0c239875f2148884c1ab",
                "title": "Golden",
                "level": 1,
                "ordinal": 0,
                "path": ["sec-0c239875f2148884c1ab"],
                "block_start": 0,
                "block_end": 3,
            }
        ],
        "assets": [],
        "page_map": [],
        "metadata": {"format": "markdown", "single_file": False},
    }


def test_rich_service_skips_unused_standard_primary_without_validator(
    tmp_path,
    monkeypatch,
):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(
        repository,
        b"# Service calls\nBody.\n",
        SourceFormat.MARKDOWN,
    )
    service = RichDocumentParserService(repository)
    calls = []

    def record_standard_call(source):
        calls.append(source.content_identity)
        raise AssertionError("standard projection is unused without a validator")

    monkeypatch.setattr(service.standard_parser, "parse_source", record_standard_call)

    outcome = service.parse(SourceBundle(primary=artifact))

    assert calls == []
    assert outcome.report.primary.content_identity == artifact.content_identity
    assert outcome.warnings == (PDF_VALIDATOR_MISSING_WARNING,)


def test_markdown_page_markers_are_metadata_not_prose(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(
        repository,
        b"""<!-- Generated by ARC. -->

<!-- Source PDF page 1 -->

# First

Body one.

<!-- Source PDF page 2 -->

<!-- PDF_PAGE: 3 -->

Body three.

<!-- hidden trailing metadata -->
""",
        SourceFormat.MARKDOWN,
    )

    document = RichDocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    assert [block.payload.get("text") for block in document.blocks] == [
        "First",
        "Body one.",
        "Body three.",
    ]
    assert {
        item.block_id: item.page_number for item in document.page_map
    } == {
        document.blocks[0].block_id: 1,
        document.blocks[1].block_id: 1,
        document.blocks[2].block_id: 3,
    }
    assert document.metadata["source_page_boundaries"] == {
        "schema_version": "ac.document.source_page_boundaries.v1",
        "items": (
            {
                "page_number": 1,
                "before_block_id": document.blocks[0].block_id,
            },
            {
                "page_number": 2,
                "before_block_id": document.blocks[2].block_id,
            },
            {
                "page_number": 3,
                "before_block_id": document.blocks[2].block_id,
            },
        ),
    }
    assert document.metadata["document_notes"] == {
            "schema_version": "ac.document.document_notes.v1",
        "items": (
            {
                "kind": "metadata",
                "text": "<!-- Generated by ARC. -->",
                "before_block_id": document.blocks[0].block_id,
            },
            {
                "kind": "source_page",
                "text": "<!-- Source PDF page 1 -->",
                "before_block_id": document.blocks[0].block_id,
                "page_number": 1,
            },
            {
                "kind": "source_page",
                "text": "<!-- Source PDF page 2 -->",
                "before_block_id": document.blocks[2].block_id,
                "page_number": 2,
            },
            {
                "kind": "source_page",
                "text": "<!-- PDF_PAGE: 3 -->",
                "before_block_id": document.blocks[2].block_id,
                "page_number": 3,
            },
            {
                "kind": "metadata",
                "text": "<!-- hidden trailing metadata -->",
                "before_block_id": None,
            },
        ),
    }


def test_mixed_markdown_html_comment_remains_authored_prose(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(
        repository,
        b"Visible <!-- authored marker --> text.\n",
        SourceFormat.MARKDOWN,
    )

    document = RichDocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    assert len(document.blocks) == 1
    assert document.blocks[0].payload["text"] == (
        "Visible <!-- authored marker --> text."
    )


def test_html_rich_parse_preserves_equation_table_figure_and_selector(tmp_path):
    source = tmp_path / "paper.html"
    image = tmp_path / "plot.svg"
    image.write_text("<svg/>", encoding="utf-8")
    source.write_text(
        """
        <article>
          <h1 id="intro">Introduction</h1>
          <p>See <a href="https://example.test">source</a>
             and <math alttext="a+b"></math>.</p>
          <table class="ltx_equation" id="eq1">
            <tr><td><math alttext="F=ma"></math></td></tr>
          </table>
          <table><caption>Inputs</caption>
            <tr><th>x</th><th>y</th></tr><tr><td>1</td><td>2</td></tr>
          </table>
          <figure><img src="plot.svg" alt="plot"><figcaption>Result</figcaption></figure>
        </article>
        """,
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")
    artifact = repository.import_path(source)

    document = RichDocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.HEADING,
        RichBlockKind.PARAGRAPH,
        RichBlockKind.EQUATION,
        RichBlockKind.TABLE,
        RichBlockKind.FIGURE,
    ]
    assert document.blocks[0].locator.selector == "#intro"
    assert next(
        item["tex"]
        for item in document.blocks[1].payload["inline_spans"]
        if item["kind"] == "math"
    ) == "a+b"
    assert [item["kind"] for item in document.blocks[1].payload["inline_spans"]] == [
        "text",
        "link",
        "text",
        "math",
        "text",
    ]
    assert document.blocks[2].payload == {
        "tex": "F=ma",
        "display": True,
        "label": "",
    }
    assert document.blocks[3].payload["headers"] == ("x", "y")
    assert document.blocks[3].payload["caption"] == "Inputs"
    assert document.blocks[4].payload["caption"] == "Result"
    assert document.blocks[4].payload["media_type"] == "image/svg+xml"
    assert document.assets[0].media_type == "image/svg+xml"


def test_html_locators_use_opening_tags_and_cover_top_level_articles_once(
    tmp_path,
):
    source = tmp_path / "articles.html"
    source.write_text(
        "\n".join(
            [
                "<nav><p id='navigation'>Outside</p></nav>",
                "<article>",
                "  <h1 data-role='title' id='first'>First</h1>",
                "    <img alt='void image' src='missing.png'>",
                "  <article><p id='nested'>Nested once</p></article>",
                "</article>",
                "<article>",
                " <p class='last'>Second article</p>",
                "</article>",
            ]
        ),
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")

    document = RichDocumentParserService(repository).parse(
        SourceBundle(primary=repository.import_path(source))
    ).document

    assert [
        block.payload.get("text", block.payload.get("alt_text"))
        for block in document.blocks
    ] == [
        "First",
        "void image",
        "Nested once",
        "Second article",
    ]
    assert [
        (
            block.locator.line_start,
            block.locator.column_start,
            block.locator.line_end,
            block.locator.column_end,
        )
        for block in document.blocks
    ] == [
        (3, 3, 3, 3),
        (4, 5, 4, 5),
        (5, 12, 5, 12),
        (8, 2, 8, 2),
    ]
    assert [block.locator.selector for block in document.blocks] == [
        "#first",
        "img:nth-block(2)",
        "#nested",
        "p:nth-block(4)",
    ]
    assert all(
        block.locator.source_id != "navigation" for block in document.blocks
    )


def test_html_without_article_preserves_visible_non_navigation_text_in_order(
    tmp_path,
):
    source = tmp_path / "body.html"
    source.write_text(
        "\n".join(
            [
                "<html><body>",
                "  <div class='chrome'><nav>Navigation only.</nav></div>",
                "  <main>",
                "    <div id='lead'>Lead <span>context</span>.</div>",
                "    <p id='middle'>Middle text.</p>",
                "    <section><div id='tail'>Tail text.</div></section>",
                "  </main>",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")

    document = RichDocumentParserService(repository).parse_source(
        repository.import_path(source)
    )

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.PARAGRAPH,
        RichBlockKind.PARAGRAPH,
        RichBlockKind.PARAGRAPH,
    ]
    assert [block.payload["text"] for block in document.blocks] == [
        "Lead context.",
        "Middle text.",
        "Tail text.",
    ]
    assert all(
        "Navigation only." not in block.payload["text"]
        for block in document.blocks
    )


def test_html_block_math_splits_paragraph_and_table_in_source_order(
    tmp_path,
):
    source = tmp_path / "block-math.html"
    source.write_text(
        "\n".join(
            [
                "<article>",
                "  <p id='prose'>Before ",
                "    <math id='paragraph-equation' display='block' alttext='x = 1'></math>",
                "    after.</p>",
                "  <table id='values'>",
                "    <tr><th>Value</th></tr>",
                "    <tr><td>Left ",
                "      <math id='table-equation' display='block' alttext='y = 2'></math>",
                "      right.</td></tr>",
                "  </table>",
                "</article>",
            ]
        ),
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")

    document = RichDocumentParserService(repository).parse_source(
        repository.import_path(source)
    )

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.PARAGRAPH,
        RichBlockKind.EQUATION,
        RichBlockKind.PARAGRAPH,
        RichBlockKind.TABLE,
        RichBlockKind.EQUATION,
        RichBlockKind.TABLE,
    ]
    before, paragraph_equation, after, left, table_equation, right = (
        document.blocks
    )
    assert before.payload["text"] == "Before"
    assert paragraph_equation.payload == {
        "tex": "x = 1",
        "display": True,
        "label": "",
    }
    assert after.payload["text"] == "after."
    assert left.payload["headers"] == ("Value",)
    assert left.payload["rows"] == (("Left",),)
    assert table_equation.payload == {
        "tex": "y = 2",
        "display": True,
        "label": "",
    }
    assert right.payload["headers"] == ("",)
    assert right.payload["rows"] == (("right.",),)
    assert before.locator == paragraph_equation.locator == after.locator
    assert left.locator == table_equation.locator == right.locator
    assert [
        block.payload["tex"]
        for block in document.blocks
        if block.kind is RichBlockKind.EQUATION
    ] == ["x = 1", "y = 2"]


@pytest.mark.parametrize("source_format", [SourceFormat.MARKDOWN, SourceFormat.HTML])
def test_inline_images_are_imported_as_figure_blocks(tmp_path, source_format):
    image = tmp_path / "inline.png"
    image.write_bytes(b"\x89PNG inline")
    if source_format is SourceFormat.MARKDOWN:
        source = tmp_path / "inline.md"
        source.write_text(
            "Before ![inline plot](inline.png) after.",
            encoding="utf-8",
        )
    else:
        source = tmp_path / "inline.html"
        source.write_text(
            "<article><p>Before <img src='inline.png' alt='inline plot'> after.</p></article>",
            encoding="utf-8",
        )
    repository = SourceRepository(tmp_path / "cache")

    document = RichDocumentParserService(repository).parse(
        SourceBundle(primary=repository.import_path(source))
    ).document

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.PARAGRAPH,
        RichBlockKind.FIGURE,
        RichBlockKind.PARAGRAPH,
    ]
    before, figure, after = document.blocks
    assert before.payload["text"] == "Before"
    assert after.payload["text"] == "after."
    assert figure.payload["alt_text"] == "inline plot"
    assert figure.payload["asset_digest"] == document.assets[0].artifact_digest
    assert before.locator == figure.locator == after.locator


@pytest.mark.parametrize("source_format", [SourceFormat.MARKDOWN, SourceFormat.HTML])
def test_list_inline_image_preserves_order_and_asset(tmp_path, source_format):
    image = tmp_path / "inline.png"
    image.write_bytes(b"\x89PNG list inline")
    if source_format is SourceFormat.MARKDOWN:
        source = tmp_path / "list.md"
        source.write_text(
            "- first\n- before ![list plot](inline.png) after\n- last",
            encoding="utf-8",
        )
    else:
        source = tmp_path / "list.html"
        source.write_text(
            "<article><ul><li>first</li><li>before "
            "<img src='inline.png' alt='list plot'> after</li>"
            "<li>last</li></ul></article>",
            encoding="utf-8",
        )
    repository = SourceRepository(tmp_path / "cache")

    document = RichDocumentParserService(repository).parse(
        SourceBundle(primary=repository.import_path(source))
    ).document

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.LIST,
        RichBlockKind.FIGURE,
        RichBlockKind.LIST,
    ]
    before, figure, after = document.blocks
    assert [item["text"] for item in before.payload["items"]] == [
        "first",
        "before",
    ]
    assert [item["text"] for item in after.payload["items"]] == [
        "after",
        "last",
    ]
    assert figure.payload["alt_text"] == "list plot"
    assert figure.payload["asset_digest"] == document.assets[0].artifact_digest
    assert before.locator == figure.locator == after.locator


@pytest.mark.parametrize("source_format", [SourceFormat.MARKDOWN, SourceFormat.HTML])
def test_table_inline_image_preserves_order_and_asset(tmp_path, source_format):
    image = tmp_path / "inline.png"
    image.write_bytes(b"\x89PNG table inline")
    if source_format is SourceFormat.MARKDOWN:
        source = tmp_path / "table.md"
        source.write_text(
            "| Result |\n| --- |\n"
            "| before ![table plot](inline.png) after |",
            encoding="utf-8",
        )
    else:
        source = tmp_path / "table.html"
        source.write_text(
            "<article><table><tr><th>Result</th></tr><tr><td>before "
            "<img src='inline.png' alt='table plot'> after</td></tr>"
            "</table></article>",
            encoding="utf-8",
        )
    repository = SourceRepository(tmp_path / "cache")

    document = RichDocumentParserService(repository).parse(
        SourceBundle(primary=repository.import_path(source))
    ).document

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.TABLE,
        RichBlockKind.FIGURE,
        RichBlockKind.TABLE,
    ]
    before, figure, after = document.blocks
    assert before.payload["headers"] == ("Result",)
    assert before.payload["rows"] == (("before",),)
    assert after.payload["headers"] == ("",)
    assert after.payload["rows"] == (("after",),)
    assert figure.payload["alt_text"] == "table plot"
    assert figure.payload["asset_digest"] == document.assets[0].artifact_digest


def test_flattened_tex_rich_parse_and_multifile_rejection(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(
        repository,
        b"\n".join(
            [
                br"\section{Model}",
                br"The \href{https://example.test}{reference} uses \(x+y\).",
                br"\begin{equation}",
                br"E = mc^2 \label{energy}",
                br"\end{equation}",
                br"\begin{enumerate}",
                br"\item First",
                br"\item Second",
                br"\end{enumerate}",
            ]
        ),
        SourceFormat.TEX,
    )

    document = RichDocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.HEADING,
        RichBlockKind.PARAGRAPH,
        RichBlockKind.EQUATION,
        RichBlockKind.LIST,
    ]
    assert next(
        item["target"]
        for item in document.blocks[1].payload["inline_spans"]
        if item["kind"] == "link"
    ) == "https://example.test"
    assert document.blocks[2].payload["label"] == "energy"
    assert document.blocks[3].payload["ordered"] is True

    project = _store(
        repository,
        br"\section{Main}" b"\n" br"\input{chapter}",
        SourceFormat.TEX,
    )
    with pytest.raises(Exception) as error:
        RichDocumentParserService(repository).parse(
            SourceBundle(primary=project)
        )
    assert getattr(error.value, "code", "") == "unsupported_tex_project"


def test_tex_headings_support_balanced_starred_short_and_multiline_titles(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(
        repository,
        "\n".join(
            [
                r"\section{Ordinary}",
                r"\section*{Starred}",
                r"\subsection[Short {with ] brace}]{Long {nested} \{literal\}}",
                r"\subsubsection{Across",
                r"  \textbf{multiple lines}}",
                r"\section{\texorpdfstring{TeX choice}{PDF choice}}",
            ]
        ).encode(),
        SourceFormat.TEX,
    )

    document = RichDocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    assert [block.payload["text"] for block in document.blocks] == [
        "Ordinary",
        "Starred",
        "Long nested {literal}",
        "Across multiple lines",
        "TeX choice",
    ]
    assert [block.payload["level"] for block in document.blocks] == [
        1,
        1,
        2,
        3,
        1,
    ]
    assert [
        (block.locator.line_start, block.locator.line_end)
        for block in document.blocks
    ] == [(1, 1), (2, 2), (3, 3), (4, 5), (6, 6)]
    assert all(
        block.locator.column_start is None
        and block.locator.column_end is None
        for block in document.blocks
    )


@pytest.mark.parametrize(
    ("source_format", "text"),
    [
        (SourceFormat.MARKDOWN, "$$\nx+y"),
        (SourceFormat.MARKDOWN, "\\[\nx+y"),
        (SourceFormat.MARKDOWN, "\\begin{align}\nx&=y"),
        (SourceFormat.TEX, "$$\nx+y"),
        (SourceFormat.TEX, "\\[\nx+y"),
        (SourceFormat.TEX, "\\begin{equation}\nx=y"),
        (SourceFormat.TEX, "\\section[Short]{Unclosed"),
    ],
)
def test_unclosed_rich_blocks_fail_before_document_creation(
    tmp_path, source_format, text
):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(repository, text.encode(), source_format)

    with pytest.raises(Exception) as error:
        RichDocumentParserService(repository).parse(
            SourceBundle(primary=artifact)
        )

    assert getattr(error.value, "code", "") == "unclosed_rich_block"


def test_unclosed_markdown_fence_extends_to_real_eof_with_line_locator(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    artifact = _store(
        repository,
        b"before\n\n```python\none\ntwo",
        SourceFormat.MARKDOWN,
    )

    document = RichDocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    code = document.blocks[-1]
    assert code.kind is RichBlockKind.CODE
    assert code.payload["text"] == "one\ntwo"
    assert (code.locator.line_start, code.locator.line_end) == (3, 5)
    assert code.locator.column_start is None
    assert code.locator.column_end is None


def test_multiline_tex_figure_preserves_asset_and_caption(tmp_path):
    image = tmp_path / "plot.png"
    image.write_bytes(b"\x89PNG plot")
    source = tmp_path / "paper.tex"
    source.write_text(
        "\n".join(
            [
                r"\section{Results}",
                r"\begin{figure}",
                r"\centering",
                r"\includegraphics[width=.8\linewidth]{plot.png}",
                r"\caption{Measured response}",
                r"\label{fig:response}",
                r"\end{figure}",
            ]
        ),
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")

    document = RichDocumentParserService(repository).parse(
        SourceBundle(primary=repository.import_path(source))
    ).document

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.HEADING,
        RichBlockKind.FIGURE,
    ]
    figure = document.blocks[1]
    assert figure.payload["caption"] == "Measured response"
    assert figure.payload["target"] == "plot.png"
    assert figure.payload["asset_digest"] == document.assets[0].artifact_digest


def test_rich_document_and_block_codecs_are_strict_and_path_free(tmp_path):
    source = tmp_path / "paper.md"
    source.write_text("# Codec\nText.", encoding="utf-8")
    repository = SourceRepository(tmp_path / "cache")
    artifact = repository.import_path(source)
    document = RichDocumentParserService(repository).parse(
        SourceBundle(primary=artifact)
    ).document

    encoded = rich_document_to_document(document)
    decoded = rich_document_from_document(encoded)
    encoded_block = rich_block_to_document(document.blocks[1])

    assert decoded.document_digest == document.document_digest
    assert decoded.source.origin.kind is SourceOriginKind.REPOSITORY
    assert encoded["blocks"][0]["locator"]["column_start"] is None
    assert encoded["blocks"][0]["locator"]["column_end"] is None
    assert decoded.blocks[0].locator.column_start is None
    assert decoded.blocks[0].locator.column_end is None
    assert str(source) not in str(encoded)
    assert (
        rich_block_from_document(encoded_block).block_id
        == document.blocks[1].block_id
    )
    with pytest.raises(ValueError, match="invalid fields"):
        rich_document_from_document({**encoded, "unknown": True})
    with pytest.raises(ValueError, match="invalid fields"):
        rich_block_from_document({**encoded_block, "unknown": True})
    invalid_payload = {
        **encoded_block,
        "payload": {**encoded_block["payload"], "unknown": True},
    }
    with pytest.raises(ValueError, match="invalid fields"):
        rich_block_from_document(invalid_payload)
    with pytest.raises(ValueError, match="arrays must be lists"):
        rich_block_from_document(
            {
                **encoded_block,
                "payload": {
                    **encoded_block["payload"],
                    "inline_spans": tuple(
                        encoded_block["payload"]["inline_spans"]
                    ),
                },
            }
        )
    with pytest.raises(ValueError, match="arrays must be lists"):
        rich_document_from_document(
            {**encoded, "blocks": tuple(encoded["blocks"])}
        )
    corrupt = {
        **encoded,
        "blocks": [
            encoded["blocks"][0],
            {
                **encoded["blocks"][1],
                "payload": {
                    **encoded["blocks"][1]["payload"],
                    "text": "changed",
                    "inline_spans": [
                        {
                            "kind": "text",
                            "start": 0,
                            "end": 7,
                            "text": "changed",
                        }
                    ],
                },
            },
        ],
    }
    with pytest.raises(ValueError, match="digest"):
        rich_document_from_document(corrupt)

    invalid_offsets = {
        **encoded_block,
        "payload": {
            **encoded_block["payload"],
            "inline_spans": [
                {
                    **encoded_block["payload"]["inline_spans"][0],
                    "start": 1,
                }
            ],
        },
    }
    with pytest.raises(ValueError, match="contiguously"):
        rich_block_from_document(invalid_offsets)


def test_rich_document_identity_excludes_source_path(tmp_path):
    first = tmp_path / "first.md"
    second = tmp_path / "nested" / "second.md"
    second.parent.mkdir()
    payload = b"# Identity\nSame body."
    first.write_bytes(payload)
    second.write_bytes(payload)
    repository = SourceRepository(tmp_path / "cache")
    service = RichDocumentParserService(repository)

    first_document = service.parse(
        SourceBundle(primary=repository.import_path(first))
    ).document
    second_document = service.parse(
        SourceBundle(primary=repository.import_path(second))
    ).document

    assert first_document.document_digest == second_document.document_digest
    assert [block.block_id for block in first_document.blocks] == [
        block.block_id for block in second_document.blocks
    ]


def test_block_and_section_identity_include_full_source_identity(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    payload = b"Plain text."
    markdown = _store(repository, payload, SourceFormat.MARKDOWN)
    tex = _store(repository, payload, SourceFormat.TEX)
    service = RichDocumentParserService(repository)

    markdown_document = service.parse(
        SourceBundle(primary=markdown)
    ).document
    tex_document = service.parse(SourceBundle(primary=tex)).document

    assert markdown_document.blocks[0].kind is RichBlockKind.PARAGRAPH
    assert tex_document.blocks[0].kind is RichBlockKind.PARAGRAPH
    assert markdown_document.blocks[0].payload == tex_document.blocks[0].payload
    assert markdown_document.blocks[0].block_id != tex_document.blocks[0].block_id
    assert (
        markdown_document.sections[0].section_id
        != tex_document.sections[0].section_id
    )


def test_rich_integer_fields_reject_booleans():
    locator = SourceLocator(SourceFormat.MARKDOWN, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="identity"):
        RichBlock(
            block_id="block",
            ordinal=False,
            kind=RichBlockKind.PARAGRAPH,
            section_path=(),
            locator=locator,
            payload={
                "text": "x",
                "inline_spans": [
                    {"kind": "text", "start": 0, "end": 1, "text": "x"}
                ],
            },
        )
    with pytest.raises(ValueError, match="metadata"):
        RichSection(
            section_id="section",
            title="Title",
            level=True,
            ordinal=0,
            path=("section",),
            block_start=0,
            block_end=1,
        )
    with pytest.raises(ValueError, match="page map"):
        RichPageMapEntry(block_id="block", page_number=True)
    with pytest.raises(ValueError, match="positions"):
        SourceLocator(SourceFormat.MARKDOWN, True, 1, 1, 1)


def test_matching_pdf_builds_page_map_and_mismatch_fails(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"# Introduction\nText.\n\n# Method\nMore text.\n",
        SourceFormat.MARKDOWN,
    )
    matching_payload = b"%PDF matching"
    matching = _store(repository, matching_payload, SourceFormat.PDF)
    mismatch_payload = b"%PDF mismatch"
    mismatch = _store(repository, mismatch_payload, SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {
            matching_payload: PDFTextLayer(
                ("Introduction\nText.", "Method\nMore text.")
            ),
            mismatch_payload: PDFTextLayer(("Completely unrelated",)),
        }
    )
    service = RichDocumentParserService(
        repository, pdf_text_extractor=extractor
    )

    outcome = service.parse(
        SourceBundle(primary=primary, validators=(matching,))
    )

    pages = {
        entry.block_id: entry.page_number for entry in outcome.document.page_map
    }
    assert pages[outcome.document.blocks[0].block_id] == 1
    assert pages[outcome.document.blocks[-1].block_id] == 2
    assert outcome.document.blocks[1].block_id not in pages
    assert PDF_VALIDATOR_MISSING_WARNING not in outcome.warnings

    with pytest.raises(RichDocumentValidationError) as error:
        service.parse(SourceBundle(primary=primary, validators=(mismatch,)))
    assert error.value.code == "pdf_validator_mismatch"


def test_heading_free_source_reconciles_by_body_text(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"A distinctive compact source body contains enough stable words for validation.",
        SourceFormat.MARKDOWN,
    )
    pdf_payload = b"%PDF heading-free"
    pdf = _store(repository, pdf_payload, SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
            {
                pdf_payload: PDFTextLayer(
                    ("A distinctive compact source body contains enough stable words for validation.",)
                )
            }
    )

    outcome = RichDocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=primary, validators=(pdf,)))

    assert len(outcome.document.sections) == 1
    assert outcome.document.sections[0].title == "Document"
    assert outcome.document.page_map[0].page_number == 1


def test_pdf_section_body_anchor_reconciles_titleless_section_and_stays_unique(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        (
            b"# Abstract\n\n"
            b"Eight stable words make this opening body anchor unique today.\n"
        ),
        SourceFormat.MARKDOWN,
    )
    pdf_payload = b"%PDF titleless abstract"
    pdf = _store(repository, pdf_payload, SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {
            pdf_payload: PDFTextLayer(
                ("Eight stable words make this opening body anchor unique today.",)
            )
        }
    )

    outcome = RichDocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=primary, validators=(pdf,)))

    section = next(
        entry
        for entry in outcome.report.entries
        if entry.subject_id.startswith("section:")
    )
    assert section.status.value == "verified"
    assert section.provenance["matching_method"] == "content_anchor"
    assert section.provenance["body_anchor"] == [
        "eight",
        "stable",
        "words",
        "make",
        "this",
        "opening",
        "body",
        "anchor",
    ]


def test_pdf_section_body_anchor_rejects_multiple_pages(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"# Summary\n\nEight stable words make this opening body anchor unique today.\n",
        SourceFormat.MARKDOWN,
    )
    pdf_payload = b"%PDF repeated body"
    pdf = _store(repository, pdf_payload, SourceFormat.PDF)
    repeated = "Eight stable words make this opening body anchor unique today."
    extractor = FakePDFTextExtractor(
        {pdf_payload: PDFTextLayer((repeated, repeated))}
    )

    with pytest.raises(RichDocumentValidationError) as error:
        RichDocumentParserService(
            repository, pdf_text_extractor=extractor
        ).parse(SourceBundle(primary=primary, validators=(pdf,)))

    assert error.value.code == "pdf_validator_ambiguous"
    assert error.value.details[0]["matching_method"] == "content_anchor"
    assert error.value.details[0]["page_candidates"] == [1, 2]


def test_pdf_section_matches_wrapped_semantic_heading_with_different_labels(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"# III.1 Tidal response beyond equilibrium\n\nSource body.\n",
        SourceFormat.MARKDOWN,
    )
    pdf_payload = b"%PDF wrapped heading"
    pdf = _store(repository, pdf_payload, SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {
            pdf_payload: PDFTextLayer(
                ("A. Tidal response beyond\nequilibrium\nSource body.",)
            )
        }
    )

    outcome = RichDocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=primary, validators=(pdf,)))

    section = next(
        entry
        for entry in outcome.report.entries
        if entry.subject_id.startswith("section:")
    )
    assert section.provenance["matching_method"] == "joined_heading_lines"


def test_pdf_section_matching_does_not_strip_article_a_from_a_title(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"# A Model\n",
        SourceFormat.MARKDOWN,
    )
    pdf_payload = b"%PDF article title false positive"
    pdf = _store(repository, pdf_payload, SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {
            pdf_payload: PDFTextLayer(
                ("This paragraph discusses a model but contains no section heading.",)
            )
        }
    )

    with pytest.raises(RichDocumentValidationError) as error:
        RichDocumentParserService(
            repository, pdf_text_extractor=extractor
        ).parse(SourceBundle(primary=primary, validators=(pdf,)))

    assert error.value.code == "pdf_validator_mismatch"


def test_html_equation_table_groups_fragments_by_visible_displayed_label(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"""
        <article><h1>Overview</h1>
        <table class="ltx_equation" id="S3.E22">
          <tr><td><math alttext="x"></math><math alttext="= y"></math></td><td><span class="ltx_tag">(4)</span></td></tr>
          <tr><td><math alttext="+ z"></math></td></tr>
        </table>
        <table class="ltx_equation" id="S3.E23">
          <tr><td><math alttext="a"></math><math alttext="= b"></math></td><td><span class="ltx_tag">(5)</span></td></tr>
          <tr><td><math alttext="c"></math><math alttext="= d"></math></td><td><span class="ltx_tag">(6)</span></td></tr>
        </table></article>
        """,
        SourceFormat.HTML,
    )

    document = RichDocumentParserService(repository).parse_source(primary)

    equations = [
        block for block in document.blocks if block.kind is RichBlockKind.EQUATION
    ]
    assert [block.payload["tex"] for block in equations] == [
        "x = y + z",
        "a = b",
        "c = d",
    ]
    assert [block.payload["label"] for block in equations] == ["4", "5", "6"]
    assert all(block.payload["label"] != "S3.E23" for block in equations)


def test_ambiguous_pdf_math_evidence_is_diagnostic_not_a_rich_parse_failure(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"# Overview\n\n$$ x = 1 $$\n",
        SourceFormat.MARKDOWN,
    )
    pdf_payload = b"%PDF repeated short formula"
    pdf = _store(repository, pdf_payload, SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {
            pdf_payload: PDFTextLayer(
                ("Overview\nx = 1", "The short formula x = 1 appears again.")
            )
        }
    )

    outcome = RichDocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=primary, validators=(pdf,)))

    math_entry = next(
        entry for entry in outcome.report.entries if entry.subject_id.startswith("math-")
    )
    assert math_entry.status is ReconciliationStatus.AMBIGUOUS
    assert any("PDF math evidence ambiguous" in warning for warning in outcome.warnings)


def test_pdf_validator_keeps_rich_equation_labels_without_canonical_inference(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"""
        <article><h1>Overview</h1>
        <table class="ltx_equation"><tr><td><math alttext="a = b"></math></td><td><span class="ltx_tag">(4)</span></td></tr></table>
        </article>
        """,
        SourceFormat.HTML,
    )
    pdf_payload = b"%PDF number only"
    pdf = _store(repository, pdf_payload, SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {pdf_payload: PDFTextLayer(("Overview\nc = d (1)",))}
    )

    outcome = RichDocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=primary, validators=(pdf,)))

    span_entry = next(
        entry
        for entry in outcome.report.entries
        if entry.subject_id.startswith("math-")
    )
    assert span_entry.status.value == "unreviewed"
    assert "equation_label_reconciliation" not in outcome.document.metadata
    equation = next(
        block
        for block in outcome.document.blocks
        if block.kind is RichBlockKind.EQUATION
    )
    assert equation.payload["label"] == "4"


def test_preface_does_not_shift_heading_page_map(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"Preface without a heading.\n\n# Introduction\nText.\n\n# Method\nMore.\n",
        SourceFormat.MARKDOWN,
    )
    pdf_payload = b"%PDF with preface"
    pdf = _store(repository, pdf_payload, SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {
            pdf_payload: PDFTextLayer(
                ("Introduction\nText.", "Method\nMore.")
            )
        }
    )

    document = RichDocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=primary, validators=(pdf,))).document

    page_by_block = {
        item.block_id: item.page_number for item in document.page_map
    }
    preface, introduction, _, method, _ = document.blocks
    assert preface.block_id not in page_by_block
    assert page_by_block[introduction.block_id] == 1
    assert page_by_block[method.block_id] == 2


def test_page_map_matches_late_block_on_second_page_of_same_section(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        (
            b"# Results\n"
            b"Opening explanation appears only near the start.\n\n"
            b"Late conclusion appears only at the end.\n"
        ),
        SourceFormat.MARKDOWN,
    )
    pdf_payload = b"%PDF multi-page section"
    pdf = _store(repository, pdf_payload, SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {
            pdf_payload: PDFTextLayer(
                (
                    "Results\nOpening explanation appears only near the start.",
                    "Late conclusion appears only at the end.",
                )
            )
        }
    )

    document = RichDocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=primary, validators=(pdf,))).document

    page_by_block = {
        item.block_id: item.page_number for item in document.page_map
    }
    heading, opening, late = document.blocks
    assert page_by_block[heading.block_id] == 1
    assert page_by_block[opening.block_id] == 1
    assert page_by_block[late.block_id] == 2


def test_page_map_omits_block_with_ambiguous_page_text(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        (
            b"# Results\n"
            b"Repeated explanatory sentence appears here.\n"
        ),
        SourceFormat.MARKDOWN,
    )
    pdf_payload = b"%PDF repeated block"
    pdf = _store(repository, pdf_payload, SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {
            pdf_payload: PDFTextLayer(
                (
                    "Results\nRepeated explanatory sentence appears here.",
                    "Repeated explanatory sentence appears here.",
                )
            )
        }
    )

    document = RichDocumentParserService(
        repository, pdf_text_extractor=extractor
    ).parse(SourceBundle(primary=primary, validators=(pdf,))).document

    page_by_block = {
        item.block_id: item.page_number for item in document.page_map
    }
    heading, repeated = document.blocks
    assert page_by_block[heading.block_id] == 1
    assert repeated.block_id not in page_by_block


def test_ambiguous_or_invalid_pdf_fails_deterministically(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"# Introduction\nText.\n",
        SourceFormat.MARKDOWN,
    )
    ambiguous_payload = b"%PDF ambiguous"
    ambiguous = _store(repository, ambiguous_payload, SourceFormat.PDF)
    invalid_payload = b"not actually PDF"
    invalid = _store(repository, invalid_payload, SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {
            ambiguous_payload: PDFTextLayer(
                ("Introduction page one", "Introduction page two")
            ),
            invalid_payload: PDFTextLayer(("Introduction",)),
        }
    )
    service = RichDocumentParserService(
        repository, pdf_text_extractor=extractor
    )

    with pytest.raises(RichDocumentValidationError) as error:
        service.parse(SourceBundle(primary=primary, validators=(ambiguous,)))
    assert error.value.code == "pdf_validator_ambiguous"

    with pytest.raises(RichDocumentValidationError) as error:
        service.parse(SourceBundle(primary=primary, validators=(invalid,)))
    assert error.value.code == "pdf_validator_invalid"
    assert invalid_payload not in extractor.calls


def test_pdf_without_text_layer_is_not_accepted_as_validated(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"# Introduction\nText.\n",
        SourceFormat.MARKDOWN,
    )
    pdf_payload = b"%PDF image-only"
    pdf = _store(repository, pdf_payload, SourceFormat.PDF)
    extractor = FakePDFTextExtractor(
        {pdf_payload: PDFTextLayer((), "no extractable text layer")}
    )

    with pytest.raises(RichDocumentValidationError) as error:
        RichDocumentParserService(
            repository, pdf_text_extractor=extractor
        ).parse(SourceBundle(primary=primary, validators=(pdf,)))

    assert error.value.code == "pdf_validator_unverifiable"


def test_source_repository_asset_manifest_is_strict_and_verified(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    stored = repository.store_asset_bytes(
        b"asset bytes", media_type="application/octet-stream"
    )

    assert repository.get_asset(stored.artifact_digest) == stored
    assert repository.read_asset_bytes(stored) == b"asset bytes"
    object_dir = repository._asset_object_dir(stored.artifact_digest)  # noqa: SLF001
    (object_dir / "asset").write_bytes(b"corrupt")

    with pytest.raises(Exception) as error:
        repository.read_asset_bytes(stored)
    assert getattr(error.value, "code", "") == "asset_corrupt"
