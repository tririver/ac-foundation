from __future__ import annotations

import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest
import ac_document.rich_document.list_paths as list_path_validation

presentation_validation = importlib.import_module(
    "ac_document.rich_document.source_presentation"
)
rich_parser = importlib.import_module("ac_document.rich_document.parser")

from ac_document import (
    PDF_VALIDATOR_MISSING_WARNING,
    DOCUMENT_DIAGNOSTICS_METADATA_KEY,
    DOCUMENT_DIAGNOSTICS_SCHEMA,
    PDFTextLayer,
    RICH_DOCUMENT_SCHEMA,
    RICH_DOCUMENT_SCHEMA_V2,
    SOURCE_FRONT_MATTER_METADATA_KEY,
    SOURCE_FRONT_MATTER_SCHEMA,
    SOURCE_NOTES_METADATA_KEY,
    SOURCE_NOTES_SCHEMA,
    SOURCE_PRESENTATION_METADATA_KEY,
    SOURCE_PRESENTATION_SCHEMA,
    SOURCE_TARGET_MANIFEST_METADATA_KEY,
    SOURCE_TARGET_MANIFEST_SCHEMA,
    RichBlock,
    RichBlockKind,
    RichDocument,
    RichDocumentParserService,
    RichDocumentValidationError,
    RichListPathEntry,
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
    document_diagnostics,
    source_front_matter,
    source_notes,
    source_presentation,
    source_target_manifest,
    validate_list_paths,
    validate_source_fidelity_metadata,
    validate_source_presentation_metadata,
    validate_source_target_manifest,
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


def _source_target(document, alias):
    manifest = source_target_manifest(document)
    assert manifest is not None
    return next(item for item in manifest["targets"] if item["alias"] == alias)


def _diagnostic_categories(document):
    diagnostics = document_diagnostics(document)
    assert diagnostics is not None
    assert diagnostics["schema_version"] == DOCUMENT_DIAGNOSTICS_SCHEMA
    assert diagnostics["visible_content"]["unaccounted"] == 0
    return [item["category"] for item in diagnostics["projections"]]


def _source_presentation_block(document, block_id):
    presentation = source_presentation(document)
    assert presentation is not None
    return next(
        item for item in presentation["blocks"] if item["block_id"] == block_id
    )


def _source_presentation_field(
    document,
    block,
    field,
    *,
    item_index=None,
    row_index=None,
    column_index=None,
):
    entry = _source_presentation_block(document, block.block_id)
    return next(
        item
        for item in entry["fields"]
        if item["field"] == field
        and item["item_index"] == item_index
        and item["row_index"] == row_index
        and item["column_index"] == column_index
    )


def _source_presentation_caption(document, block):
    presentation = source_presentation(document)
    assert presentation is not None
    return next(
        item
        for item in presentation["captions"]
        if item["block_id"] == block.block_id
    )


def _source_presentation_figure(document, block):
    presentation = source_presentation(document)
    assert presentation is not None
    return next(
        item
        for item in presentation["figures"]
        if item["block_id"] == block.block_id
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
        "schema_version": "ac.document.rich_document.v3",
        "document_digest": (
                "4208069eb3255b0a04517825b119858b32ec1e1044c3e9d3bbed7dc62f706979"
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
                "list_path": [],
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
                "list_path": [],
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
                "list_path": [],
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


def test_html_source_target_manifest_preserves_exact_sections_and_blocks(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"""
        <article>
          <section id="S2"><h2 id="S2.H1">Model</h2>
            <p id="S2.p1">Model prose.</p>
            <section id="S2.SS1"><h3 id="S2.SS1.H1">Nested</h3>
              <p id="S2.SS1.p1">Nested prose.</p>
            </section>
          </section>
          <section id="S3"><h2 id="S3.H1">Data</h2>
            <p id="S3.p1">Data prose.</p>
          </section>
        </article>
        """,
        SourceFormat.HTML,
    )

    document = RichDocumentParserService(repository).parse_source(primary)
    manifest = source_target_manifest(document)

    assert manifest is not None
    assert manifest["schema_version"] == SOURCE_TARGET_MANIFEST_SCHEMA
    aliases = [item["alias"] for item in manifest["targets"]]
    assert len(aliases) == len(set(aliases))
    section_2 = _source_target(document, "S2")
    nested = _source_target(document, "S2.SS1")
    section_3 = _source_target(document, "S3")
    heading = _source_target(document, "S2.H1")
    assert section_2 == {
        "alias": "S2",
        "selector": "#S2",
        "kind": "section",
        "block_id": document.blocks[0].block_id,
        "block_start": 0,
        "block_end": 4,
        "section_id": document.sections[0].section_id,
        "panels": (),
    }
    assert nested["block_start"] == 2
    assert nested["block_end"] == 4
    assert nested["section_id"] == document.sections[1].section_id
    assert section_3["block_start"] == 4
    assert section_3["block_end"] == 6
    assert section_3["section_id"] == document.sections[2].section_id
    assert heading["kind"] == "heading"
    assert heading["block_start"] == 0
    assert heading["block_end"] == 1
    assert heading["section_id"] == ""
    assert document.blocks[0].locator.source_id == "S2.H1"
    assert document.blocks[0].locator.selector == "#S2.H1"
    decoded = rich_document_from_document(rich_document_to_document(document))
    assert decoded.document_digest == document.document_digest
    assert source_target_manifest(decoded) == manifest


def test_html_source_target_manifest_binds_referenced_nested_table_target(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article>
              <p id="P1">See <a href="#T1.note">Table note</a>.</p>
              <figure id="T1" class="ltx_table">
                <figcaption>Table 1: Values.</figcaption>
                <table><tr><th>Value</th></tr><tr><td>1</td></tr></table>
                <div id="T1.note"><p>Authored note.</p></div>
              </figure>
            </article>
            """,
            SourceFormat.HTML,
        )
    )

    table = next(
        block for block in document.blocks if block.kind is RichBlockKind.TABLE
    )
    target = _source_target(document, "T1.note")

    assert target["selector"] == "#T1.note"
    assert target["kind"] == "table"
    assert target["block_id"] == table.block_id
    assert target["block_start"] == table.ordinal
    assert target["block_end"] == table.ordinal + 1


def test_source_target_manifest_is_optional_for_legacy_documents(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(repository, b"# Legacy\nBody.\n", SourceFormat.MARKDOWN)
    )

    assert SOURCE_TARGET_MANIFEST_METADATA_KEY not in document.metadata
    assert source_target_manifest(document) is None


def test_source_target_manifest_rejects_tampering(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><section id="S2"><h2 id="S2.H1">Model</h2>
              <p id="S2.p1">Text.</p>
            </section></article>
            """,
            SourceFormat.HTML,
        )
    )
    original = rich_document_to_document(document)["metadata"][
        SOURCE_TARGET_MANIFEST_METADATA_KEY
    ]

    cases = []
    duplicate = json.loads(json.dumps(original))
    duplicate["targets"].append(dict(duplicate["targets"][0]))
    cases.append(duplicate)
    unknown_kind = json.loads(json.dumps(original))
    unknown_kind["targets"][0]["kind"] = "unknown"
    cases.append(unknown_kind)
    bad_selector = json.loads(json.dumps(original))
    bad_selector["targets"][0]["selector"] = "#different"
    cases.append(bad_selector)
    missing_block = json.loads(json.dumps(original))
    missing_block["targets"][0]["block_id"] = "block-missing"
    cases.append(missing_block)
    bad_bounds = json.loads(json.dumps(original))
    bad_bounds["targets"][0]["block_end"] = len(document.blocks) + 1
    cases.append(bad_bounds)
    bad_section = json.loads(json.dumps(original))
    section = next(
        item for item in bad_section["targets"] if item["kind"] == "section"
    )
    section["block_end"] -= 1
    cases.append(bad_section)
    unknown_field = json.loads(json.dumps(original))
    unknown_field["targets"][0]["extra"] = True
    cases.append(unknown_field)
    panels_on_section = json.loads(json.dumps(original))
    section = next(
        item
        for item in panels_on_section["targets"]
        if item["kind"] == "section"
    )
    section["panels"] = [
        {
            "panel_index": 0,
            "source_id": "",
            "selector": "",
            "target": "",
            "media_type": "",
            "alt_text": "",
            "status": "unsupported",
            "asset_digest": "",
            "logical_name": "",
            "size": 0,
        }
    ]
    cases.append(panels_on_section)

    for value in cases:
        with pytest.raises(ValueError, match="source target manifest"):
            validate_source_target_manifest(
                value,
                blocks=document.blocks,
                sections=document.sections,
                assets=document.assets,
            )

    encoded = rich_document_to_document(document)
    encoded["metadata"][SOURCE_TARGET_MANIFEST_METADATA_KEY]["unexpected"] = True
    with pytest.raises(ValueError, match="source target manifest"):
        rich_document_from_document(encoded)


def test_duplicate_section_alias_is_omitted_from_authoritative_manifest(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article>
              <section id="S2"><h2 id="S2.H1">First</h2></section>
              <section id="S2"><h2 id="S2.H2">Second</h2></section>
            </article>
            """,
            SourceFormat.HTML,
        )
    )

    manifest = source_target_manifest(document)
    assert manifest is not None
    aliases = {item["alias"] for item in manifest["targets"]}
    assert "S2" not in aliases
    assert {"S2.H1", "S2.H2"} <= aliases


def test_source_target_manifest_rejects_panel_tampering(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    (media / "plot.svg").write_text("<svg>panel</svg>", encoding="utf-8")
    source = tmp_path / "paper.html"
    source.write_text(
        """
        <article><figure class="ltx_figure" id="S4.F1">
          <object id="S4.F1.p1" type="image/svg+xml"
                  data="media/plot.svg"></object>
        </figure></article>
        """,
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        repository.import_path(source)
    )
    original = rich_document_to_document(document)["metadata"][
        SOURCE_TARGET_MANIFEST_METADATA_KEY
    ]
    figure_index = next(
        index
        for index, item in enumerate(original["targets"])
        if item["alias"] == "S4.F1"
    )

    cases = []
    bad_index = json.loads(json.dumps(original))
    bad_index["targets"][figure_index]["panels"][0]["panel_index"] = 1
    cases.append(bad_index)
    bad_status = json.loads(json.dumps(original))
    bad_status["targets"][figure_index]["panels"][0]["status"] = "unknown"
    cases.append(bad_status)
    bad_selector = json.loads(json.dumps(original))
    bad_selector["targets"][figure_index]["panels"][0]["selector"] = (
        "#different"
    )
    cases.append(bad_selector)
    missing_asset = json.loads(json.dumps(original))
    missing_asset["targets"][figure_index]["panels"][0]["asset_digest"] = (
        "f" * 64
    )
    cases.append(missing_asset)
    false_missing = json.loads(json.dumps(original))
    false_missing["targets"][figure_index]["panels"][0]["status"] = "missing"
    cases.append(false_missing)
    bad_logical_name = json.loads(json.dumps(original))
    bad_logical_name["targets"][figure_index]["panels"][0]["logical_name"] = (
        "media/different.svg"
    )
    cases.append(bad_logical_name)
    bad_media_type = json.loads(json.dumps(original))
    bad_media_type["targets"][figure_index]["panels"][0]["media_type"] = (
        "image/png"
    )
    cases.append(bad_media_type)
    bad_size = json.loads(json.dumps(original))
    bad_size["targets"][figure_index]["panels"][0]["size"] += 1
    cases.append(bad_size)
    unknown_field = json.loads(json.dumps(original))
    unknown_field["targets"][figure_index]["panels"][0]["extra"] = True
    cases.append(unknown_field)

    for value in cases:
        with pytest.raises(ValueError, match="source target manifest"):
            validate_source_target_manifest(
                value,
                blocks=document.blocks,
                sections=document.sections,
                assets=document.assets,
            )


def test_html_source_front_matter_preserves_structured_authored_order(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><h1 id="title">Paper</h1>
              <div class="ltx_authors">
                <span class="ltx_creator ltx_role_author">
                  <span class="ltx_personname">Ada Lovelace<sup>1</sup>
                    <a class="ltx_ref ltx_orcid"
                       href="https://orcid.org/0000-0001-2345-6789"
                       title="ORCID 0000-0001-2345-6789"></a>
                  </span>
                  <span class="ltx_author_notes"><span class="ltx_author_notes_content">
                    <span class="ltx_contact ltx_role_email">
                      <span class="ltx_contact_name">Email:</span>
                      <a href="mailto:ada@example.test">ada@example.test</a>
                    </span>
                    <span class="ltx_contact ltx_role_affiliation">
                      <span class="ltx_contact_name">Affiliation:</span><sup>1</sup>
                      Analytical Engines Institute
                    </span>
                  </span></span>
                </span>
                <span class="ltx_author_before">; </span>
                <span class="ltx_creator ltx_role_author">
                  <span class="ltx_personname">Grace Hopper<sup>2</sup>
                    <a class="ltx_ref ltx_orcid"
                       href="https://orcid.org/0000-0002-3456-7890"
                       title="ORCID 0000-0002-3456-7890"></a>
                  </span>
                  <span class="ltx_author_notes"><span class="ltx_author_notes_content">
                    <span class="ltx_contact ltx_role_email">
                      <span class="ltx_contact_name">Email:</span>
                      <a href="mailto:grace@example.test,%20corresponding%20author">grace@example.test, corresponding author</a>
                    </span>
                    <span class="ltx_contact ltx_role_affiliation">
                      <span class="ltx_contact_name">Affiliation:</span><sup>2</sup>
                      Compiler Laboratory
                    </span>
                  </span></span>
                </span>
              </div>
              <div class="ltx_abstract"><h6>Abstract</h6><p>Summary.</p></div>
            </article>
            """,
            SourceFormat.HTML,
        )
    )

    front_matter = source_front_matter(document)
    assert front_matter is not None
    assert front_matter["schema_version"] == SOURCE_FRONT_MATTER_SCHEMA
    assert len(front_matter["entries"]) == 1
    group = front_matter["entries"][0]
    assert group["kind"] == "authors"
    assert group["block_index"] == 1
    assert group["locator"]["source_format"] == "html"
    assert group["locator"]["source_id"] == ""
    assert [author["name"] for author in group["authors"]] == [
        "Ada Lovelace",
        "Grace Hopper",
    ]
    assert [author["markers"] for author in group["authors"]] == [("1",), ("2",)]
    assert [author["orcid"] for author in group["authors"]] == [
        "0000-0001-2345-6789",
        "0000-0002-3456-7890",
    ]
    assert [author["contacts"][0]["kind"] for author in group["authors"]] == [
        "email",
        "email",
    ]
    assert group["authors"][1]["contacts"][0] == {
        "kind": "email",
        "label": "Email:",
        "value": "grace@example.test, corresponding author",
        "target": "mailto:grace@example.test,%20corresponding%20author",
    }
    assert [item["marker"] for item in group["affiliations"]] == ["1", "2"]
    assert [item["text"] for item in group["affiliations"]] == [
        "Analytical Engines Institute",
        "Compiler Laboratory",
    ]
    flow = group["creator_flow"]
    assert flow["creator_count"] == 2
    assert flow["slot_count"] == 6
    assert [creator["ordinal"] for creator in flow["creators"]] == [0, 1]
    assert [creator["author_id"] for creator in flow["creators"]] == [
        author["author_id"] for author in group["authors"]
    ]
    assert [
        [slot["kind"] for slot in creator["slots"]]
        for creator in flow["creators"]
    ] == [
        ["author", "contact", "affiliation"],
        ["author", "contact", "affiliation"],
    ]
    assert [
        creator["slots"][1]["contact_index"]
        for creator in flow["creators"]
    ] == [0, 0]
    assert [
        creator["slots"][2]["affiliation_id"]
        for creator in flow["creators"]
    ] == [
        affiliation["affiliation_id"] for affiliation in group["affiliations"]
    ]
    assert all(
        name not in " ".join(str(block.payload) for block in document.blocks)
        for name in ("Ada Lovelace", "Grace Hopper")
    )
    decoded = rich_document_from_document(rich_document_to_document(document))
    assert source_front_matter(decoded) == front_matter
    assert decoded.document_digest == document.document_digest


def test_html_front_matter_ignores_known_pubnotes_and_empty_contact_scaffolding(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article>
              <h1 class="ltx_title ltx_title_document">Paper
                <span class="ltx_pubnotes"><span class="ltx_pubnote
                  ltx_role_thanks"><span class="ltx_note_name">Facilities:</span>
                  Telescope</span></span>
              </h1>
              <span class="ltx_pubnotes ltx_pubnotes_meta">
                <span class="ltx_pubnote ltx_role_software">Pipeline</span>
              </span>
              <div class="ltx_authors" id="authors">
                <span class="ltx_creator ltx_role_author" id="creator">
                  <span class="ltx_annotated_personname">
                    <span class="ltx_personname" id="person">Ada Author</span>
                    <span class="ltx_contact ltx_role_orcid">
                      <a class="ltx_ref ltx_orcid"
                         href="https://orcid.org/0000-0001-2345-6789"></a>
                    </span>
                  </span>
                  <span class="ltx_author_notes"><span
                    class="ltx_author_notes_content">
                    <span class="ltx_contact ltx_role_affiliation" id="aff">
                      <span class="ltx_contact_name">Affiliation:</span>
                      Institute
                    </span>
                    <span class="ltx_contact ltx_role_email">
                      <span class="ltx_contact_name">Email:</span>
                      <a href="mailto:"></a>
                    </span>
                  </span></span>
                </span>
              </div>
              <p>Body.</p>
            </article>
            """,
            SourceFormat.HTML,
        )
    )

    assert document.blocks[0].payload["text"] == "Paper"
    assert all(
        "Facilities" not in str(block.payload)
        and "Pipeline" not in str(block.payload)
        for block in document.blocks
    )
    entry = source_front_matter(document)["entries"][0]
    assert [author["name"] for author in entry["authors"]] == ["Ada Author"]
    assert entry["authors"][0]["orcid"] == "0000-0001-2345-6789"
    assert entry["authors"][0]["contacts"] == ()
    assert [item["text"] for item in entry["affiliations"]] == ["Institute"]
    assert [
        slot["kind"] for slot in entry["creator_flow"]["creators"][0]["slots"]
    ] == ["author", "affiliation"]


def test_html_source_front_matter_preserves_creator_occurrence_flow(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><div class="ltx_authors" id="authors">
              <span class="ltx_creator ltx_role_author" id="C0">
                <span class="ltx_personname" id="P0">Ada<sup>1</sup></span>
                <span class="ltx_contact ltx_role_email" id="E0">a@test</span>
                <span class="ltx_contact ltx_role_homepage" id="H0">site.test</span>
                <span class="ltx_contact ltx_role_affiliation" id="A10">
                  <sup>1</sup>Shared Institute</span>
              </span>
              <span class="ltx_creator ltx_role_author" id="C1">
                <span class="ltx_personname" id="P1">Grace<sup>2</sup></span>
                <span class="ltx_contact ltx_role_email" id="E1">g@test</span>
                <span class="ltx_contact ltx_role_affiliation" id="A11">
                  <sup>1</sup>Shared Institute</span>
              </span>
              <span class="ltx_creator ltx_role_author" id="C2">
                <span class="ltx_personname" id="P2">Lin<sup>3</sup></span>
                <span class="ltx_contact ltx_role_email" id="E2">l@test</span>
                <span class="ltx_contact ltx_role_affiliation" id="A12">
                  <sup>1</sup>Shared Institute</span>
                <span class="ltx_contact ltx_role_affiliation" id="A20">
                  <sup>2</sup>Compiler Laboratory</span>
                <span class="ltx_contact ltx_role_affiliation" id="A30">
                  <sup>3</sup>Physics Department</span>
              </span>
              <span class="ltx_creator ltx_role_author" id="C3">
                <span class="ltx_personname" id="P3">No Aff<sup>1</sup></span>
                <span class="ltx_contact ltx_role_phone" id="T3">+1 555</span>
              </span>
            </div></article>
            """,
            SourceFormat.HTML,
        )
    )

    entry = source_front_matter(document)["entries"][0]
    flow = entry["creator_flow"]
    assert flow["creator_count"] == 4
    assert flow["slot_count"] == 14
    assert [author["markers"] for author in entry["authors"]] == [
        ("1",),
        ("2",),
        ("3",),
        ("1",),
    ]
    assert [creator["locator"]["source_id"] for creator in flow["creators"]] == [
        "C0",
        "C1",
        "C2",
        "C3",
    ]
    assert [
        [slot["kind"] for slot in creator["slots"]]
        for creator in flow["creators"]
    ] == [
        ["author", "contact", "contact", "affiliation"],
        ["author", "contact", "affiliation"],
        ["author", "contact", "affiliation", "affiliation", "affiliation"],
        ["author", "contact"],
    ]
    assert [
        [
            slot["contact_index"]
            for slot in creator["slots"]
            if slot["kind"] == "contact"
        ]
        for creator in flow["creators"]
    ] == [[0, 1], [0], [0], [0]]
    affiliation_ids = [
        affiliation["affiliation_id"] for affiliation in entry["affiliations"]
    ]
    assert [
        [
            slot["affiliation_id"]
            for slot in creator["slots"]
            if slot["kind"] == "affiliation"
        ]
        for creator in flow["creators"]
    ] == [
        [affiliation_ids[0]],
        [affiliation_ids[0]],
        affiliation_ids,
        [],
    ]
    assert flow["creators"][1]["slots"][2]["affiliation_id"] == (
        affiliation_ids[0]
    )
    assert entry["authors"][1]["markers"] == ("2",)
    assert [slot["locator"]["source_id"] for slot in flow["creators"][2]["slots"]] == [
        "P2",
        "E2",
        "A12",
        "A20",
        "A30",
    ]
    assert source_front_matter(
        rich_document_from_document(rich_document_to_document(document))
    ) == source_front_matter(document)
    digest_protected = rich_document_to_document(document)
    digest_entry = digest_protected["metadata"][SOURCE_FRONT_MATTER_METADATA_KEY][
        "entries"
    ][0]
    first_affiliation_slot = next(
        slot
        for slot in digest_entry["creator_flow"]["creators"][0]["slots"]
        if slot["kind"] == "affiliation"
    )
    first_affiliation_slot["affiliation_id"] = affiliation_ids[1]
    validate_source_fidelity_metadata(
        digest_protected["metadata"],
        blocks=document.blocks,
        source=document.source,
    )
    with pytest.raises(ValueError, match="digest"):
        rich_document_from_document(digest_protected)


def test_html_article_flow_preserves_acknowledgement_and_visible_fallbacks(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><h1>Paper</h1>
              <div class="ltx_classification">
                <h6 class="ltx_title_classification">pacs</h6>
                <span id="pacs-value">Dark energy.</span>
              </div>
              <div id="custom-flow">Visible <em>fallback</em> content.</div>
              <div class="ltx_acknowledgements" id="ACK">
                <h6 class="ltx_title_acknowledgements">Acknowledgements.</h6>
                Direct authored <span id="ACK.1"><em>complete</em> body.</span>
              </div>
            </article>
            """,
            SourceFormat.HTML,
        )
    )

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.HEADING,
        RichBlockKind.HEADING,
        RichBlockKind.PARAGRAPH,
        RichBlockKind.PARAGRAPH,
        RichBlockKind.HEADING,
        RichBlockKind.PARAGRAPH,
    ]
    assert [
        block.payload["text"]
        for block in document.blocks
        if block.kind is RichBlockKind.PARAGRAPH
    ] == [
        "Dark energy.",
        "Visible fallback content.",
        "Direct authored complete body.",
    ]
    acknowledgement = document.blocks[-1]
    assert acknowledgement.locator.source_id == "ACK.1"
    assert acknowledgement.locator.selector == "#ACK.1"
    classification = document.blocks[1]
    assert classification.payload == {"text": "pacs", "level": 6}
    assert _source_presentation_block(document, classification.block_id)[
        "roles"
    ] == ("classification",)
    assert _source_target(document, "ACK.1")["block_id"] == (
        acknowledgement.block_id
    )


def test_html_source_presentation_preserves_roles_inline_math_and_overlapping_marks(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><h1 id="title"><strong>Paper</strong></h1>
              <div class="ltx_classification">
                <h6 class="ltx_title ltx_title_classification"><em>Classification</em></h6>
                <span id="classification-value">Dark energy.</span>
              </div>
              <h3 class="ltx_title_subsection" id="S2.SS1">
                Curvature-<math alttext="\\Lambda"><semantics><mi>Lambda</mi>
                  <annotation encoding="application/x-tex">\\Lambda</annotation>
                </semantics></math>-Cold Dark Matter
              </h3>
              <p id="P1">A <a href="#bib.bib1">linked
                <strong>volume <em>38</em></strong></a> follows.</p>
              <ul id="L1"><li id="L1.i1"><em>Marked item</em></li></ul>
              <div class="ltx_acknowledgements" id="ACK">
                <h6 class="ltx_title_acknowledgements">Acknowledgements.</h6>
                <p id="ACK.1">Body.</p>
              </div>
            </article>
            """,
            SourceFormat.HTML,
        )
    )

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.HEADING,
        RichBlockKind.HEADING,
        RichBlockKind.PARAGRAPH,
        RichBlockKind.HEADING,
        RichBlockKind.PARAGRAPH,
        RichBlockKind.LIST,
        RichBlockKind.HEADING,
        RichBlockKind.PARAGRAPH,
    ]
    classification = document.blocks[1]
    assert classification.payload == {"text": "Classification", "level": 6}
    assert _source_presentation_block(document, classification.block_id)[
        "roles"
    ] == ("classification",)
    assert document.blocks[2].payload["text"] == "Dark energy."

    subsection = document.blocks[3]
    assert subsection.payload["text"] == "Curvature-Lambda-Cold Dark Matter"
    heading_view = _source_presentation_field(document, subsection, "text")
    math_span = next(
        span for span in heading_view["inline_spans"] if span["kind"] == "math"
    )
    assert math_span["text"] == "Lambda"
    assert math_span["tex"] == "\\Lambda"
    assert "\\Lambda" not in subsection.payload["text"]

    paragraph = document.blocks[4]
    paragraph_view = _source_presentation_field(document, paragraph, "text")
    links = [
        span for span in paragraph_view["inline_spans"] if span["kind"] == "link"
    ]
    assert {span["target"] for span in links} == {"#bib.bib1"}
    assert "".join(span["text"] for span in links) == "linked volume 38"
    assert paragraph_view["marks"] == (
        {"kind": "strong", "start": 9, "end": 18},
        {"kind": "emphasis", "start": 16, "end": 18},
    )
    list_view = _source_presentation_field(
        document,
        document.blocks[5],
        "list_item",
        item_index=0,
    )
    assert list_view["marks"] == (
        {"kind": "emphasis", "start": 0, "end": 11},
    )
    acknowledgement = document.blocks[6]
    assert acknowledgement.payload["level"] == 2
    assert _source_presentation_block(document, acknowledgement.block_id)[
        "roles"
    ] == ("acknowledgements",)

    presentation = source_presentation(document)
    assert presentation is not None
    assert presentation["schema_version"] == SOURCE_PRESENTATION_SCHEMA
    assert presentation["classifications"] == (
        {
            "classification_id": presentation["classifications"][0][
                "classification_id"
            ],
            "locator": presentation["classifications"][0]["locator"],
            "heading_block_id": classification.block_id,
            "value_block_ids": (document.blocks[2].block_id,),
            "composition": "inline",
            "separator": ": ",
            "separator_source": "latexml_ar5iv_classification_after",
        },
    )
    assert presentation["classifications"][0]["classification_id"].startswith(
        "classification-"
    )
    assert source_presentation(
        rich_document_from_document(rich_document_to_document(document))
    ) == presentation


def test_html_latexml_special_headings_use_semantic_section_hierarchy(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><h1 id="title">Paper</h1>
              <div class="ltx_abstract" id="abstract">
                <h6 class="ltx_title ltx_title_abstract" id="abstract-title">
                  Abstract</h6><p>Summary.</p>
              </div>
              <section id="S2"><h2 id="S2-title">Section</h2>
                <div class="ltx_acknowledgements" id="ack-section">
                  <h6 id="ack-section-title">Acknowledgements.</h6><p>Body.</p>
                </div>
              </section>
              <div class="ltx_acknowledgements" id="ack-root">
                <h6 class="ltx_title_acknowledgements" id="ack-root-title">
                  Root acknowledgements.</h6><p>Body.</p>
              </div>
              <section id="S3"><h2 id="S3-title">Second section</h2>
                <section id="S3.SS1"><h3 id="S3.SS1-title">Nested section</h3>
                  <div class="ltx_acknowledgements" id="ack-nested">
                    <h6 id="ack-nested-title">Nested acknowledgements.</h6>
                    <p>Body.</p>
                  </div>
                  <h6 id="ordinary">Ordinary h6</h6>
                  <h6 id="literal">Abstract</h6>
                  <h6 class="ltx_title_unknown" id="unknown">
                    Acknowledgements</h6>
                </section>
              </section>
              <div class="ltx_classification">
                <h6 class="ltx_title_classification" id="classification">
                  Classification</h6><span>Value.</span>
              </div>
              <h2 id="references">References</h2>
            </article>
            """,
            SourceFormat.HTML,
        )
    )

    headings = {
        block.locator.source_id: block
        for block in document.blocks
        if block.kind is RichBlockKind.HEADING
    }
    expected = {
        "abstract-title": (2, ("abstract",)),
        "ack-section-title": (3, ("acknowledgements",)),
        "ack-root-title": (2, ("acknowledgements",)),
        "ack-nested-title": (4, ("acknowledgements",)),
        "ordinary": (6, ()),
        "literal": (6, ()),
        "unknown": (6, ()),
        "classification": (6, ("classification",)),
    }
    for source_id, (level, roles) in expected.items():
        block = headings[source_id]
        assert block.payload["level"] == level
        assert (
            _source_presentation_block(document, block.block_id)["roles"]
            == roles
        )

    title_path = headings["title"].section_path
    assert headings["abstract-title"].section_path[:-1] == title_path
    assert headings["ack-section-title"].section_path[:-1] == headings[
        "S2-title"
    ].section_path
    assert headings["ack-root-title"].section_path[:-1] == title_path
    assert headings["ack-nested-title"].section_path[:-1] == headings[
        "S3.SS1-title"
    ].section_path
    assert headings["references"].section_path[:-1] == title_path
    assert source_presentation(
        rich_document_from_document(rich_document_to_document(document))
    ) == source_presentation(document)


def test_html_semantic_wrapper_roles_only_its_unique_authored_title(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><h1>Paper</h1>
              <div class="ltx_abstract">
                <h6 id="abstract-title">Abstract</h6>
                <div><h4 id="abstract-body-heading">Body heading</h4></div>
              </div>
              <section><h2>Section</h2>
                <div class="ltx_acknowledgements">
                  <h6 id="ack-title">Acknowledgements</h6>
                  <div><h5 id="ack-body-heading">Body heading</h5></div>
                </div>
              </section>
            </article>
            """,
            SourceFormat.HTML,
        )
    )
    headings = {
        block.locator.source_id: block
        for block in document.blocks
        if block.kind is RichBlockKind.HEADING and block.locator.source_id
    }
    assert headings["abstract-title"].payload["level"] == 2
    assert headings["ack-title"].payload["level"] == 3
    assert headings["abstract-body-heading"].payload["level"] == 4
    assert headings["ack-body-heading"].payload["level"] == 5
    assert _source_presentation_block(
        document,
        headings["abstract-title"].block_id,
    )["roles"] == ("abstract",)
    assert _source_presentation_block(
        document,
        headings["ack-title"].block_id,
    )["roles"] == ("acknowledgements",)
    assert _source_presentation_block(
        document,
        headings["abstract-body-heading"].block_id,
    )["roles"] == ()
    assert _source_presentation_block(
        document,
        headings["ack-body-heading"].block_id,
    )["roles"] == ()


def test_html_semantic_wrapper_neutralizes_multiple_eligible_titles(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><div class="ltx_abstract">
              <h6>First title</h6><h5>Second title</h5>
            </div></article>
            """,
            SourceFormat.HTML,
        )
    )
    assert [block.payload["text"] for block in document.blocks] == [
        "First title",
        "Second title",
    ]
    assert "source_presentation" in _diagnostic_categories(document)


@pytest.mark.parametrize(
    "markup",
    (
        """
        <div class="ltx_abstract"><h6 class="ltx_title_acknowledgements">
          Conflicting convention</h6></div>
        """,
        """
        <div class="ltx_acknowledgements">
          <div class="ltx_acknowledgements"><h6>Nested convention</h6></div>
        </div>
        """,
        """
        <section><h6>Parent at maximum depth</h6>
          <div class="ltx_acknowledgements"><h6>Unrepresentable child</h6></div>
        </section>
        """,
        """
        <section><div class="ltx_acknowledgements">
          <h6>Missing authored parent</h6></div></section>
        """,
        """
        <section><h2>First parent</h2><h3>Second parent</h3>
          <div class="ltx_acknowledgements"><h6>Ambiguous parent</h6></div>
        </section>
        """,
    ),
)
def test_html_latexml_special_heading_ambiguity_degrades_locally(tmp_path, markup):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            f"<article><h1>Paper</h1>{markup}</article>".encode(),
            SourceFormat.HTML,
        )
    )
    assert document.blocks[0].payload["text"] == "Paper"
    assert "source_presentation" in _diagnostic_categories(document)


def test_source_presentation_semantic_heading_roles_are_validated_and_digested(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><h1>Paper</h1>
              <div class="ltx_abstract"><h6 id="abstract-title">Abstract</h6>
                <p>Summary.</p></div>
              <h6 id="ordinary">Ordinary heading</h6>
            </article>
            """,
            SourceFormat.HTML,
        )
    )
    original = rich_document_to_document(document)
    presentation = original["metadata"][SOURCE_PRESENTATION_METADATA_KEY]
    block_by_source_id = {
        block["locator"]["source_id"]: block for block in original["blocks"]
    }

    bad_level_binding = json.loads(json.dumps(original))
    ordinary_id = block_by_source_id["ordinary"]["block_id"]
    next(
        entry
        for entry in bad_level_binding["metadata"][
            SOURCE_PRESENTATION_METADATA_KEY
        ]["blocks"]
        if entry["block_id"] == ordinary_id
    )["roles"] = ["abstract"]
    with pytest.raises(ValueError, match="source presentation"):
        validate_source_presentation_metadata(
            bad_level_binding["metadata"],
            blocks=document.blocks,
            source=document.source,
        )
    with pytest.raises(ValueError, match="source presentation"):
        rich_document_from_document(bad_level_binding)

    bad_acknowledgement_level = json.loads(json.dumps(original))
    title_id = original["blocks"][0]["block_id"]
    next(
        entry
        for entry in bad_acknowledgement_level["metadata"][
            SOURCE_PRESENTATION_METADATA_KEY
        ]["blocks"]
        if entry["block_id"] == title_id
    )["roles"] = ["acknowledgements"]
    with pytest.raises(ValueError, match="source presentation"):
        validate_source_presentation_metadata(
            bad_acknowledgement_level["metadata"],
            blocks=document.blocks,
            source=document.source,
        )

    digest_protected = json.loads(json.dumps(original))
    abstract_id = block_by_source_id["abstract-title"]["block_id"]
    next(
        entry
        for entry in digest_protected["metadata"][
            SOURCE_PRESENTATION_METADATA_KEY
        ]["blocks"]
        if entry["block_id"] == abstract_id
    )["roles"] = ["acknowledgements"]
    validate_source_presentation_metadata(
        digest_protected["metadata"],
        blocks=document.blocks,
        source=document.source,
    )
    with pytest.raises(ValueError, match="digest"):
        rich_document_from_document(digest_protected)

    assert presentation == rich_document_to_document(
        rich_document_from_document(original)
    )["metadata"][SOURCE_PRESENTATION_METADATA_KEY]


def test_html_source_presentation_preserves_table_order_rich_cells_and_spans(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article>
              <figure class="ltx_table" id="T-before">
                <figcaption class="ltx_caption" style="text-align:start"><em>Before</em>
                  <a href="#bib.bib17">reference</a>
                  <math alttext="1\\sigma"><semantics><mn>1 sigma</mn>
                    <annotation encoding="application/x-tex">1\\sigma</annotation>
                  </semantics></math>.</figcaption>
                <table><tr><th id="H1" class="ltx_align_left ltx_border_t"
                  colspan="2"><strong>Estimate</strong></th></tr>
                  <tr><td id="C1" class="ltx_align_center ltx_border_b"
                    rowspan="2"><math alttext="0.686^{+0.039}_{-0.039}">
                    <semantics><mn>0.686 +0.039 -0.039</mn><annotation
                      encoding="application/x-tex">0.686^{+0.039}_{-0.039}</annotation>
                    </semantics></math></td><td>A</td></tr>
                  <tr><td><em>B</em></td></tr>
                </table>
              </figure>
              <figure class="ltx_table" id="T-after">
                <table><tr><td>Only</td></tr></table>
                <figcaption class="ltx_caption ltx_centering">After.</figcaption>
              </figure>
            </article>
            """,
            SourceFormat.HTML,
        )
    )

    before, after = document.blocks
    assert before.payload == {
        "headers": ("Estimate", ""),
        "rows": (
            ("0.686 +0.039 -0.039", "A"),
            ("", "B"),
        ),
        "caption": "Before reference 1 sigma.",
    }
    caption = _source_presentation_field(document, before, "caption")
    assert next(
        span for span in caption["inline_spans"] if span["kind"] == "link"
    )["target"] == "#bib.bib17"
    assert next(
        span for span in caption["inline_spans"] if span["kind"] == "math"
    )["tex"] == "1\\sigma"
    assert caption["marks"] == (
        {"kind": "emphasis", "start": 0, "end": 6},
    )
    header_view = _source_presentation_field(
        document,
        before,
        "table_header",
        column_index=0,
    )
    assert header_view["marks"] == (
        {"kind": "strong", "start": 0, "end": 8},
    )
    rich_cell = _source_presentation_field(
        document,
        before,
        "table_cell",
        row_index=0,
        column_index=0,
    )
    assert rich_cell["text"] == before.payload["rows"][0][0]
    assert rich_cell["inline_spans"][0]["tex"] == "0.686^{+0.039}_{-0.039}"
    marked_cell = _source_presentation_field(
        document,
        before,
        "table_cell",
        row_index=1,
        column_index=1,
    )
    assert marked_cell["marks"] == (
        {"kind": "emphasis", "start": 0, "end": 1},
    )
    table_presentation = next(
        item
        for item in source_presentation(document)["tables"]
        if item["block_id"] == before.block_id
    )
    assert set(table_presentation) == {"block_id", "cells"}
    assert len(table_presentation["cells"]) == 4
    header_geometry, value_geometry = table_presentation["cells"][:2]
    assert {
        key: header_geometry[key]
        for key in ("row_index", "column_index", "row_span", "column_span", "kind")
    } == {
        "row_index": 0,
        "column_index": 0,
        "row_span": 1,
        "column_span": 2,
        "kind": "header",
    }
    assert header_geometry["locator"]["source_id"] == "H1"
    assert header_geometry["locator"]["selector"] == "#H1"
    assert header_geometry["horizontal_alignment"] == "left"
    assert header_geometry["horizontal_alignment_sources"] == (
        "class:ltx_align_left",
    )
    assert header_geometry["rule_edges"] == (
        {"edge": "top", "source": "class:ltx_border_t"},
    )
    assert {
        key: value_geometry[key]
        for key in ("row_index", "column_index", "row_span", "column_span", "kind")
    } == {
        "row_index": 1,
        "column_index": 0,
        "row_span": 2,
        "column_span": 1,
        "kind": "data",
    }
    assert value_geometry["locator"]["source_id"] == "C1"
    assert value_geometry["locator"]["selector"] == "#C1"
    assert value_geometry["horizontal_alignment"] == "center"
    assert value_geometry["rule_edges"] == (
        {"edge": "bottom", "source": "class:ltx_border_b"},
    )
    assert _source_presentation_caption(document, before) == {
        "block_id": before.block_id,
        "kind": "table",
        "placement": "before_content",
        "alignment": "start",
        "alignment_sources": ("style:text-align:start",),
    }
    assert _source_presentation_caption(document, after) == {
        "block_id": after.block_id,
        "kind": "table",
        "placement": "after_content",
        "alignment": "center",
        "alignment_sources": ("class:ltx_centering",),
    }


def test_html_source_presentation_groups_exact_classification_flow_only(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article>
              <div class="ltx_classification" id="C1">
                <h6 class="ltx_title ltx_title_classification">
                  <strong>Type</strong></h6>
                <span id="C1.V1">First value.</span>
                <p id="C1.V2">Second <em>value</em>.</p>
              </div>
              <h6 id="ordinary">Ordinary heading</h6>
              <p id="ordinary-value">Adjacent prose.</p>
              <div class="ltx_classification" id="empty">
                <h6 class="ltx_title ltx_title_classification">Empty</h6>
              </div>
              <div class="ltx_classification" id="ambiguous">
                <h6 class="ltx_title ltx_title_classification">One</h6>
                <h6 class="ltx_title ltx_title_classification">Two</h6>
                <span>Value.</span>
              </div>
              <div class="ltx_classification" id="unknown-convention">
                <h6>Unknown convention</h6><span>Unknown value.</span>
              </div>
              <div class="ltx_classification" id="nested-outer">
                <h6 class="ltx_title ltx_title_classification">Outer</h6>
                <div class="ltx_classification" id="nested-inner">
                  <h6 class="ltx_title ltx_title_classification">Inner</h6>
                  <span>Nested value.</span>
                </div>
              </div>
            </article>
            """,
            SourceFormat.HTML,
        )
    )

    presentation = source_presentation(document)
    assert presentation is not None
    assert len(presentation["classifications"]) == 1
    relation = presentation["classifications"][0]
    heading = next(
        block
        for block in document.blocks
        if block.kind is RichBlockKind.HEADING
        and block.payload["text"] == "Type"
    )
    values = [
        block
        for block in document.blocks
        if block.kind is RichBlockKind.PARAGRAPH
        and block.payload["text"] in {"First value.", "Second value."}
    ]
    assert relation == {
        "classification_id": relation["classification_id"],
        "locator": {
            "source_format": "html",
            "line_start": 3,
            "column_start": 15,
            "line_end": 3,
            "column_end": 15,
            "selector": "#C1",
            "source_id": "C1",
        },
        "heading_block_id": heading.block_id,
        "value_block_ids": tuple(block.block_id for block in values),
        "composition": "inline",
        "separator": ": ",
        "separator_source": "latexml_ar5iv_classification_after",
    }
    assert [block.payload["text"] for block in values] == [
        "First value.",
        "Second value.",
    ]
    relation_block_ids = {
        relation["heading_block_id"],
        *relation["value_block_ids"],
    }
    assert all(
        block.block_id not in relation_block_ids
        for block in document.blocks
        if block.payload.get("text")
        in {
            "Ordinary heading",
            "Adjacent prose.",
            "Empty",
            "One",
            "Two",
            "Value.",
            "Unknown convention",
            "Unknown value.",
            "Outer",
            "Inner",
            "Nested value.",
        }
    )


def test_html_source_presentation_unifies_figure_and_table_captions(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article>
              <figure class="ltx_figure" id="F-start">
                <figcaption class="ltx_caption" style="text-align:start">
                  Start.</figcaption>
                <img src="start.png">
              </figure>
              <figure class="ltx_figure" id="F-center">
                <img src="center.png">
                <figcaption class="ltx_caption ltx_centering">Center.</figcaption>
              </figure>
              <figure class="ltx_figure" id="F-end">
                <img src="end.png">
                <figcaption style="text-align:end">End.</figcaption>
              </figure>
              <figure class="ltx_table" id="T-start">
                <figcaption style="text-align:start">Start table.</figcaption>
                <table><tr><td>A</td></tr></table>
              </figure>
              <figure class="ltx_table" id="T-center">
                <table><tr><td>B</td></tr></table>
                <figcaption class="ltx_centering">Center table.</figcaption>
              </figure>
              <table id="T-end"><caption style="text-align:end">End table.</caption>
                <tr><td>C</td></tr></table>
              <figure class="ltx_figure" id="F-neutral">
                <img src="neutral.png"><figcaption>Neutral.</figcaption>
              </figure>
            </article>
            """,
            SourceFormat.HTML,
        )
    )

    presentation = source_presentation(document)
    assert presentation is not None
    by_source_id = {block.locator.source_id: block for block in document.blocks}
    captions = {
        block.locator.source_id: _source_presentation_caption(document, block)
        for block in document.blocks
        if block.payload.get("caption")
    }
    assert list(captions) == [
        "F-start",
        "F-center",
        "F-end",
        "T-start",
        "T-center",
        "T-end",
        "F-neutral",
    ]
    assert [captions[key]["alignment"] for key in captions] == [
        "start",
        "center",
        "end",
        "start",
        "center",
        "end",
        None,
    ]
    assert [captions[key]["placement"] for key in captions] == [
        "before_content",
        "after_content",
        "after_content",
        "before_content",
        "after_content",
        "embedded",
        "after_content",
    ]
    assert captions["F-end"]["alignment_sources"] == (
        "style:text-align:end",
    )
    assert captions["F-neutral"]["alignment_sources"] == ()
    assert {
        item["block_id"] for item in presentation["captions"]
    } == {block.block_id for block in by_source_id.values()}

    conflict = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><figure class="ltx_figure" id="conflict">
              <img src="plot.png"><figcaption class="ltx_centering"
                style="text-align:end">Conflict.</figcaption>
            </figure></article>
            """,
            SourceFormat.HTML,
        )
    )
    assert conflict.blocks[0].payload["caption"] == "Conflict."
    assert "caption_presentation" in _diagnostic_categories(conflict)


def test_html_multiple_authored_captions_preserve_source_flow(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><figure class="ltx_figure" id="grouped">
              <div class="ltx_flex_figure"><figure class="ltx_figure_panel">
                <img src="one.png">
              </figure></div>
              <figcaption>Figure 1: One with <math display="inline">
                <annotation encoding="application/x-tex">x_1</annotation>
              </math>.</figcaption>
              <div class="ltx_flex_figure"><figure class="ltx_figure_panel">
                <img src="two.png">
              </figure></div>
              <figcaption>Figure 2: Two.</figcaption>
            </figure></article>
            """,
            SourceFormat.HTML,
        )
    )

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.FIGURE,
        RichBlockKind.PARAGRAPH,
        RichBlockKind.FIGURE,
        RichBlockKind.PARAGRAPH,
    ]
    assert [
        block.payload.get("target")
        for block in document.blocks
        if block.kind is RichBlockKind.FIGURE
    ] == ["one.png", "two.png"]
    assert [
        block.payload["text"]
        for block in document.blocks
        if block.kind is RichBlockKind.PARAGRAPH
    ] == ["Figure 1: One with x_1.", "Figure 2: Two."]
    first_caption = document.blocks[1]
    assert any(
        span["kind"] == "math" and span["tex"] == "x_1"
        for span in first_caption.payload["inline_spans"]
    )
    assert "figure_layout" in _diagnostic_categories(document)
    diagnostics = document.metadata[DOCUMENT_DIAGNOSTICS_METADATA_KEY]
    assert diagnostics["visible_content"]["unaccounted"] == 0


def test_html_list_figure_with_multiple_captions_preserves_source_flow(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><ul><li><figure id="grouped">
              <img src="one.png"><figcaption>Figure 1: One.</figcaption>
              <img src="two.png"><figcaption>Figure 2: Two.</figcaption>
            </figure></li></ul></article>
            """,
            SourceFormat.HTML,
        )
    )

    assert [
        block.payload.get("target")
        for block in document.blocks
        if block.kind is RichBlockKind.FIGURE
    ] == ["one.png", "two.png"]
    assert "Figure 1: One." in str(document.blocks)
    assert "Figure 2: Two." in str(document.blocks)


def test_html_caption_alignment_accepts_case_insensitive_important(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article>
              <figure id="F"><img src="plot.png"><figcaption
                style="text-align:center !IMPORTANT">Figure.</figcaption></figure>
              <table id="T"><caption style="text-align:end !ImPoRtAnT">
                Table.</caption><tr><td>A</td></tr></table>
            </article>
            """,
            SourceFormat.HTML,
        )
    )
    blocks = {block.locator.source_id: block for block in document.blocks}
    assert _source_presentation_caption(document, blocks["F"])["alignment"] == (
        "center"
    )
    assert _source_presentation_caption(document, blocks["T"])["alignment"] == (
        "end"
    )


def test_html_source_presentation_preserves_authored_table_cell_rules(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article>
              <table id="booktabs"><tr>
                <th class="ltx_align_left ltx_border_t">Left</th>
                <th class="ltx_align_center ltx_border_t">Center</th></tr>
                <tr><td class="ltx_border_b" style="text-align:end">End</td>
                  <td class="ltx_border_b">Neutral</td></tr>
              </table>
              <table id="boxed"><tr><td class="ltx_border_t ltx_border_r
                ltx_border_b ltx_border_l">Boxed</td></tr></table>
              <table id="merged"><tr><td colspan="2" class="ltx_align_center
                ltx_border_t ltx_border_b">Merged</td></tr>
                <tr><td>A</td><td>B</td></tr></table>
            </article>
            """,
            SourceFormat.HTML,
        )
    )

    presentation = source_presentation(document)
    assert presentation is not None
    tables = {
        block.locator.source_id: next(
            item
            for item in presentation["tables"]
            if item["block_id"] == block.block_id
        )
        for block in document.blocks
    }
    booktabs = tables["booktabs"]["cells"]
    assert [cell["horizontal_alignment"] for cell in booktabs] == [
        "left",
        "center",
        "end",
        None,
    ]
    assert [cell["rule_edges"] for cell in booktabs] == [
        ({"edge": "top", "source": "class:ltx_border_t"},),
        ({"edge": "top", "source": "class:ltx_border_t"},),
        ({"edge": "bottom", "source": "class:ltx_border_b"},),
        ({"edge": "bottom", "source": "class:ltx_border_b"},),
    ]
    assert booktabs[2]["horizontal_alignment_sources"] == (
        "style:text-align:end",
    )
    assert tables["boxed"]["cells"][0]["rule_edges"] == (
        {"edge": "top", "source": "class:ltx_border_t"},
        {"edge": "right", "source": "class:ltx_border_r"},
        {"edge": "bottom", "source": "class:ltx_border_b"},
        {"edge": "left", "source": "class:ltx_border_l"},
    )
    merged = tables["merged"]["cells"]
    assert len(merged) == 3
    assert {
        key: merged[0][key]
        for key in ("row_index", "column_index", "row_span", "column_span")
    } == {
        "row_index": 0,
        "column_index": 0,
        "row_span": 1,
        "column_span": 2,
    }
    assert merged[0]["rule_edges"] == (
        {"edge": "top", "source": "class:ltx_border_t"},
        {"edge": "bottom", "source": "class:ltx_border_b"},
    )
    assert all(
        not (cell["row_index"] == 0 and cell["column_index"] == 1)
        for cell in merged
    )

    for markup in (
        '<td class="ltx_align_left" style="text-align:center">Conflict</td>',
        '<td class="ltx_border_t ltx_border_tt">Conflict</td>',
    ):
        degraded = RichDocumentParserService(repository).parse_source(
            _store(
                repository,
                f"<article><table><tr>{markup}</tr></table></article>".encode(),
                SourceFormat.HTML,
            )
        )
        assert degraded.blocks[0].payload["rows"] == (("Conflict",),)
        assert "table_presentation" in _diagnostic_categories(degraded)


def test_source_presentation_rejects_tamper_and_remains_optional(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
                <article>
                  <div class="ltx_classification" id="C1">
                    <h6 class="ltx_title ltx_title_classification">
                      <strong>Label</strong></h6>
                    <span id="C1.V1">Value one.</span>
                    <p id="C1.V2">Value two.</p>
                  </div>
                  <p id="P1"><a href="#target"><strong>Alpha</strong></a>
                  beta.</p>
                  <table id="T1"><caption class="ltx_centering">Table.</caption>
                    <tr><td id="T1.C0" colspan="2">Wide</td></tr>
                    <tr><td id="T1.C1">Left</td>
                    <td id="T1.C2">Right</td></tr></table>
                  <figure id="F1"><img src="plot.png">
                    <figcaption style="text-align:start">Figure.</figcaption>
                  </figure>
            </article>
            """,
            SourceFormat.HTML,
        )
    )
    original = rich_document_to_document(document)
    presentation = original["metadata"][SOURCE_PRESENTATION_METADATA_KEY]
    paragraph_entry = next(
        entry
        for entry in presentation["blocks"]
        if next(
            block
            for block in original["blocks"]
            if block["block_id"] == entry["block_id"]
        )["locator"]["source_id"]
        == "P1"
    )
    paragraph_index = presentation["blocks"].index(paragraph_entry)
    cases = []

    unknown = json.loads(json.dumps(original))
    unknown["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["extra"] = True
    cases.append(unknown)
    duplicate = json.loads(json.dumps(original))
    duplicate["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["blocks"].append(
        duplicate["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["blocks"][0]
    )
    cases.append(duplicate)
    bad_range = json.loads(json.dumps(original))
    bad_range["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["blocks"][
        paragraph_index
    ][
        "fields"
    ][0]["marks"][0]["end"] = 999
    cases.append(bad_range)
    mismatch = json.loads(json.dumps(original))
    mismatch["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["blocks"][
        paragraph_index
    ][
        "fields"
    ][0]["text"] = "Changed"
    cases.append(mismatch)
    empty_link = json.loads(json.dumps(original))
    link_spans = empty_link["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "blocks"
    ][paragraph_index]["fields"][0]["inline_spans"]
    next(span for span in link_spans if span["kind"] == "link")["target"] = ""
    cases.append(empty_link)
    duplicate_field = json.loads(json.dumps(original))
    fields = duplicate_field["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "blocks"
    ][paragraph_index]["fields"]
    fields.append(dict(fields[0]))
    cases.append(duplicate_field)
    unknown_role = json.loads(json.dumps(original))
    unknown_role["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["blocks"][
        paragraph_index
    ]["roles"] = ["display-heading"]
    cases.append(unknown_role)
    conflicting_roles = json.loads(json.dumps(original))
    conflicting_roles["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["blocks"][
        paragraph_index
    ]["roles"] = ["classification", "acknowledgements"]
    cases.append(conflicting_roles)
    overlap = json.loads(json.dumps(original))
    cells = overlap["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["tables"][0][
        "cells"
    ]
    cells.append(dict(cells[0]))
    cases.append(overlap)
    duplicate_cell_id = json.loads(json.dumps(original))
    cells = duplicate_cell_id["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "tables"
    ][0]["cells"]
    cells[1]["locator"]["source_id"] = cells[0]["locator"]["source_id"]
    cells[1]["locator"]["selector"] = cells[0]["locator"]["selector"]
    cases.append(duplicate_cell_id)
    bad_cell_alignment = json.loads(json.dumps(original))
    bad_cell_alignment["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["tables"][
        0
    ]["cells"][0]["horizontal_alignment"] = "middle"
    cases.append(bad_cell_alignment)
    bad_cell_alignment_source = json.loads(json.dumps(original))
    bad_cell_alignment_source["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "tables"
    ][0]["cells"][0]["horizontal_alignment"] = "left"
    bad_cell_alignment_source["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "tables"
    ][0]["cells"][0]["horizontal_alignment_sources"] = [
        "class:ltx_align_center"
    ]
    cases.append(bad_cell_alignment_source)
    bad_rule_source = json.loads(json.dumps(original))
    bad_rule_source["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["tables"][0][
        "cells"
    ][0]["rule_edges"] = [{"edge": "top", "source": "class:unknown"}]
    cases.append(bad_rule_source)
    duplicate_rule = json.loads(json.dumps(original))
    duplicate_rule["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["tables"][0][
        "cells"
    ][0]["rule_edges"] = [
        {"edge": "top", "source": "class:ltx_border_t"},
        {"edge": "top", "source": "class:ltx_border_tt"},
    ]
    cases.append(duplicate_rule)
    covered_cell_style = json.loads(json.dumps(original))
    cells = covered_cell_style["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "tables"
    ][0]["cells"]
    covered = dict(cells[0])
    covered.update(
        {
            "row_index": 0,
            "column_index": 1,
            "row_span": 1,
            "column_span": 1,
            "locator": {
                "source_format": "html",
                "line_start": None,
                "column_start": None,
                "line_end": None,
                "column_end": None,
                "selector": "",
                "source_id": "",
            },
            "horizontal_alignment": "center",
            "horizontal_alignment_sources": ["class:ltx_align_center"],
            "rule_edges": [],
        }
    )
    cells.append(covered)
    cells.sort(key=lambda cell: (cell["row_index"], cell["column_index"]))
    cases.append(covered_cell_style)
    obsolete_table_placement = json.loads(json.dumps(original))
    obsolete_table_placement["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "tables"
    ][0]["caption_placement"] = "before_table"
    cases.append(obsolete_table_placement)
    duplicate_caption = json.loads(json.dumps(original))
    captions = duplicate_caption["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "captions"
    ]
    captions.append(dict(captions[0]))
    cases.append(duplicate_caption)
    reversed_captions = json.loads(json.dumps(original))
    reversed_captions["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "captions"
    ].reverse()
    cases.append(reversed_captions)
    bad_alignment = json.loads(json.dumps(original))
    bad_alignment["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["captions"][0][
        "alignment"
    ] = "middle"
    cases.append(bad_alignment)
    bad_caption_placement = json.loads(json.dumps(original))
    bad_caption_placement["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "captions"
    ][0]["placement"] = "around_content"
    cases.append(bad_caption_placement)
    bad_caption_kind = json.loads(json.dumps(original))
    bad_caption_kind["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["captions"][
        0
    ]["kind"] = "figure"
    cases.append(bad_caption_kind)
    bad_alignment_source = json.loads(json.dumps(original))
    bad_alignment_source["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "captions"
    ][0]["alignment_sources"] = ["style:text-align:end"]
    cases.append(bad_alignment_source)
    unknown_caption_field = json.loads(json.dumps(original))
    unknown_caption_field["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "captions"
    ][0]["extra"] = True
    cases.append(unknown_caption_field)
    bad_separator = json.loads(json.dumps(original))
    bad_separator["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "classifications"
    ][0]["separator"] = " - "
    cases.append(bad_separator)
    bad_classification_identity = json.loads(json.dumps(original))
    bad_classification_identity["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "classifications"
    ][0]["classification_id"] = "classification-guessed"
    cases.append(bad_classification_identity)
    bad_separator_source = json.loads(json.dumps(original))
    bad_separator_source["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "classifications"
    ][0]["separator_source"] = "guessed"
    cases.append(bad_separator_source)
    bad_composition = json.loads(json.dumps(original))
    bad_composition["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "classifications"
    ][0]["composition"] = "adjacent"
    cases.append(bad_composition)
    bad_heading = json.loads(json.dumps(original))
    bad_heading["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "classifications"
    ][0]["heading_block_id"] = paragraph_entry["block_id"]
    cases.append(bad_heading)
    reversed_values = json.loads(json.dumps(original))
    reversed_values["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "classifications"
    ][0]["value_block_ids"].reverse()
    cases.append(reversed_values)
    unknown_value = json.loads(json.dumps(original))
    unknown_value["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "classifications"
    ][0]["value_block_ids"][0] = "block-unknown"
    cases.append(unknown_value)
    duplicate_classification = json.loads(json.dumps(original))
    relations = duplicate_classification["metadata"][
        SOURCE_PRESENTATION_METADATA_KEY
    ]["classifications"]
    relations.append(dict(relations[0]))
    cases.append(duplicate_classification)
    unknown_classification_field = json.loads(json.dumps(original))
    unknown_classification_field["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "classifications"
    ][0]["extra"] = True
    cases.append(unknown_classification_field)

    for value in cases:
        with pytest.raises(ValueError, match="source presentation"):
            validate_source_presentation_metadata(
                value["metadata"],
                blocks=document.blocks,
                source=document.source,
            )
        with pytest.raises(ValueError, match="source presentation"):
            rich_document_from_document(value)

    digest_protected = json.loads(json.dumps(original))
    digest_protected["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["captions"][
        0
    ]["alignment"] = "end"
    digest_protected["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["captions"][
        0
    ]["alignment_sources"] = ["style:text-align:end"]
    validate_source_presentation_metadata(
        digest_protected["metadata"],
        blocks=document.blocks,
        source=document.source,
    )
    with pytest.raises(ValueError, match="digest"):
        rich_document_from_document(digest_protected)

    legacy = dict(document.metadata)
    legacy.pop(SOURCE_PRESENTATION_METADATA_KEY)
    legacy_document = RichDocument(
        source=document.source,
        blocks=document.blocks,
        sections=document.sections,
        assets=document.assets,
        page_map=document.page_map,
        metadata=legacy,
    )
    assert source_presentation(legacy_document) is None
    legacy_encoded = rich_document_to_document(legacy_document)
    legacy_decoded = rich_document_from_document(legacy_encoded)
    assert source_presentation(legacy_decoded) is None
    assert rich_document_to_document(legacy_decoded) == legacy_encoded


def test_html_malformed_structured_authors_fall_back_to_visible_core_flow(tmp_path):
    repository = SourceRepository(tmp_path / "cache")

    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"<article><div class='ltx_authors'>Visible unstructured byline.</div></article>",
            SourceFormat.HTML,
        )
    )
    assert document.blocks[0].payload["text"] == "Visible unstructured byline."
    assert "source_front_matter" in _diagnostic_categories(document)


def test_html_plain_structural_fallback_keeps_other_source_presentation(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article>
              <figure id="T-bad" class="ltx_table">
                <figcaption>Complex table retained as text.</figcaption>
                <span class="ltx_tabular">
                  <span class="ltx_tr"><span class="ltx_td">Value</span></span>
                </span>
                <span class="ltx_tabular">
                  <span class="ltx_tr"><span class="ltx_td">Other</span></span>
                </span>
              </figure>
              <table id="T-good">
                <caption>Ordinary table.</caption>
                <tr><td>Cell</td></tr>
              </table>
            </article>
            """,
            SourceFormat.HTML,
        )
    )

    fallback = next(
        block for block in document.blocks if block.locator.source_id == "T-bad"
    )
    assert fallback.kind is RichBlockKind.PARAGRAPH
    presentation = source_presentation(document)
    assert presentation is not None
    fallback_view = _source_presentation_field(document, fallback, "text")
    assert fallback_view["text"] == fallback.payload["text"]
    ordinary = next(
        block for block in document.blocks if block.locator.source_id == "T-good"
    )
    assert _source_presentation_caption(document, ordinary)["placement"] == (
        "embedded"
    )
    assert "html_structure" in _diagnostic_categories(document)


def test_html_latexml_span_tabular_recovers_safe_table_grid(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article>
              <figure id="S4.T6" class="ltx_table">
                <figcaption class="ltx_caption ltx_centering">
                  Table 6: Member variable stars.
                </figcaption>
                <span class="ltx_transformed_outer">
                  <span class="ltx_transformed_inner">
                    <span class="ltx_tabular ltx_align_middle">
                      <span class="ltx_tr">
                        <span class="ltx_td ltx_align_left ltx_border_tt">No.</span>
                        <span class="ltx_td ltx_align_center ltx_border_tt">
                          <span class="ltx_tabular"><span class="ltx_tr">
                            <span class="ltx_td">Source ID</span>
                          </span></span>
                        </span>
                        <span class="ltx_td ltx_align_center ltx_border_tt">
                          <math alttext="\\alpha"><semantics><mi>alpha</mi>
                          <annotation encoding="application/x-tex">\\alpha</annotation>
                          </semantics></math>
                        </span>
                      </span>
                      <span class="ltx_tr">
                        <span class="ltx_td">01</span>
                        <span class="ltx_td">2014635675271057280</span>
                        <span class="ltx_td">343.608</span>
                      </span>
                    </span>
                  </span>
                </span>
              </figure>
            </article>
            """,
            SourceFormat.HTML,
        )
    )

    assert len(document.blocks) == 1
    table = document.blocks[0]
    assert table.kind is RichBlockKind.TABLE
    assert table.locator.source_id == "S4.T6"
    assert table.payload == {
        "headers": ("No.", "Source ID", "alpha"),
        "rows": (("01", "2014635675271057280", "343.608"),),
        "caption": "Table 6: Member variable stars.",
    }
    presentation = source_presentation(document)
    assert presentation is not None
    cells = presentation["tables"][0]["cells"]
    assert [cell["kind"] for cell in cells] == [
        "header",
        "header",
        "header",
        "data",
        "data",
        "data",
    ]
    assert cells[0]["horizontal_alignment"] == "left"
    assert cells[1]["horizontal_alignment"] == "center"
    assert cells[0]["rule_edges"] == (
        {"edge": "top", "source": "class:ltx_border_tt"},
    )


def test_html_source_notes_preserve_marker_body_owner_and_inline_semantics(
    tmp_path,
):
    image = tmp_path / "figure.png"
    image.write_bytes(b"figure")
    source = tmp_path / "paper.html"
    source.write_text(
        """
        <article><p id="P1">Alpha<span class="ltx_note ltx_role_footnote" id="footnote1">
          <sup class="ltx_note_mark">1</sup><span class="ltx_note_outer">
            <span class="ltx_note_content"><sup class="ltx_note_mark"><span class="ltx_tag ltx_tag_note">1</span></sup>
              First body.
            </span>
          </span></span> omega.</p>
          <table id="T1"><tr><th>Header</th></tr>
            <tr><th id="T1.H1">Value<span class="ltx_note ltx_role_footnote" id="footnote2">
            <sup class="ltx_note_mark">2</sup><span class="ltx_note_outer">
              <span class="ltx_note_content"><sup class="ltx_note_mark"><span class="ltx_tag ltx_tag_note">2</span></sup>
                Table body.
              </span>
            </span></span></th></tr></table>
          <p id="P2">Beta<span class="ltx_note ltx_role_footnote" id="footnote3">
            <sup class="ltx_note_mark">3</sup><span class="ltx_note_outer">
              <span class="ltx_note_content"><sup class="ltx_note_mark"><span class="ltx_tag ltx_tag_note">3</span></sup>
                See <math alttext="x+y"><semantics><mrow><mi>x</mi><mo>+</mo><mi>y</mi></mrow>
                  <annotation encoding="application/x-tex">x+y</annotation></semantics></math>
                and <a href="#F1">Figure</a>.
              </span>
            </span></span> tail.</p>
          <figure id="F1"><img src="figure.png" alt="Figure"></figure>
        </article>
        """,
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        repository.import_path(source)
    )

    assert document.blocks[0].payload["text"] == "Alpha1 omega."
    assert document.blocks[1].payload["headers"] == ("Header",)
    assert document.blocks[1].payload["rows"] == (("Value2",),)
    assert document.blocks[2].payload["text"] == "Beta3 tail."
    serialized_payloads = " ".join(str(block.payload) for block in document.blocks)
    assert "First body." not in serialized_payloads
    assert "Table body." not in serialized_payloads
    assert "See x+y and Figure." not in serialized_payloads

    notes = source_notes(document)
    assert notes is not None
    assert notes["schema_version"] == SOURCE_NOTES_SCHEMA
    assert [note["note_id"] for note in notes["notes"]] == [
        "footnote1",
        "footnote2",
        "footnote3",
    ]
    assert [note["ordinal"] for note in notes["notes"]] == [0, 1, 2]
    assert [note["marker"] for note in notes["notes"]] == ["1", "2", "3"]
    assert [note["body"] for note in notes["notes"]] == [
        "First body.",
        "Table body.",
        "See x+y and Figure.",
    ]
    assert [note["owner_block_id"] for note in notes["notes"]] == [
        document.blocks[0].block_id,
        document.blocks[1].block_id,
        document.blocks[2].block_id,
    ]
    assert [note["owner_locator"]["source_id"] for note in notes["notes"]] == [
        "P1",
        "T1.H1",
        "P2",
    ]
    assert [note["locator"]["source_id"] for note in notes["notes"]] == [
        "footnote1",
        "footnote2",
        "footnote3",
    ]
    assert [note["anchor"]["field"] for note in notes["notes"]] == [
        "text",
        "table_cell",
        "text",
    ]
    for note in notes["notes"]:
        owner = next(
            block
            for block in document.blocks
            if block.block_id == note["owner_block_id"]
        )
        anchor = note["anchor"]
        target = (
            owner.payload["headers"][anchor["column_index"]]
            if anchor["field"] == "table_header"
            else owner.payload["rows"][anchor["row_index"]][
                anchor["column_index"]
            ]
            if anchor["field"] == "table_cell"
            else owner.payload["text"]
        )
        assert target[anchor["start"] : anchor["end"]] == note["marker"]
    assert [span["kind"] for span in notes["notes"][2]["inline_spans"]] == [
        "text",
        "math",
        "text",
        "link",
        "text",
    ]
    assert notes["notes"][2]["inline_spans"][1]["tex"] == "x+y"
    assert notes["notes"][2]["inline_spans"][3]["target"] == "#F1"
    decoded = rich_document_from_document(rich_document_to_document(document))
    assert source_notes(decoded) == notes
    assert decoded.document_digest == document.document_digest


def test_html_source_notes_ignore_bodyless_footnotemark_placeholders(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article>
              <p id="P1">Alpha<span class="ltx_note ltx_role_footnote" id="footnote1">
                <sup class="ltx_note_mark">1</sup><span class="ltx_note_outer">
                  <span class="ltx_note_content"><sup class="ltx_note_mark">1</sup>
                    Authored body.
                  </span>
                </span>
              </span>.</p>
              <table id="T1"><tr><td id="T1.1">Value
                <span class="ltx_note ltx_role_footnotemark" id="T1.1.1">
                  <sup class="ltx_note_mark">a</sup><span class="ltx_note_outer">
                    <span class="ltx_note_content"><sup class="ltx_note_mark">a</sup>
                      <span class="ltx_note_type">footnotemark: </span>
                    </span>
                  </span>
                </span>
              </td></tr></table>
            </article>
            """,
            SourceFormat.HTML,
        )
    )

    notes = source_notes(document)
    table = next(
        block for block in document.blocks if block.kind is RichBlockKind.TABLE
    )

    assert notes is not None
    assert [note["note_id"] for note in notes["notes"]] == ["footnote1"]
    assert table.payload["rows"] == (("Value a",),)
    assert "footnotemark:" not in str(document.metadata)


@pytest.mark.parametrize(
    "markup",
    (
        "<h2>Heading {note}</h2>",
        "<figure><img src='plot.png'><figcaption>Figure {note}</figcaption></figure>",
        "<table><caption>Table {note}</caption><tr><td>A</td></tr></table>",
    ),
)
def test_html_notes_in_titles_and_captions_degrade_locally(tmp_path, markup):
    repository = SourceRepository(tmp_path / "cache")
    note = """
      <span class="ltx_note ltx_role_footnote" id="footnote1">
        <sup class="ltx_note_mark">1</sup><span class="ltx_note_outer">
          <span class="ltx_note_content"><sup class="ltx_note_mark">
            <span class="ltx_tag ltx_tag_note">1</span></sup>Body.</span>
        </span>
      </span>
    """
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            f"<article>{markup.format(note=note)}</article>".encode(),
            SourceFormat.HTML,
        )
    )
    assert document.blocks
    assert "source_notes" in _diagnostic_categories(document)
    assert "Body." in " ".join(
        str(block.payload) for block in document.blocks
    )


def test_html_note_anchors_use_final_normalized_visible_text(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    note = """
      <span class="ltx_note ltx_role_footnote" id="{note_id}">
        <sup class="ltx_note_mark">{marker}</sup><span class="ltx_note_outer">
          <span class="ltx_note_content"><sup class="ltx_note_mark">
            <span class="ltx_tag ltx_tag_note">{marker}</span></sup>{body}</span>
        </span>
      </span>
    """
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            (
                "<article><p id='P'>  "
                + note.format(note_id="n1", marker="1", body="Paragraph body.")
                + " tail.</p><ul><li>  "
                + note.format(note_id="n2", marker="2", body="List body.")
                + " tail.</li></ul><table><tr><td id='C'>  "
                + note.format(note_id="n3", marker="3", body="Table body.")
                + " tail.</td></tr></table></article>"
            ).encode(),
            SourceFormat.HTML,
        )
    )
    notes = source_notes(document)
    assert notes is not None
    assert [item["note_id"] for item in notes["notes"]] == ["n1", "n2", "n3"]
    assert [item["anchor"]["start"] for item in notes["notes"]] == [0, 0, 0]
    for item in notes["notes"]:
        owner = next(
            block for block in document.blocks if block.block_id == item["owner_block_id"]
        )
        anchor = item["anchor"]
        if anchor["field"] == "text":
            target = owner.payload["text"]
        elif anchor["field"] == "list_item":
            target = owner.payload["items"][anchor["item_index"]]["text"]
        else:
            target = owner.payload["rows"][anchor["row_index"]][
                anchor["column_index"]
            ]
        assert target[anchor["start"] : anchor["end"]] == item["marker"]


def test_source_fidelity_metadata_rejects_tampering_and_preserves_legacy(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><h1>Paper</h1><div class="ltx_authors">
              <span class="ltx_creator ltx_role_author">
                <span class="ltx_personname">Author<sup>1</sup>
                  <a class="ltx_ref ltx_orcid" href="https://orcid.org/0000-0001-2345-6789"
                     title="ORCID 0000-0001-2345-6789"></a></span>
                <span class="ltx_author_notes"><span class="ltx_author_notes_content">
                  <span class="ltx_contact ltx_role_email"><span class="ltx_contact_name">Email:</span>
                    <a href="mailto:a@example.test">a@example.test</a></span>
                  <span class="ltx_contact ltx_role_affiliation"><span class="ltx_contact_name">Affiliation:</span>
                    <sup>1</sup> Institute</span>
                </span></span>
              </span></div>
                  <p id="P">Text<span class="ltx_note ltx_role_footnote" id="footnote1">
                    <sup class="ltx_note_mark">1</sup><span class="ltx_note_outer">
                      <span class="ltx_note_content"><sup class="ltx_note_mark"><span class="ltx_tag ltx_tag_note">1</span></sup>
                        Body.</span></span></span> middle
                    <span class="ltx_note ltx_role_footnote" id="footnote2">
                    <sup class="ltx_note_mark">2</sup><span class="ltx_note_outer">
                      <span class="ltx_note_content"><sup class="ltx_note_mark">
                        <span class="ltx_tag ltx_tag_note">2</span></sup>
                        Second <a href="https://example.test/note">body</a>.</span>
                    </span></span>.</p>
            </article>
            """,
            SourceFormat.HTML,
        )
    )
    original = rich_document_to_document(document)
    cases = []
    front_unknown = json.loads(json.dumps(original))
    front_unknown["metadata"][SOURCE_FRONT_MATTER_METADATA_KEY]["extra"] = True
    cases.append(front_unknown)
    bad_block_index = json.loads(json.dumps(original))
    bad_block_index["metadata"][SOURCE_FRONT_MATTER_METADATA_KEY]["entries"][0][
        "block_index"
    ] = len(document.blocks) + 1
    cases.append(bad_block_index)
    bad_orcid = json.loads(json.dumps(original))
    bad_orcid["metadata"][SOURCE_FRONT_MATTER_METADATA_KEY]["entries"][0][
        "authors"
    ][0]["orcid_url"] = "https://orcid.org/0000-0000-0000-0000"
    cases.append(bad_orcid)
    bad_author_marker = json.loads(json.dumps(original))
    bad_author_marker["metadata"][SOURCE_FRONT_MATTER_METADATA_KEY]["entries"][0][
        "authors"
    ][0]["markers"] = ["X"]
    cases.append(bad_author_marker)
    duplicate_author = json.loads(json.dumps(original))
    authors = duplicate_author["metadata"][SOURCE_FRONT_MATTER_METADATA_KEY][
        "entries"
    ][0]["authors"]
    authors.append(dict(authors[0]))
    cases.append(duplicate_author)
    duplicate_affiliation = json.loads(json.dumps(original))
    affiliations = duplicate_affiliation["metadata"][
        SOURCE_FRONT_MATTER_METADATA_KEY
    ]["entries"][0]["affiliations"]
    affiliations.append(dict(affiliations[0]))
    cases.append(duplicate_affiliation)
    flow_unknown = json.loads(json.dumps(original))
    flow_unknown["metadata"][SOURCE_FRONT_MATTER_METADATA_KEY]["entries"][0][
        "creator_flow"
    ]["extra"] = True
    cases.append(flow_unknown)
    bad_creator_count = json.loads(json.dumps(original))
    bad_creator_count["metadata"][SOURCE_FRONT_MATTER_METADATA_KEY]["entries"][
        0
    ]["creator_flow"]["creator_count"] += 1
    cases.append(bad_creator_count)
    bad_slot_count = json.loads(json.dumps(original))
    bad_slot_count["metadata"][SOURCE_FRONT_MATTER_METADATA_KEY]["entries"][0][
        "creator_flow"
    ]["slot_count"] += 1
    cases.append(bad_slot_count)
    duplicate_creator = json.loads(json.dumps(original))
    creators = duplicate_creator["metadata"][SOURCE_FRONT_MATTER_METADATA_KEY][
        "entries"
    ][0]["creator_flow"]["creators"]
    creators.append(dict(creators[0]))
    cases.append(duplicate_creator)
    bad_creator_author = json.loads(json.dumps(original))
    bad_creator_author["metadata"][SOURCE_FRONT_MATTER_METADATA_KEY]["entries"][
        0
    ]["creator_flow"]["creators"][0]["author_id"] = "front-author-unknown"
    cases.append(bad_creator_author)
    bad_creator_provenance = json.loads(json.dumps(original))
    creator_locator = bad_creator_provenance["metadata"][
        SOURCE_FRONT_MATTER_METADATA_KEY
    ]["entries"][0]["creator_flow"]["creators"][0]["locator"]
    creator_locator["source_id"] = "different-creator"
    creator_locator["selector"] = "#different-creator"
    cases.append(bad_creator_provenance)
    reversed_slots = json.loads(json.dumps(original))
    reversed_slots["metadata"][SOURCE_FRONT_MATTER_METADATA_KEY]["entries"][0][
        "creator_flow"
    ]["creators"][0]["slots"].reverse()
    cases.append(reversed_slots)
    bad_contact_index = json.loads(json.dumps(original))
    contact_slot = next(
        slot
        for slot in bad_contact_index["metadata"][
            SOURCE_FRONT_MATTER_METADATA_KEY
        ]["entries"][0]["creator_flow"]["creators"][0]["slots"]
        if slot["kind"] == "contact"
    )
    contact_slot["contact_index"] = 99
    cases.append(bad_contact_index)
    unknown_affiliation_ref = json.loads(json.dumps(original))
    affiliation_slot = next(
        slot
        for slot in unknown_affiliation_ref["metadata"][
            SOURCE_FRONT_MATTER_METADATA_KEY
        ]["entries"][0]["creator_flow"]["creators"][0]["slots"]
        if slot["kind"] == "affiliation"
    )
    affiliation_slot["affiliation_id"] = "front-affiliation-unknown"
    cases.append(unknown_affiliation_ref)
    duplicate_slot_id = json.loads(json.dumps(original))
    slots = duplicate_slot_id["metadata"][SOURCE_FRONT_MATTER_METADATA_KEY][
        "entries"
    ][0]["creator_flow"]["creators"][0]["slots"]
    slots[1]["slot_id"] = slots[0]["slot_id"]
    cases.append(duplicate_slot_id)
    unknown_slot_field = json.loads(json.dumps(original))
    unknown_slot_field["metadata"][SOURCE_FRONT_MATTER_METADATA_KEY]["entries"][
        0
    ]["creator_flow"]["creators"][0]["slots"][0]["extra"] = True
    cases.append(unknown_slot_field)
    omitted_occurrence = json.loads(json.dumps(original))
    omitted_occurrence["metadata"][SOURCE_FRONT_MATTER_METADATA_KEY]["entries"][
        0
    ]["creator_flow"]["creators"][0]["slots"].pop()
    cases.append(omitted_occurrence)
    note_unknown = json.loads(json.dumps(original))
    note_unknown["metadata"][SOURCE_NOTES_METADATA_KEY]["notes"][0]["extra"] = True
    cases.append(note_unknown)
    duplicate_note = json.loads(json.dumps(original))
    duplicate_note["metadata"][SOURCE_NOTES_METADATA_KEY]["notes"].append(
        dict(duplicate_note["metadata"][SOURCE_NOTES_METADATA_KEY]["notes"][0])
    )
    cases.append(duplicate_note)
    bad_owner = json.loads(json.dumps(original))
    bad_owner["metadata"][SOURCE_NOTES_METADATA_KEY]["notes"][0][
        "owner_block_id"
    ] = "block-missing"
    cases.append(bad_owner)
    bad_note_selector = json.loads(json.dumps(original))
    bad_note_selector["metadata"][SOURCE_NOTES_METADATA_KEY]["notes"][0][
        "locator"
    ]["selector"] = "#different"
    cases.append(bad_note_selector)
    bad_note_identity = json.loads(json.dumps(original))
    bad_note_identity["metadata"][SOURCE_NOTES_METADATA_KEY]["notes"][0][
        "note_id"
    ] = "different"
    cases.append(bad_note_identity)
    bad_note_body = json.loads(json.dumps(original))
    bad_note_body["metadata"][SOURCE_NOTES_METADATA_KEY]["notes"][0]["body"] = (
        "Changed."
    )
    cases.append(bad_note_body)
    bad_link_target = json.loads(json.dumps(original))
    second_spans = bad_link_target["metadata"][SOURCE_NOTES_METADATA_KEY]["notes"][
        1
    ]["inline_spans"]
    next(span for span in second_spans if span["kind"] == "link")["target"] = ""
    cases.append(bad_link_target)
    bad_anchor = json.loads(json.dumps(original))
    bad_anchor["metadata"][SOURCE_NOTES_METADATA_KEY]["notes"][0]["anchor"][
        "start"
    ] += 1
    cases.append(bad_anchor)
    bad_note_order = json.loads(json.dumps(original))
    note_values = bad_note_order["metadata"][SOURCE_NOTES_METADATA_KEY]["notes"]
    note_values.reverse()
    for ordinal, note in enumerate(note_values):
        note["ordinal"] = ordinal
    cases.append(bad_note_order)

    for value in cases:
        with pytest.raises(ValueError, match="source front matter|source notes"):
            rich_document_from_document(value)

    validate_source_fidelity_metadata(
        document.metadata,
        blocks=document.blocks,
        source=document.source,
    )
    legacy = RichDocumentParserService(repository).parse_source(
        _store(repository, b"Legacy.\n", SourceFormat.MARKDOWN)
    )
    assert source_front_matter(legacy) is None
    assert source_notes(legacy) is None
    legacy_encoded = rich_document_to_document(legacy)
    legacy_decoded = rich_document_from_document(legacy_encoded)
    assert source_front_matter(legacy_decoded) is None
    assert rich_document_to_document(legacy_decoded) == legacy_encoded


@pytest.mark.parametrize(
    "metadata_key",
    (
        SOURCE_TARGET_MANIFEST_METADATA_KEY,
        SOURCE_FRONT_MATTER_METADATA_KEY,
        SOURCE_NOTES_METADATA_KEY,
        SOURCE_PRESENTATION_METADATA_KEY,
    ),
)
def test_optional_source_metadata_rejects_explicit_null(tmp_path, metadata_key):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(repository, b"Body.\n", SourceFormat.MARKDOWN)
    )
    with pytest.raises(ValueError):
        RichDocument(
            source=document.source,
            blocks=document.blocks,
            sections=document.sections,
            assets=document.assets,
            page_map=document.page_map,
            metadata={metadata_key: None},
            schema_version=document.schema_version,
        )


def test_source_note_owner_locator_is_provenance_not_binding(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><p id="P1">One<span class="ltx_note" id="footnote1">
              <sup class="ltx_note_mark">1</sup><span class="ltx_note_outer">
                <span class="ltx_note_content"><sup class="ltx_note_mark">
                  <span class="ltx_tag ltx_tag_note">1</span></sup>Body one.</span>
              </span></span>.</p>
              <p id="P2">Two<span class="ltx_note" id="footnote2">
                <sup class="ltx_note_mark">2</sup><span class="ltx_note_outer">
                  <span class="ltx_note_content"><sup class="ltx_note_mark">
                    <span class="ltx_tag ltx_tag_note">2</span></sup>Body two.</span>
                </span></span>.</p>
            </article>
            """,
            SourceFormat.HTML,
        )
    )
    encoded = rich_document_to_document(document)
    changed = json.loads(json.dumps(encoded))
    notes = changed["metadata"][SOURCE_NOTES_METADATA_KEY]["notes"]
    notes[0]["owner_locator"] = dict(notes[1]["owner_locator"])

    validate_source_fidelity_metadata(
        changed["metadata"],
        blocks=document.blocks,
        source=document.source,
    )
    assert notes[0]["owner_block_id"] == document.blocks[0].block_id
    assert notes[0]["anchor"] == encoded["metadata"][SOURCE_NOTES_METADATA_KEY][
        "notes"
    ][0]["anchor"]
    with pytest.raises(ValueError, match="digest"):
        rich_document_from_document(changed)


def test_source_front_matter_creator_flow_rejects_cross_entry_collision(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><div class="ltx_authors"><span class="ltx_creator
              ltx_role_author"><span class="ltx_personname">One<sup>1</sup></span>
              <span class="ltx_contact ltx_role_affiliation"><sup>1</sup>
                First Institute</span></span></div>
              <p>Between.</p>
              <div class="ltx_authors"><span class="ltx_creator
                ltx_role_author"><span class="ltx_personname">Two<sup>1</sup></span>
                <span class="ltx_contact ltx_role_affiliation"><sup>1</sup>
                  Second Institute</span></span></div></article>
            """,
            SourceFormat.HTML,
        )
    )
    encoded = rich_document_to_document(document)
    entries = encoded["metadata"][SOURCE_FRONT_MATTER_METADATA_KEY]["entries"]
    assert len(entries) == 2
    entries[1]["creator_flow"]["creators"][0]["creator_id"] = entries[0][
        "creator_flow"
    ]["creators"][0]["creator_id"]

    with pytest.raises(ValueError, match="source front matter"):
        validate_source_fidelity_metadata(
            encoded["metadata"],
            blocks=document.blocks,
            source=document.source,
        )
    with pytest.raises(ValueError, match="source front matter"):
        rich_document_from_document(encoded)


def test_html_latexml_table_figure_preserves_wrapper_and_visible_grid(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"""
        <article>
          <figure class="ltx_table" id="S3.T1">
            <figcaption>Table 1: Parameters for
              <math alttext="\\Lambda"><semantics><mi>Lambda</mi>
                <annotation encoding="application/x-tex">\\Lambda</annotation>
              </semantics></math>CDM.
            </figcaption>
            <table id="S3.T1.2">
              <tr><th>Model</th><th colspan="2">Scores</th></tr>
              <tr>
                <th><math alttext="\\Lambda"><semantics><mi>Lambda</mi>
                  <annotation encoding="application/x-tex">\\Lambda</annotation>
                </semantics></math>CDM</th>
                <td><math alttext="20.6"><semantics><mn>20.6</mn>
                  <annotation encoding="application/x-tex">20.6</annotation>
                </semantics></math></td>
                <td>0.0</td>
              </tr>
              <tr>
                <th id="age-header">Age <span class="ltx_note" id="footnote2">
                  <sup class="ltx_note_mark">2</sup><span class="ltx_note_outer">
                    <span class="ltx_note_content"><sup class="ltx_note_mark">
                      <span class="ltx_tag ltx_tag_note">2</span></sup>
                      hidden note body
                    </span>
                  </span>
                </span></th>
                <td>13.8</td><td></td>
              </tr>
              <tr><th rowspan="2">Shared</th><td>1</td><td>2</td></tr>
              <tr><td>3</td><td>4</td></tr>
            </table>
          </figure>
        </article>
        """,
        SourceFormat.HTML,
    )

    document = RichDocumentParserService(repository).parse_source(primary)

    assert len(document.blocks) == 1
    table = document.blocks[0]
    assert table.kind is RichBlockKind.TABLE
    assert table.locator.source_id == "S3.T1"
    assert table.locator.selector == "#S3.T1"
    assert table.payload == {
        "headers": ("Model", "Scores", ""),
        "rows": (
            ("LambdaCDM", "20.6", "0.0"),
            ("Age 2", "13.8", ""),
            ("Shared", "1", "2"),
            ("", "3", "4"),
        ),
        "caption": "Table 1: Parameters for LambdaCDM.",
    }
    assert "\\Lambda" not in table.payload["caption"]
    assert all(
        "hidden note body" not in cell
        for row in table.payload["rows"]
        for cell in row
    )
    notes = source_notes(document)
    assert notes is not None
    assert notes["notes"][0]["note_id"] == "footnote2"
    assert notes["notes"][0]["body"] == "hidden note body"
    assert notes["notes"][0]["owner_locator"]["source_id"] == "age-header"
    target = _source_target(document, "S3.T1")
    assert target["kind"] == "table"
    assert target["block_id"] == table.block_id
    assert target["block_start"] == 0
    assert target["block_end"] == 1
    assert target["panels"] == ()


@pytest.mark.parametrize(
    "table_markup",
    [
        "<table><tr><td>first</td></tr></table>"
        "<table><tr><td>second</td></tr></table>",
        "<table><tr><td>outer<table><tr><td>nested</td></tr></table>"
        "</td></tr></table>",
    ],
)
def test_html_latexml_table_figure_with_multiple_tables_preserves_plain_flow(
    tmp_path,
    table_markup,
):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        f"""
        <article><h1>Overview</h1>
          <figure class="ltx_table" id="S3.T1">
            {table_markup}
            <figcaption>Table 1: Ambiguous wrapper.</figcaption>
          </figure>
        </article>
        """.encode(),
        SourceFormat.HTML,
    )

    document = RichDocumentParserService(repository).parse_source(primary)

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.HEADING,
        RichBlockKind.PARAGRAPH,
    ]
    assert "Table 1: Ambiguous wrapper." in document.blocks[-1].payload["text"]
    assert "html_structure" in _diagnostic_categories(document)


@pytest.mark.parametrize("span", ["0", "-1", "invalid", "5000"])
def test_html_latexml_table_figure_with_unsafe_span_preserves_plain_flow(
    tmp_path,
    span,
):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        f"""
        <article><h1>Overview</h1>
          <figure class="ltx_table" id="S3.T1">
            <table><tr><th colspan="{span}">unsafe</th></tr></table>
            <figcaption>Table 1: Unsafe span.</figcaption>
          </figure>
        </article>
        """.encode(),
        SourceFormat.HTML,
    )

    document = RichDocumentParserService(repository).parse_source(primary)

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.HEADING,
        RichBlockKind.PARAGRAPH,
    ]
    assert "unsafe" in document.blocks[-1].payload["text"]
    assert "table_presentation" in _diagnostic_categories(document)


def test_html_table_rejects_aggregate_span_coverage_before_expansion(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    trailing_rows = "<tr></tr>" * 256
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            (
                "<article><h1>Overview</h1><table id='large'>"
                "<tr><td rowspan='257' colspan='256'>unsafe</td></tr>"
                f"{trailing_rows}</table></article>"
            ).encode(),
            SourceFormat.HTML,
        )
    )

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.HEADING,
        RichBlockKind.PARAGRAPH,
    ]
    assert document.blocks[-1].payload["text"] == "unsafe"
    assert "table_presentation" in _diagnostic_categories(document)


def test_table_presentation_rejects_aggregate_span_before_grid_expansion(
    monkeypatch,
):
    cell = {
        "row_index": 0,
        "column_index": 0,
        "row_span": 257,
        "column_span": 256,
        "kind": "header",
        "locator": {
            "source_format": "html",
            "line_start": None,
            "column_start": None,
            "line_end": None,
            "column_end": None,
            "selector": "",
            "source_id": "",
        },
        "horizontal_alignment": None,
        "horizontal_alignment_sources": [],
        "rule_edges": [],
    }
    block = SimpleNamespace(
        payload={
            "headers": tuple("" for _ in range(256)),
            "rows": tuple(
                tuple("" for _ in range(256)) for _ in range(256)
            ),
        }
    )
    builtin_range = range
    expanded = 0

    def guarded_range(*args):
        nonlocal expanded
        value = builtin_range(*args)
        expanded += len(value)
        if expanded > 10_000:
            raise AssertionError("unbounded Table coverage expansion")
        return value

    monkeypatch.setattr(
        presentation_validation,
        "range",
        guarded_range,
        raising=False,
    )
    with pytest.raises(ValueError, match="coverage"):
        presentation_validation._validate_table_cells(
            [cell],
            block=block,
            source_format="html",
            authored_ids=set(),
        )


def test_html_duplicate_wrapper_ids_remain_distinct_and_ambiguous(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"""
        <article>
          <figure class="ltx_table" id="S3.T1">
            <table><tr><td>first</td></tr></table>
          </figure>
          <figure class="ltx_table" id="S3.T1">
            <table><tr><td>second</td></tr></table>
          </figure>
        </article>
        """,
        SourceFormat.HTML,
    )

    document = RichDocumentParserService(repository).parse_source(primary)

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.TABLE,
        RichBlockKind.TABLE,
    ]
    assert [block.locator.source_id for block in document.blocks] == [
        "S3.T1",
        "S3.T1",
    ]
    assert len({block.block_id for block in document.blocks}) == 2
    assert [block.payload["rows"][0][0] for block in document.blocks] == [
        "first",
        "second",
    ]
    manifest = source_target_manifest(document)
    assert manifest is not None
    assert "S3.T1" not in {item["alias"] for item in manifest["targets"]}


def test_html_source_presentation_preserves_exact_figure_panel_layout(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    cells = "".join(
        f"""
        <div class="ltx_flex_cell ltx_flex_size_3">
          <img class="ltx_graphics ltx_figure_panel" id="F3.g{index}"
               src="panel-{index}.png"{dimensions}>
        </div>{break_markup}
        """
        for index, dimensions, break_markup in (
            (0, ' width="300" height="200" style="aspect-ratio:3/2;"', ""),
            (1, "", ""),
            (2, "", '<div class="ltx_flex_break"></div>'),
            (3, "", ""),
            (4, "", ""),
        )
    )
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            f"""
            <article>
              <figure class="ltx_figure" id="F1">
                <img class="ltx_graphics" id="F1.g1" src="single.png"
                     width="320" height="180" style="aspect-ratio:16/9;">
              </figure>
              <figure class="ltx_figure" id="F2">
                <div class="ltx_flex_figure">
                  <div class="ltx_flex_cell ltx_flex_size_2">
                    <img class="ltx_graphics ltx_figure_panel" id="F2.g1"
                         src="left.png">
                  </div>
                  <div class="ltx_flex_cell ltx_flex_size_2">
                    <object class="ltx_graphics ltx_figure_panel" id="F2.g2"
                            type="image/svg+xml" data="right.svg"></object>
                  </div>
                </div>
              </figure>
              <figure class="ltx_figure" id="F3">
                <div class="ltx_flex_figure">{cells}</div>
              </figure>
              <figure class="ltx_figure" id="F-mixed">
                <div class="ltx_flex_figure">
                  <div class="ltx_flex_cell ltx_flex_size_2">
                    <img class="ltx_graphics ltx_figure_panel" id="F-mixed.g1"
                         src="mixed-left.png">
                  </div>
                  <div class="ltx_flex_cell ltx_flex_size_2">
                    <img class="ltx_graphics ltx_figure_panel" id="F-mixed.g2"
                         src="mixed-right.png">
                  </div>
                  <div class="ltx_flex_break"></div>
                  <div class="ltx_flex_cell ltx_flex_size_1">
                    <img class="ltx_graphics ltx_figure_panel" id="F-mixed.g3"
                         src="mixed-full.png">
                  </div>
                </div>
              </figure>
              <figure class="ltx_figure" id="F-neutral">
                <img id="F-neutral.g1" src="a.png">
                <img id="F-neutral.g2" src="b.png">
              </figure>
            </article>
            """.encode(),
            SourceFormat.HTML,
        )
    )
    figures = {
        block.locator.source_id: block
        for block in document.blocks
        if block.kind is RichBlockKind.FIGURE
    }
    layouts = {
        source_id: _source_presentation_figure(document, block)
        for source_id, block in figures.items()
    }

    assert list(layouts) == ["F1", "F2", "F3", "F-mixed", "F-neutral"]
    assert layouts["F1"]["layout"] == {
        "kind": "single",
        "column_count": 1,
        "row_count": 1,
        "rows": ((0,),),
        "column_source": "latexml_ar5iv_direct_graphic",
        "row_sources": ("latexml_ar5iv_direct_graphic",),
        "break_after_panel_indexes": (),
        "break_source": None,
    }
    assert layouts["F1"]["panels"] == (
        {
            "panel_index": 0,
            "source_id": "F1.g1",
            "row_index": 0,
            "column_index": 0,
            "display_width": 320,
            "display_height": 180,
            "dimension_source": "attributes:width,height",
            "aspect_ratio": (16, 9),
            "aspect_ratio_source": "style:aspect-ratio",
        },
    )
    assert layouts["F2"]["layout"] == {
        "kind": "flex",
        "column_count": 2,
        "row_count": 1,
        "rows": ((0, 1),),
        "column_source": "class:ltx_flex_size_2",
        "row_sources": ("class:ltx_flex_size_2",),
        "break_after_panel_indexes": (),
        "break_source": None,
    }
    assert all(
        panel["display_width"] is None
        and panel["display_height"] is None
        and panel["dimension_source"] is None
        and panel["aspect_ratio"] is None
        and panel["aspect_ratio_source"] is None
        for panel in layouts["F2"]["panels"]
    )
    assert layouts["F3"]["layout"] == {
        "kind": "flex",
        "column_count": 3,
        "row_count": 2,
        "rows": ((0, 1, 2), (3, 4)),
        "column_source": "class:ltx_flex_size_3",
        "row_sources": (
            "class:ltx_flex_size_3",
            "class:ltx_flex_size_3",
        ),
        "break_after_panel_indexes": (2,),
        "break_source": "class:ltx_flex_break",
    }
    assert layouts["F3"]["panels"][3]["row_index"] == 1
    assert layouts["F3"]["panels"][3]["column_index"] == 0
    assert layouts["F-mixed"]["layout"] == {
        "kind": "flex",
        "column_count": 2,
        "row_count": 2,
        "rows": ((0, 1), (2,)),
        "column_source": None,
        "row_sources": (
            "class:ltx_flex_size_2",
            "class:ltx_flex_size_1",
        ),
        "break_after_panel_indexes": (1,),
        "break_source": "class:ltx_flex_break",
    }
    assert [
        (panel["row_index"], panel["column_index"])
        for panel in layouts["F-mixed"]["panels"]
    ] == [(0, 0), (0, 1), (1, 0)]
    assert layouts["F-neutral"]["layout"] == {
        "kind": "neutral",
        "column_count": None,
        "row_count": None,
        "rows": (),
        "column_source": None,
        "row_sources": (),
        "break_after_panel_indexes": (),
        "break_source": None,
    }
    assert all(
        panel["row_index"] is None and panel["column_index"] is None
        for panel in layouts["F-neutral"]["panels"]
    )
    assert source_presentation(
        rich_document_from_document(rich_document_to_document(document))
    ) == source_presentation(document)


def test_html_latexml_single_figure_accepts_neutral_p_span_wrappers(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article>
              <figure class="ltx_figure" id="S4.F1">
                <p class="ltx_p ltx_align_center">
                  <span class="ltx_text">
                    <img class="ltx_graphics ltx_img_landscape"
                         id="S4.F1.g1" src="paper/P1.jpg"
                         width="320" height="180">
                  </span>
                </p>
                <figcaption class="ltx_caption ltx_centering">
                  Figure 1: wrapped graphic.
                </figcaption>
              </figure>
            </article>
            """,
            SourceFormat.HTML,
        )
    )

    figure = next(
        block for block in document.blocks if block.kind is RichBlockKind.FIGURE
    )
    presentation = _source_presentation_figure(document, figure)

    assert presentation["layout"] == {
        "kind": "single",
        "column_count": 1,
        "row_count": 1,
        "rows": ((0,),),
        "column_source": "latexml_ar5iv_direct_graphic",
        "row_sources": ("latexml_ar5iv_direct_graphic",),
        "break_after_panel_indexes": (),
        "break_source": None,
    }
    assert presentation["panels"] == (
        {
            "panel_index": 0,
            "source_id": "S4.F1.g1",
            "row_index": 0,
            "column_index": 0,
            "display_width": 320,
            "display_height": 180,
            "dimension_source": "attributes:width,height",
            "aspect_ratio": None,
            "aspect_ratio_source": None,
        },
    )


@pytest.mark.parametrize(
    "flex_markup",
    (
        '<div class="ltx_flex_figure"></div>'
        '<div class="ltx_flex_figure"></div>',
        '<div><div class="ltx_flex_figure"></div></div>',
        '<div class="ltx_flex_figure">'
        '<div class="ltx_flex_cell ltx_flex_size_4">'
        '<img class="ltx_graphics ltx_figure_panel" src="a.png"></div></div>',
        '<div class="ltx_flex_figure">'
        '<div class="ltx_flex_cell ltx_flex_size_2">'
        '<img class="ltx_graphics ltx_figure_panel" src="a.png"></div>'
        '<div class="ltx_flex_cell ltx_flex_size_3">'
        '<img class="ltx_graphics ltx_figure_panel" src="b.png"></div></div>',
        '<div class="ltx_flex_figure"><div class="ltx_flex_break"></div>'
        '<div class="ltx_flex_cell ltx_flex_size_2">'
        '<img class="ltx_graphics ltx_figure_panel" src="a.png"></div></div>',
        '<div class="ltx_flex_figure">'
        '<div class="ltx_flex_cell ltx_flex_size_2">'
        '<img class="ltx_graphics ltx_figure_panel" src="a.png"></div>'
        '<div class="ltx_flex_break"></div></div>',
        '<div class="ltx_flex_figure">'
        '<div class="ltx_flex_cell ltx_flex_size_2">'
        '<img class="ltx_graphics ltx_figure_panel" src="a.png">'
        '<img class="ltx_graphics ltx_figure_panel" src="b.png"></div></div>',
        '<div class="ltx_flex_figure ltx_flex_size_2">'
        '<div class="ltx_flex_cell ltx_flex_size_2">'
        '<img class="ltx_graphics ltx_figure_panel" src="a.png"></div></div>',
        '<div class="ltx_flex_figure">'
        '<div class="ltx_flex_cell ltx_flex_size_2">'
        '<img class="ltx_graphics ltx_figure_panel" src="a.png"></div>'
        '<div class="ltx_flex_break ltx_flex_size_2"></div>'
        '<div class="ltx_flex_cell ltx_flex_size_2">'
        '<img class="ltx_graphics ltx_figure_panel" src="b.png"></div></div>',
        '<img class="ltx_graphics" src="a.png">'
        '<img class="ltx_graphics" src="b.png">',
        '<div><img class="ltx_graphics ltx_figure_panel" src="a.png"></div>',
        '<p><span>prefix<img class="ltx_graphics" src="a.png"></span></p>',
        '<p><span><img class="ltx_graphics" src="a.png"></span>'
        '<em>extra</em></p>',
    ),
)
def test_html_latexml_figure_layout_structure_degrades_to_neutral(
    tmp_path,
    flex_markup,
):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            (
                '<article><figure class="ltx_figure" id="F1">'
                f"{flex_markup}</figure></article>"
            ).encode(),
            SourceFormat.HTML,
        )
    )
    assert "figure_layout" in _diagnostic_categories(document)


@pytest.mark.parametrize(
    "attributes",
    (
        'width="0" height="2"',
        'width="3.5" height="2"',
        'width="3"',
        'width="3" height="2" style="aspect-ratio:2/1;"',
        'width="3" height="2" style="aspect-ratio:auto;"',
        'width="3" height="2" style="aspect-ratio:0/1;"',
        'width="3" height="2" style="aspect-ratio:3/2;aspect-ratio:1/1;"',
        'width="1000001" height="2"',
    ),
)
def test_html_latexml_figure_panel_dimensions_degrade_to_neutral(tmp_path, attributes):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            (
                '<article><figure class="ltx_figure" id="F1">'
                '<img class="ltx_graphics" id="F1.g1" src="a.png" '
                f"{attributes}></figure></article>"
            ).encode(),
            SourceFormat.HTML,
        )
    )
    assert document.blocks[0].kind is RichBlockKind.FIGURE
    assert "figure_layout" in _diagnostic_categories(document)


def test_source_presentation_figure_layout_rejects_tamper_and_is_digested(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article>
              <figure class="ltx_figure" id="F1">
                <img class="ltx_graphics" id="F1.g1" src="single.png"
                     width="320" height="180" style="aspect-ratio:16/9;">
              </figure>
              <figure class="ltx_figure" id="F2">
                <div class="ltx_flex_figure">
                  <div class="ltx_flex_cell ltx_flex_size_2">
                    <img class="ltx_graphics ltx_figure_panel" id="F2.g1"
                         src="left.png" width="200" height="100"
                         style="aspect-ratio:2/1;">
                  </div>
                  <div class="ltx_flex_cell ltx_flex_size_2">
                    <img class="ltx_graphics ltx_figure_panel" id="F2.g2"
                         src="right.png">
                  </div>
                </div>
              </figure>
            </article>
            """,
            SourceFormat.HTML,
        )
    )
    original = rich_document_to_document(document)
    figures = original["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["figures"]
    cases = []

    missing = json.loads(json.dumps(original))
    missing["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["figures"].pop()
    cases.append(missing)
    duplicate = json.loads(json.dumps(original))
    duplicate_figures = duplicate["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "figures"
    ]
    duplicate_figures.append(dict(duplicate_figures[0]))
    cases.append(duplicate)
    reversed_order = json.loads(json.dumps(original))
    reversed_order["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "figures"
    ].reverse()
    cases.append(reversed_order)
    unknown_field = json.loads(json.dumps(original))
    unknown_field["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["figures"][0][
        "extra"
    ] = True
    cases.append(unknown_field)
    wrong_block = json.loads(json.dumps(original))
    wrong_block["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["figures"][0][
        "block_id"
    ] = original["blocks"][1]["block_id"]
    cases.append(wrong_block)
    missing_panel = json.loads(json.dumps(original))
    missing_panel["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["figures"][1][
        "panels"
    ].pop()
    cases.append(missing_panel)
    wrong_panel_index = json.loads(json.dumps(original))
    wrong_panel_index["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["figures"][
        1
    ]["panels"][0]["panel_index"] = 1
    cases.append(wrong_panel_index)
    wrong_source_id = json.loads(json.dumps(original))
    wrong_source_id["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["figures"][1][
        "panels"
    ][0]["source_id"] = "F2.unknown"
    cases.append(wrong_source_id)
    overlap = json.loads(json.dumps(original))
    overlap["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["figures"][1][
        "panels"
    ][1]["column_index"] = 0
    cases.append(overlap)
    out_of_bounds = json.loads(json.dumps(original))
    out_of_bounds["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["figures"][1][
        "panels"
    ][1]["column_index"] = 2
    cases.append(out_of_bounds)
    bad_rows = json.loads(json.dumps(original))
    bad_rows["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["figures"][1][
        "layout"
    ]["rows"] = [[0, 0]]
    cases.append(bad_rows)
    bad_column_source = json.loads(json.dumps(original))
    bad_column_source["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["figures"][
        1
    ]["layout"]["column_source"] = "class:ltx_flex_size_3"
    cases.append(bad_column_source)
    missing_row_sources = json.loads(json.dumps(original))
    missing_row_sources["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["figures"][
        1
    ]["layout"].pop("row_sources")
    cases.append(missing_row_sources)
    bad_row_source = json.loads(json.dumps(original))
    bad_row_source["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["figures"][1][
        "layout"
    ]["row_sources"][0] = "class:ltx_flex_size_3"
    cases.append(bad_row_source)
    bad_break_source = json.loads(json.dumps(original))
    bad_break_source["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["figures"][1][
        "layout"
    ]["break_source"] = "class:ltx_flex_break"
    cases.append(bad_break_source)
    bad_dimension = json.loads(json.dumps(original))
    bad_dimension["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["figures"][1][
        "panels"
    ][0]["display_width"] = 0
    cases.append(bad_dimension)
    bad_dimension_source = json.loads(json.dumps(original))
    bad_dimension_source["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "figures"
    ][1]["panels"][1]["dimension_source"] = "attributes:width,height"
    cases.append(bad_dimension_source)
    nonnormalized_ratio = json.loads(json.dumps(original))
    nonnormalized_ratio["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["figures"][
        1
    ]["panels"][0]["aspect_ratio"] = [4, 2]
    cases.append(nonnormalized_ratio)
    bad_ratio_source = json.loads(json.dumps(original))
    bad_ratio_source["metadata"][SOURCE_PRESENTATION_METADATA_KEY]["figures"][1][
        "panels"
    ][0]["aspect_ratio_source"] = "style:unknown"
    cases.append(bad_ratio_source)

    for value in cases:
        with pytest.raises(ValueError, match="source presentation"):
            validate_source_presentation_metadata(
                value["metadata"],
                blocks=document.blocks,
                source=document.source,
            )
        with pytest.raises(ValueError, match="source presentation"):
            rich_document_from_document(value)

    digest_protected = json.loads(json.dumps(original))
    panel = digest_protected["metadata"][SOURCE_PRESENTATION_METADATA_KEY][
        "figures"
    ][0]["panels"][0]
    panel["display_width"] = 640
    panel["display_height"] = 360
    validate_source_presentation_metadata(
        digest_protected["metadata"],
        blocks=document.blocks,
        source=document.source,
    )
    with pytest.raises(ValueError, match="digest"):
        rich_document_from_document(digest_protected)
    assert len(figures) == 2


def test_html_single_svg_object_figure_imports_wrapper_asset_and_caption(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    (media / "plot.svg").write_text("<svg/>", encoding="utf-8")
    source = tmp_path / "paper.html"
    source.write_text(
        """
        <article><figure class="ltx_figure" id="S4.F1">
          <object type="image/svg+xml" data="media/plot.svg"
                  aria-label="DE density"></object>
          <figcaption><em>Figure 1</em>: <a href="#details">Density</a>.</figcaption>
        </figure></article>
        """,
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")

    outcome = RichDocumentParserService(repository).parse(
        SourceBundle(primary=repository.import_path(source))
    )

    assert not any("local asset" in warning for warning in outcome.warnings)
    assert len(outcome.document.blocks) == 1
    figure = outcome.document.blocks[0]
    assert figure.kind is RichBlockKind.FIGURE
    assert figure.locator.source_id == "S4.F1"
    assert figure.locator.selector == "#S4.F1"
    assert figure.payload["target"] == "media/plot.svg"
    assert figure.payload["alt_text"] == "DE density"
    assert figure.payload["caption"] == "Figure 1: Density."
    caption_view = _source_presentation_field(
        outcome.document,
        figure,
        "caption",
    )
    assert caption_view["marks"] == (
        {"kind": "emphasis", "start": 0, "end": 8},
    )
    assert next(
        span for span in caption_view["inline_spans"] if span["kind"] == "link"
    )["target"] == "#details"
    assert figure.payload["media_type"] == "image/svg+xml"
    assert figure.payload["asset_digest"] == (
        outcome.document.assets[0].artifact_digest
    )
    target = _source_target(outcome.document, "S4.F1")
    assert target["block_id"] == figure.block_id
    assert [panel["status"] for panel in target["panels"]] == ["available"]
    assert target["panels"][0]["target"] == "media/plot.svg"
    assert target["panels"][0]["asset_digest"] == figure.payload["asset_digest"]


def test_html_single_svg_object_figure_missing_asset_is_explicit(tmp_path):
    source = tmp_path / "paper.html"
    source.write_text(
        """
        <article><figure class="ltx_figure" id="S4.F1">
          <object type="image/svg+xml" data="media/missing.svg"></object>
          <figcaption>Figure 1: Missing locally.</figcaption>
        </figure></article>
        """,
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")

    outcome = RichDocumentParserService(repository).parse(
        SourceBundle(primary=repository.import_path(source))
    )

    assert [
        warning for warning in outcome.warnings if "local asset" in warning
    ] == ["local asset was not found: media/missing.svg"]
    assert len(outcome.document.blocks) == 1
    figure = outcome.document.blocks[0]
    assert figure.kind is RichBlockKind.FIGURE
    assert figure.locator.source_id == "S4.F1"
    assert figure.payload["target"] == "media/missing.svg"
    assert figure.payload["asset_digest"] == ""
    assert figure.payload["media_type"] == ""
    assert figure.payload["size"] == 0
    assert outcome.document.assets == ()
    panel = _source_target(outcome.document, "S4.F1")["panels"][0]
    assert panel["status"] == "missing"
    assert panel["target"] == "media/missing.svg"
    assert panel["asset_digest"] == ""
    assert panel["size"] == 0


def test_html_multi_object_figure_preserves_parent_and_ordered_panels(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    (media / "left.svg").write_text("<svg/>", encoding="utf-8")
    (media / "right.svg").write_text("<svg/>", encoding="utf-8")
    source = tmp_path / "paper.html"
    source.write_text(
        """
        <article><h1>Overview</h1>
          <figure class="ltx_figure" id="S4.F4">
            <object type="image/svg+xml" data="media/left.svg"></object>
            <object type="image/svg+xml" data="media/right.svg"></object>
            <figcaption>Figure 4: Two panels.</figcaption>
          </figure>
        </article>
        """,
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")

    outcome = RichDocumentParserService(repository).parse(
        SourceBundle(primary=repository.import_path(source))
    )

    assert not any("local asset" in warning for warning in outcome.warnings)
    assert [block.kind for block in outcome.document.blocks] == [
        RichBlockKind.HEADING,
        RichBlockKind.FIGURE,
    ]
    parent = outcome.document.blocks[1]
    assert parent.locator.source_id == "S4.F4"
    assert parent.payload["target"] == ""
    assert parent.payload["asset_digest"] == ""
    assert parent.payload["caption"] == "Figure 4: Two panels."
    assert len(outcome.document.assets) == 1
    target = _source_target(outcome.document, "S4.F4")
    assert target["block_id"] == parent.block_id
    assert [panel["panel_index"] for panel in target["panels"]] == [0, 1]
    assert [panel["target"] for panel in target["panels"]] == [
        "media/left.svg",
        "media/right.svg",
    ]
    assert [panel["status"] for panel in target["panels"]] == [
        "available",
        "available",
    ]
    assert len({panel["asset_digest"] for panel in target["panels"]}) == 1
    assert [panel["logical_name"] for panel in target["panels"]] == [
        "media/left.svg",
        "media/right.svg",
    ]


@pytest.mark.parametrize(
    ("media_markup", "expected_target", "expected_media_type"),
    [
        (
            '<object type="application/pdf" data="media/plot.svg"></object>',
            "media/plot.svg",
            "application/pdf",
        ),
        ('<object type="image/svg+xml"></object>', "", "image/svg+xml"),
        (
            '<object data="media/plot.svg"></object>',
            "media/plot.svg",
            "image/svg+xml",
        ),
    ],
)
def test_html_unsupported_object_preserves_parent_and_panel_status(
    tmp_path,
    media_markup,
    expected_target,
    expected_media_type,
):
    media = tmp_path / "media"
    media.mkdir()
    (media / "plot.svg").write_text("<svg/>", encoding="utf-8")
    (media / "left.png").write_bytes(b"left")
    source = tmp_path / "paper.html"
    source.write_text(
        f"""
        <article><h1>Overview</h1>
          <figure class="ltx_figure" id="S4.F4">
            {media_markup}
            <figcaption>Figure 4: Unsupported or ambiguous.</figcaption>
          </figure>
        </article>
        """,
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")

    outcome = RichDocumentParserService(repository).parse(
        SourceBundle(primary=repository.import_path(source))
    )

    assert [block.kind for block in outcome.document.blocks] == [
        RichBlockKind.HEADING,
        RichBlockKind.FIGURE,
    ]
    assert any(
        "unsupported figure panel" in warning for warning in outcome.warnings
    )
    parent = outcome.document.blocks[1]
    assert parent.locator.source_id == "S4.F4"
    assert parent.payload["target"] == ""
    assert outcome.document.assets == ()
    panel = _source_target(outcome.document, "S4.F4")["panels"][0]
    assert panel["status"] == "unsupported"
    assert panel["target"] == expected_target
    assert panel["media_type"] == expected_media_type


def test_html_mixed_figure_preserves_parent_and_panel_source_order(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    (media / "left.svg").write_text("<svg/>", encoding="utf-8")
    (media / "right.png").write_bytes(b"right")
    source = tmp_path / "paper.html"
    source.write_text(
        """
        <article><figure class="ltx_figure" id="S4.F4">
          <object id="S4.F4.p1" type="image/svg+xml"
                  data="media/left.svg"></object>
          <img id="S4.F4.p2" src="media/right.png" alt="Right">
          <figcaption>Figure 4: Mixed panels.</figcaption>
        </figure></article>
        """,
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")

    outcome = RichDocumentParserService(repository).parse(
        SourceBundle(primary=repository.import_path(source))
    )

    assert not any("figure panel" in warning for warning in outcome.warnings)
    assert len(outcome.document.blocks) == 1
    parent = outcome.document.blocks[0]
    assert parent.kind is RichBlockKind.FIGURE
    assert parent.locator.source_id == "S4.F4"
    assert parent.payload["target"] == ""
    target = _source_target(outcome.document, "S4.F4")
    assert [panel["panel_index"] for panel in target["panels"]] == [0, 1]
    assert [panel["source_id"] for panel in target["panels"]] == [
        "S4.F4.p1",
        "S4.F4.p2",
    ]
    assert [panel["selector"] for panel in target["panels"]] == [
        "#S4.F4.p1",
        "#S4.F4.p2",
    ]
    assert [panel["target"] for panel in target["panels"]] == [
        "media/left.svg",
        "media/right.png",
    ]
    assert [panel["status"] for panel in target["panels"]] == [
        "available",
        "available",
    ]


def test_html_fifteen_panel_figure_preserves_order_and_missing_state(tmp_path):
    panels = "".join(
        f'<object id="S4.F13.p{index}" type="image/svg+xml" '
        f'data="media/panel-{index}.svg"></object>'
        for index in range(15)
    )
    source = tmp_path / "paper.html"
    source.write_text(
        f"""
        <article><figure class="ltx_figure" id="S4.F13">
          {panels}<figcaption>Figure 13: Fifteen panels.</figcaption>
        </figure></article>
        """,
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")

    outcome = RichDocumentParserService(repository).parse(
        SourceBundle(primary=repository.import_path(source))
    )

    parent = outcome.document.blocks[0]
    assert parent.kind is RichBlockKind.FIGURE
    assert parent.locator.source_id == "S4.F13"
    assert parent.payload["target"] == ""
    target = _source_target(outcome.document, "S4.F13")
    assert [panel["panel_index"] for panel in target["panels"]] == list(range(15))
    assert [panel["source_id"] for panel in target["panels"]] == [
        f"S4.F13.p{index}" for index in range(15)
    ]
    assert [panel["target"] for panel in target["panels"]] == [
        f"media/panel-{index}.svg" for index in range(15)
    ]
    assert {panel["status"] for panel in target["panels"]} == {"missing"}
    assert len(
        [warning for warning in outcome.warnings if "local asset" in warning]
    ) == 15
    assert outcome.document.assets == ()


@pytest.mark.parametrize(
    "panels",
    [
        '<object id="panel" type="image/svg+xml" data="media/a.svg"></object>'
        '<object id="panel" type="image/svg+xml" data="media/b.svg"></object>',
        '<object id="panel-a" type="image/svg+xml" data="media/a.svg"></object>'
        '<object id="panel-b" type="image/svg+xml" data="media/a.svg"></object>',
        '<object id="S4.F4" type="image/svg+xml" data="media/a.svg"></object>',
    ],
)
def test_html_compound_figure_isolates_panel_identity_collisions(
    tmp_path,
    panels,
):
    source = tmp_path / "paper.html"
    source.write_text(
        f"""
        <article><figure class="ltx_figure" id="S4.F4">{panels}</figure></article>
        """,
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")

    outcome = RichDocumentParserService(repository).parse(
        SourceBundle(primary=repository.import_path(source))
    )
    assert outcome.document.blocks[0].kind is RichBlockKind.FIGURE
    assert source_target_manifest(outcome.document) is None
    assert "source_target_manifest" in _diagnostic_categories(outcome.document)


def test_html_figures_isolate_cross_parent_panel_identity_collisions(tmp_path):
    source = tmp_path / "paper.html"
    source.write_text(
        """
        <article>
          <figure class="ltx_figure" id="S4.F4">
            <object id="shared-panel" type="image/svg+xml"
                    data="media/a.svg"></object>
          </figure>
          <figure class="ltx_figure" id="S4.F7">
            <object id="shared-panel" type="image/svg+xml"
                    data="media/b.svg"></object>
          </figure>
        </article>
        """,
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")

    outcome = RichDocumentParserService(repository).parse(
        SourceBundle(primary=repository.import_path(source))
    )
    assert len(outcome.document.blocks) == 2
    assert source_target_manifest(outcome.document) is None
    assert "source_target_manifest" in _diagnostic_categories(outcome.document)


@pytest.mark.parametrize(
    "wrapper",
    (
        "<span>Alpha <em>beta</em>.</span>",
        "<!-- converter note --><span data-publisher='unknown'>Alpha <em>beta</em>.</span>",
        "\n <div class='publisher-shell'> Alpha <em>beta</em>. </div>\n",
    ),
)
def test_html_nonsemantic_wrappers_attributes_and_comments_preserve_core_content(
    tmp_path,
    wrapper,
):
    repository = SourceRepository(tmp_path / "cache")
    baseline = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"<article><p>Alpha <em>beta</em>.</p></article>",
            SourceFormat.HTML,
        )
    )
    varied = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            f"<article><p>{wrapper}</p></article>".encode(),
            SourceFormat.HTML,
        )
    )

    assert [block.kind for block in varied.blocks] == [block.kind for block in baseline.blocks]
    assert [block.payload for block in varied.blocks] == [block.payload for block in baseline.blocks]
    assert document_diagnostics(varied)["visible_content"]["unaccounted"] == 0


def test_optional_projection_corruption_preserves_core_blocks_and_records_fallback(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    valid = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><p>Before.</p><figure class="ltx_figure" id="F1">
              <img class="ltx_graphics" id="F1.g1" src="plot.png">
            </figure><table id="T1"><tr><td>Cell</td></tr></table><p>After.</p></article>
            """,
            SourceFormat.HTML,
        )
    )
    corrupted = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><p>Before.</p><figure class="ltx_figure" id="F1">
              <p>prefix<img class="ltx_graphics" id="F1.g1" src="plot.png"></p>
            </figure><table id="T1"><tr><td class="ltx_align_left" style="text-align:center">Cell</td></tr></table><p>After.</p></article>
            """,
            SourceFormat.HTML,
        )
    )

    assert [block.kind for block in corrupted.blocks] == [
        block.kind for block in valid.blocks
    ]
    assert [block.payload for block in corrupted.blocks] == [
        block.payload for block in valid.blocks
    ]
    categories = _diagnostic_categories(corrupted)
    assert "figure_layout" in categories
    assert "table_presentation" in categories
    assert source_presentation(corrupted) is not None


def test_target_projection_failure_isolated_from_visible_core_content(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><p>Before.</p><figure id="F1"><object id="panel"
              type="image/svg+xml" data="a.svg"></object></figure><p>After.</p>
              <figure id="F2"><object id="panel" type="image/svg+xml"
              data="b.svg"></object></figure></article>
            """,
            SourceFormat.HTML,
        )
    )

    assert [block.payload["text"] for block in document.blocks if block.kind is RichBlockKind.PARAGRAPH] == [
        "Before.",
        "After.",
    ]
    assert source_target_manifest(document) is None
    diagnostics = document_diagnostics(document)
    assert diagnostics is not None
    assert diagnostics["visible_content"] == {
        "visible_units": 4,
        "emitted": 4,
        "documented_exclusions": 0,
        "opaque": 0,
        "unaccounted": 0,
    }


def test_document_diagnostics_are_typed_serializable_and_reject_unaccounted_flow(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"<article><p>Visible.</p></article>",
            SourceFormat.HTML,
        )
    )
    encoded = rich_document_to_document(document)

    decoded = rich_document_from_document(encoded)
    assert document_diagnostics(decoded) == document_diagnostics(document)
    assert decoded.metadata[
        DOCUMENT_DIAGNOSTICS_METADATA_KEY
    ] == document.metadata[DOCUMENT_DIAGNOSTICS_METADATA_KEY]

    metadata = dict(encoded["metadata"])
    diagnostics = dict(metadata[DOCUMENT_DIAGNOSTICS_METADATA_KEY])
    visible = dict(diagnostics["visible_content"])
    visible["unaccounted"] = 1
    diagnostics["visible_content"] = visible
    metadata[DOCUMENT_DIAGNOSTICS_METADATA_KEY] = diagnostics
    with pytest.raises(ValueError, match="reconcile|unaccounted"):
        RichDocument(
            source=document.source,
            blocks=document.blocks,
            sections=document.sections,
            assets=document.assets,
            page_map=document.page_map,
            metadata=metadata,
        )


def test_html_visible_ownership_accounts_once_for_navigation_and_article_siblings(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <header><div>Publisher chrome.</div></header>
            <nav>Outside navigation.</nav>
            <article>
              <nav><span>Reader controls.</span></nav>
              <div class="publisher-wrapper"><p>First source paragraph.</p></div>
              <div class="publisher-wrapper"><p>Second source paragraph.</p></div>
            </article>
            <footer><div>Publisher footer.</div></footer>
            """,
            SourceFormat.HTML,
        )
    )

    assert [block.payload["text"] for block in document.blocks] == [
        "First source paragraph.",
        "Second source paragraph.",
    ]
    diagnostics = document_diagnostics(document)
    assert diagnostics is not None
    assert diagnostics["visible_content"] == {
        "visible_units": 6,
        "emitted": 2,
        "documented_exclusions": 4,
        "opaque": 0,
        "unaccounted": 0,
    }
    categories = [item["category"] for item in diagnostics["projections"]]
    assert categories.count("navigation") == 2
    assert categories.count("outside_content") == 2
    excluded = [
        item["scope"]
        for item in diagnostics["projections"]
        if item["category"] in {"navigation", "outside_content"}
    ]
    assert len(excluded) == len(set(excluded))


def test_visible_ownership_reports_a_real_set_difference_before_validation():
    locator = SourceLocator(source_format=SourceFormat.HTML)
    unit = rich_parser._HTMLVisibleOwnershipUnit(
        identity="visible-test-unit",
        locator=locator,
        kind="content",
    )
    ledger = rich_parser._ProjectionLedger()
    ledger.register(unit)

    assert ledger.metadata()["visible_content"] == {
        "visible_units": 1,
        "emitted": 0,
        "documented_exclusions": 0,
        "opaque": 0,
        "unaccounted": 1,
    }


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
                "    <caption>Values.</caption>",
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
    assert left.payload["caption"] == "Values."
    assert table_equation.payload == {
        "tex": "y = 2",
        "display": True,
        "label": "",
    }
    assert right.payload["headers"] == ("",)
    assert right.payload["rows"] == (("right.",),)
    assert right.payload["caption"] == ""
    assert before.locator == paragraph_equation.locator == after.locator
    assert left.locator == table_equation.locator == right.locator
    assert [
        block.payload["tex"]
        for block in document.blocks
        if block.kind is RichBlockKind.EQUATION
    ] == ["x = 1", "y = 2"]


@pytest.mark.parametrize(
    "embedded",
    (
        '<math display="block" alttext="x = 1"></math>',
        '<img src="plot.png" alt="Plot">',
    ),
)
def test_html_split_table_has_one_caption_owner(tmp_path, embedded):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            (
                "<article><table id='T'><caption>Caption.</caption>"
                f"<tr><td>Before {embedded} After.</td></tr>"
                "</table></article>"
            ).encode(),
            SourceFormat.HTML,
        )
    )
    tables = [block for block in document.blocks if block.kind is RichBlockKind.TABLE]
    assert len(tables) == 2
    assert [block.payload["caption"] for block in tables] == ["Caption.", ""]
    presentation = source_presentation(document)
    assert presentation is not None
    assert [entry["block_id"] for entry in presentation["captions"]] == [
        tables[0].block_id
    ]


def test_html_block_math_only_table_keeps_caption_owner(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article><table id="T"><caption>Caption.</caption><tr><td>
              <math display="block" alttext="x = 1"></math>
            </td></tr></table></article>
            """,
            SourceFormat.HTML,
        )
    )
    tables = [block for block in document.blocks if block.kind is RichBlockKind.TABLE]
    assert len(tables) == 1
    assert tables[0].payload["caption"] == "Caption."
    assert any(block.kind is RichBlockKind.EQUATION for block in document.blocks)


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

    if source_format is SourceFormat.MARKDOWN:
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
        assert not any(block.list_path for block in document.blocks)
    else:
        assert [block.kind for block in document.blocks] == [
            RichBlockKind.LIST,
            RichBlockKind.LIST,
            RichBlockKind.FIGURE,
            RichBlockKind.LIST,
            RichBlockKind.LIST,
        ]
        first, before, figure, after, last = document.blocks
        assert [
            block.payload["items"][0]["text"]
            for block in (first, before, after, last)
        ] == ["first", "before", "after", "last"]
        assert [block.list_path[0].item_index for block in document.blocks] == [
            0,
            1,
            1,
            1,
            2,
        ]
    assert figure.payload["alt_text"] == "list plot"
    assert figure.payload["asset_digest"] == document.assets[0].artifact_digest


def test_html_list_nested_equation_table_preserves_identity_label_and_order(
    tmp_path,
):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"""
        <article>
          <ul id="S3.I2">
            <li id="S3.I2.i1">
              <p>Before the equation.</p>
              <table class="ltx_equation" id="S3.E13">
                <tr>
                  <td><math alttext="x = 13" display="block"></math></td>
                  <td><span class="ltx_tag">(13)</span></td>
                </tr>
              </table>
              <p>where the residual is defined.</p>
            </li>
            <li id="S3.I2.i2"><p>Second item.</p></li>
          </ul>
        </article>
        """,
        SourceFormat.HTML,
    )

    document = RichDocumentParserService(repository).parse_source(primary)

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.LIST,
        RichBlockKind.EQUATION,
        RichBlockKind.LIST,
        RichBlockKind.LIST,
    ]
    before, equation, after, second = document.blocks
    assert [item["text"] for item in before.payload["items"]] == [
        "Before the equation."
    ]
    assert equation.payload == {
        "tex": "x = 13",
        "display": True,
        "label": "13",
    }
    assert equation.locator.source_id == "S3.E13"
    assert equation.locator.selector == "#S3.E13"
    assert [item["text"] for item in after.payload["items"]] == [
        "where the residual is defined."
    ]
    assert [item["text"] for item in second.payload["items"]] == [
        "Second item."
    ]
    assert [block.list_path[0].segment_index for block in document.blocks] == [
        0,
        1,
        2,
        0,
    ]
    assert all(
        "(13)" not in item["text"]
        for block in (before, after, second)
        for item in block.payload["items"]
    )


def test_html_list_path_preserves_item_ownership_and_continuations(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"""
        <article>
          <ul id="S3.I2">
            <li id="S3.I2.i1">
              <span class="ltx_tag ltx_tag_item">&#x2022;</span>
              <p id="S3.I2.i1.p1">Before the equation.</p>
              <table class="ltx_equation" id="S3.E13">
                <tr>
                  <td><math alttext="x = 13" display="block"></math></td>
                  <td><span class="ltx_tag">(13)</span></td>
                </tr>
              </table>
              <p id="S3.I2.i1.p2">where the residual is defined.</p>
            </li>
            <li id="S3.I2.i2"><p>Second item.</p></li>
          </ul>
        </article>
        """,
        SourceFormat.HTML,
    )

    document = RichDocumentParserService(repository).parse_source(primary)

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.LIST,
        RichBlockKind.EQUATION,
        RichBlockKind.LIST,
        RichBlockKind.LIST,
    ]
    assert [
        block.payload["items"][0]["text"]
        for block in document.blocks
        if block.kind is RichBlockKind.LIST
    ] == ["Before the equation.", "where the residual is defined.", "Second item."]
    entries = [block.list_path[0] for block in document.blocks]
    assert all(isinstance(entry, RichListPathEntry) for entry in entries)
    assert len({entry.container_id for entry in entries}) == 1
    assert [entry.container_source_id for entry in entries] == ["S3.I2"] * 4
    assert [entry.item_source_id for entry in entries] == [
        "S3.I2.i1",
        "S3.I2.i1",
        "S3.I2.i1",
        "S3.I2.i2",
    ]
    assert len({entry.item_id for entry in entries[:3]}) == 1
    assert entries[3].item_id != entries[0].item_id
    assert [entry.item_index for entry in entries] == [0, 0, 0, 1]
    assert [entry.item_count for entry in entries] == [2, 2, 2, 2]
    assert [entry.depth for entry in entries] == [0, 0, 0, 0]
    assert [entry.ordered for entry in entries] == [False] * 4
    assert [entry.segment_index for entry in entries] == [0, 1, 2, 0]
    assert [entry.continuation for entry in entries] == [
        False,
        True,
        True,
        False,
    ]
    assert document.blocks[0].locator.source_id == "S3.I2.i1.p1"
    assert document.blocks[1].locator.source_id == "S3.E13"
    assert document.blocks[2].locator.source_id == "S3.I2.i1.p2"
    validate_list_paths(
        document.blocks,
        source_target_manifest=source_target_manifest(document),
    )


def test_html_list_path_covers_table_figure_code_and_tail(tmp_path):
    image = tmp_path / "panel.png"
    image.write_bytes(b"panel")
    source = tmp_path / "paper.html"
    source.write_text(
        """
        <article><ol id="L1"><li id="L1.I1">
          <p id="L1.p1">Intro.</p>
          <table id="L1.T1"><tr><td>cell</td></tr></table>
          <figure id="L1.F1"><img src="panel.png" alt="Panel"></figure>
          <pre id="L1.C1"><code class="language-python">print(1)</code></pre>
          <p id="L1.p2">Tail.</p>
        </li></ol></article>
        """,
        encoding="utf-8",
    )
    repository = SourceRepository(tmp_path / "cache")

    document = RichDocumentParserService(repository).parse_source(
        repository.import_path(source)
    )

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.LIST,
        RichBlockKind.TABLE,
        RichBlockKind.FIGURE,
        RichBlockKind.CODE,
        RichBlockKind.LIST,
    ]
    assert [block.locator.source_id for block in document.blocks] == [
        "L1.p1",
        "L1.T1",
        "L1.F1",
        "L1.C1",
        "L1.p2",
    ]
    entries = [block.list_path[0] for block in document.blocks]
    assert [entry.segment_index for entry in entries] == list(range(5))
    assert [entry.continuation for entry in entries] == [
        False,
        True,
        True,
        True,
        True,
    ]
    assert all(entry.ordered for entry in entries)
    assert document.blocks[2].payload["asset_digest"] == (
        document.assets[0].artifact_digest
    )
    assert document.blocks[3].payload == {
        "text": "print(1)",
        "language": "python",
    }


def test_html_nested_lists_preserve_ordered_paths_and_parent_segments(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        b"""
        <article><ul id="L0">
          <li id="L0.I0"><p>Outer intro.</p>
            <ol id="L1">
              <li id="L1.I0"><p>Inner one.</p>
                <table class="ltx_equation" id="L1.E1">
                  <tr><td><math alttext="a = b"></math></td></tr>
                </table>
              </li>
              <li id="L1.I1"><p>Inner two.</p></li>
            </ol>
            <p>Outer tail.</p>
          </li>
          <li id="L0.I1"><p>Next outer.</p></li>
        </ul></article>
        """,
        SourceFormat.HTML,
    )

    document = RichDocumentParserService(repository).parse_source(primary)

    assert [block.kind for block in document.blocks] == [
        RichBlockKind.LIST,
        RichBlockKind.LIST,
        RichBlockKind.EQUATION,
        RichBlockKind.LIST,
        RichBlockKind.LIST,
        RichBlockKind.LIST,
    ]
    assert [len(block.list_path) for block in document.blocks] == [1, 2, 2, 2, 1, 1]
    outer = [block.list_path[0] for block in document.blocks]
    assert [entry.item_source_id for entry in outer] == [
        "L0.I0",
        "L0.I0",
        "L0.I0",
        "L0.I0",
        "L0.I0",
        "L0.I1",
    ]
    assert [entry.segment_index for entry in outer] == [0, 1, 2, 3, 4, 0]
    assert [entry.continuation for entry in outer] == [
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    inner = [document.blocks[index].list_path[1] for index in (1, 2, 3)]
    assert [entry.container_source_id for entry in inner] == ["L1"] * 3
    assert [entry.item_source_id for entry in inner] == ["L1.I0", "L1.I0", "L1.I1"]
    assert [entry.item_index for entry in inner] == [0, 0, 1]
    assert [entry.item_count for entry in inner] == [2, 2, 2]
    assert [entry.depth for entry in inner] == [1, 1, 1]
    assert [entry.ordered for entry in inner] == [True, True, True]
    assert [entry.segment_index for entry in inner] == [0, 1, 0]
    assert [entry.continuation for entry in inner] == [False, True, False]


def test_html_idless_list_identities_are_stable_and_path_derived(tmp_path):
    payload = b"""
    <article><ul><li><p>Same.</p><p>Tail.</p></li><li><p>Same.</p></li></ul></article>
    """
    first = tmp_path / "first.html"
    second = tmp_path / "nested" / "second.html"
    second.parent.mkdir()
    first.write_bytes(payload)
    second.write_bytes(payload)
    repository = SourceRepository(tmp_path / "cache")
    service = RichDocumentParserService(repository)

    first_document = service.parse_source(repository.import_path(first))
    second_document = service.parse_source(repository.import_path(second))

    first_paths = [block.list_path for block in first_document.blocks]
    assert first_paths == [block.list_path for block in second_document.blocks]
    assert first_document.document_digest == second_document.document_digest
    entries = [path[0] for path in first_paths]
    assert len({entry.container_id for entry in entries}) == 1
    assert entries[0].container_id.startswith("list-")
    assert entries[0].item_id.startswith("list-item-")
    assert entries[0].item_id == entries[1].item_id
    assert entries[2].item_id != entries[0].item_id
    assert all(not entry.container_source_id for entry in entries)
    assert all(not entry.container_selector for entry in entries)
    assert all(not entry.item_source_id for entry in entries)
    assert all(not entry.item_selector for entry in entries)


@pytest.mark.parametrize(
    "markup",
    [
        "<ul id='duplicate'><li id='I1'>One.</li></ul>"
        "<ul id='duplicate'><li id='I2'>Two.</li></ul>",
        "<ul id='L'><li id='duplicate'>One.</li>"
        "<li id='duplicate'>Two.</li></ul>",
    ],
)
def test_html_list_paths_reject_duplicate_authored_owner_ids(tmp_path, markup):
    repository = SourceRepository(tmp_path / "cache")

    with pytest.raises(ValueError, match="rich list path"):
        RichDocumentParserService(repository).parse_source(
            _store(
                repository,
                f"<article>{markup}</article>".encode(),
                SourceFormat.HTML,
            )
        )


def test_html_list_path_preserves_empty_authored_items(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"<article><ul id='L'><li id='I0'></li><li id='I1'>Text.</li></ul></article>",
            SourceFormat.HTML,
        )
    )

    assert [block.payload["items"][0]["text"] for block in document.blocks] == [
        "",
        "Text.",
    ]
    assert [block.list_path[0].item_index for block in document.blocks] == [0, 1]
    assert [block.list_path[0].segment_index for block in document.blocks] == [0, 0]


def test_rich_document_v3_list_path_codec_and_v2_compatibility(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    html = _store(
        repository,
        b"<article><ul id='L'><li id='I'><p>One.</p><p>Two.</p></li></ul></article>",
        SourceFormat.HTML,
    )
    document = RichDocumentParserService(repository).parse_source(html)

    encoded = rich_document_to_document(document)
    decoded = rich_document_from_document(encoded)
    assert RICH_DOCUMENT_SCHEMA == "ac.document.rich_document.v3"
    assert encoded["schema_version"] == RICH_DOCUMENT_SCHEMA
    assert encoded["blocks"][0]["list_path"][0]["item_source_id"] == "I"
    assert decoded.document_digest == document.document_digest
    assert decoded.blocks[0].list_path == document.blocks[0].list_path
    assert rich_block_from_document(
        rich_block_to_document(document.blocks[0])
    ).list_path == document.blocks[0].list_path

    markdown = RichDocumentParserService(repository).parse_source(
        _store(repository, b"Legacy body.\n", SourceFormat.MARKDOWN)
    )
    legacy = RichDocument(
        source=markdown.source,
        blocks=markdown.blocks,
        sections=markdown.sections,
        assets=markdown.assets,
        page_map=markdown.page_map,
        metadata=markdown.metadata,
        schema_version=RICH_DOCUMENT_SCHEMA_V2,
    )
    legacy_encoded = rich_document_to_document(legacy)
    assert legacy_encoded["schema_version"] == RICH_DOCUMENT_SCHEMA_V2
    assert all("list_path" not in block for block in legacy_encoded["blocks"])
    assert not rich_block_from_document(legacy_encoded["blocks"][0]).list_path
    legacy_decoded = rich_document_from_document(legacy_encoded)
    assert legacy_decoded.schema_version == RICH_DOCUMENT_SCHEMA_V2
    assert source_presentation(legacy_decoded) is None
    assert legacy_decoded.document_digest == legacy.document_digest
    assert rich_document_to_document(legacy_decoded) == legacy_encoded
    assert all(not block.list_path for block in legacy_decoded.blocks)
    invalid_legacy = json.loads(json.dumps(legacy_encoded))
    invalid_legacy["blocks"][0]["list_path"] = []
    with pytest.raises(ValueError, match="invalid fields"):
        rich_document_from_document(invalid_legacy)

    html_without_lists = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"<article><h1 id='H1'>Heading</h1><p id='P1'>Body.</p></article>",
            SourceFormat.HTML,
        )
    )
    legacy_with_targets = RichDocument(
        source=html_without_lists.source,
        blocks=html_without_lists.blocks,
        sections=html_without_lists.sections,
        assets=html_without_lists.assets,
        page_map=html_without_lists.page_map,
        metadata=html_without_lists.metadata,
        schema_version=RICH_DOCUMENT_SCHEMA_V2,
    )
    decoded_with_targets = rich_document_from_document(
        rich_document_to_document(legacy_with_targets)
    )
    assert source_target_manifest(decoded_with_targets) is not None
    assert source_presentation(decoded_with_targets) is not None
    assert _source_target(decoded_with_targets, "H1")["kind"] == "heading"

    with pytest.raises(ValueError, match="v2 cannot carry list paths"):
        RichDocument(
            source=document.source,
            blocks=document.blocks,
            sections=document.sections,
            assets=document.assets,
            page_map=document.page_map,
            metadata=document.metadata,
            schema_version=RICH_DOCUMENT_SCHEMA_V2,
        )


def test_rich_document_rejects_untrusted_list_item_count_without_large_range(
    tmp_path,
    monkeypatch,
):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"<article><ul><li>One.</li></ul></article>",
            SourceFormat.HTML,
        )
    )
    encoded = rich_document_to_document(document)
    encoded["blocks"][0]["list_path"][0]["item_count"] = 10**12
    builtin_range = range

    def guarded_range(*args):
        if any(isinstance(value, int) and value > 10_000 for value in args):
            raise AssertionError("untrusted item_count reached range()")
        return builtin_range(*args)

    monkeypatch.setattr(
        list_path_validation,
        "range",
        guarded_range,
        raising=False,
    )
    with pytest.raises(ValueError, match="item count"):
        rich_document_from_document(encoded)


def test_rich_document_list_path_rejects_tampering(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    document = RichDocumentParserService(repository).parse_source(
        _store(
            repository,
            b"""
            <article>
              <section><h2>First</h2><ul id="L0">
                <li id="L0.I0"><p>One.</p><ol id="L1">
                  <li id="L1.I0"><p>Nested.</p></li>
                </ol><p>Two.</p></li>
                <li id="L0.I1"><p>Next.</p></li>
              </ul></section>
              <section><h2>Second</h2><p id="outside">Outside.</p></section>
            </article>
            """,
            SourceFormat.HTML,
        )
    )
    original = rich_document_to_document(document)
    owned = [
        index
        for index, block in enumerate(original["blocks"])
        if block["list_path"]
    ]
    first, nested, tail, next_item = owned

    cases = []
    missing_block_path = json.loads(json.dumps(original))
    del missing_block_path["blocks"][first]["list_path"]
    cases.append(missing_block_path)
    unknown_field = json.loads(json.dumps(original))
    unknown_field["blocks"][first]["list_path"][0]["extra"] = True
    cases.append(unknown_field)
    missing_field = json.loads(json.dumps(original))
    del missing_field["blocks"][first]["list_path"][0]["depth"]
    cases.append(missing_field)
    bad_depth = json.loads(json.dumps(original))
    bad_depth["blocks"][nested]["list_path"][1]["depth"] = 0
    cases.append(bad_depth)
    bad_index = json.loads(json.dumps(original))
    bad_index["blocks"][next_item]["list_path"][0]["item_index"] = 2
    cases.append(bad_index)
    bad_item_count = json.loads(json.dumps(original))
    for index in owned:
        bad_item_count["blocks"][index]["list_path"][0]["item_count"] = 3
    cases.append(bad_item_count)
    duplicate_item = json.loads(json.dumps(original))
    duplicate_item["blocks"][next_item]["list_path"][0]["item_id"] = (
        duplicate_item["blocks"][first]["list_path"][0]["item_id"]
    )
    cases.append(duplicate_item)
    segment_gap = json.loads(json.dumps(original))
    segment_gap["blocks"][tail]["list_path"][0]["segment_index"] += 1
    cases.append(segment_gap)
    bad_continuation = json.loads(json.dumps(original))
    bad_continuation["blocks"][tail]["list_path"][0]["continuation"] = False
    cases.append(bad_continuation)
    wrong_parent = json.loads(json.dumps(original))
    wrong_parent["blocks"][nested]["list_path"][0]["item_id"] = (
        wrong_parent["blocks"][next_item]["list_path"][0]["item_id"]
    )
    cases.append(wrong_parent)
    wrong_section = json.loads(json.dumps(original))
    wrong_section["blocks"][tail]["section_path"] = list(
        wrong_section["blocks"][-1]["section_path"]
    )
    cases.append(wrong_section)
    missing_owner = json.loads(json.dumps(original))
    missing_owner["blocks"][tail]["list_path"] = []
    cases.append(missing_owner)
    all_wrong_section = json.loads(json.dumps(original))
    second_section_path = list(all_wrong_section["blocks"][-1]["section_path"])
    for index in owned:
        all_wrong_section["blocks"][index]["section_path"] = second_section_path
    cases.append(all_wrong_section)
    manifest_collision = json.loads(json.dumps(original))
    manifest_collision["blocks"][next_item]["list_path"][0]["item_source_id"] = (
        "outside"
    )
    manifest_collision["blocks"][next_item]["list_path"][0]["item_selector"] = (
        "#outside"
    )
    cases.append(manifest_collision)

    for value in cases:
        with pytest.raises(ValueError, match="list path|rich block"):
            rich_document_from_document(value)


def test_rich_list_path_entry_rejects_non_string_identity():
    with pytest.raises(ValueError, match="list path identity"):
        RichListPathEntry(
            container_id=1,
            container_source_id="",
            container_selector="",
            item_id="item",
            item_source_id="",
            item_selector="",
            item_index=0,
            item_count=1,
            depth=0,
            ordered=False,
            segment_index=0,
            continuation=False,
        )


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


def test_html_unlabelled_equation_table_groups_math_cells_by_row(tmp_path):
    repository = SourceRepository(tmp_path / "cache")
    primary = _store(
        repository,
        br"""
        <article><h1>Overview</h1>
        <table class="ltx_equation" id="S3.EG1">
          <tr id="S3.Ex1">
            <td><math alttext="\lambda_1"></math></td>
            <td><math alttext="="></math></td>
            <td><math alttext="a"></math></td>
          </tr>
          <tr id="S3.Ex2">
            <td><math alttext="\lambda_2"></math></td>
            <td><math alttext="="></math></td>
            <td><math alttext="b"></math></td>
          </tr>
        </table></article>
        """,
        SourceFormat.HTML,
    )

    document = RichDocumentParserService(repository).parse_source(primary)

    equations = [
        block for block in document.blocks
        if block.kind is RichBlockKind.EQUATION
    ]
    assert [block.payload["tex"] for block in equations] == [
        r"\lambda_1 = a",
        r"\lambda_2 = b",
    ]
    assert [block.payload["label"] for block in equations] == ["", ""]


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
