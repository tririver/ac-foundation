# ac-document

Host-independent document infrastructure for AC Foundation: immutable source storage,
deterministic parsing, rich-document contracts, cached search, document
structure, and terminology workflows. It contains no paper-provider behavior.

`AcDocumentService` accepts local files and repository artifacts. Academic
identifiers and providers belong to a consumer package, not `ac-document`.

HTML MathML TeX projections are normalized before entering a `RichDocument`.
LaTeXML-only line-breaking hints, redundant default-black color commands, and
empty `array` option lists are removed while semantic TeX is preserved.
LaTeXML equation tables without visible equation numbers retain one logical
equation unit per authored table row instead of emitting each MathML cell as a
separate display equation.

## Resilient optional HTML projections

HTML parses record `metadata.document_diagnostics` with schema
`ac.document.document_diagnostics.v1`. It is a codec-validated local ledger
for optional source presentation, Figure/Table layout, source-target
navigation, front matter, and notes. Each entry has a stable category, source
scope and locator, `exact`, `neutral`, or `unavailable` status, a fallback
action, and bounded evidence. Its visible-content accounting reconciles every
visible source unit to emitted content, a documented exclusion, or a safe
plain fallback; `unaccounted` is always zero.

Ownership is enumerated from the DOM before RichDocument projection: nested
and repeated wrappers receive one non-overlapping unit, article-internal
navigation is a documented exclusion, and visible siblings outside selected
article roots are recorded as source chrome rather than guessed as paper body.

Exact projection validators remain fail-closed: malformed optional layout,
target, or presentation data is never admitted as authoritative. The parser
instead keeps safe core blocks where possible, uses a neutral projection or
source-preserving plain block where it is not, and records the degradation.
An HTML structure that degrades to a plain paragraph still publishes that
paragraph's exact plain rich field, so one local fallback cannot invalidate
otherwise valid presentation metadata for the rest of the document.
Source identity, UTF-8 decoding, source digest, unsafe input, and core
`RichBlock`/codec contract failures remain hard errors. Existing v2/v3
documents and Markdown/TeX canonical output remain compatible when this
optional key is absent.

Explicit Markdown, TeX, and HTML keyword/index metadata is normalized before
terminology workflows consume it. A value made only of Unicode dash punctuation
is an empty publisher placeholder and is omitted; an authored Keywords section
remains ordinary document content and can still supply the real term list.
Model-proposed terms enter the public keyword result only when the term or one
of its aliases has a literal source occurrence. Ungrounded semantic guesses
remain durable model evidence but do not become glossary entries or Reader
omission warnings. Cached legacy inventories receive the same grounded result
projection.

```bash
ac-document --help
ac-document export-rich-document source.md --output-dir publication
ac-document acquire-html-bundle https://example.org/document.html --output-dir document-bundle
python -m pytest packages/ac-document/tests
```

## Explicit HTML bundle acquisition

`ac-document acquire-html-bundle` is the only URL-fetching document command.
It performs a bounded public-HTTPS acquisition, stores the original HTML plus
available same-origin image dependencies in the AC document cache, and emits an
`ac.document.html_source_export.v1` `manifest.json` into the required output
directory. Its nested `bundle` is an `ac.document.html_source_bundle.v1` document.
Local import, parse, and export
operations remain network-free. Dependencies that cannot be fetched safely are
represented as structured bundle warnings; their bytes are never fabricated.
Dependencies are same-final-origin by default. A caller can set
`same_origin_dependencies=False` only with a nonempty explicit `allowed_origins` policy
that includes every additional public HTTPS origin it intends to acquire from.

The public `HTMLSourceAcquisitionService.acquire_dependencies()` API accepts a
caller-verified HTML primary and a storage adapter, so consumers can keep their
own cache/archive implementation. `materialize_html_source_bundle()` publishes
an atomically verified local export. Safe relative authored targets retain the
original primary HTML bytes; only targets that cannot be safely materialized in
place are rewritten and recorded in the export manifest.
`verify_html_source_bundle_export()` lets an offline consumer reload that
manifest and verify the staged `source.html` and each referenced resource before
use.

## RichDocument list ancestry

RichDocument v3 keeps authored list content as flat, independently addressable
blocks. Each block inside an HTML list item carries an ordered `list_path`.
Entries preserve deterministic container/item identities, authored IDs and
selectors when present, item index/count, nesting depth, ordered semantics, and
an exact segment index. `continuation` is true precisely after segment zero, so
consumers can draw one marker per authored item without inspecting block text.

Construction and codec decoding reject conflicting owners, duplicate authored
IDs, invalid nesting/indexes, discontinuous segments, section mismatches, and
source-target alias collisions. Declared item counts must equal the bounded
emitted item coverage and are never expanded as an untrusted numeric range.
Existing v2 documents remain decodable and
round-trip as v2 with no `list_path`; reparsing is required to migrate them to
v3 and changes the document digest, so document-bound derived artifacts must be
rebuilt.

## Authored front matter and notes

HTML RichDocuments may expose two independently versioned metadata contracts:
`source_front_matter` (`ac.document.source_front_matter.v1`) and
`source_notes` (`ac.document.source_notes.v1`). Front matter preserves the
exact insertion point, locator, ordered authors, authored markers, ORCIDs,
contacts, and affiliations. Notes preserve one marker in owner content plus a
separate rich body, exact note and owner locators, final owner block ID,
validated paragraph/list/table marker anchor, and source order. Note bodies
retain inline links and math without duplicating the
body or LaTeXML's nested marker markup inside paragraph/table payloads.
LaTeXML `ltx_role_footnotemark` nodes that contain only a marker and the
generated `footnotemark:` accessibility label are not source notes: their
visible marker remains in the owner content, while no empty standalone note is
invented. Separate authored Table-note paragraphs remain part of the Table.
Known LaTeXML publication-note containers adjacent to the byline are excluded
from the primary title and ordinary body flow. Empty contact scaffolding is
omitted, while a creator-owned ORCID link remains attached to its normalized
author even when LaTeXML nests it beside the person name.
Consumers bind a note only through `owner_block_id` plus `anchor`.
`owner_locator` is immutable source provenance, not a routing or binding key;
serialized locator changes are covered by the RichDocument digest.
The current anchor contract covers paragraph text, LIST items, and Table
headers/cells. A note in a heading, Figure caption, or Table caption cannot
claim an exact note projection: the parser emits its safe plain body fallback
and records an unavailable `source_notes` diagnostic rather than dropping
visible source content or rejecting the whole document.

Each authored front-matter entry also carries an exact `creator_flow` for
source-faithful presentation. Ordered creator groups reference one normalized
author and contain ordered typed slots: the author occurrence, normalized
contacts by stable per-author index, and normalized affiliations by identity.
Slots keep deterministic identities and source locators but do not copy names,
emails, ORCIDs, or affiliation text. The same affiliation identity may occur in
several creator groups, including a trailing registry inside the final source
creator. Such repetition records presentation occurrence only; semantic
author-to-affiliation association remains exclusively the normalized author
markers plus affiliation registry.

Consumers render the source front matter by creator-flow group/slot order, but
use normalized authors/contacts/affiliations for lookup and association. A
translation surface may reuse the source grouping while leaving person names,
emails, ORCIDs, and other identifiers untranslated, and it must not infer
affiliation ownership from the group containing an occurrence. Responsive
layout and pixel styling remain consumer concerns. The producer recognizes
only direct LaTeXML/ar5iv `ltx_creator ltx_role_author` structure; it never
groups creators by names, marker text, institution similarity, coordinates, or
adjacency.

Both contracts have exact nested field sets and are validated during
RichDocument construction and decoding. Documents without these optional keys,
including existing v2 and v3 documents, retain their original codec behavior.
An absent key is the legacy case; an explicitly present `null` value is invalid.
The earlier unpublished `source_front_matter.v1` draft lacked creator flow and
is intentionally rejected when present; producer and consumer artifacts from
that draft must be reparsed/rebuilt together.
Consumers must reparse and rebuild document-bound artifacts to acquire the new
metadata. HTML parsing requires every visible article flow event to emit
content, emit structured front matter, use a documented structural exclusion,
or receive a safe plain fallback; the diagnostics ledger rejects a nonzero
unaccounted count. Figure, Table, and panel order remains the authored HTML
order.

## Authored source presentation

HTML RichDocuments may also expose `metadata.source_presentation` with schema
`ac.document.source_presentation.v1`. When present, it is the authoritative
rich view of otherwise plain block fields: heading and paragraph text, LIST
items, Figure and Table captions, and every Table header/cell. Typed link and
math spans reconstruct the unchanged plain value; independent `strong` and
`emphasis` ranges preserve source-authored marks even when they overlap a link
or math span. Closed semantic heading roles distinguish `abstract`,
`classification`, and `acknowledgements` from a bare HTML heading level without
inspecting displayed text.

For exact LaTeXML/ar5iv abstract and acknowledgements conventions, the heading
block's plain `payload.level` is the semantic document level rather than the
presentational `h1`...`h6` tag number. A document-front-matter abstract is a
level-2 child of the document title. A root acknowledgement is also level 2;
an acknowledgement under an authored HTML `section` is one level below that
section's unique preceding direct heading. A parent already at level 6,
conflicting abstract/acknowledgement conventions, repeated nested convention
ancestors, or a missing/ambiguous section parent prevents that optional
semantic projection from being admitted. Ordinary h6
elements, literal `Abstract`/`Acknowledgements` text, and unknown classes retain
their authored numeric tag level and receive no semantic role. Classification
headings retain their authored level and remain outside the outline. Consumers
use the semantic block level for outlines and Markdown; they must not recover
roles or levels from heading text, neighboring blocks, or the raw HTML tag.

An exact `classifications` relation binds one classification heading to its
ordered value blocks and declares inline composition. The `": "` separator is
only declared for the LaTeXML/ar5iv semantic pair `ltx_classification` plus
`ltx_title ltx_title_classification`; it comes from that stylesheet profile's
title `:after` rule, not from displayed text. Unknown, missing, nested, or
ambiguous structures expose no relation, so consumers must not merge adjacent
blocks heuristically. The exact provenance token is
`latexml_ar5iv_classification_after`; the producer does not apply the separator
to arbitrary HTML headings or parse remote CSS. Classification headings remain
outside the outline.

The unified `captions` registry covers every visible Figure and Table caption.
It preserves `before_content`, `after_content`, or Table-only `embedded`
placement and a nullable logical alignment. Alignment is authoritative only
when backed by exact semantic `text-align:start|center|end` style or LaTeXML
`ltx_centering`/`ltx_align_center` class tokens; unknown evidence is explicit
neutral metadata and conflicting evidence degrades only that optional
presentation projection. Translation consumers
reuse source placement/alignment while keeping translated caption content
independent. Table entries separately preserve ordered authored cell origins,
including `rowspan`/`colspan`, source cell kind, and locator. Each origin also
carries a nullable horizontal alignment with exact class/style evidence and an
ordered set of authored physical rule edges. Alignment preserves physical
`left`/`right`, logical `start`/`end`, and `center` without converting between
them. Rule edges preserve physical `top`/`right`/`bottom`/`left` provenance;
covered span positions never receive independent style.

Consumers treat present Table-cell metadata as authoritative: start with no
synthetic grid, then draw only declared rule edges and apply only declared
alignment. The producer recognizes a closed set of LaTeXML alignment/border
classes and safe `text-align` keywords. Unknown alignment remains neutral;
recognized conflicts, duplicate physical edges, unsupported inline border CSS,
and unknown `ltx_border_*` classes are kept out of an exact presentation
projection. Table raw padding, arbitrary
style, pixel dimensions, and TeX lengths are deliberately outside the Table
contract, so consumers retain their safe default cell padding. Authored span
coverage is aggregate-bounded before grid expansion to reject hostile or
accidentally enormous rectangular spans.

LaTeXML transformed Tables that use one outer `span.ltx_tabular` with
`span.ltx_tr` and direct `span.ltx_td` children are normalized through the
same bounded grid contract as native HTML Tables. Nested `ltx_tabular`
structures remain cell content; multiple independent outer grids, malformed
rows, or unsafe spans retain the existing visible plain-text fallback.

The ordered `figures` registry covers every Figure that has an authoritative
source-target panel manifest. Each descriptor joins by final Figure `block_id`;
its panels join the existing asset/status manifest by contiguous `panel_index`
and exact authored `source_id`. An exact LaTeXML/ar5iv single
`img|object.ltx_graphics` is a one-column `single` layout. The graphic may be
directly owned by the Figure or nested through a pure single-child `p`/`span`
wrapper chain; wrappers with authored text, extra elements, or other layout
structure remain ambiguous and produce a neutral Figure projection with a
diagnostic. An exact direct
`ltx_flex_figure` preserves ordered rows, exact `ltx_flex_break` positions,
and each row's authored `ltx_flex_size_1|2|3` source. Cells within one row
must use one size, while explicitly separated rows may use different sizes;
the root `column_source` is null for that mixed-row case and `column_count`
is the maximum authored row capacity. The
producer never derives columns from panel count, filenames, captions, or Figure
numbers. Addressable generic HTML Figures outside that closed profile receive a
`neutral` descriptor with no row/column claim; Figures with no exact target
alias remain outside the registry.
If one authored Figure wrapper owns multiple direct captions, the producer
does not guess a caption-to-panel association. It emits every media and caption
inline flow in original DOM order as source-preserving blocks and records a
`figure_layout` diagnostic; caption math, links, and marks remain structured.

Recognized Figure panels preserve positive bounded integer `width`/`height`
attributes and a reduced positive `style:aspect-ratio` pair with closed
provenance tokens. Either source may be absent and is then explicitly null; if
both dimensions and aspect ratio exist they must agree. Unknown or nested flex
structure, multiple flex roots, mixed size classes within one row, unknown
size classes, empty row breaks,
cells without exactly one direct panel graphic, malformed dimensions, and
conflicting aspect ratios are not admitted as exact layout. Consumers treat a
present exact layout
as authoritative, but use their own responsive sizing policy; raw remote CSS,
arbitrary style, pixel typography, and acquisition behavior are not part of
this contract.

Construction and codec decoding reject unknown or duplicate fields,
view/plain reconstruction mismatches, invalid spans or marks, missing block
fields, classification binding/order/separator errors, caption identity/order/
evidence conflicts, Table-cell presentation conflicts, and overlapping or
out-of-bounds cell geometry. Figure validation additionally rejects registry
coverage/order errors, panel-manifest mismatches, incomplete/overlapping grid
placement, invalid row breaks, dimensions, ratios, or provenance. Existing
v2/v3 documents without this optional metadata remain valid; consumers must not
heuristically reconstruct absent presentation. The unpublished earlier v1
drafts stored Table placement inside `tables` and later omitted `figures`;
those shapes are intentionally rejected now, with placement owned only by
`captions` and Figure layout owned only by `figures`. Producer and consumer
artifacts from those drafts must be reparsed/rebuilt together. Reparsing changes
Figure block IDs and the document digest, so translations and other
document-bound artifacts must also be rebuilt.
As with the other optional source contracts, explicit `null` is invalid rather
than equivalent to absence.

## Authoritative source targets

HTML RichDocuments may expose `metadata.source_target_manifest` with schema
`ac.document.source_target_manifest.v1`. Each exact authored alias maps to an
existing canonical block and a validated half-open block range. Section aliases
map to declared outline ranges without rewriting heading locators. Figure
targets may include ordered panel descriptors whose status is `available`,
`missing`, or `unsupported`; a compound wrapper always targets its parent
Figure block rather than panel zero.
An exact internal link may also target a uniquely owned descendant of one
emitted block, such as an authored note paragraph inside a Table wrapper. The
manifest binds that descendant alias to its containing block; ambiguous split
ownership is omitted rather than inferred. Source-note IDs remain governed by
`source_notes` and are not duplicated into this registry.

Consumers should prefer a present, valid manifest and fail closed on conflicts,
unknown kinds, missing blocks, invalid ranges, or inconsistent panels. Parser
admission drops a malformed optional manifest and records an unavailable
diagnostic while preserving core blocks. Documents without the metadata remain
compatible with a unique exact-locator fallback.
An explicitly present `null` manifest is invalid; only an absent key selects
the fallback.
