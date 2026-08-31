"""Validated authored front matter and source-note metadata."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


SOURCE_FRONT_MATTER_METADATA_KEY = "source_front_matter"
SOURCE_FRONT_MATTER_SCHEMA = "ac.document.source_front_matter.v1"
SOURCE_NOTES_METADATA_KEY = "source_notes"
SOURCE_NOTES_SCHEMA = "ac.document.source_notes.v1"

_FRONT_MATTER_FIELDS = {"schema_version", "entries"}
_FRONT_MATTER_ENTRY_FIELDS = {
    "front_matter_id",
    "kind",
    "block_index",
    "locator",
    "authors",
    "affiliations",
    "creator_flow",
}
_AUTHOR_FIELDS = {
    "author_id",
    "source_id",
    "selector",
    "name",
    "markers",
    "orcid",
    "orcid_url",
    "contacts",
}
_CONTACT_FIELDS = {"kind", "label", "value", "target"}
_AFFILIATION_FIELDS = {
    "affiliation_id",
    "source_id",
    "selector",
    "marker",
    "text",
}
_CREATOR_FLOW_FIELDS = {"creator_count", "slot_count", "creators"}
_CREATOR_FIELDS = {
    "creator_id",
    "ordinal",
    "locator",
    "author_id",
    "slots",
}
_CREATOR_SLOT_FIELDS = {
    "slot_id",
    "ordinal",
    "kind",
    "locator",
    "contact_index",
    "affiliation_id",
}
_SOURCE_NOTES_FIELDS = {"schema_version", "notes"}
_NOTE_FIELDS = {
    "note_id",
    "ordinal",
    "marker",
    "body",
    "inline_spans",
    "locator",
    "owner_block_id",
    "owner_locator",
    "anchor",
}
_ANCHOR_FIELDS = {
    "field",
    "item_index",
    "row_index",
    "column_index",
    "start",
    "end",
}
_LOCATOR_FIELDS = {
    "source_format",
    "line_start",
    "column_start",
    "line_end",
    "column_end",
    "selector",
    "source_id",
}
_ORCID_RE = re.compile(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]")
_CREATOR_ID_RE = re.compile(r"front-creator-[0-9a-f]{24}")
_CREATOR_SLOT_ID_RE = re.compile(r"front-creator-slot-[0-9a-f]{24}")


def source_front_matter(document: Any) -> Mapping[str, Any] | None:
    """Return and revalidate structured authored front matter when present."""

    if SOURCE_FRONT_MATTER_METADATA_KEY not in document.metadata:
        return None
    value = document.metadata[SOURCE_FRONT_MATTER_METADATA_KEY]
    validate_source_fidelity_metadata(
        document.metadata,
        blocks=document.blocks,
        source=document.source,
    )
    return value


def source_notes(document: Any) -> Mapping[str, Any] | None:
    """Return authored notes; bind through owner block + anchor only.

    ``owner_locator`` preserves source provenance. It is not a routing key and
    cannot replace the validated ``owner_block_id`` plus payload ``anchor``.
    """

    if SOURCE_NOTES_METADATA_KEY not in document.metadata:
        return None
    value = document.metadata[SOURCE_NOTES_METADATA_KEY]
    validate_source_fidelity_metadata(
        document.metadata,
        blocks=document.blocks,
        source=document.source,
    )
    return value


def validate_source_fidelity_metadata(
    metadata: Mapping[str, Any],
    *,
    blocks: Sequence[Any],
    source: Any,
) -> None:
    """Validate source-fidelity metadata against its RichDocument."""

    if SOURCE_FRONT_MATTER_METADATA_KEY in metadata:
        front_matter = metadata[SOURCE_FRONT_MATTER_METADATA_KEY]
        _validate_front_matter(
            front_matter,
            block_count=len(blocks),
            source_format=source.source_format.value,
        )
    if SOURCE_NOTES_METADATA_KEY in metadata:
        notes = metadata[SOURCE_NOTES_METADATA_KEY]
        _validate_notes(
            notes,
            blocks=blocks,
            source_format=source.source_format.value,
        )


def _validate_front_matter(
    value: Any,
    *,
    block_count: int,
    source_format: str,
) -> None:
    envelope = _mapping(value, "source front matter must be an object")
    _require_fields(envelope, _FRONT_MATTER_FIELDS, "source front matter")
    if envelope.get("schema_version") != SOURCE_FRONT_MATTER_SCHEMA:
        _front_invalid("has an unsupported schema")
    entries = _sequence(envelope.get("entries"), "entries must be a list")
    if not entries:
        _front_invalid("entries are empty")
    front_ids: set[str] = set()
    author_ids: set[str] = set()
    affiliation_ids: set[str] = set()
    authored_ids: set[str] = set()
    creator_ids: set[str] = set()
    creator_slot_ids: set[str] = set()
    creator_source_ids: set[str] = set()
    previous_block_index = -1
    for raw in entries:
        entry = _mapping(raw, "entry must be an object")
        _require_fields(entry, _FRONT_MATTER_ENTRY_FIELDS, "source front matter entry")
        front_id = _nonempty_string(entry, "front_matter_id", "source front matter")
        if front_id in front_ids:
            _front_invalid("contains a duplicate entry identity")
        front_ids.add(front_id)
        if _string(entry, "kind", "source front matter") != "authors":
            _front_invalid("entry kind is unknown")
        block_index = _integer(entry, "block_index", "source front matter")
        if not 0 <= block_index <= block_count or block_index < previous_block_index:
            _front_invalid("entry insertion order is invalid")
        previous_block_index = block_index
        _validate_locator(entry.get("locator"), source_format, "source front matter")
        authors = _sequence(entry.get("authors"), "authors must be a list")
        affiliations = _sequence(
            entry.get("affiliations"), "affiliations must be a list"
        )
        if not authors:
            _front_invalid("authors entry is empty")
        affiliation_markers: set[str] = set()
        entry_affiliations: dict[str, Mapping[str, Any]] = {}
        for raw_affiliation in affiliations:
            affiliation = _mapping(raw_affiliation, "affiliation must be an object")
            _require_fields(
                affiliation,
                _AFFILIATION_FIELDS,
                "source front matter affiliation",
            )
            affiliation_id = _nonempty_string(
                affiliation,
                "affiliation_id",
                "source front matter",
            )
            if affiliation_id in affiliation_ids:
                _front_invalid("contains a duplicate affiliation identity")
            affiliation_ids.add(affiliation_id)
            entry_affiliations[affiliation_id] = affiliation
            _validate_source_selector(
                affiliation,
                source_key="source_id",
                selector_key="selector",
                description="source front matter affiliation",
                authored_ids=authored_ids,
            )
            marker = _string(affiliation, "marker", "source front matter")
            if marker and marker in affiliation_markers:
                _front_invalid("contains a duplicate affiliation marker")
            if marker:
                affiliation_markers.add(marker)
            _nonempty_string(affiliation, "text", "source front matter")
        entry_authors: dict[str, Mapping[str, Any]] = {}
        author_order: list[str] = []
        for raw_author in authors:
            author = _mapping(raw_author, "author must be an object")
            _require_fields(author, _AUTHOR_FIELDS, "source front matter author")
            author_id = _nonempty_string(author, "author_id", "source front matter")
            if author_id in author_ids:
                _front_invalid("contains a duplicate author identity")
            author_ids.add(author_id)
            entry_authors[author_id] = author
            author_order.append(author_id)
            _validate_source_selector(
                author,
                source_key="source_id",
                selector_key="selector",
                description="source front matter author",
                authored_ids=authored_ids,
            )
            _nonempty_string(author, "name", "source front matter")
            markers = _sequence(author.get("markers"), "markers must be a list")
            if any(not isinstance(marker, str) or not marker for marker in markers):
                _front_invalid("author markers are invalid")
            if any(marker not in affiliation_markers for marker in markers):
                _front_invalid("author marker has no declared affiliation")
            orcid = _string(author, "orcid", "source front matter")
            orcid_url = _string(author, "orcid_url", "source front matter")
            if bool(orcid) != bool(orcid_url):
                _front_invalid("author ORCID fields are incomplete")
            if orcid and (
                _ORCID_RE.fullmatch(orcid) is None
                or orcid_url != f"https://orcid.org/{orcid}"
            ):
                _front_invalid("author ORCID fields are inconsistent")
            contacts = _sequence(author.get("contacts"), "contacts must be a list")
            for raw_contact in contacts:
                contact = _mapping(raw_contact, "contact must be an object")
                _require_fields(contact, _CONTACT_FIELDS, "source front matter contact")
                _nonempty_string(contact, "kind", "source front matter")
                _string(contact, "label", "source front matter")
                _nonempty_string(contact, "value", "source front matter")
                _string(contact, "target", "source front matter")
        _validate_creator_flow(
            entry.get("creator_flow"),
            authors=entry_authors,
            author_order=tuple(author_order),
            affiliations=entry_affiliations,
            source_format=source_format,
            creator_ids=creator_ids,
            slot_ids=creator_slot_ids,
            source_ids=creator_source_ids,
        )


def _validate_creator_flow(
    value: Any,
    *,
    authors: Mapping[str, Mapping[str, Any]],
    author_order: tuple[str, ...],
    affiliations: Mapping[str, Mapping[str, Any]],
    source_format: str,
    creator_ids: set[str],
    slot_ids: set[str],
    source_ids: set[str],
) -> None:
    flow = _mapping(value, "creator flow must be an object")
    _require_fields(flow, _CREATOR_FLOW_FIELDS, "source front matter creator flow")
    creators = _sequence(flow.get("creators"), "creators must be a list")
    creator_count = _integer(flow, "creator_count", "source front matter")
    slot_count = _integer(flow, "slot_count", "source front matter")
    if creator_count != len(creators) or creator_count != len(authors):
        _front_invalid("creator flow count differs from normalized authors")
    if creator_count == 0 or slot_count < creator_count:
        _front_invalid("creator flow counts are empty or invalid")

    actual_author_order: list[str] = []
    referenced_affiliations: set[str] = set()
    actual_slot_count = 0
    previous_creator_position: tuple[int, int] | None = None
    for creator_ordinal, raw_creator in enumerate(creators):
        creator = _mapping(raw_creator, "creator must be an object")
        _require_fields(creator, _CREATOR_FIELDS, "source front matter creator")
        creator_id = _nonempty_string(
            creator,
            "creator_id",
            "source front matter",
        )
        if (
            _CREATOR_ID_RE.fullmatch(creator_id) is None
            or creator_id in creator_ids
        ):
            _front_invalid("creator identity is invalid or duplicate")
        creator_ids.add(creator_id)
        if _integer(creator, "ordinal", "source front matter") != creator_ordinal:
            _front_invalid("creator ordinals are not contiguous")
        creator_locator = _validate_locator(
            creator.get("locator"),
            source_format,
            "source front matter creator",
        )
        previous_creator_position = _validate_flow_position(
            creator_locator,
            previous=previous_creator_position,
            description="creator",
        )
        _bind_flow_source_id(
            creator_locator,
            source_ids=source_ids,
        )
        author_id = _nonempty_string(
            creator,
            "author_id",
            "source front matter",
        )
        author = authors.get(author_id)
        if author is None:
            _front_invalid("creator refers to an unknown author")
        if (
            creator_locator["source_id"] != author["source_id"]
            or creator_locator["selector"] != author["selector"]
        ):
            _front_invalid("creator provenance differs from its author source")
        actual_author_order.append(author_id)
        slots = _sequence(creator.get("slots"), "creator slots must be a list")
        if not slots:
            _front_invalid("creator slots are empty")
        actual_slot_count += len(slots)
        contact_indexes: list[int] = []
        author_slot_count = 0
        previous_slot_position: tuple[int, int] | None = None
        for slot_ordinal, raw_slot in enumerate(slots):
            slot = _mapping(raw_slot, "creator slot must be an object")
            _require_fields(
                slot,
                _CREATOR_SLOT_FIELDS,
                "source front matter creator slot",
            )
            slot_id = _nonempty_string(
                slot,
                "slot_id",
                "source front matter",
            )
            if (
                _CREATOR_SLOT_ID_RE.fullmatch(slot_id) is None
                or slot_id in slot_ids
            ):
                _front_invalid("creator slot identity is invalid or duplicate")
            slot_ids.add(slot_id)
            if _integer(slot, "ordinal", "source front matter") != slot_ordinal:
                _front_invalid("creator slot ordinals are not contiguous")
            slot_locator = _validate_locator(
                slot.get("locator"),
                source_format,
                "source front matter creator slot",
            )
            previous_slot_position = _validate_flow_position(
                slot_locator,
                previous=previous_slot_position,
                description="creator slot",
            )
            _bind_flow_source_id(slot_locator, source_ids=source_ids)
            kind = _string(slot, "kind", "source front matter")
            contact_index = slot.get("contact_index")
            affiliation_id = _string(
                slot,
                "affiliation_id",
                "source front matter",
            )
            if kind == "author":
                author_slot_count += 1
                if (
                    slot_ordinal != 0
                    or contact_index is not None
                    or affiliation_id
                ):
                    _front_invalid("author creator slot binding is invalid")
            elif kind == "contact":
                if (
                    not isinstance(contact_index, int)
                    or isinstance(contact_index, bool)
                    or not 0 <= contact_index < len(author["contacts"])
                    or affiliation_id
                ):
                    _front_invalid("contact creator slot binding is invalid")
                contact_indexes.append(contact_index)
            elif kind == "affiliation":
                if contact_index is not None or affiliation_id not in affiliations:
                    _front_invalid("affiliation creator slot binding is invalid")
                referenced_affiliations.add(affiliation_id)
            else:
                _front_invalid("creator slot kind is unknown")
        if author_slot_count != 1:
            _front_invalid("creator does not contain one author slot")
        if contact_indexes != list(range(len(author["contacts"]))):
            _front_invalid("creator contact occurrences are incomplete or unordered")
    if tuple(actual_author_order) != author_order:
        _front_invalid("creator author references differ from normalized order")
    if actual_slot_count != slot_count:
        _front_invalid("creator slot count differs from its occurrences")
    if referenced_affiliations != set(affiliations):
        _front_invalid("creator affiliation occurrences are incomplete")


def _validate_flow_position(
    locator: Mapping[str, Any],
    *,
    previous: tuple[int, int] | None,
    description: str,
) -> tuple[int, int] | None:
    line = locator["line_start"]
    if line is None:
        return previous
    position = (int(line), int(locator["column_start"] or 0))
    if previous is not None and position <= previous:
        _front_invalid(f"{description} locators are out of source order")
    return position


def _bind_flow_source_id(
    locator: Mapping[str, Any],
    *,
    source_ids: set[str],
) -> None:
    source_id = str(locator["source_id"])
    if not source_id:
        return
    if source_id in source_ids:
        _front_invalid("creator flow contains a duplicate source ID")
    source_ids.add(source_id)


def _validate_notes(
    value: Any,
    *,
    blocks: Sequence[Any],
    source_format: str,
) -> None:
    envelope = _mapping(value, "source notes must be an object")
    _require_fields(envelope, _SOURCE_NOTES_FIELDS, "source notes")
    if envelope.get("schema_version") != SOURCE_NOTES_SCHEMA:
        _notes_invalid("has an unsupported schema")
    notes = _sequence(envelope.get("notes"), "notes must be a list")
    if not notes:
        _notes_invalid("notes are empty")
    block_by_id = {block.block_id: block for block in blocks}
    note_ids: set[str] = set()
    authored_ids: set[str] = set()
    previous_owner_ordinal = -1
    previous_owner_block_id = ""
    previous_anchor_order: tuple[int, ...] = ()
    for ordinal, raw in enumerate(notes):
        note = _mapping(raw, "note must be an object")
        _require_fields(note, _NOTE_FIELDS, "source notes")
        note_id = _nonempty_string(note, "note_id", "source notes")
        if note_id in note_ids:
            _notes_invalid("contains a duplicate note identity")
        note_ids.add(note_id)
        if _integer(note, "ordinal", "source notes") != ordinal:
            _notes_invalid("ordinals are not contiguous")
        marker = _nonempty_string(note, "marker", "source notes")
        body = _nonempty_string(note, "body", "source notes")
        locator = _validate_locator(note.get("locator"), source_format, "source notes")
        source_id = locator["source_id"]
        if source_id:
            if note_id != source_id:
                _notes_invalid("note identity differs from its authored ID")
            if source_id in authored_ids:
                _notes_invalid("contains a duplicate authored note ID")
            authored_ids.add(source_id)
        owner_block_id = _nonempty_string(
            note,
            "owner_block_id",
            "source notes",
        )
        owner = block_by_id.get(owner_block_id)
        if owner is None:
            _notes_invalid("refers to an unknown owner block")
        if owner.ordinal < previous_owner_ordinal:
            _notes_invalid("owners are out of source order")
        anchor_order = _validate_note_anchor(
            note.get("anchor"), owner=owner, marker=marker
        )
        if (
            owner.ordinal == previous_owner_ordinal
            and owner_block_id == previous_owner_block_id
            and anchor_order <= previous_anchor_order
        ):
            _notes_invalid("anchors are out of source order within one owner")
        previous_owner_ordinal = owner.ordinal
        previous_owner_block_id = owner_block_id
        previous_anchor_order = anchor_order
        _validate_locator(note.get("owner_locator"), source_format, "source notes")
        _validate_inline_spans(note.get("inline_spans"), text=body)
        if not marker.strip():
            _notes_invalid("marker is blank")


def _validate_note_anchor(
    value: Any,
    *,
    owner: Any,
    marker: str,
) -> tuple[int, ...]:
    anchor = _mapping(value, "source notes anchor must be an object")
    _require_fields(anchor, _ANCHOR_FIELDS, "source notes anchor")
    field = _string(anchor, "field", "source notes")
    item_index = anchor.get("item_index")
    row_index = anchor.get("row_index")
    column_index = anchor.get("column_index")
    owner_kind = getattr(owner.kind, "value", owner.kind)
    target = ""
    if field == "text":
        if owner_kind != "paragraph" or any(
            value is not None for value in (item_index, row_index, column_index)
        ):
            _notes_invalid("text anchor differs from its owner")
        target = owner.payload["text"]
        order = (0,)
    elif field == "list_item":
        if owner_kind != "list" or row_index is not None or column_index is not None:
            _notes_invalid("list anchor differs from its owner")
        index = _optional_index(item_index, "item index")
        items = owner.payload["items"]
        if index >= len(items):
            _notes_invalid("list anchor item is out of bounds")
        target = items[index]["text"]
        order = (0, index)
    elif field == "table_header":
        if owner_kind != "table" or item_index is not None or row_index is not None:
            _notes_invalid("table header anchor differs from its owner")
        column = _optional_index(column_index, "column index")
        headers = owner.payload["headers"]
        if column >= len(headers):
            _notes_invalid("table header anchor is out of bounds")
        target = headers[column]
        order = (0, column)
    elif field == "table_cell":
        if owner_kind != "table" or item_index is not None:
            _notes_invalid("table cell anchor differs from its owner")
        row = _optional_index(row_index, "row index")
        column = _optional_index(column_index, "column index")
        rows = owner.payload["rows"]
        if row >= len(rows) or column >= len(rows[row]):
            _notes_invalid("table cell anchor is out of bounds")
        target = rows[row][column]
        order = (1, row, column)
    else:
        _notes_invalid("anchor field is unknown")
    start = _integer(anchor, "start", "source notes")
    end = _integer(anchor, "end", "source notes")
    if not 0 <= start < end <= len(target) or target[start:end] != marker:
        _notes_invalid("anchor range does not identify its marker")
    return (*order, start, end)


def _optional_index(value: Any, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _notes_invalid(f"anchor {description} is invalid")
    return value


def _validate_locator(
    value: Any,
    source_format: str,
    description: str,
) -> Mapping[str, Any]:
    locator = _mapping(value, f"{description} locator must be an object")
    _require_fields(locator, _LOCATOR_FIELDS, f"{description} locator")
    if locator.get("source_format") != source_format:
        _invalid(description, "locator format differs from its source")
    positions = [
        locator.get("line_start"),
        locator.get("column_start"),
        locator.get("line_end"),
        locator.get("column_end"),
    ]
    if any(
        item is not None
        and (not isinstance(item, int) or isinstance(item, bool) or item < 1)
        for item in positions
    ):
        _invalid(description, "locator positions are invalid")
    if (positions[0] is None) != (positions[2] is None) or (
        positions[1] is None
    ) != (positions[3] is None):
        _invalid(description, "locator ranges are incomplete")
    if positions[1] is not None and positions[0] is None:
        _invalid(description, "locator columns require line positions")
    if positions[0] is not None and positions[2] < positions[0]:
        _invalid(description, "locator range is reversed")
    if (
        positions[0] == positions[2]
        and positions[1] is not None
        and positions[3] < positions[1]
    ):
        _invalid(description, "locator range is reversed")
    source_id = _string(locator, "source_id", description)
    selector = _string(locator, "selector", description)
    if source_id and selector != f"#{source_id}":
        _invalid(description, "locator selector differs from its source ID")
    if not source_id and selector:
        _invalid(description, "locator selector has no source ID")
    return locator


def _validate_source_selector(
    value: Mapping[str, Any],
    *,
    source_key: str,
    selector_key: str,
    description: str,
    authored_ids: set[str],
) -> None:
    source_id = _string(value, source_key, description)
    selector = _string(value, selector_key, description)
    if source_id:
        if selector != f"#{source_id}" or source_id in authored_ids:
            _invalid(description, "source ID or selector conflicts")
        authored_ids.add(source_id)
    elif selector:
        _invalid(description, "selector has no source ID")


def _validate_inline_spans(value: Any, *, text: str) -> None:
    spans = _sequence(value, "inline spans must be a list")
    cursor = 0
    reconstructed: list[str] = []
    fields = {
        "text": {"kind", "start", "end", "text"},
        "link": {"kind", "start", "end", "text", "target"},
        "math": {"kind", "start", "end", "text", "tex", "source"},
    }
    for raw in spans:
        span = _mapping(raw, "inline span must be an object")
        kind = span.get("kind")
        if kind not in fields:
            _notes_invalid("inline span kind is invalid")
        _require_fields(span, fields[str(kind)], "source notes inline span")
        start = _integer(span, "start", "source notes")
        end = _integer(span, "end", "source notes")
        span_text = _string(span, "text", "source notes")
        if start != cursor or end <= start or end - start != len(span_text):
            _notes_invalid("inline spans do not contiguously cover the body")
        cursor = end
        reconstructed.append(span_text)
        if kind == "link":
            _nonempty_string(span, "target", "source notes")
        elif kind == "math":
            _nonempty_string(span, "tex", "source notes")
            _nonempty_string(span, "source", "source notes")
    if "".join(reconstructed) != text:
        _notes_invalid("inline spans do not reconstruct the body")


def _mapping(value: Any, message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(message)
    return value


def _sequence(value: Any, message: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(message)
    return value


def _require_fields(
    value: Mapping[str, Any],
    fields: set[str],
    description: str,
) -> None:
    if set(value) != fields:
        _invalid(description, "has invalid fields")


def _string(value: Mapping[str, Any], key: str, description: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        _invalid(description, f"{key} must be a string")
    return item


def _nonempty_string(
    value: Mapping[str, Any],
    key: str,
    description: str,
) -> str:
    item = _string(value, key, description)
    if not item:
        _invalid(description, f"{key} is empty")
    return item


def _integer(value: Mapping[str, Any], key: str, description: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        _invalid(description, f"{key} must be an integer")
    return item


def _front_invalid(message: str) -> None:
    raise ValueError(f"source front matter {message}")


def _notes_invalid(message: str) -> None:
    raise ValueError(f"source notes {message}")


def _invalid(description: str, message: str) -> None:
    raise ValueError(f"{description} {message}")


__all__ = [
    "SOURCE_FRONT_MATTER_METADATA_KEY",
    "SOURCE_FRONT_MATTER_SCHEMA",
    "SOURCE_NOTES_METADATA_KEY",
    "SOURCE_NOTES_SCHEMA",
    "source_front_matter",
    "source_notes",
    "validate_source_fidelity_metadata",
]
