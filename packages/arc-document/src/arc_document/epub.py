"""Deterministic EPUB-to-readable-HTML derivation in publication spine order."""

from __future__ import annotations

import html
import io
import posixpath
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import unquote

from bs4 import BeautifulSoup
from lxml import etree


EPUB_MEDIA_TYPE = "application/epub+zip"
EPUB_READABLE_MEDIA_TYPE = "text/html"
MAX_EPUB_FILES = 20_000
MAX_EPUB_UNCOMPRESSED_BYTES = 500 * 1024 * 1024


class EpubDerivationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class EpubReadableDocument:
    payload: bytes
    spine_paths: tuple[str, ...]
    title: str


def derive_epub_readable_html(payload: bytes) -> EpubReadableDocument:
    """Build one readable HTML document from XHTML items in EPUB spine order."""

    if not isinstance(payload, bytes):
        raise TypeError("EPUB payload must be bytes")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except (OSError, zipfile.BadZipFile) as exc:
        raise EpubDerivationError("epub_invalid", "EPUB is not a readable ZIP archive") from exc
    with archive:
        _validate_archive(archive)
        container = _parse_xml(
            _read_member(archive, "META-INF/container.xml"),
            code="epub_container_invalid",
        )
        rootfiles = container.xpath(
            "//*[local-name()='rootfile']/@full-path"
        )
        if not rootfiles:
            raise EpubDerivationError(
                "epub_container_invalid", "EPUB container has no package document"
            )
        package_path = _safe_member_name(str(rootfiles[0]))
        package = _parse_xml(
            _read_member(archive, package_path),
            code="epub_package_invalid",
        )
        package_dir = posixpath.dirname(package_path)
        manifest: dict[str, tuple[str, str]] = {}
        for item in package.xpath("//*[local-name()='manifest']/*[local-name()='item']"):
            item_id = str(item.get("id") or "")
            href = str(item.get("href") or "")
            media_type = str(item.get("media-type") or "").casefold()
            if item_id and href:
                member = _join_member(package_dir, href)
                manifest[item_id] = (member, media_type)
        spine_paths: list[str] = []
        fragments: list[str] = []
        for itemref in package.xpath("//*[local-name()='spine']/*[local-name()='itemref']"):
            item_id = str(itemref.get("idref") or "")
            entry = manifest.get(item_id)
            if entry is None:
                raise EpubDerivationError(
                    "epub_spine_invalid",
                    f"EPUB spine references missing manifest item: {item_id or '<empty>'}",
                )
            member, media_type = entry
            if media_type not in {
                "application/xhtml+xml",
                "text/html",
                "application/xml",
            }:
                continue
            chapter = _read_member(archive, member)
            fragments.append(_readable_fragment(chapter, member))
            spine_paths.append(member)
        if not fragments:
            raise EpubDerivationError(
                "epub_spine_empty", "EPUB spine contains no readable document items"
            )
        title_values = package.xpath(
            "//*[local-name()='metadata']/*[local-name()='title']/text()"
        )
        title = " ".join(str(title_values[0]).split()) if title_values else ""
        heading = f"<h1>{html.escape(title)}</h1>" if title else ""
        document = (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            f"<title>{html.escape(title or 'EPUB')}</title></head><body>"
            f"{heading}{''.join(fragments)}</body></html>"
        )
        return EpubReadableDocument(
            payload=document.encode("utf-8"),
            spine_paths=tuple(spine_paths),
            title=title,
        )


def _validate_archive(archive: zipfile.ZipFile) -> None:
    infos = archive.infolist()
    if len(infos) > MAX_EPUB_FILES:
        raise EpubDerivationError(
            "epub_too_large", f"EPUB contains more than {MAX_EPUB_FILES} files"
        )
    total = 0
    seen: set[str] = set()
    for info in infos:
        member = _safe_member_name(info.filename)
        if member in seen:
            raise EpubDerivationError(
                "epub_member_invalid", "EPUB contains a duplicate member path"
            )
        seen.add(member)
        total += info.file_size
        if total > MAX_EPUB_UNCOMPRESSED_BYTES:
            raise EpubDerivationError(
                "epub_too_large",
                f"EPUB expands beyond {MAX_EPUB_UNCOMPRESSED_BYTES} bytes",
            )


def _parse_xml(payload: bytes, *, code: str) -> etree._Element:
    try:
        return etree.fromstring(
            payload,
            parser=etree.XMLParser(
                resolve_entities=False,
                no_network=True,
                recover=False,
                huge_tree=False,
            ),
        )
    except (ValueError, etree.XMLSyntaxError) as exc:
        raise EpubDerivationError(code, "EPUB XML is malformed") from exc


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    safe = _safe_member_name(name)
    try:
        return archive.read(safe)
    except (KeyError, OSError, RuntimeError) as exc:
        raise EpubDerivationError(
            "epub_member_unreadable", f"EPUB member is missing or unreadable: {safe}"
        ) from exc


def _join_member(base: str, href: str) -> str:
    path = unquote(href.split("#", 1)[0].split("?", 1)[0])
    return _safe_member_name(posixpath.join(base, path))


def _safe_member_name(value: str) -> str:
    normalized = str(PurePosixPath(str(value).replace("\\", "/")))
    path = PurePosixPath(normalized)
    if (
        not normalized
        or normalized == "."
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise EpubDerivationError(
            "epub_member_invalid", "EPUB contains an unsafe member path"
        )
    return normalized


def _readable_fragment(payload: bytes, member: str) -> str:
    try:
        soup = BeautifulSoup(payload, "lxml")
    except (ValueError, UnicodeError) as exc:
        raise EpubDerivationError(
            "epub_content_invalid", f"EPUB content is malformed: {member}"
        ) from exc
    for element in soup.find_all(["script", "style", "template"]):
        element.decompose()
    body = soup.body
    contents = body.decode_contents() if body is not None else soup.decode()
    return (
        f'<section data-epub-spine="{html.escape(member, quote=True)}">'
        f"{contents}</section>"
    )


__all__ = [
    "EPUB_MEDIA_TYPE",
    "EPUB_READABLE_MEDIA_TYPE",
    "EpubDerivationError",
    "EpubReadableDocument",
    "derive_epub_readable_html",
]
