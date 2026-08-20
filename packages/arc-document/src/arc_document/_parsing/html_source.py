from __future__ import annotations

from bs4 import BeautifulSoup, Tag


def html_roots(soup: BeautifulSoup) -> list[Tag | BeautifulSoup]:
    """Return ordered content roots without repeating nested articles."""

    articles = [
        node
        for node in soup.find_all("article")
        if not isinstance(node.find_parent("article"), Tag)
    ]
    return articles or [soup.body or soup]


def html_source_position(
    node: Tag,
) -> tuple[int | None, int | None, int | None, int | None]:
    """Return a real opening-tag point anchor when the parser provides one."""

    source_line = getattr(node, "sourceline", None)
    source_position = getattr(node, "sourcepos", None)
    has_position = (
        isinstance(source_line, int)
        and not isinstance(source_line, bool)
        and source_line >= 1
        and isinstance(source_position, int)
        and not isinstance(source_position, bool)
        and source_position >= 0
    )
    if not has_position:
        return None, None, None, None
    return source_line, source_position + 1, source_line, source_position + 1


def rich_html_selector(node: Tag, ordinal: int) -> str:
    """Return the Rich parser's current source selector."""

    if node.get("id"):
        return f"#{node['id']}"
    return f"{node.name}:nth-block({ordinal + 1})"


__all__ = [
    "html_roots",
    "html_source_position",
    "rich_html_selector",
]
