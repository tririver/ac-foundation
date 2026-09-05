from __future__ import annotations

import re

from ..sources import SourceArtifact
from .errors import ParseError


def normalize_tex(value: str) -> str:
    """Return the TeX normalization currently shared by both projections."""

    value = re.sub(r"(?<!\\)%[^\n]*", "", value)
    value = re.sub(r"\\(?:label|tag)\s*\{[^{}]*\}", "", value)
    # LaTeXML includes line-breaking and default-black presentation commands
    # in MathML TeX projections.  They carry no mathematical semantics and are
    # not portable to renderers such as KaTeX.
    value = re.sub(r"\\penalty(?![A-Za-z@])", "", value)
    value = re.sub(
        r"\\color\s*\[\s*rgb\s*\]\s*"
        r"\{\s*0(?:\.0+)?\s*,\s*0(?:\.0+)?\s*,\s*0(?:\.0+)?\s*\}",
        "",
        value,
    )
    value = re.sub(
        r"(\\begin\s*\{array\})\s*\[\s*\]\s*(?=\{)",
        r"\1",
        value,
    )
    # Remove only outer display wrappers.  Content environments such as
    # ``array`` and ``matrix`` carry layout semantics and must survive into the
    # RichDocument and renderer.
    value = re.sub(
        r"\\begin\s*\{(?:equation|align|gather|multline|eqnarray)\*?\}"
        r"|\\end\s*\{(?:equation|align|gather|multline|eqnarray)\*?\}",
        "",
        value,
    )
    value = value.strip()
    for left, right in (
        (r"\[", r"\]"),
        (r"\(", r"\)"),
        ("$$", "$$"),
        ("$", "$"),
    ):
        if (
            value.startswith(left)
            and value.endswith(right)
            and len(value) >= len(left) + len(right)
        ):
            value = value[len(left) : len(value) - len(right)]
            break
    return " ".join(value.split())


def tex_without_comments(text: str) -> str:
    """Remove TeX comments while preserving the current source line count."""

    text = re.sub(
        r"\\begin\{comment\*?\}.*?\\end\{comment\*?\}",
        lambda match: "\n" * match.group(0).count("\n"),
        text,
        flags=re.DOTALL,
    )
    return "\n".join(
        re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines()
    )


def tex_structural_text(text: str) -> str:
    """Expose structural TeX while preserving original line coordinates.

    Comments, literal environments, inline ``\\verb`` values, and material
    outside an explicit document environment are replaced by spaces. Newlines
    are retained so scanners can report source line numbers without offsets.
    """

    active = tex_without_comments(text)
    literal = re.compile(
        r"\\begin\{(?P<env>verbatim\*?|lstlisting|minted)\}"
        r".*?\\end\{(?P=env)\}",
        re.DOTALL,
    )
    active = literal.sub(lambda match: _mask_tex_text(match.group(0)), active)
    active = re.sub(
        r"\\verb\*?(?P<delimiter>[^\sA-Za-z]).*?(?P=delimiter)",
        lambda match: _mask_tex_text(match.group(0)),
        active,
    )
    begin = re.search(r"\\begin\{document\}", active)
    if begin is None:
        return active
    end = re.search(r"\\end\{document\}", active[begin.end() :])
    body_end = begin.end() + end.start() if end is not None else len(active)
    return (
        _mask_tex_text(active[: begin.end()])
        + active[begin.end() : body_end]
        + _mask_tex_text(active[body_end:])
    )


def _mask_tex_text(value: str) -> str:
    return "".join("\n" if character == "\n" else " " for character in value)


def skip_tex_whitespace(value: str, cursor: int) -> int:
    while cursor < len(value) and value[cursor].isspace():
        cursor += 1
    return cursor


def scan_tex_balanced(
    value: str,
    cursor: int,
    *,
    opening: str,
    closing: str,
    artifact: SourceArtifact,
    description: str,
) -> tuple[str, int]:
    """Scan the Rich parser's current balanced TeX argument subset."""

    if cursor >= len(value) or value[cursor] != opening:
        raise ValueError("balanced TeX scan must start at the opening delimiter")
    depth = 1
    brace_depth = 0
    current = cursor + 1
    while current < len(value):
        character = value[current]
        if character == "\\":
            current += 2
            continue
        if opening == "[" and character == "{":
            brace_depth += 1
            current += 1
            continue
        if opening == "[" and character == "}" and brace_depth:
            brace_depth -= 1
            current += 1
            continue
        if brace_depth:
            current += 1
            continue
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return value[cursor + 1 : current], current + 1
        current += 1
    raise ParseError(
        "unclosed_rich_block",
        f"unclosed {description}",
        artifact=artifact,
    )


def scan_tex_balanced_text(
    value: str,
    cursor: int,
    *,
    opening: str,
    closing: str,
) -> tuple[str, int]:
    """Scan a best-effort balanced text argument without raising at EOF."""

    depth = 1
    current = cursor + 1
    while current < len(value):
        character = value[current]
        if character == "\\":
            current += 2
            continue
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return value[cursor + 1 : current], current + 1
        current += 1
    return value[cursor + 1 :], len(value)


def unwrap_texorpdfstring(value: str) -> str:
    """Replace each valid ``\\texorpdfstring`` with its TeX argument."""

    cursor = 0
    output: list[str] = []
    marker = r"\texorpdfstring"
    while True:
        start = value.find(marker, cursor)
        if start < 0:
            output.append(value[cursor:])
            break
        output.append(value[cursor:start])
        argument = skip_tex_whitespace(value, start + len(marker))
        if argument >= len(value) or value[argument] != "{":
            output.append(marker)
            cursor = start + len(marker)
            continue
        first, after_first = scan_tex_balanced_text(
            value, argument, opening="{", closing="}"
        )
        second_start = skip_tex_whitespace(value, after_first)
        if second_start >= len(value) or value[second_start] != "{":
            output.append(value[start:after_first])
            cursor = after_first
            continue
        _, after_second = scan_tex_balanced_text(
            value, second_start, opening="{", closing="}"
        )
        output.append(first)
        cursor = after_second
    return "".join(output)


def scan_tex_heading(
    lines: list[str],
    index: int,
    artifact: SourceArtifact,
    *,
    cursor: int = 0,
) -> tuple[int, int, str, str] | None:
    """Scan the supported balanced section-heading syntax."""

    command_match = re.search(
        r"\\(section|subsection|subsubsection)(?![A-Za-z@])\*?",
        lines[index][cursor:],
    )
    if command_match is None:
        return None
    remainder = lines[index][cursor:] + (
        ("\n" + "\n".join(lines[index + 1 :]))
        if index + 1 < len(lines)
        else ""
    )
    remainder_cursor = skip_tex_whitespace(remainder, command_match.end())
    if remainder_cursor < len(remainder) and remainder[remainder_cursor] == "[":
        _, remainder_cursor = scan_tex_balanced(
            remainder,
            remainder_cursor,
            opening="[",
            closing="]",
            artifact=artifact,
            description=f"{command_match.group(1)} short title",
        )
        remainder_cursor = skip_tex_whitespace(remainder, remainder_cursor)
    if (
        remainder_cursor >= len(remainder)
        or remainder[remainder_cursor] != "{"
    ):
        raise ParseError(
            "unclosed_rich_block",
            f"{command_match.group(1)} heading has no complete title argument",
            artifact=artifact,
        )
    title, remainder_cursor = scan_tex_balanced(
        remainder,
        remainder_cursor,
        opening="{",
        closing="}",
        artifact=artifact,
        description=f"{command_match.group(1)} title",
    )
    consumed = remainder[:remainder_cursor]
    line_delta = consumed.count("\n")
    end_cursor = (
        cursor + remainder_cursor
        if line_delta == 0
        else len(consumed.rsplit("\n", 1)[-1])
    )
    return (
        index + line_delta,
        end_cursor,
        command_match.group(1),
        title,
    )


__all__ = [
    "normalize_tex",
    "scan_tex_heading",
    "scan_tex_balanced",
    "scan_tex_balanced_text",
    "skip_tex_whitespace",
    "tex_structural_text",
    "tex_without_comments",
    "unwrap_texorpdfstring",
]
