"""Logical displayed-equation extraction for ar5iv-style HTML tables."""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import Tag

from .tex_lex import normalize_tex


_TAG_CLASS_RE = re.compile(r"(?:^|\s)ltx_tag(?:\s|$)")


@dataclass(frozen=True)
class HtmlEquationUnit:
    """One rendered equation unit, including every MathML fragment it contains."""

    math_nodes: tuple[Tag, ...]
    label: str
    locator_node: Tag


def html_top_level_math_nodes(node: Tag) -> list[Tag]:
    return [
        math
        for math in node.find_all("math")
        if isinstance(math, Tag) and math.find_parent("math") is None
    ]


def html_equation_table_units(table: Tag) -> tuple[HtmlEquationUnit, ...]:
    """Group a rendered equation table by its visible equation-number units.

    ar5iv frequently emits one MathML node per visual fragment of a multiline
    formula while placing only one ``ltx_tag`` in the outer equation table.
    Treating each node as an equation duplicates that visible label.  A table
    with one explicit tag is therefore one logical unit.  If a table contains
    several tagged rows, each tag remains a distinct unit; untagged continuation
    rows are assigned to their nearest tagged row.
    """

    math_nodes = html_top_level_math_nodes(table)
    if not math_nodes:
        return ()
    order = {id(math): index for index, math in enumerate(math_nodes)}
    rows = [
        row
        for row in table.find_all("tr")
        if isinstance(row, Tag) and row.find_parent("table") is table
    ]
    row_maths = [
        tuple(
            math
            for math in html_top_level_math_nodes(row)
            if math.find_parent("tr") is row
        )
        for row in rows
    ]
    row_labels = [_row_equation_labels(row) for row in rows]
    labelled_rows = [
        (index, labels[0])
        for index, labels in enumerate(row_labels)
        if len(labels) == 1
    ]

    if len(labelled_rows) == 1:
        row_index, label = labelled_rows[0]
        return (
            HtmlEquationUnit(
                math_nodes=tuple(math_nodes),
                label=label,
                locator_node=rows[row_index],
            ),
        )
    if labelled_rows:
        grouped: dict[int, list[Tag]] = {
            row_index: list(row_maths[row_index])
            for row_index, _ in labelled_rows
        }
        assigned = {
            id(math)
            for values in grouped.values()
            for math in values
        }
        for row_index, values in enumerate(row_maths):
            if row_labels[row_index] or not values:
                continue
            target_row, _ = min(
                labelled_rows,
                key=lambda candidate: (
                    abs(candidate[0] - row_index),
                    candidate[0] > row_index,
                    candidate[0],
                ),
            )
            grouped[target_row].extend(values)
            assigned.update(id(math) for math in values)
        output = [
            HtmlEquationUnit(
                math_nodes=tuple(
                    sorted(grouped[row_index], key=lambda math: order[id(math)])
                ),
                label=label,
                locator_node=rows[row_index],
            )
            for row_index, label in labelled_rows
            if grouped[row_index]
        ]
        output.extend(
            HtmlEquationUnit((math,), "", math)
            for math in math_nodes
            if id(math) not in assigned
        )
        return tuple(
            sorted(
                output,
                key=lambda unit: min(order[id(math)] for math in unit.math_nodes),
            )
        )

    table_labels = _equation_labels(table)
    if len(table_labels) == 1:
        return (
            HtmlEquationUnit(tuple(math_nodes), table_labels[0], table),
        )
    if not table_labels:
        grouped = [
            HtmlEquationUnit(values, "", row)
            for row, values in zip(rows, row_maths, strict=True)
            if values
        ]
        assigned = {
            id(math)
            for unit in grouped
            for math in unit.math_nodes
        }
        grouped.extend(
            HtmlEquationUnit((math,), "", math)
            for math in math_nodes
            if id(math) not in assigned
        )
        if grouped:
            return tuple(
                sorted(
                    grouped,
                    key=lambda unit: min(
                        order[id(math)] for math in unit.math_nodes
                    ),
                )
            )
    return tuple(HtmlEquationUnit((math,), "", math) for math in math_nodes)


def html_displayed_equation_label(math: Tag) -> str:
    """Return a visible displayed label without falling back across many tags."""

    row = math.find_parent("tr")
    if isinstance(row, Tag):
        labels = _row_equation_labels(row)
        if len(labels) == 1:
            return labels[0]
    table = math.find_parent("table")
    if isinstance(table, Tag):
        labels = _equation_labels(table)
        if len(labels) == 1:
            return labels[0]
    parent = math.parent
    if isinstance(parent, Tag):
        labels = _equation_labels(parent)
        if len(labels) == 1:
            return labels[0]
    return ""


def html_math_tex(node: Tag) -> str:
    """Read the preferred TeX projection from one HTML MathML node."""

    tex = str(node.get("alttext") or node.get("alt") or "")
    if not tex:
        annotation = node.find(
            "annotation", attrs={"encoding": re.compile("tex", re.I)}
        )
        tex = (
            annotation.get_text(" ", strip=True)
            if isinstance(annotation, Tag)
            else ""
        )
    return normalize_tex(tex or node.get_text(" ", strip=True))


def _row_equation_labels(row: Tag) -> list[str]:
    return [
        _normalize_displayed_equation_label(tag.get_text(" ", strip=True))
        for tag in row.find_all(class_=_TAG_CLASS_RE)
        if isinstance(tag, Tag) and tag.find_parent("tr") is row
    ]


def _equation_labels(node: Tag) -> list[str]:
    return [
        _normalize_displayed_equation_label(tag.get_text(" ", strip=True))
        for tag in node.find_all(class_=_TAG_CLASS_RE)
        if isinstance(tag, Tag)
    ]


def _normalize_displayed_equation_label(value: str) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    match = re.fullmatch(r"\(\s*([^()]+?)\s*\)", compact)
    return match.group(1).strip() if match else compact


__all__ = [
    "HtmlEquationUnit",
    "html_displayed_equation_label",
    "html_equation_table_units",
    "html_math_tex",
    "html_top_level_math_nodes",
]
