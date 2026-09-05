from __future__ import annotations

import pytest

from ac_document._parsing import normalize_tex


def test_normalize_tex_preserves_content_environments() -> None:
    assert normalize_tex(
        r"\begin{array}[]{cc}a&b\\c&d\end{array}"
    ) == r"\begin{array}{cc}a&b\\c&d\end{array}"


def test_normalize_tex_removes_only_display_environment() -> None:
    assert normalize_tex(
        r"\begin{equation}\begin{matrix}a&b\end{matrix}\end{equation}"
    ) == r"\begin{matrix}a&b\end{matrix}"


def test_normalize_tex_removes_latexml_penalty_hints() -> None:
    assert normalize_tex(
        r"H\sim 10^{14}\penalty\ {\rm GeV}"
    ) == r"H\sim 10^{14}\ {\rm GeV}"


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (r"x\penalty100y", "xy"),
        (r"x\penalty+7y", "xy"),
        (r"x\penalty -50 y", "x y"),
        (r"x\penalty - 50 y", "x y"),
        (r"x\penalty -- 50 y", "x y"),
        (r"x\penalty y 42", "x y 42"),
        (r"x\penalty - \alpha 42", r"x - \alpha 42"),
        (r"x\penalty -- \alpha 42", r"x -- \alpha 42"),
        (r"x\\penalty100y", r"x\\penalty100y"),
        (r"x\penaltyvalue 42", r"x\penaltyvalue 42"),
    ),
)
def test_normalize_tex_removes_only_penalty_integer_arguments(
    value: str,
    expected: str,
) -> None:
    assert normalize_tex(value) == expected


def test_normalize_tex_removes_redundant_latexml_black_color() -> None:
    assert normalize_tex(
        r"a{\color[rgb]{0,0,0}+}b{\color[rgb]{0.0, 0.00, 0}-}c"
    ) == r"a{+}b{-}c"


def test_normalize_tex_preserves_nonblack_rgb_color() -> None:
    assert normalize_tex(
        r"{\color[rgb]{1,0,0}x}"
    ) == r"{\color[rgb]{1,0,0}x}"
