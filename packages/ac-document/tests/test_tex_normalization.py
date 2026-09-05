from __future__ import annotations

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


def test_normalize_tex_removes_redundant_latexml_black_color() -> None:
    assert normalize_tex(
        r"a{\color[rgb]{0,0,0}+}b{\color[rgb]{0.0, 0.00, 0}-}c"
    ) == r"a{+}b{-}c"


def test_normalize_tex_preserves_nonblack_rgb_color() -> None:
    assert normalize_tex(
        r"{\color[rgb]{1,0,0}x}"
    ) == r"{\color[rgb]{1,0,0}x}"
