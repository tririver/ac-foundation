from __future__ import annotations

from arc_document._parsing import normalize_tex


def test_normalize_tex_preserves_content_environments() -> None:
    assert normalize_tex(
        r"\begin{array}[]{cc}a&b\\c&d\end{array}"
    ) == r"\begin{array}[]{cc}a&b\\c&d\end{array}"


def test_normalize_tex_removes_only_display_environment() -> None:
    assert normalize_tex(
        r"\begin{equation}\begin{matrix}a&b\end{matrix}\end{equation}"
    ) == r"\begin{matrix}a&b\end{matrix}"
