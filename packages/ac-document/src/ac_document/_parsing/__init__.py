"""Private lexical and source-location primitives shared by AC Foundation parsers."""

from .errors import ParseError
from .tex_lex import normalize_tex

__all__ = ["ParseError", "normalize_tex"]
