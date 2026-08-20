"""Provider-neutral document infrastructure for ARC."""

__version__ = "1.1.0"

from .cached_document import *
from .cached_full_text_search import *
from .document_search import *
from .document_structure import *
from .epub import *
from .parse import *
from .rich_document import *
from .service import ArcDocumentService, DocumentInputError, default_cache_root
from .source_repository import *
from .sources import *
from .terms import *
from .workflows import *

# Export public imported names while keeping implementation modules private.
__all__ = [name for name in globals() if not name.startswith("_")]
