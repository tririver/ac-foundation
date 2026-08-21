from __future__ import annotations

import ac_document
from ac_document import workflow_support


def test_workflow_support_is_public() -> None:
    assert ac_document.workflow_support is workflow_support
    assert workflow_support.DocumentWorkflowError.__module__.startswith("ac_document")
