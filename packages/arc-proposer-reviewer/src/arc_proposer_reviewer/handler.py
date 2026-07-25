from __future__ import annotations

from arc_jobs import RunContext, RunOutcome

from .models import ExecutionOptions
from .protocol import decode_batch_request
from .service import ProposerReviewerService


class ProposerReviewerHandler:
    name = "arc.proposer_reviewer.batch.v3"

    def __init__(
        self,
        service: ProposerReviewerService,
        *,
        options: ExecutionOptions = ExecutionOptions(),
    ) -> None:
        self.service = service
        self.options = options

    def execute(self, context: RunContext) -> RunOutcome:
        request = decode_batch_request(context.semantic_input)
        return self.service.execute(context, request, options=self.options)
