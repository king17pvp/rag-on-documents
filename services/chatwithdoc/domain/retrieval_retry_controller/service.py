from __future__ import annotations

from base import AsyncBaseService, BaseModel


class RetrievalRetryControllerInput(BaseModel):
    retry_count: int
    max_retries: int = 3


class RetrievalRetryControllerOutput(BaseModel):
    retry_count: int
    should_retry: bool


class RetrievalRetryControllerService(AsyncBaseService):
    """Tracks retrieval retry attempts and decides whether to retry or escalate."""

    max_retries: int = 3

    async def process(
        self, inputs: RetrievalRetryControllerInput
    ) -> RetrievalRetryControllerOutput:
        new_retry_count = inputs.retry_count + 1
        should_retry = new_retry_count <= self.max_retries
        return RetrievalRetryControllerOutput(
            retry_count=new_retry_count,
            should_retry=should_retry,
        )
