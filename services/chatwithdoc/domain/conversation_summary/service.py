from __future__ import annotations

from typing import Optional

from base import AsyncBaseService, BaseModel
from llm import LiteLLMClient

from .prompts.conversation_summary import conversation_summary_prompt


class ConversationSummaryInput(BaseModel):
    question: str
    answer: str
    current_summary: Optional[str] = None


class ConversationSummaryOutput(BaseModel):
    updated_summary: str


class ConversationSummaryService(AsyncBaseService):
    """Incrementally updates the running conversation summary after each turn."""

    llm_client: LiteLLMClient

    async def process(
        self, inputs: ConversationSummaryInput
    ) -> ConversationSummaryOutput:
        prompt = conversation_summary_prompt.format(
            current_summary=inputs.current_summary or "No previous summary.",
            question=inputs.question,
            answer=inputs.answer,
        )
        messages = [{"role": "user", "content": prompt}]

        response = await self.llm_client.completion(messages=messages)
        updated_summary = response.choices[0].message.content.strip()
        return ConversationSummaryOutput(updated_summary=updated_summary)
