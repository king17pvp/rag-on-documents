from __future__ import annotations

from typing import List, Optional

from base import AsyncBaseService, BaseModel
from llm import LiteLLMClient

from .prompts.answer_generator import answer_generator_prompt
from ...utils import format_context_docs


class AnswerGeneratorInput(BaseModel):
    question: str
    reranked_docs: List[dict]
    conversation_summary: Optional[str] = None


class AnswerGeneratorOutput(BaseModel):
    answer: str


class AnswerGeneratorService(AsyncBaseService):
    """Generates a grounded answer from the retrieved context."""

    llm_client: LiteLLMClient

    async def process(self, inputs: AnswerGeneratorInput) -> AnswerGeneratorOutput:
        context_text = format_context_docs(inputs.reranked_docs)
        system_msg = answer_generator_prompt.format(
            conversation_summary=inputs.conversation_summary or "N/A",
        )
        user_msg = f"Context:\n{context_text}\n\nQuestion: {inputs.question}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        response = await self.llm_client.completion(messages=messages)
        answer = response.choices[0].message.content.strip()
        return AnswerGeneratorOutput(answer=answer)
