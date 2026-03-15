from __future__ import annotations

from typing import List, Optional

from base import AsyncBaseService, BaseModel
from llm import LiteLLMClient

from .prompts.rephrase_question import rephrase_question_prompt
from ...utils import format_conversation_history


class RephraseQuestionInput(BaseModel):
    question: str
    conversation_history: List[dict]
    conversation_summary: Optional[str] = None


class RephraseQuestionOutput(BaseModel):
    rephrased_question: str


class RephraseQuestionService(AsyncBaseService):
    """Rewrites a follow-up question as a self-contained standalone question."""

    llm_client: LiteLLMClient

    async def process(self, inputs: RephraseQuestionInput) -> RephraseQuestionOutput:
        if not inputs.conversation_history and not inputs.conversation_summary:
            return RephraseQuestionOutput(rephrased_question=inputs.question)

        history_text = format_conversation_history(inputs.conversation_history)
        user_content = (
            f"Conversation Summary:\n{inputs.conversation_summary or 'N/A'}\n\n"
            f"Recent History:\n{history_text}\n\n"
            f"Follow-up Question: {inputs.question}\n\n"
            f"Standalone Question:"
        )

        messages = [
            {"role": "system", "content": rephrase_question_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            response = await self.llm_client.completion(messages=messages, temperature=0)
            rephrased = response.choices[0].message.content.strip()
            return RephraseQuestionOutput(
                rephrased_question=rephrased or inputs.question
            )
        except Exception:
            return RephraseQuestionOutput(rephrased_question=inputs.question)
