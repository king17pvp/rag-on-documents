from __future__ import annotations

from typing import List, Optional

from base import AsyncBaseService, BaseModel
from llm import LiteLLMClient

from .prompts.answer_confidence_checker import answer_confidence_checker_prompt
from ...utils import format_context_docs, parse_json_response

VALID_CONFIDENCE = {"high", "low"}


class AnswerConfidenceCheckerInput(BaseModel):
    question: str
    answer: str
    reranked_docs: List[dict]


class AnswerConfidenceCheckerOutput(BaseModel):
    answer_confidence: str  # "high" | "low"
    reasoning: Optional[str] = None


class AnswerConfidenceCheckerService(AsyncBaseService):
    """Checks whether the generated answer is complete and well-supported."""

    llm_client: LiteLLMClient

    async def process(
        self, inputs: AnswerConfidenceCheckerInput
    ) -> AnswerConfidenceCheckerOutput:
        context_text = format_context_docs(inputs.reranked_docs)
        prompt = answer_confidence_checker_prompt.format(
            question=inputs.question,
            answer=inputs.answer,
            context=context_text,
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.llm_client.completion(messages=messages, temperature=0)
            content = response.choices[0].message.content
            data = parse_json_response(content)
            confidence = data.get("confidence", "low")
            if confidence not in VALID_CONFIDENCE:
                confidence = "low"
            return AnswerConfidenceCheckerOutput(
                answer_confidence=confidence,
                reasoning=data.get("reasoning"),
            )
        except Exception:
            return AnswerConfidenceCheckerOutput(
                answer_confidence="low",
                reasoning="Could not evaluate answer confidence.",
            )
