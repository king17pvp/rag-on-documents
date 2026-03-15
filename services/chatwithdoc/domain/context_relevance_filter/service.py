from __future__ import annotations

from typing import List, Optional

from base import AsyncBaseService, BaseModel
from llm import LiteLLMClient

from .prompts.context_relevance_filter import context_relevance_filter_prompt
from ...utils import format_context_docs, parse_json_response

VALID_RELEVANCE = {"strong", "weak", "no_context"}


class ContextRelevanceFilterInput(BaseModel):
    question: str
    reranked_docs: List[dict]


class ContextRelevanceFilterOutput(BaseModel):
    context_relevance: str  # "strong" | "weak" | "no_context"
    reasoning: Optional[str] = None


class ContextRelevanceFilterService(AsyncBaseService):
    """Evaluates whether the retrieved context is sufficient to answer the question."""

    llm_client: LiteLLMClient

    async def process(
        self, inputs: ContextRelevanceFilterInput
    ) -> ContextRelevanceFilterOutput:
        if not inputs.reranked_docs:
            return ContextRelevanceFilterOutput(
                context_relevance="no_context",
                reasoning="No documents were retrieved.",
            )

        context_text = format_context_docs(inputs.reranked_docs)
        prompt = context_relevance_filter_prompt.format(
            question=inputs.question,
            context=context_text,
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.llm_client.completion(messages=messages, temperature=0)
            content = response.choices[0].message.content
            data = parse_json_response(content)
            relevance = data.get("relevance", "no_context")
            if relevance not in VALID_RELEVANCE:
                relevance = "no_context"
            return ContextRelevanceFilterOutput(
                context_relevance=relevance,
                reasoning=data.get("reasoning"),
            )
        except Exception:
            return ContextRelevanceFilterOutput(
                context_relevance="no_context",
                reasoning="Failed to evaluate context relevance.",
            )
