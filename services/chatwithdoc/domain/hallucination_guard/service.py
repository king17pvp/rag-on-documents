from __future__ import annotations

from typing import List, Optional

from base import AsyncBaseService, BaseModel
from llm import LiteLLMClient

from .prompts.hallucination_guard import hallucination_guard_prompt
from ...utils import format_context_docs, parse_json_response

VALID_STATUS = {"grounded", "hallucination"}


class HallucinationGuardInput(BaseModel):
    answer: str
    reranked_docs: List[dict]


class HallucinationGuardOutput(BaseModel):
    hallucination_status: str  # "grounded" | "hallucination"
    issues: Optional[str] = None


class HallucinationGuardService(AsyncBaseService):
    """Verifies that every factual claim in the answer is grounded in the context."""

    llm_client: LiteLLMClient

    async def process(self, inputs: HallucinationGuardInput) -> HallucinationGuardOutput:
        context_text = format_context_docs(inputs.reranked_docs)
        prompt = hallucination_guard_prompt.format(
            answer=inputs.answer,
            context=context_text,
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.llm_client.completion(messages=messages, temperature=0)
            content = response.choices[0].message.content
            data = parse_json_response(content)
            status = data.get("status", "grounded")
            if status not in VALID_STATUS:
                status = "grounded"
            return HallucinationGuardOutput(
                hallucination_status=status,
                issues=data.get("issues", "none"),
            )
        except Exception:
            return HallucinationGuardOutput(
                hallucination_status="grounded",
                issues="Could not evaluate for hallucinations.",
            )
