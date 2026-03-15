from __future__ import annotations

from base import AsyncBaseService, BaseModel
from llm import LiteLLMClient

from .prompts.clarification_generator import clarification_generator_prompt


class ClarificationGeneratorInput(BaseModel):
    question: str
    situation: str


class ClarificationGeneratorOutput(BaseModel):
    clarification: str


class ClarificationGeneratorService(AsyncBaseService):
    """Generates a user-facing clarification when the system cannot answer confidently."""

    llm_client: LiteLLMClient

    async def process(
        self, inputs: ClarificationGeneratorInput
    ) -> ClarificationGeneratorOutput:
        prompt = clarification_generator_prompt.format(
            situation=inputs.situation,
            question=inputs.question,
        )
        messages = [{"role": "user", "content": prompt}]

        response = await self.llm_client.completion(messages=messages)
        clarification = response.choices[0].message.content.strip()
        return ClarificationGeneratorOutput(clarification=clarification)
