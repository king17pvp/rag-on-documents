from __future__ import annotations

from base import AsyncBaseService, BaseModel
from llm import LiteLLMClient

from .prompts.interrupt_checker import interrupt_checker_prompt
from ...utils import parse_json_response


class InterruptCheckerInput(BaseModel):
    question: str


class InterruptCheckerOutput(BaseModel):
    is_interrupted: bool


class InterruptCheckerService(AsyncBaseService):
    """Detects whether the user intends to stop the conversation."""

    llm_client: LiteLLMClient

    async def process(self, inputs: InterruptCheckerInput) -> InterruptCheckerOutput:
        prompt = interrupt_checker_prompt.format(question=inputs.question)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.llm_client.completion(messages=messages, temperature=0)
            content = response.choices[0].message.content
            data = parse_json_response(content)
            return InterruptCheckerOutput(
                is_interrupted=bool(data.get("is_interrupted", False))
            )
        except Exception:
            return InterruptCheckerOutput(is_interrupted=False)
