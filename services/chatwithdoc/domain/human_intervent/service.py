from __future__ import annotations

from base import AsyncBaseService, BaseModel


class HumanInterventInput(BaseModel):
    clarification: str


class HumanInterventOutput(BaseModel):
    final_response: str
    response_type: str = "clarification"


class HumanInterventService(AsyncBaseService):
    """Terminal node that packages the clarification as the final response.

    In a production system this node could trigger a human-in-the-loop
    workflow (e.g. routing to a live agent queue).
    """

    async def process(self, inputs: HumanInterventInput) -> HumanInterventOutput:
        return HumanInterventOutput(
            final_response=inputs.clarification,
            response_type="clarification",
        )
