from __future__ import annotations

from typing import List

from base import AsyncBaseService, BaseModel
from llm import LiteLLMClient

from .prompts.query_expansion import query_expansion_prompt
from ...utils import parse_json_response


class QueryExpansionInput(BaseModel):
    question: str
    num_queries: int = 3


class QueryExpansionOutput(BaseModel):
    expanded_queries: List[str]


class QueryExpansionService(AsyncBaseService):
    """Generates multiple diverse search queries from a single question."""

    llm_client: LiteLLMClient

    async def process(self, inputs: QueryExpansionInput) -> QueryExpansionOutput:
        prompt = query_expansion_prompt.format(
            question=inputs.question,
            num_queries=inputs.num_queries,
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.llm_client.completion(
                messages=messages, temperature=0.3
            )
            content = response.choices[0].message.content
            data = parse_json_response(content)
            queries: List[str] = data.get("queries", [])
        except Exception:
            queries = []

        if not queries:
            queries = [inputs.question]
        elif inputs.question not in queries:
            queries.insert(0, inputs.question)

        return QueryExpansionOutput(expanded_queries=queries)
