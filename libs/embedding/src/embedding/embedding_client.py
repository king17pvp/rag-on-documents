from __future__ import annotations

from typing import List, Optional

import litellm

from base import AsyncBaseService, BaseModel

from .settings import EmbeddingSettings


class EmbeddingInput(BaseModel):
    texts: List[str]


class EmbeddingOutput(BaseModel):
    embeddings: List[List[float]]


class EmbeddingClient(AsyncBaseService):
    """Async embedding client backed by LiteLLM for multi-provider support."""

    settings: EmbeddingSettings

    def _get_kwargs(self) -> dict:
        kwargs: dict = {
            "model": self.settings.model,
            "api_key": self.settings.api_key,
        }
        if self.settings.api_base:
            kwargs["api_base"] = self.settings.api_base
        return kwargs

    async def process(self, inputs: EmbeddingInput) -> EmbeddingOutput:
        embeddings: List[List[float]] = []

        for i in range(0, len(inputs.texts), self.settings.batch_size):
            batch = inputs.texts[i : i + self.settings.batch_size]
            response = await litellm.aembedding(input=batch, **self._get_kwargs())
            batch_embeddings = [item["embedding"] for item in response.data]
            embeddings.extend(batch_embeddings)

        return EmbeddingOutput(embeddings=embeddings)

    async def embed_query(self, text: str) -> List[float]:
        result = await self.process(EmbeddingInput(texts=[text]))
        return result.embeddings[0]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = await self.process(EmbeddingInput(texts=texts))
        return result.embeddings

    async def check_health(self) -> bool:
        try:
            await self.embed_query("health check")
            return True
        except Exception:
            return False
