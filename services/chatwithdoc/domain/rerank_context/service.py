from __future__ import annotations

from typing import List

from base import AsyncBaseService, BaseModel


class RerankContextInput(BaseModel):
    retrieved_docs: List[dict]
    question: str
    top_k: int = 5


class RerankContextOutput(BaseModel):
    reranked_docs: List[dict]


class RerankContextService(AsyncBaseService):
    """Re-ranks and deduplicates retrieved document chunks.

    Uses score-based ordering and removes near-duplicate chunks based on
    word-overlap similarity. A future enhancement could swap this for a
    dedicated cross-encoder reranker model.
    """

    similarity_threshold: float = 0.8

    @staticmethod
    def _word_overlap(a: str, b: str) -> float:
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / max(len(words_a), len(words_b))

    async def process(self, inputs: RerankContextInput) -> RerankContextOutput:
        docs = sorted(
            inputs.retrieved_docs, key=lambda d: d.get("score", 0.0), reverse=True
        )

        seen_texts: List[str] = []
        unique_docs: List[dict] = []

        for doc in docs:
            text = doc.get("text", "").strip()
            is_duplicate = any(
                self._word_overlap(text, seen) >= self.similarity_threshold
                for seen in seen_texts
            )
            if not is_duplicate:
                seen_texts.append(text)
                unique_docs.append(doc)

            if len(unique_docs) >= inputs.top_k:
                break

        return RerankContextOutput(reranked_docs=unique_docs)
