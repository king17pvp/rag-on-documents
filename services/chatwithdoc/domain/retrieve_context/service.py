from __future__ import annotations

from typing import List, Optional

from base import AsyncBaseService, BaseModel
from open_search import OpenSearchInput, OpenSearchService


class RetrieveContextInput(BaseModel):
    expanded_queries: List[str]
    conversation_id: str
    index_name: str = "rag-documents"
    top_k: int = 5
    query_embeddings: Optional[List[List[float]]] = None


class RetrieveContextOutput(BaseModel):
    retrieved_docs: List[dict]


class RetrieveContextService(AsyncBaseService):
    """Retrieves relevant document chunks via hybrid search (KNN + BM25).

    For each expanded query:
      - Runs a KNN (semantic) search when embeddings are available.
      - Always runs a BM25 (keyword) search.
      - Merges the two ranked lists with Reciprocal Rank Fusion (RRF).

    Results across all expanded queries are deduplicated and ranked by
    their RRF score.

    Notes
    -----
    • The conversation_id filter uses ``metadata.conversation_id`` (a
      ``keyword`` field in the index mapping).  The ``.keyword`` sub-field
      suffix must NOT be used — that sub-field only exists on ``text`` fields.
    • For KNN, the ``filter`` is placed inside the ``knn`` clause (supported
      from OpenSearch 2.4+) rather than in an outer ``bool.filter``, which
      does not apply reliably to approximate-KNN in OpenSearch 2.x.
    """

    opensearch_service: OpenSearchService
    # RRF constant — 60 is the standard default
    rrf_k: int = 60

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _knn_search(
        self,
        embedding: List[float],
        conversation_id: str,
        index_name: str,
        top_k: int,
    ) -> List[dict]:
        """Semantic search using the KNN index.

        ``filter`` is placed inside the ``knn`` clause so that OpenSearch
        applies it during the approximate-nearest-neighbour walk rather than
        as a post-filter (avoids retrieving k candidates that all get
        discarded by the filter).
        """
        query_body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": top_k,
                        "filter": {
                            "term": {"metadata.conversation_id": conversation_id}
                        },
                    }
                }
            },
        }
        try:
            result = await self.opensearch_service.process(
                OpenSearchInput(index_name=index_name, query_body=query_body)
            )
            return result.results
        except Exception:
            return []

    async def _bm25_search(
        self,
        query: str,
        conversation_id: str,
        index_name: str,
        top_k: int,
    ) -> List[dict]:
        """Full-text BM25 search."""
        query_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [{"match": {"text": query}}],
                    # metadata.conversation_id is a keyword field — no .keyword suffix
                    "filter": [
                        {"term": {"metadata.conversation_id": conversation_id}}
                    ],
                }
            },
        }
        try:
            result = await self.opensearch_service.process(
                OpenSearchInput(index_name=index_name, query_body=query_body)
            )
            return result.results
        except Exception:
            return []

    def _rrf_merge(
        self,
        knn_hits: List[dict],
        bm25_hits: List[dict],
    ) -> List[dict]:
        """Reciprocal Rank Fusion of two ranked hit lists.

        Score formula: sum over lists of  1 / (rrf_k + rank + 1).
        The merged list is sorted descending by fused score.
        """
        scores: dict[str, float] = {}
        docs_by_id: dict[str, dict] = {}

        for rank, hit in enumerate(knn_hits):
            doc_id = hit.get("_id", "")
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            docs_by_id[doc_id] = hit

        for rank, hit in enumerate(bm25_hits):
            doc_id = hit.get("_id", "")
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            docs_by_id.setdefault(doc_id, hit)

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [
            {**docs_by_id[doc_id], "_score": scores[doc_id]}
            for doc_id in sorted_ids
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    async def process(self, inputs: RetrieveContextInput) -> RetrieveContextOutput:
        seen_ids: set[str] = set()
        all_docs: List[dict] = []

        for i, query in enumerate(inputs.expanded_queries):
            embedding: Optional[List[float]] = (
                inputs.query_embeddings[i]
                if inputs.query_embeddings and i < len(inputs.query_embeddings)
                else None
            )

            # BM25 always runs
            bm25_hits = await self._bm25_search(
                query=query,
                conversation_id=inputs.conversation_id,
                index_name=inputs.index_name,
                top_k=inputs.top_k,
            )

            if embedding:
                # KNN + BM25 → fuse with RRF
                knn_hits = await self._knn_search(
                    embedding=embedding,
                    conversation_id=inputs.conversation_id,
                    index_name=inputs.index_name,
                    top_k=inputs.top_k,
                )
                hits = self._rrf_merge(knn_hits, bm25_hits)
            else:
                # BM25 only
                hits = bm25_hits

            for hit in hits:
                doc_id = hit.get("_id", "")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    source = hit.get("_source", {})
                    all_docs.append(
                        {
                            "_id": doc_id,
                            "text": source.get("text", ""),
                            "metadata": source.get("metadata", {}),
                            "score": hit.get("_score", 0.0),
                        }
                    )

        all_docs.sort(key=lambda d: d["score"], reverse=True)
        return RetrieveContextOutput(retrieved_docs=all_docs)
