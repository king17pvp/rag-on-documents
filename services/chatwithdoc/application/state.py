from __future__ import annotations

from typing import List, Optional, TypedDict


class GraphState(TypedDict, total=False):
    """Shared mutable state threaded through every node of the LangGraph pipeline."""

    # ── Input ─────────────────────────────────────────────────────────────────
    conversation_id: str
    user_id: str
    question: str

    # ── Flow control ──────────────────────────────────────────────────────────
    is_interrupted: bool
    retry_count: int

    # ── Conversation history ──────────────────────────────────────────────────
    conversation_history: List[dict]
    conversation_summary: Optional[str]

    # ── Retrieval pipeline ────────────────────────────────────────────────────
    rephrased_question: str
    expanded_queries: List[str]
    retrieved_docs: List[dict]
    reranked_docs: List[dict]

    # ── Evaluation ────────────────────────────────────────────────────────────
    context_relevance: str          # "strong" | "weak" | "no_context"
    answer: str
    answer_confidence: str          # "high" | "low"
    hallucination_status: str       # "grounded" | "hallucination"

    # ── Clarification ─────────────────────────────────────────────────────────
    clarification: Optional[str]
    clarification_reason: Optional[str]

    # ── Final output ──────────────────────────────────────────────────────────
    final_response: str
    response_type: str              # "answer" | "clarification" | "interrupted"
