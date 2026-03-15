from __future__ import annotations

import logging
import time
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from ..domain.answer_confidence_checker.service import (
    AnswerConfidenceCheckerInput,
    AnswerConfidenceCheckerService,
)
from ..domain.answer_generator.service import AnswerGeneratorInput, AnswerGeneratorService
from ..domain.clarification_generator.service import (
    ClarificationGeneratorInput,
    ClarificationGeneratorService,
)
from ..domain.context_relevance_filter.service import (
    ContextRelevanceFilterInput,
    ContextRelevanceFilterService,
)
from ..domain.conversation_summary.service import (
    ConversationSummaryInput,
    ConversationSummaryService,
)
from ..domain.hallucination_guard.service import (
    HallucinationGuardInput,
    HallucinationGuardService,
)
from ..domain.history_retrieval.service import (
    HistoryRetrievalInput,
    HistoryRetrievalService,
)
from ..domain.human_intervent.service import HumanInterventInput, HumanInterventService
from ..domain.interrupt_checker.service import (
    InterruptCheckerInput,
    InterruptCheckerService,
)
from ..domain.query_expansion.service import QueryExpansionInput, QueryExpansionService
from ..domain.rerank_context.service import RerankContextInput, RerankContextService
from ..domain.rephrase_question.service import (
    RephraseQuestionInput,
    RephraseQuestionService,
)
from ..domain.retrieve_context.service import RetrieveContextInput, RetrieveContextService
from ..domain.retrieval_retry_controller.service import (
    RetrievalRetryControllerInput,
    RetrievalRetryControllerService,
)
from ..domain.store_conversation.service import (
    StoreConversationInput,
    StoreConversationService,
)
from .state import GraphState

# Situation messages surfaced to the clarification generator
_SITUATION = {
    "no_context": (
        "The system could not find relevant information in the uploaded documents "
        "to answer this question."
    ),
    "max_retries": (
        "After multiple retrieval attempts, the system was still unable to find "
        "sufficient context to answer this question confidently."
    ),
    "low_confidence": (
        "The system generated an answer but it lacks sufficient confidence in its "
        "accuracy based on the available context."
    ),
    "hallucination": (
        "The system detected that the generated answer may contain information not "
        "supported by the source documents."
    ),
}


class ChatWithDocAssembler:
    """Builds and runs the LangGraph execution graph for the ChatWithDoc pipeline.

    Graph topology
    --------------
    START → interrupt_checker
    interrupt_checker  → history_retrieval (not interrupted)
                       → END (interrupted)
    history_retrieval  → rephrase_question
    rephrase_question  → query_expansion
    query_expansion    → retrieve_context
    retrieve_context   → rerank_context
    rerank_context     → context_relevance_filter
    context_relevance_filter → answer_generator          (strong)
                             → retrieval_retry_controller (weak)
                             → clarification_generator   (no_context)
    retrieval_retry_controller → query_expansion          (retry)
                               → clarification_generator  (max retries)
    answer_generator   → answer_confidence_checker
    answer_confidence_checker → hallucination_guard       (high)
                              → clarification_generator   (low)
    hallucination_guard → conversation_summary            (grounded)
                        → clarification_generator         (hallucination)
    clarification_generator → human_intervent
    conversation_summary    → store_conversation
    store_conversation      → END
    human_intervent         → END
    """

    def __init__(
        self,
        interrupt_checker: InterruptCheckerService,
        history_retrieval: HistoryRetrievalService,
        rephrase_question: RephraseQuestionService,
        query_expansion: QueryExpansionService,
        retrieve_context: RetrieveContextService,
        rerank_context: RerankContextService,
        context_relevance_filter: ContextRelevanceFilterService,
        retrieval_retry_controller: RetrievalRetryControllerService,
        answer_generator: AnswerGeneratorService,
        answer_confidence_checker: AnswerConfidenceCheckerService,
        hallucination_guard: HallucinationGuardService,
        clarification_generator: ClarificationGeneratorService,
        conversation_summary: ConversationSummaryService,
        store_conversation: StoreConversationService,
        human_intervent: HumanInterventService,
        index_name: str = "rag-documents",
        top_k: int = 5,
        query_expansion_count: int = 3,
        embedding_client: Optional[Any] = None,
    ) -> None:
        self._interrupt_checker = interrupt_checker
        self._history_retrieval = history_retrieval
        self._rephrase_question = rephrase_question
        self._query_expansion = query_expansion
        self._retrieve_context = retrieve_context
        self._rerank_context = rerank_context
        self._context_relevance_filter = context_relevance_filter
        self._retrieval_retry_controller = retrieval_retry_controller
        self._answer_generator = answer_generator
        self._answer_confidence_checker = answer_confidence_checker
        self._hallucination_guard = hallucination_guard
        self._clarification_generator = clarification_generator
        self._conversation_summary = conversation_summary
        self._store_conversation = store_conversation
        self._human_intervent = human_intervent
        self._index_name = index_name
        self._top_k = top_k
        self._query_expansion_count = query_expansion_count
        self._embedding_client = embedding_client
        self._graph = self._build_graph()

    # ── Node implementations ──────────────────────────────────────────────────

    async def _node_interrupt_checker(self, state: GraphState) -> dict:
        result = await self._interrupt_checker.process(
            InterruptCheckerInput(question=state["question"])
        )
        if result.is_interrupted:
            return {
                "is_interrupted": True,
                "final_response": "Understood. The conversation has been stopped.",
                "response_type": "interrupted",
            }
        return {"is_interrupted": False}

    async def _node_history_retrieval(self, state: GraphState) -> dict:
        result = await self._history_retrieval.process(
            HistoryRetrievalInput(conversation_id=state["conversation_id"])
        )
        return {
            "conversation_history": result.conversation_history,
            "conversation_summary": result.conversation_summary,
        }

    async def _node_rephrase_question(self, state: GraphState) -> dict:
        result = await self._rephrase_question.process(
            RephraseQuestionInput(
                question=state["question"],
                conversation_history=state.get("conversation_history", []),
                conversation_summary=state.get("conversation_summary"),
            )
        )
        return {"rephrased_question": result.rephrased_question}

    async def _node_query_expansion(self, state: GraphState) -> dict:
        result = await self._query_expansion.process(
            QueryExpansionInput(
                question=state.get("rephrased_question", state["question"]),
                num_queries=self._query_expansion_count,
            )
        )
        return {"expanded_queries": result.expanded_queries}

    async def _node_retrieve_context(self, state: GraphState) -> dict:
        # Optionally embed the expanded queries for KNN search
        query_embeddings = None
        if self._embedding_client:
            queries = state.get("expanded_queries", [state["question"]])
            query_embeddings = await self._embedding_client.embed_documents(queries)

        result = await self._retrieve_context.process(
            RetrieveContextInput(
                expanded_queries=state.get("expanded_queries", [state["question"]]),
                conversation_id=state["conversation_id"],
                index_name=self._index_name,
                top_k=self._top_k,
                query_embeddings=query_embeddings,
            )
        )
        return {"retrieved_docs": result.retrieved_docs}

    async def _node_rerank_context(self, state: GraphState) -> dict:
        result = await self._rerank_context.process(
            RerankContextInput(
                retrieved_docs=state.get("retrieved_docs", []),
                question=state.get("rephrased_question", state["question"]),
                top_k=self._top_k,
            )
        )
        return {"reranked_docs": result.reranked_docs}

    async def _node_context_relevance_filter(self, state: GraphState) -> dict:
        result = await self._context_relevance_filter.process(
            ContextRelevanceFilterInput(
                question=state.get("rephrased_question", state["question"]),
                reranked_docs=state.get("reranked_docs", []),
            )
        )
        return {"context_relevance": result.context_relevance}

    async def _node_retrieval_retry_controller(self, state: GraphState) -> dict:
        result = await self._retrieval_retry_controller.process(
            RetrievalRetryControllerInput(retry_count=state.get("retry_count", 0))
        )
        updates: dict = {"retry_count": result.retry_count}
        if not result.should_retry:
            updates["clarification_reason"] = "max_retries"
        return updates

    async def _node_answer_generator(self, state: GraphState) -> dict:
        result = await self._answer_generator.process(
            AnswerGeneratorInput(
                question=state.get("rephrased_question", state["question"]),
                reranked_docs=state.get("reranked_docs", []),
                conversation_summary=state.get("conversation_summary"),
            )
        )
        return {"answer": result.answer}

    async def _node_answer_confidence_checker(self, state: GraphState) -> dict:
        result = await self._answer_confidence_checker.process(
            AnswerConfidenceCheckerInput(
                question=state.get("rephrased_question", state["question"]),
                answer=state.get("answer", ""),
                reranked_docs=state.get("reranked_docs", []),
            )
        )
        updates: dict = {"answer_confidence": result.answer_confidence}
        if result.answer_confidence == "low":
            updates["clarification_reason"] = "low_confidence"
        return updates

    async def _node_hallucination_guard(self, state: GraphState) -> dict:
        result = await self._hallucination_guard.process(
            HallucinationGuardInput(
                answer=state.get("answer", ""),
                reranked_docs=state.get("reranked_docs", []),
            )
        )
        updates: dict = {"hallucination_status": result.hallucination_status}
        if result.hallucination_status == "hallucination":
            updates["clarification_reason"] = "hallucination"
        return updates

    async def _node_clarification_generator(self, state: GraphState) -> dict:
        reason = state.get("clarification_reason", "no_context")
        situation = _SITUATION.get(reason, _SITUATION["no_context"])
        result = await self._clarification_generator.process(
            ClarificationGeneratorInput(
                question=state["question"],
                situation=situation,
            )
        )
        return {"clarification": result.clarification}

    async def _node_conversation_summary(self, state: GraphState) -> dict:
        result = await self._conversation_summary.process(
            ConversationSummaryInput(
                question=state["question"],
                answer=state.get("answer", ""),
                current_summary=state.get("conversation_summary"),
            )
        )
        return {"conversation_summary": result.updated_summary}

    async def _node_store_conversation(self, state: GraphState) -> dict:
        await self._store_conversation.process(
            StoreConversationInput(
                conversation_id=state["conversation_id"],
                question=state["question"],
                answer=state.get("answer", ""),
                updated_summary=state.get("conversation_summary"),
            )
        )
        return {
            "final_response": state.get("answer", ""),
            "response_type": "answer",
        }

    async def _node_human_intervent(self, state: GraphState) -> dict:
        result = await self._human_intervent.process(
            HumanInterventInput(clarification=state.get("clarification", ""))
        )
        return {
            "final_response": result.final_response,
            "response_type": result.response_type,
        }

    # ── Routing functions ─────────────────────────────────────────────────────

    @staticmethod
    def _route_interrupt_checker(state: GraphState) -> str:
        return END if state.get("is_interrupted") else "history_retrieval"

    @staticmethod
    def _route_context_relevance(state: GraphState) -> str:
        relevance = state.get("context_relevance", "no_context")
        if relevance == "strong":
            return "answer_generator"
        if relevance == "weak":
            return "retrieval_retry_controller"
        return "clarification_generator"

    @staticmethod
    def _route_retry_controller(state: GraphState) -> str:
        # should_retry is implicit: retry if retry_count <= max_retries
        # The controller node sets clarification_reason = "max_retries" when exhausted
        if state.get("clarification_reason") == "max_retries":
            return "clarification_generator"
        return "query_expansion"

    @staticmethod
    def _route_answer_confidence(state: GraphState) -> str:
        return (
            "hallucination_guard"
            if state.get("answer_confidence") == "high"
            else "clarification_generator"
        )

    @staticmethod
    def _route_hallucination_guard(state: GraphState) -> str:
        return (
            "conversation_summary"
            if state.get("hallucination_status") == "grounded"
            else "clarification_generator"
        )

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self):
        graph: StateGraph = StateGraph(GraphState)

        # Register nodes
        graph.add_node("interrupt_checker", self._node_interrupt_checker)
        graph.add_node("history_retrieval", self._node_history_retrieval)
        graph.add_node("rephrase_question", self._node_rephrase_question)
        graph.add_node("query_expansion", self._node_query_expansion)
        graph.add_node("retrieve_context", self._node_retrieve_context)
        graph.add_node("rerank_context", self._node_rerank_context)
        graph.add_node("context_relevance_filter", self._node_context_relevance_filter)
        graph.add_node(
            "retrieval_retry_controller", self._node_retrieval_retry_controller
        )
        graph.add_node("answer_generator", self._node_answer_generator)
        graph.add_node(
            "answer_confidence_checker", self._node_answer_confidence_checker
        )
        graph.add_node("hallucination_guard", self._node_hallucination_guard)
        graph.add_node("clarification_generator", self._node_clarification_generator)
        graph.add_node("conversation_summary", self._node_conversation_summary)
        graph.add_node("store_conversation", self._node_store_conversation)
        graph.add_node("human_intervent", self._node_human_intervent)

        # Entry point
        graph.set_entry_point("interrupt_checker")

        # Conditional: interrupt_checker
        graph.add_conditional_edges(
            "interrupt_checker",
            self._route_interrupt_checker,
            {"history_retrieval": "history_retrieval", END: END},
        )

        # Linear pipeline
        graph.add_edge("history_retrieval", "rephrase_question")
        graph.add_edge("rephrase_question", "query_expansion")
        graph.add_edge("query_expansion", "retrieve_context")
        graph.add_edge("retrieve_context", "rerank_context")
        graph.add_edge("rerank_context", "context_relevance_filter")

        # Conditional: context_relevance_filter
        graph.add_conditional_edges(
            "context_relevance_filter",
            self._route_context_relevance,
            {
                "answer_generator": "answer_generator",
                "retrieval_retry_controller": "retrieval_retry_controller",
                "clarification_generator": "clarification_generator",
            },
        )

        # Conditional: retrieval_retry_controller
        graph.add_conditional_edges(
            "retrieval_retry_controller",
            self._route_retry_controller,
            {
                "query_expansion": "query_expansion",
                "clarification_generator": "clarification_generator",
            },
        )

        # Linear: answer pipeline
        graph.add_edge("answer_generator", "answer_confidence_checker")

        # Conditional: answer_confidence_checker
        graph.add_conditional_edges(
            "answer_confidence_checker",
            self._route_answer_confidence,
            {
                "hallucination_guard": "hallucination_guard",
                "clarification_generator": "clarification_generator",
            },
        )

        # Conditional: hallucination_guard
        graph.add_conditional_edges(
            "hallucination_guard",
            self._route_hallucination_guard,
            {
                "conversation_summary": "conversation_summary",
                "clarification_generator": "clarification_generator",
            },
        )

        # Terminal paths
        graph.add_edge("clarification_generator", "human_intervent")
        graph.add_edge("conversation_summary", "store_conversation")
        graph.add_edge("store_conversation", END)
        graph.add_edge("human_intervent", END)

        return graph.compile()

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(
        self,
        conversation_id: str,
        user_id: str,
        question: str,
    ) -> GraphState:
        """Execute the full RAG pipeline and return the final state.

        Logs a single structured record with total execution time and the
        final response type, which is enough for high‑level observability
        without per‑node tracing.
        """
        logger = logging.getLogger("chatwithdoc.graph")
        initial_state: GraphState = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "question": question,
            "retry_count": 0,
        }
        start = time.perf_counter()
        success = False
        state: GraphState | None = None
        try:
            state = await self._graph.ainvoke(initial_state)
            success = True
            return state
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "graph_execution_completed",
                extra={
                    "event": "graph_execution_completed",
                    "success": success,
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "duration_ms": duration_ms,
                    "response_type": (state or {}).get("response_type"),
                },
            )
