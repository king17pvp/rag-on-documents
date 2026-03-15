"""
ChatWithDoc service entry-point.

Run from the project root with:
    uvicorn services.chatwithdoc.main:app --reload

Importing this module triggers ``services/chatwithdoc/__init__.py``, which adds
all library source directories to ``sys.path`` before any lib imports occur.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.router import router
from .application.assembler import ChatWithDocAssembler
from .application.dependencies import (
    build_db_client,
    build_embedding_client,
    build_minio_client,
    build_opensearch_service,
    build_llm_client,
)
from .application.document_indexer import DocumentIndexer
from .domain.answer_confidence_checker.service import AnswerConfidenceCheckerService
from .domain.answer_generator.service import AnswerGeneratorService
from .domain.clarification_generator.service import ClarificationGeneratorService
from .domain.context_relevance_filter.service import ContextRelevanceFilterService
from .domain.conversation_summary.service import ConversationSummaryService
from .domain.hallucination_guard.service import HallucinationGuardService
from .domain.history_retrieval.service import HistoryRetrievalService
from .domain.human_intervent.service import HumanInterventService
from .domain.interrupt_checker.service import InterruptCheckerService
from .domain.query_expansion.service import QueryExpansionService
from .domain.rerank_context.service import RerankContextService
from .domain.rephrase_question.service import RephraseQuestionService
from .domain.retrieve_context.service import RetrieveContextService
from .domain.retrieval_retry_controller.service import RetrievalRetryControllerService
from .domain.store_conversation.service import StoreConversationService
from .settings import ChatWithDocSettings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build all infrastructure clients and domain services at startup."""
    settings = ChatWithDocSettings()

    llm_client = build_llm_client(settings)
    db_client = build_db_client(settings)
    opensearch_service = build_opensearch_service(settings)
    embedding_client = build_embedding_client(settings)
    minio_client = build_minio_client(settings)
    minio_client.init_bucket()

    assembler = ChatWithDocAssembler(
        interrupt_checker=InterruptCheckerService(llm_client=llm_client),
        history_retrieval=HistoryRetrievalService(db_client=db_client),
        rephrase_question=RephraseQuestionService(llm_client=llm_client),
        query_expansion=QueryExpansionService(llm_client=llm_client),
        retrieve_context=RetrieveContextService(
            opensearch_service=opensearch_service
        ),
        rerank_context=RerankContextService(),
        context_relevance_filter=ContextRelevanceFilterService(
            llm_client=llm_client
        ),
        retrieval_retry_controller=RetrievalRetryControllerService(
            max_retries=settings.max_retries
        ),
        answer_generator=AnswerGeneratorService(llm_client=llm_client),
        answer_confidence_checker=AnswerConfidenceCheckerService(
            llm_client=llm_client
        ),
        hallucination_guard=HallucinationGuardService(llm_client=llm_client),
        clarification_generator=ClarificationGeneratorService(
            llm_client=llm_client
        ),
        conversation_summary=ConversationSummaryService(llm_client=llm_client),
        store_conversation=StoreConversationService(db_client=db_client),
        human_intervent=HumanInterventService(),
        index_name=settings.opensearch_index,
        top_k=settings.retrieval_top_k,
        query_expansion_count=settings.query_expansion_count,
        embedding_client=embedding_client,
    )

    document_indexer = DocumentIndexer(
        minio_client=minio_client,
        opensearch_service=opensearch_service,
        embedding_client=embedding_client,
        index_name=settings.opensearch_index,
    )

    app.state.assembler = assembler
    app.state.db_client = db_client
    app.state.minio_client = minio_client
    app.state.document_indexer = document_indexer
    app.state.settings = settings

    yield

    # Cleanup (add teardown logic here if needed)


def create_app() -> FastAPI:
    app = FastAPI(
        title="ChatWithDoc",
        description=(
            "RAG-powered document Q&A service. "
            "Upload PDF/DOCX files and ask questions against them."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
