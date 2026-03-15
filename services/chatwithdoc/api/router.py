from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse

from database import Conversation, Message, PostgresClient
from storage_handler import MinioClient

from ..application.assembler import ChatWithDocAssembler
from ..application.document_indexer import DocumentIndexer
from .schemas import (
    ChatRequest,
    ChatResponse,
    CreateConversationRequest,
    CreateConversationResponse,
    DocumentListResponse,
    HealthResponse,
    HistoryResponse,
    MessageRecord,
    UploadDocumentsResponse,
    UploadedFile,
)

router = APIRouter()


# ── Dependency helpers ────────────────────────────────────────────────────────

def get_assembler(request: Request) -> ChatWithDocAssembler:
    return request.app.state.assembler


def get_db_client(request: Request) -> PostgresClient:
    return request.app.state.db_client


def get_minio_client(request: Request) -> MinioClient:
    return request.app.state.minio_client


def get_document_indexer(request: Request) -> DocumentIndexer:
    return request.app.state.document_indexer


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request) -> HealthResponse:
    details: dict = {}

    try:
        db: PostgresClient = get_db_client(request)
        await db.check_health()
        details["database"] = "ok"
    except Exception as exc:
        details["database"] = f"error: {exc}"

    try:
        minio: MinioClient = get_minio_client(request)
        await minio.check_health()
        details["storage"] = "ok"
    except Exception as exc:
        details["storage"] = f"error: {exc}"

    overall = "ok" if all(v == "ok" for v in details.values()) else "degraded"
    return HealthResponse(status=overall, details=details)


# ── Chat ──────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    body: ChatRequest,
    assembler: ChatWithDocAssembler = Depends(get_assembler),
) -> ChatResponse:
    """Run the full RAG pipeline and return a grounded answer or clarification."""
    try:
        state = await assembler.run(
            conversation_id=body.conversation_id,
            user_id=body.user_id,
            question=body.question,
        )
        return ChatResponse(
            conversation_id=body.conversation_id,
            response=state.get("final_response", ""),
            response_type=state.get("response_type", "answer"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Conversations ─────────────────────────────────────────────────────────────

@router.post(
    "/conversations",
    response_model=CreateConversationResponse,
    status_code=201,
    tags=["Conversations"],
)
async def create_conversation(
    body: CreateConversationRequest,
    db: PostgresClient = Depends(get_db_client),
    minio: MinioClient = Depends(get_minio_client),
) -> CreateConversationResponse:
    """Create a new conversation and its MinIO storage folder."""
    with db.get_session() as session:
        conv = Conversation(
            conv_id=str(uuid.uuid4()),
            user_id=body.user_id,
            title=body.title,
        )
        session.add(conv)
        session.commit()
        session.refresh(conv)

        try:
            minio.create_conversation_folder(conv.conv_id)
        except Exception:
            pass  # Storage folder is best-effort

        return CreateConversationResponse(
            conversation_id=conv.conv_id,
            user_id=conv.user_id,
            title=conv.title,
            created_at=conv.created_at,
        )


@router.get(
    "/conversations/{conversation_id}/history",
    response_model=HistoryResponse,
    tags=["Conversations"],
)
async def get_history(
    conversation_id: str,
    db: PostgresClient = Depends(get_db_client),
) -> HistoryResponse:
    """Return all messages and the running summary for a conversation."""
    with db.get_session() as session:
        conv = (
            session.query(Conversation)
            .filter(Conversation.conv_id == conversation_id)
            .first()
        )
        if not conv:
            raise HTTPException(
                status_code=404, detail=f"Conversation {conversation_id!r} not found"
            )

        messages = (
            session.query(Message)
            .filter(Message.conv_id == conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )

        return HistoryResponse(
            conversation_id=conversation_id,
            summary=conv.summary,
            messages=[
                MessageRecord(
                    msg_id=msg.msg_id,
                    question=msg.question,
                    answer=msg.answer,
                    created_at=msg.created_at,
                )
                for msg in messages
            ],
        )


# ── Documents ─────────────────────────────────────────────────────────────────

@router.get(
    "/conversations/{conversation_id}/documents",
    response_model=DocumentListResponse,
    tags=["Documents"],
)
async def list_documents(
    conversation_id: str,
    db: PostgresClient = Depends(get_db_client),
    minio: MinioClient = Depends(get_minio_client),
) -> DocumentListResponse:
    """List all files already uploaded for a conversation (read from MinIO)."""
    with db.get_session() as session:
        conv = (
            session.query(Conversation)
            .filter(Conversation.conv_id == conversation_id)
            .first()
        )
        if not conv:
            raise HTTPException(
                status_code=404, detail=f"Conversation {conversation_id!r} not found"
            )

    files = minio.list_files(conversation_id)
    return DocumentListResponse(conversation_id=conversation_id, files=files)


@router.post(
    "/documents/upload",
    response_model=UploadDocumentsResponse,
    tags=["Documents"],
)
async def upload_documents(
    conversation_id: str = Form(...),
    files: List[UploadFile] = File(...),
    indexer: DocumentIndexer = Depends(get_document_indexer),
    db: PostgresClient = Depends(get_db_client),
) -> UploadDocumentsResponse:
    """Upload one or more PDF/DOCX files and index them into OpenSearch."""
    # Verify conversation exists
    with db.get_session() as session:
        conv = (
            session.query(Conversation)
            .filter(Conversation.conv_id == conversation_id)
            .first()
        )
        if not conv:
            raise HTTPException(
                status_code=404, detail=f"Conversation {conversation_id!r} not found"
            )

    results: List[UploadedFile] = []
    for upload in files:
        filename = upload.filename or "unknown"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ("pdf", "docx", "doc"):
            results.append(
                UploadedFile(
                    filename=filename,
                    chunks=0,
                    status="unsupported_format",
                )
            )
            continue

        try:
            file_bytes = await upload.read()
            summary = await indexer.index_document(
                conversation_id=conversation_id,
                filename=filename,
                file_bytes=file_bytes,
            )
            results.append(
                UploadedFile(
                    filename=summary["filename"],
                    chunks=summary["chunks"],
                    status=summary["status"],
                )
            )
        except Exception as exc:
            results.append(
                UploadedFile(filename=filename, chunks=0, status=f"error: {exc}")
            )

    return UploadDocumentsResponse(
        conversation_id=conversation_id,
        uploaded=results,
    )
