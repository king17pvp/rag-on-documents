from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    conversation_id: str = Field(..., description="UUID of the conversation")
    user_id: str = Field(..., description="UUID of the user")
    question: str = Field(..., min_length=1, description="The user's question")


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    response_type: str = Field(
        ..., description="One of: 'answer', 'clarification', 'interrupted'"
    )


# ── Conversations ─────────────────────────────────────────────────────────────

class CreateConversationRequest(BaseModel):
    user_id: str
    title: Optional[str] = None


class CreateConversationResponse(BaseModel):
    conversation_id: str
    user_id: str
    title: Optional[str]
    created_at: Optional[datetime] = None


# ── Documents ─────────────────────────────────────────────────────────────────

class UploadedFile(BaseModel):
    filename: str
    chunks: int
    status: str


class UploadDocumentsResponse(BaseModel):
    conversation_id: str
    uploaded: List[UploadedFile]


class DocumentListResponse(BaseModel):
    conversation_id: str
    files: List[str]


# ── History ───────────────────────────────────────────────────────────────────

class MessageRecord(BaseModel):
    msg_id: str
    question: str
    answer: str
    created_at: Optional[datetime] = None


class HistoryResponse(BaseModel):
    conversation_id: str
    summary: Optional[str]
    messages: List[MessageRecord]


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    details: dict
