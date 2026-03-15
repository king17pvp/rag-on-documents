from __future__ import annotations

from typing import Optional

from base import AsyncBaseService, BaseModel
from database import Conversation, Message, PostgresClient


class StoreConversationInput(BaseModel):
    conversation_id: str
    question: str
    answer: str
    updated_summary: Optional[str] = None


class StoreConversationOutput(BaseModel):
    success: bool
    message: str = ""


class StoreConversationService(AsyncBaseService):
    """Persists a Q&A turn and updated conversation summary to PostgreSQL."""

    db_client: PostgresClient

    async def process(self, inputs: StoreConversationInput) -> StoreConversationOutput:
        try:
            with self.db_client.get_session() as session:
                session.add(
                    Message(
                        conv_id=inputs.conversation_id,
                        question=inputs.question,
                        answer=inputs.answer,
                    )
                )

                if inputs.updated_summary:
                    conv = (
                        session.query(Conversation)
                        .filter(Conversation.conv_id == inputs.conversation_id)
                        .first()
                    )
                    if conv:
                        conv.summary = inputs.updated_summary

                session.commit()
            return StoreConversationOutput(success=True)
        except Exception as exc:
            return StoreConversationOutput(success=False, message=str(exc))
