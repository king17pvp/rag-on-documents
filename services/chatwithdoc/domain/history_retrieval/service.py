from __future__ import annotations

from typing import List, Optional

from base import AsyncBaseService, BaseModel
from database import Conversation, Message, PostgresClient


class HistoryRetrievalInput(BaseModel):
    conversation_id: str
    max_messages: int = 10


class HistoryRetrievalOutput(BaseModel):
    conversation_history: List[dict]
    conversation_summary: Optional[str]


class HistoryRetrievalService(AsyncBaseService):
    """Fetches recent conversation messages and the running summary from PostgreSQL."""

    db_client: PostgresClient

    async def process(self, inputs: HistoryRetrievalInput) -> HistoryRetrievalOutput:
        with self.db_client.get_session() as session:
            conv = (
                session.query(Conversation)
                .filter(Conversation.conv_id == inputs.conversation_id)
                .first()
            )

            if not conv:
                return HistoryRetrievalOutput(
                    conversation_history=[],
                    conversation_summary=None,
                )

            messages = (
                session.query(Message)
                .filter(Message.conv_id == inputs.conversation_id)
                .order_by(Message.created_at.desc())
                .limit(inputs.max_messages)
                .all()
            )
            messages = list(reversed(messages))

            history = [
                {"question": msg.question, "answer": msg.answer}
                for msg in messages
            ]

            return HistoryRetrievalOutput(
                conversation_history=history,
                conversation_summary=conv.summary,
            )
