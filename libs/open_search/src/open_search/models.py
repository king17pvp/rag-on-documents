from __future__ import annotations

from typing import Any
from typing import Optional

from base import BaseModel


class AddDocumentInput(BaseModel):
    text: str
    embedding: list[float]
    metadata: Optional[dict[str, Any]] = None
