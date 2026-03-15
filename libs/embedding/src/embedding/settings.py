from __future__ import annotations

from typing import Optional

from base import BaseModel


class EmbeddingSettings(BaseModel):
    api_key: str
    model: str
    api_base: Optional[str] = None
    dimensions: int = 1536
    batch_size: int = 100
