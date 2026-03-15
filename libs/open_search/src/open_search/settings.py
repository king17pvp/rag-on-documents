from __future__ import annotations

from base import BaseModel


class OpenSearchSettings(BaseModel):
    host: str
    port: int
    knn_size: int
    dimensions: int
    embedding_model: str
    encoding_format: str
