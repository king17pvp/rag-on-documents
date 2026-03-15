from __future__ import annotations

from base import BaseModel


class MinioSettings(BaseModel):
    """Minio connection settings.

    Attributes:
        endpoint: Minio endpoint URL (e.g. "localhost:9000" or "play.min.io").
        access_key: Minio access key (username).
        secret_key: Minio secret key (password).
        secure: Whether to use HTTPS (default: False for local).
        bucket_name: Default bucket name to use for storage.
    """

    endpoint: str
    access_key: str
    secret_key: str
    secure: bool = False
    bucket_name: str = "rag-documents"
