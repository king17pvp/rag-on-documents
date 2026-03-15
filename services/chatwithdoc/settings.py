from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatWithDocSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_api_key: str = Field(..., alias="LLM_API_KEY")
    llm_model: str = Field(..., alias="LLM_MODEL")
    llm_api_base: Optional[str] = Field(None, alias="LLM_API_BASE")
    llm_temperature: float = Field(0.7, alias="LLM_TEMPERATURE")
    llm_max_tokens: Optional[int] = Field(None, alias="LLM_MAX_TOKENS")

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_api_key: Optional[str] = Field(None, alias="EMBEDDING_API_KEY")
    embedding_model: str = Field("text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_api_base: Optional[str] = Field(None, alias="EMBEDDING_API_BASE")

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    pg_username: str = Field(..., alias="PG_USERNAME")
    pg_password: str = Field(..., alias="PG_PASSWORD")
    pg_host: str = Field("localhost", alias="PG_HOST")
    pg_db: str = Field("ragdb", alias="PG_DB")

    # ── OpenSearch ────────────────────────────────────────────────────────────
    opensearch_host: str = Field("localhost", alias="OPENSEARCH_HOST")
    opensearch_port: int = Field(9200, alias="OPENSEARCH_PORT")
    opensearch_knn_size: int = Field(10, alias="OPENSEARCH_KNN_SIZE")
    opensearch_dimensions: int = Field(1536, alias="OPENSEARCH_DIMENSIONS")
    opensearch_index: str = Field("rag-documents", alias="OPENSEARCH_INDEX")

    # ── MinIO ─────────────────────────────────────────────────────────────────
    minio_endpoint: str = Field("localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(..., alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(..., alias="MINIO_SECRET_KEY")
    minio_secure: bool = Field(False, alias="MINIO_SECURE")
    minio_bucket_name: str = Field("rag-documents", alias="MINIO_BUCKET_NAME")

    # ── RAG pipeline ──────────────────────────────────────────────────────────
    max_retries: int = Field(3, alias="MAX_RETRIES")
    query_expansion_count: int = Field(3, alias="QUERY_EXPANSION_COUNT")
    retrieval_top_k: int = Field(5, alias="RETRIEVAL_TOP_K")
