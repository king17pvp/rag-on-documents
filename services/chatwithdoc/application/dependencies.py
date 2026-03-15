from __future__ import annotations

from database import PostgresClient, PGSettings
from embedding import EmbeddingClient, EmbeddingSettings
from llm import LiteLLMClient, LLMSettings
from open_search import OpenSearchService, OpenSearchSettings
from storage_handler import MinioClient, MinioSettings

from ..settings import ChatWithDocSettings


def build_llm_client(settings: ChatWithDocSettings) -> LiteLLMClient:
    return LiteLLMClient(
        settings=LLMSettings(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            api_base=settings.llm_api_base,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    )


def build_db_client(settings: ChatWithDocSettings) -> PostgresClient:
    return PostgresClient(
        settings=PGSettings(
            username=settings.pg_username,
            password=settings.pg_password,
            host=settings.pg_host,
            db=settings.pg_db,
        )
    )


def build_opensearch_service(settings: ChatWithDocSettings) -> OpenSearchService:
    return OpenSearchService(
        settings=OpenSearchSettings(
            host=settings.opensearch_host,
            port=settings.opensearch_port,
            knn_size=settings.opensearch_knn_size,
            dimensions=settings.opensearch_dimensions,
            embedding_model=settings.embedding_model,
            encoding_format="float",
        )
    )


def build_embedding_client(settings: ChatWithDocSettings) -> EmbeddingClient:
    return EmbeddingClient(
        settings=EmbeddingSettings(
            api_key=settings.embedding_api_key or settings.llm_api_key,
            model=settings.embedding_model,
            api_base=settings.embedding_api_base,
            dimensions=settings.opensearch_dimensions,
        )
    )


def build_minio_client(settings: ChatWithDocSettings) -> MinioClient:
    return MinioClient(
        settings=MinioSettings(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            bucket_name=settings.minio_bucket_name,
        )
    )
