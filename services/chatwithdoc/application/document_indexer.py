from __future__ import annotations

import io
from typing import BinaryIO, List

from document_parser import DocxParser, PDFParser
from embedding import EmbeddingClient
from open_search import AddDocumentInput, OpenSearchService
from storage_handler import MinioClient

from ..utils import chunk_text

# OpenSearch index mapping for KNN + BM25 hybrid search
KNN_INDEX_BODY = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 100,
        }
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1536,
                "method": {
                    "name": "hnsw",
                    # lucene engine supports inline filter in KNN queries (OpenSearch 2.4+)
                    # and is the current OpenSearch default engine.
                    "engine": "lucene",
                    "space_type": "cosinesimil",
                    "parameters": {"m": 16, "ef_construction": 100},
                },
            },
            "metadata": {
                "properties": {
                    "conversation_id": {"type": "keyword"},
                    "filename": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                }
            },
        }
    },
}


class DocumentIndexer:
    """Orchestrates the full upload-parse-embed-index pipeline for a document."""

    def __init__(
        self,
        minio_client: MinioClient,
        opensearch_service: OpenSearchService,
        embedding_client: EmbeddingClient,
        index_name: str = "rag-documents",
        chunk_size: int = 400,
        chunk_overlap: int = 50,
    ) -> None:
        self.minio_client = minio_client
        self.opensearch_service = opensearch_service
        self.embedding_client = embedding_client
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _ensure_index(self, dimensions: int) -> None:
        """Create the KNN index if it does not yet exist."""
        body = dict(KNN_INDEX_BODY)
        body["mappings"]["properties"]["embedding"]["dimension"] = dimensions  # type: ignore[index]
        self.opensearch_service.create_index(self.index_name, body)

    def _parse_document(self, filename: str, file_data: BinaryIO) -> str:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            return PDFParser().parse_file(file_data)
        if ext in ("docx", "doc"):
            return DocxParser().parse_file(file_data)
        raise ValueError(f"Unsupported file type: {ext}")

    async def index_document(
        self,
        conversation_id: str,
        filename: str,
        file_bytes: bytes,
    ) -> dict:
        """Upload a document, parse it, embed its chunks, and index them.

        Returns a summary dict with ``filename``, ``chunks``, and ``status``.
        """
        # 1. Upload to MinIO
        object_path = self.minio_client.upload_file(
            conversation_id=conversation_id,
            filename=filename,
            file_data=io.BytesIO(file_bytes),
            length=len(file_bytes),
        )

        # 2. Parse
        text = self._parse_document(filename, io.BytesIO(file_bytes))
        if not text.strip():
            return {"filename": filename, "chunks": 0, "status": "empty_document"}

        # 3. Chunk
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return {"filename": filename, "chunks": 0, "status": "no_chunks"}

        # 4. Embed (batch)
        embeddings = await self.embedding_client.embed_documents(chunks)

        # 5. Ensure index exists
        dimensions = len(embeddings[0]) if embeddings else 1536
        self._ensure_index(dimensions)

        # 6. Index documents
        docs: List[AddDocumentInput] = [
            AddDocumentInput(
                text=chunk,
                embedding=emb,
                metadata={
                    "conversation_id": conversation_id,
                    "filename": filename,
                    "object_path": object_path,
                    "chunk_index": idx,
                },
            )
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        self.opensearch_service.add_documents(docs, self.index_name)

        return {"filename": filename, "chunks": len(chunks), "status": "indexed"}
