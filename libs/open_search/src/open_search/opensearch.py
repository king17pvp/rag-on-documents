from __future__ import annotations

from functools import cached_property
from typing import Any
from typing import Optional
from uuid import uuid4

from base import BaseModel
from base import BaseService
from opensearchpy import OpenSearch

from .models import AddDocumentInput
from .settings import OpenSearchSettings


class OpenSearchInput(BaseModel):
    index_name: str
    query_body: dict[str, Any]
    params: Optional[dict[str, Any]] = None


class OpenSearchOutput(BaseModel):
    results: list[dict]


class OpenSearchService(BaseService):
    settings: OpenSearchSettings

    @cached_property
    def client(self) -> OpenSearch:
        """Create and return an OpenSearch client (cached — one connection per service instance)."""
        return OpenSearch(
            hosts=[{'host': self.settings.host, 'port': self.settings.port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )

    def index_exists(self, index_name: str) -> bool:
        """Check if an index exists in OpenSearch.

        Args:
            index_name (str): The name of the index to check.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        return self.client.indices.exists(index=index_name)

    def create_index(self, index_name: str, index_body: dict[str, Any]) -> bool:
        """Create an index in OpenSearch.

        Args:
            index_name (str): The name of the index to create.
            index_body (dict[str, Any]): The body of the index to create.

        Returns:
            bool: True if the index was created, False if it already exists.
        """
        if self.index_exists(index_name=index_name):
            return False

        response = self.client.indices.create(index=index_name, body=index_body)
        return response.get('acknowledged', False)

    def delete_index(self, index_name: str) -> bool:
        """
        Delete the specified index from OpenSearch.

        Args:
            index_name (str): The name of the index to delete.

        Returns:
            bool: True if the index was deleted, False if it did not exist.
        """
        if not self.index_exists(index_name=index_name):
            return False

        response = self.client.indices.delete(index=index_name)
        return response.get('acknowledged', False)

    def search_pipeline_exists(self, pipeline_id: str) -> bool:
        """
        Check if a search pipeline exists in OpenSearch.

        Args:
            pipeline_id (str): The ID of the pipeline to check.
        Returns:
            bool: True if the pipeline exists, False otherwise.
        """
        return True if self.client.search_pipeline.get(id=pipeline_id, ignore=[404]) else False

    def create_search_pipeline(self, pipeline_body: dict[str, Any], pipeline_id: str) -> bool:
        """
        Create a search pipeline in OpenSearch.

        Args:
            pipeline_body (dict[str, Any]): The body of the pipeline to create.
            pipeline_id (str): The ID of the pipeline to create.
        Returns:
            bool: True if the pipeline was created successfully, False otherwise.
        """
        if self.search_pipeline_exists(pipeline_id=pipeline_id):
            return False

        response = self.client.search_pipeline.put(id=pipeline_id, body=pipeline_body)
        return response.get('acknowledged', False)

    def delete_search_pipeline(self, pipeline_id: str) -> bool:
        """
        Delete a search pipeline from OpenSearch.

        Args:
            pipeline_id (str): The ID of the pipeline to delete.
        Returns:
            bool: True if the pipeline was deleted successfully, False otherwise.
        """
        if not self.search_pipeline_exists(pipeline_id=pipeline_id):
            return False

        response = self.client.search_pipeline.delete(id=pipeline_id)
        return response.get('acknowledged', False)

    def add_documents(self, documents: list[AddDocumentInput], index_name: str) -> bool:
        """
        Add documents to OpenSearch. Generic method that works with any Document type.

        Args:
            documents (list[AddDocumentInput]): A list of AddDocumentInput objects.
            index_name (str): The name of the index to add documents to.
        Returns:
            bool: True if documents were added successfully, False otherwise.
        """
        if not self.index_exists(index_name=index_name) or not documents:
            return False

        for doc in documents:
            _ = self.client.index(
                index=index_name,
                body=doc.model_dump(),
                id=str(uuid4()),
                refresh=True,
            )

        return True

    async def process(self, inputs: OpenSearchInput) -> OpenSearchOutput:
        """
        Search for documents in the specified index.

        Args:
            inputs (OpenSearchInput): The input containing the search query and parameters.

        Returns:
            OpenSearchOutput: The output containing the search results.
        """
        response = self.client.search(
            index=inputs.index_name,
            body=inputs.query_body,
            params=inputs.params or {},
        )
        return OpenSearchOutput(results=response.get('hits', {}).get('hits', []))
