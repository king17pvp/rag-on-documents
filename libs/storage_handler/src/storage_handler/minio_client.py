from __future__ import annotations

import io
from functools import cached_property
from typing import BinaryIO

from minio import Minio

from base import AsyncBaseService

from .settings import MinioSettings


class MinioClient(AsyncBaseService):
    """Minio storage client service."""

    settings: MinioSettings

    @cached_property
    def client(self) -> Minio:
        """Create and return a Minio client."""
        return Minio(
            endpoint=self.settings.endpoint,
            access_key=self.settings.access_key,
            secret_key=self.settings.secret_key,
            secure=self.settings.secure,
        )

    def init_bucket(self) -> None:
        """Initialize the default bucket if it does not exist."""
        if not self.client.bucket_exists(self.settings.bucket_name):
            self.client.make_bucket(self.settings.bucket_name)

    async def check_health(self) -> bool:
        """Asynchronously check storage connectivity."""
        try:
            # We can check health by ensuring the bucket exists or we can list buckets.
            self.client.bucket_exists(self.settings.bucket_name)
            return True
        except Exception:
            return False

    def create_conversation_folder(self, conversation_id: str) -> None:
        """
        Create a "folder" for a conversation.
        In object storage (like S3/Minio), folders don't strictly exist unless they contain an object.
        But we can put a zero-byte object with a trailing slash to explicitly represent a folder.
        """
        self.init_bucket()
        folder_path = f"{conversation_id}/"
        empty_stream = io.BytesIO(b"")
        self.client.put_object(
            bucket_name=self.settings.bucket_name,
            object_name=folder_path,
            data=empty_stream,
            length=0,
        )

    def upload_file(self, conversation_id: str, filename: str, file_data: BinaryIO, length: int) -> str:
        """
        Upload a file into a specific conversation folder.

        Args:
            conversation_id: The folder prefix.
            filename: The name of the file being uploaded.
            file_data: The binary stream of the file content.
            length: The size of the file in bytes.

        Returns:
            The full object path in the bucket.
        """
        self.init_bucket()
        object_name = f"{conversation_id}/{filename}"
        self.client.put_object(
            bucket_name=self.settings.bucket_name,
            object_name=object_name,
            data=file_data,
            length=length,
        )
        return object_name

    def list_files(self, conversation_id: str) -> list[str]:
        """Return the filenames stored under a conversation's prefix.

        Skips the zero-byte folder placeholder object (``conv_id/``).
        """
        self.init_bucket()
        prefix = f"{conversation_id}/"
        objects = self.client.list_objects(self.settings.bucket_name, prefix=prefix)
        return [
            obj.object_name.removeprefix(prefix)
            for obj in objects
            if obj.object_name and obj.object_name != prefix
        ]

    def download_file(self, object_name: str) -> BinaryIO:
        """
        Download a file by its object name.

        Args:
            object_name: The full path of the object, e.g. "conv_id/filename.pdf"

        Returns:
            A binary stream of the downloaded object data.
        """
        response = self.client.get_object(self.settings.bucket_name, object_name)
        return io.BytesIO(response.read())

    def get_file_url(self, object_name: str, expires: int = 3600) -> str:
        """
        Generate a presigned URL to access the file conditionally without exposing credentials.

        Args:
            object_name: The full path of the object.
            expires: Link expiration time in seconds.
        """
        return self.client.presigned_get_object(
            bucket_name=self.settings.bucket_name,
            object_name=object_name,
            expires=expires,
        )

    async def process(self, inputs) -> bool:
        """
        Minimal async `process` implementation to satisfy `AsyncBaseService`.

        Currently proxies to `check_health`, which is sufficient for generic
        health-check style usage.
        """
        return await self.check_health()
