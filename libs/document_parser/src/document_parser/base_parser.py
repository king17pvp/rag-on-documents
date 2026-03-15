from __future__ import annotations

from abc import abstractmethod
from typing import Any
from typing import BinaryIO
from typing import Iterator

from base import BaseService


class BaseParser(BaseService):
    """Abstract base class for all document parsers."""

    @abstractmethod
    def parse_file(self, file_stream: BinaryIO) -> str:
        """
        Parse the entire document stream into a single string.

        Args:
            file_stream: A binary stream of the document.

        Returns:
            Extracted text from the document.
        """
        pass

    @abstractmethod
    def parse_file_stream(self, file_stream: BinaryIO) -> Iterator[str]:
        """
        Parse the document stream page-by-page or block-by-block.

        Args:
            file_stream: A binary stream of the document.

        Yields:
            Extracted text from each page/block.
        """
        pass

    def process(self, inputs: Any) -> str:
        """Satisfy BaseService by delegating to `parse_file`.

        Parsers are synchronous; `inputs` is expected to be a binary
        file-like object compatible with `parse_file`.
        """
        # Consumers that use the generic `process` API can still rely on
        # subclasses implementing `parse_file`, so we simply route through.
        return self.parse_file(inputs)
