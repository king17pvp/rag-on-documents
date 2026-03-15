from __future__ import annotations

import io
from typing import BinaryIO
from typing import Iterator

import fitz

from .base_parser import BaseParser


class PDFParser(BaseParser):
    """Document parser for conventional PDF files using PyMuPDF (fitz)."""

    def parse_file(self, file_stream: BinaryIO) -> str:
        """
        Parse the entire PDF document stream into a single string.
        """
        text = []
        # PyMuPDF expects a filename or bytes. For a stream, we can read it directly.
        file_bytes = file_stream.read()
        
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text.append(page_text)
                    
        return "\n".join(text)

    def parse_file_stream(self, file_stream: BinaryIO) -> Iterator[str]:
        """
        Parse the PDF document stream page-by-page.
        """
        file_bytes = file_stream.read()
        
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    yield page_text
