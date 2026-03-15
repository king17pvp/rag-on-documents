from __future__ import annotations

import json
import re
from typing import Any


def parse_json_response(response: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response string.

    Handles plain JSON, JSON wrapped in markdown code blocks, and JSON embedded
    within free-form text.
    """
    text = response.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract from markdown code block (```json ... ``` or ``` ... ```)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Extract the first JSON object from mixed text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:300]}")


def format_conversation_history(history: list[dict]) -> str:
    """Render a list of ``{question, answer}`` dicts as a human-readable string."""
    if not history:
        return "No previous conversation."

    lines: list[str] = []
    for msg in history:
        lines.append(f"User: {msg.get('question', '')}")
        lines.append(f"Assistant: {msg.get('answer', '')}")
    return "\n".join(lines)


def format_context_docs(docs: list[dict]) -> str:
    """Render a list of retrieved document chunks as a numbered context block."""
    if not docs:
        return "No relevant context found."

    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        # Support both raw dicts and OpenSearch hit dicts (_source wrapper)
        source = doc.get("_source", doc)
        text = source.get("text", "")
        metadata = source.get("metadata") or {}
        label = metadata.get("filename", f"Document {i}")
        parts.append(f"[{i}] Source: {label}\n{text}")

    return "\n\n".join(parts)


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Split *text* into overlapping word-based chunks.

    Args:
        text: The full document text.
        chunk_size: Approximate number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        A list of non-empty chunk strings.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)
        i += step

    return chunks
