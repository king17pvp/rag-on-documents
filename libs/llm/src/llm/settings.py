from __future__ import annotations

from typing import Optional

from base import BaseModel


class LLMSettings(BaseModel):
    """LLM Provider Settings following OpenAI API structure.

    Attributes:
        api_key: The API key for the LLM provider.
        api_base: The base URL for the API (if using customized endpoints like Ollama/Azure).
        model: The model name to use (e.g. "gpt-4-turbo", "claude-3-opus-20240229", "ollama/llama3").
        temperature: Default temperature for the model.
        max_tokens: Maximum tokens to generate.
        api_version: API version (useful for Azure OpenAI).
        organization: Organization ID if applicable.
    """

    api_key: str
    model: str
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_version: Optional[str] = None
    organization: Optional[str] = None
