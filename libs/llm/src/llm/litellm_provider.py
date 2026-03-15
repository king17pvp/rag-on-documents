from __future__ import annotations

from typing import Any
from typing import AsyncGenerator
from typing import Dict
from typing import List
from typing import Optional

import litellm

from base import AsyncBaseService

from .settings import LLMSettings


class LiteLLMClient(AsyncBaseService):
    """LLM client service using LiteLLM to support multiple providers under OpenAI format."""

    settings: LLMSettings

    def get_completion_kwargs(self, custom_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Construct the kwargs for litellm completions."""
        kwargs = {
            "model": self.settings.model,
            "api_key": self.settings.api_key,
            "temperature": self.settings.temperature,
        }

        if self.settings.api_base:
            kwargs["api_base"] = self.settings.api_base
        if self.settings.max_tokens:
            kwargs["max_tokens"] = self.settings.max_tokens
        if self.settings.api_version:
            kwargs["api_version"] = self.settings.api_version

        if custom_kwargs:
            kwargs.update(custom_kwargs)

        return kwargs

    async def check_health(self) -> bool:
        """Asynchronously check LLM provider connectivity."""
        try:
            # We send a tiny request to verify connection and API key validity
            response = await litellm.acompletion(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=1,
                **self.get_completion_kwargs(),
            )
            return bool(response)
        except Exception:
            return False

    async def completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> Any:
        """
        Generate a text completion.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional parameters (e.g., temperature overrides).

        Returns:
            The raw response object from LiteLLM.
        """
        completion_kwargs = self.get_completion_kwargs(kwargs)
        return await litellm.acompletion(messages=messages, **completion_kwargs)

    async def stream_completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> AsyncGenerator[str, None]:
        """
        Generate a streaming text completion.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional parameters.

        Yields:
            Text chunks as they are generated.
        """
        completion_kwargs = self.get_completion_kwargs(kwargs)
        completion_kwargs["stream"] = True

        response = await litellm.acompletion(messages=messages, **completion_kwargs)

        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def process(self, inputs: Dict[str, Any]) -> Any:
        """
        Generic async `process` implementation to satisfy `AsyncBaseService`.

        Expects an `inputs` dict with:
        - `messages`: required list of OpenAI-style messages
        - any additional LiteLLM kwargs (optional)
        """
        messages = inputs.get("messages")
        if not messages:
            raise ValueError("LiteLLMClient.process requires a 'messages' key in inputs")

        extra_kwargs = {k: v for k, v in inputs.items() if k != "messages"}
        return await self.completion(messages=messages, **extra_kwargs)
