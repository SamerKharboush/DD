"""
LLM Client for CellType-Agent.

Provides unified interface for multiple LLM providers:
- Anthropic Claude
- OpenAI (fallback)
- Local models (via vLLM)
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("ct.models.llm")


class LLMClient:
    """
    Unified LLM client.

    Supports:
    - Anthropic Claude (primary)
    - OpenAI (fallback)
    - Local models via vLLM

    Usage:
        client = LLMClient()
        response = client.chat([{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
    ):
        """
        Initialize LLM client.

        Args:
            provider: Provider name (anthropic, openai, local)
            model: Model identifier
            api_key: API key (uses env var if not provided)
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key or self._get_api_key(provider)

        self._client = None

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment."""
        key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        return os.environ.get(key_map.get(provider, ""))

    @property
    def client(self):
        """Lazy-load the underlying client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self):
        """Create the provider-specific client."""
        if self.provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.warning("anthropic package not installed")

        elif self.provider == "openai":
            try:
                import openai
                return openai.OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("openai package not installed")

        elif self.provider == "local":
            # Use requests to call vLLM server
            return None  # Handled in chat()

        return None

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> dict:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts
            model: Override model
            system_prompt: System prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature

        Returns:
            Response dict with 'content' key
        """
        model = model or self.model

        if self.provider == "anthropic" and self.client:
            return self._chat_anthropic(
                messages, model, system_prompt, max_tokens, temperature
            )

        elif self.provider == "openai" and self.client:
            return self._chat_openai(
                messages, model, system_prompt, max_tokens, temperature
            )

        elif self.provider == "local":
            return self._chat_local(
                messages, model, max_tokens, temperature
            )

        else:
            # Fallback to mock response
            logger.warning(f"No valid client for provider {self.provider}")
            return {"content": "Error: No LLM client available"}

    def _chat_anthropic(
        self,
        messages: list[dict],
        model: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Call Anthropic API."""
        try:
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
                "temperature": temperature,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = self.client.messages.create(**kwargs)

            return {
                "content": response.content[0].text,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return {"content": f"Error: {e}", "error": True}

    def _chat_openai(
        self,
        messages: list[dict],
        model: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Call OpenAI API."""
        try:
            formatted_messages = []
            if system_prompt:
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            formatted_messages.extend(messages)

            response = self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=formatted_messages,
                temperature=temperature,
            )

            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                },
            }

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {"content": f"Error: {e}", "error": True}

    def _chat_local(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Call local vLLM server."""
        import requests

        try:
            response = requests.post(
                "http://localhost:8001/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=120,
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "content": data["choices"][0]["message"]["content"],
                    "model": model,
                }
            else:
                return {"content": f"Error: {response.status_code}", "error": True}

        except requests.exceptions.ConnectionError:
            return {"content": "Error: Local LLM server not running", "error": True}
        except Exception as e:
            return {"content": f"Error: {e}", "error": True}


# Singleton client
_default_client: Optional[LLMClient] = None


def get_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMClient:
    """
    Get the default LLM client.

    Args:
        provider: Override provider
        model: Override model

    Returns:
        LLMClient instance
    """
    global _default_client

    if _default_client is None or provider or model:
        provider = provider or os.environ.get("LLM_PROVIDER", "anthropic")
        model = model or os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
        _default_client = LLMClient(provider=provider, model=model)

    return _default_client