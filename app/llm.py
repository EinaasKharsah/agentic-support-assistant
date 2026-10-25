"""Small client for the Ollama chat API.

We call the REST API directly with httpx instead of using a big framework
(like LangChain). It is only a few lines, easy to read and easy to test.
"""

import json
from typing import Any, Protocol

import httpx


class LLMError(Exception):
    """Ollama is down, too slow, or returned something we cannot use."""


class LLM(Protocol):
    def chat(self, messages: list[dict[str, str]], json_schema: dict | None = None) -> str: ...


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0, temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Low temperature = stable, factual answers. Support needs accuracy, not creativity.
        self.temperature = temperature

    def chat(self, messages: list[dict[str, str]], json_schema: dict | None = None) -> str:
        """Send the messages and return the model's reply as text.

        If json_schema is given, Ollama forces the reply to follow it
        (structured output). This makes small models much more reliable.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_schema is not None:
            payload["format"] = json_schema

        try:
            response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except (httpx.HTTPError, KeyError, ValueError) as error:
            raise LLMError(f"Ollama request failed: {error}") from error

    def chat_json(self, messages: list[dict[str, str]], json_schema: dict) -> dict:
        """Like chat(), but parse the reply as JSON."""
        text = self.chat(messages, json_schema=json_schema)
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise LLMError(f"Model did not return valid JSON: {text[:200]}") from error
