"""Small client for the Ollama chat API.

We call the REST API directly with httpx instead of using a big framework
(like LangChain). It is only a few lines, easy to read and easy to test.
"""

import json
import time
from typing import Any, Protocol

import httpx


class LLMError(Exception):
    """Ollama is down, too slow, or returned something we cannot use."""


class LLM(Protocol):
    def chat(self, messages: list[dict[str, str]], json_schema: dict | None = None) -> str: ...


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        temperature: float = 0.0,
        seed: int = 42,
        retries: int = 1,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Low temperature = stable, factual answers. Support needs accuracy, not creativity.
        self.temperature = temperature
        # A fixed seed makes runs repeatable, so evaluation results can be compared.
        self.seed = seed
        # Retry once on short hiccups (Ollama restarting, model still loading).
        self.retries = retries

    def chat(self, messages: list[dict[str, str]], json_schema: dict | None = None) -> str:
        """Send the messages and return the model's reply as text.

        If json_schema is given, Ollama forces the reply to follow it
        (structured output). This makes small models much more reliable.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        if json_schema is not None:
            payload["format"] = json_schema

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()["message"]["content"]
            except (httpx.ConnectError, httpx.RemoteProtocolError) as error:
                last_error = error  # connection problem: worth one more try
            except httpx.HTTPStatusError as error:
                last_error = error
                if error.response.status_code < 500:
                    break  # our request is wrong; retrying will not help
            except (httpx.HTTPError, KeyError, ValueError) as error:
                last_error = error  # timeout or bad reply: do not wait twice as long
                break
            if attempt < self.retries:
                time.sleep(1.0)
        raise LLMError(f"Ollama request failed: {last_error}") from last_error

    def status(self) -> str:
        """'ok', 'model_missing' or 'unreachable'. Used by /health and the chat page."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            response.raise_for_status()
            names = {m.get("name", "") for m in response.json().get("models", [])}
        except (httpx.HTTPError, ValueError):
            return "unreachable"
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return "ok" if wanted in names else "model_missing"

    def chat_json(self, messages: list[dict[str, str]], json_schema: dict) -> dict:
        """Like chat(), but parse the reply as JSON."""
        text = self.chat(messages, json_schema=json_schema)
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise LLMError(f"Model did not return valid JSON: {text[:200]}") from error
