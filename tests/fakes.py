"""Test helpers that replace the real models, so tests run fast and offline."""

import hashlib
import math
import re


class FakeEmbedder:
    """Bag-of-words embedder: texts that share words get similar vectors.

    It does not understand meaning like the real model, but it is enough
    to test that indexing and search work correctly.
    """

    dimension = 256

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in re.findall(r"[a-z]+", text.lower()):
            index = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeLLM:
    """Returns a fixed reply (or raises an error) instead of calling Ollama."""

    def __init__(self, reply: str = "", error: Exception | None = None, json_reply: dict | None = None):
        self.reply = reply
        self.error = error
        self.json_reply = json_reply
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages, json_schema=None):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.reply

    def chat_json(self, messages, json_schema):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.json_reply or {}
