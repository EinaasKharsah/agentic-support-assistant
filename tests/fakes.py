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
