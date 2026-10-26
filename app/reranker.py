"""A re-ranking model: reads the question and an article TOGETHER and scores how well the
article answers the question.

The embedding model compares two separate "meaning fingerprints", so "How do I reset my
iPhone passcode?" looks close to our password article: both are about resetting a password.
A cross-encoder reads both texts at once, so it can notice that the iPhone is not our product.
It is slower, so it only scores the few articles the vector search already found.
"""

from typing import Protocol

from fastembed.rerank.cross_encoder import TextCrossEncoder


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, documents: list[str]) -> list[float]:
        # Raw model scores (logits): higher = better answer. Not limited to 0..1.
        return [float(s) for s in self._model.rerank(query, documents)]
