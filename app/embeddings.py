"""Turn text into vectors (lists of numbers) with an open-source model.

We use fastembed instead of sentence-transformers: it runs the model with
ONNX, so it does not need PyTorch. The Docker image stays small and it runs
fast on a normal CPU.
"""

from typing import Protocol

from fastembed import TextEmbedding

# BGE models search better when the *question* starts with this instruction.
# The stored documents are embedded without it. (Recommended by the model authors.)
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    """Anything that can embed documents and queries.

    Using a small interface lets the tests use a fake embedder,
    so they run without downloading the model.
    """

    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        # Ask the model once for its vector size (384 for bge-small).
        self.dimension = len(self.embed_query("dimension check"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        prefix = BGE_QUERY_INSTRUCTION if "bge" in self.model_name.lower() else ""
        return next(iter(self._model.embed([prefix + text]))).tolist()
