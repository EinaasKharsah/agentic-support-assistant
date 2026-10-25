"""Search the best card and let the LLM write the answer from it.

Usage:  python scripts/try_answer.py "Do you take Bitcoin?"
Needs Ollama running with the model from OLLAMA_MODEL (default qwen2.5:3b).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.answering import generate_answer  # noqa: E402
from app.config import settings  # noqa: E402
from app.embeddings import FastEmbedEmbedder  # noqa: E402
from app.knowledge_base import load_knowledge_base  # noqa: E402
from app.llm import OllamaClient  # noqa: E402
from app.vector_store import VectorStore, make_client  # noqa: E402


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Do you take Bitcoin?"
    store = VectorStore(
        make_client(settings.qdrant_url),
        FastEmbedEmbedder(settings.embedding_model),
        settings.qdrant_collection,
    )
    store.index(load_knowledge_base(settings.knowledge_base_path))
    llm = OllamaClient(settings.ollama_url, settings.ollama_model, settings.llm_timeout_seconds)

    best = store.search(question, top_k=1)[0]
    start = time.perf_counter()
    answer, note = generate_answer(question, best.entry, llm)
    seconds = time.perf_counter() - start

    print(f"\nQuestion: {question}")
    print(f"Source:   {best.entry.id} (score {best.score:.3f})")
    print(f"Model:    {settings.ollama_model if note is None else 'NOT USED: ' + note}")
    print(f"Time:     {seconds:.1f}s\n")
    print(answer, "\n")


if __name__ == "__main__":
    main()
