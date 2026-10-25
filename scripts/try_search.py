"""Try the search from the terminal and see the similarity scores.

Usage:  python scripts/try_search.py "I forgot my password"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.embeddings import FastEmbedEmbedder  # noqa: E402
from app.knowledge_base import load_knowledge_base  # noqa: E402
from app.vector_store import VectorStore, make_client  # noqa: E402


def main() -> None:
    question = " ".join(sys.argv[1:]) or "I forgot my password"
    store = VectorStore(
        make_client(settings.qdrant_url),
        FastEmbedEmbedder(settings.embedding_model),
        settings.qdrant_collection,
    )
    store.index(load_knowledge_base(settings.knowledge_base_path))

    print(f"\nQuestion: {question}\n")
    for result in store.search(question, top_k=3):
        flag = "  (out of scope)" if result.entry.out_of_scope else ""
        print(f"  {result.score:.3f}  {result.entry.id:<4} {result.entry.question}{flag}")
    print()


if __name__ == "__main__":
    main()
