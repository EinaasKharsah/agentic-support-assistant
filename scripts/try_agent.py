"""Ask the agent a question and see its decision.

Usage:
  python scripts/try_agent.py "I want to stop paying"
  python scripts/try_agent.py "cancel" --previous "I want to stop paying"
  python scripts/try_agent.py "I want to stop paying" --mode rules
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import SupportAgent  # noqa: E402
from app.config import settings  # noqa: E402
from app.embeddings import FastEmbedEmbedder  # noqa: E402
from app.knowledge_base import load_knowledge_base  # noqa: E402
from app.llm import OllamaClient  # noqa: E402
from app.vector_store import VectorStore, make_client  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    parser.add_argument("--previous", default=None, help="the customer's previous message")
    parser.add_argument("--mode", default=settings.agent_mode, choices=["llm", "rules"])
    args = parser.parse_args()

    entries = load_knowledge_base(settings.knowledge_base_path)
    store = VectorStore(
        make_client(settings.qdrant_url),
        FastEmbedEmbedder(settings.embedding_model),
        settings.qdrant_collection,
    )
    store.index(entries)
    llm = OllamaClient(settings.ollama_url, settings.ollama_model, settings.llm_timeout_seconds)
    agent = SupportAgent(
        store, llm,
        min_relevance=settings.min_relevance,
        ambiguity_gap=settings.ambiguity_gap,
        second_check_gap=settings.second_check_gap,
        rewrite_weak_queries=settings.rewrite_weak_queries,
        mode=args.mode,
    )

    start = time.perf_counter()
    response = agent.respond(args.message, previous_message=args.previous)
    seconds = time.perf_counter() - start

    print(f"\nMessage:    {args.message}" + (f"  (previous: {args.previous})" if args.previous else ""))
    print("Matches:")
    for match in response.matches:
        print(f"  {match.score:.3f}  {match.entry.id:<4} {match.entry.question}")
    print(f"\nAction:     {response.action.value.upper()}  (decided by {response.decided_by})")
    print(f"Reason:     {response.reason}")
    if response.source:
        print(f"Source:     {response.source}")
    print(f"Time:       {seconds:.1f}s\n")
    print(response.message)
    for i, option in enumerate(response.options, 1):
        print(f"  {i}. {option.label}  [{option.card_id}]")
    print()


if __name__ == "__main__":
    main()
