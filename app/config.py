"""All settings in one place, read from environment variables.

Changing the model or a threshold only needs a change in the .env file,
not in the code.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    knowledge_base_path: str = os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base.json")

    # Embedding model used to turn text into vectors.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # Qdrant server address. Leave empty to run Qdrant in memory (no Docker needed).
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "support_kb")

    # Ollama (the LLM that writes answers and makes decisions).
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

    # Agent decision settings (first values from manual tests; tuned on the evaluation set).
    # "llm": the LLM decides, code guards. "rules": scores only (baseline for comparison).
    agent_mode: str = os.getenv("AGENT_MODE", "llm")
    # Below this similarity, no card is relevant -> decline without asking the LLM.
    # In manual tests, a clear question scored ~0.72 and "weather in Berlin" ~0.41.
    min_relevance: float = float(os.getenv("MIN_RELEVANCE", "0.55"))
    # Rules mode only: if the top two cards are closer than this, ask a clarifying question.
    ambiguity_gap: float = float(os.getenv("AMBIGUITY_GAP", "0.05"))
    # LLM mode: if a second card scores within this gap of the chosen card, the agent
    # asks the LLM a focused "A, B or either?" question before answering. 0 = off.
    second_check_gap: float = float(os.getenv("SECOND_CHECK_GAP", "0.1"))
    # LLM mode: if the first search finds nothing relevant, rewrite the message as an English
    # search query (fixes typos, translates) and search again.
    rewrite_weak_queries: bool = os.getenv("REWRITE_WEAK_QUERIES", "true").lower() == "true"

    # Thumbs up/down from the chat page are appended to this file (one JSON per line).
    feedback_path: str = os.getenv("FEEDBACK_PATH", "feedback/feedback.jsonl")


settings = Settings()
