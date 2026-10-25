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
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))


settings = Settings()
