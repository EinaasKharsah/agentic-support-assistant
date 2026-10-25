"""All settings in one place, read from environment variables.

Changing the model or a threshold only needs a change in the .env file,
not in the code.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    knowledge_base_path: str = os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base.json")


settings = Settings()
