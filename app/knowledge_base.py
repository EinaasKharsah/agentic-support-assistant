"""Load the customer support knowledge base from a JSON file."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Entry:
    """One Q&A card from the knowledge base."""

    id: str
    category: str
    question: str
    answer: str
    out_of_scope: bool = False

    def to_text(self) -> str:
        """Text that gets embedded for search.

        We combine category, question and answer, because customers do not
        always use the words of the stored question. Example: "Do you accept
        Bitcoin?" only matches "Cryptocurrency" in the answer of Q4.
        """
        return (
            f"Category: {self.category}\n"
            f"Question: {self.question}\n"
            f"Answer: {self.answer}"
        )


REQUIRED_FIELDS = ("id", "category", "question", "answer")


def load_knowledge_base(path: str | Path) -> list[Entry]:
    """Read the JSON file and return a list of entries.

    Fails early with a clear message if the file is broken, so we never
    start the assistant with a bad knowledge base.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    entries: list[Entry] = []
    for i, item in enumerate(raw):
        missing = [f for f in REQUIRED_FIELDS if not item.get(f)]
        if missing:
            raise ValueError(f"Entry #{i} is missing fields: {missing}")
        entries.append(
            Entry(
                id=item["id"],
                category=item["category"],
                question=item["question"],
                answer=item["answer"],
                out_of_scope=bool(item.get("out_of_scope", False)),
            )
        )

    ids = [e.id for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Entry ids must be unique")

    return entries
