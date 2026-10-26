"""Turn saved feedback into cases to review and examples to train on.

Usage:
  python scripts/export_feedback.py                       # reads feedback/feedback.jsonl
  python scripts/export_feedback.py path/to/feedback.jsonl

With Docker, copy the file out of the container first:
  docker compose cp app:/app/feedback ./feedback

Writes two files next to the feedback file. The feedback folder is in .gitignore,
because it contains customer messages.

  review_queue.json         Thumbs-down replies. A person fills in the right action and
                            cards; reviewed cases can then join an evaluation set.
  finetune_decisions.jsonl  Thumbs-up decisions made by the LLM or the rules, in chat
                            format: the exact prompt the agent sends for its decision step,
                            and the decision as the answer. This is training data for
                            fine-tuning the decision model later.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import decision_messages  # noqa: E402
from app.config import settings  # noqa: E402
from app.knowledge_base import Entry, load_knowledge_base  # noqa: E402
from app.vector_store import SearchResult  # noqa: E402


def load_records(path: Path) -> list[dict]:
    """Read one JSON record per line. Damaged lines are skipped, not fatal."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("rating") in ("up", "down"):
            records.append(record)
    return records


def build_review_queue(records: list[dict]) -> list[dict]:
    """Thumbs-down replies, in the same shape as the evaluation sets, with the expected
    answer left empty for a person to fill in."""
    queue = []
    for record in records:
        if record["rating"] != "down":
            continue
        queue.append(
            {
                "id": f"F{len(queue) + 1:03d}",
                "category": "feedback",
                "question": record.get("question", ""),
                "previous_message": record.get("previous_message"),
                "system_action": record.get("action"),
                "system_cards": [record["source_card"]] if record.get("source_card") else record.get("options", []),
                "system_reply": record.get("answer", ""),
                "comment": record.get("comment"),
                "expected_action": None,
                "expected_cards": [],
                "note": "Fill in expected_action and expected_cards after reviewing.",
            }
        )
    return queue


def decision_target(record: dict) -> dict:
    """The decision the agent made, in the JSON shape the decision step returns."""
    action = record["action"]
    if action == "answer":
        cards = [record["source_card"]]
    elif action == "clarify":
        cards = list(record.get("options", []))
    else:
        cards = []
    # Keep only the first decision's reason, not the notes of later checks.
    reason = (record.get("reason") or "").split(" Second check:")[0].split(" Answer check:")[0].strip()
    return {
        "reason": reason,
        "is_support_question": action != "decline",
        "matching_cards": cards,
        "clarifying_question": record.get("answer", "") if action == "clarify" else "",
    }


def build_training_examples(records: list[dict], entries: list[Entry]) -> list[dict]:
    """Thumbs-up decisions that went through the decision step, as chat examples."""
    by_id = {entry.id: entry for entry in entries}
    examples = []
    for record in records:
        if record["rating"] != "up" or record.get("decided_by") not in ("llm", "rules"):
            continue  # guardrail decisions never reach the LLM, so they teach it nothing
        if record.get("action") not in ("answer", "clarify", "decline"):
            continue
        if record["action"] == "answer" and not record.get("source_card"):
            continue
        matches = record.get("matches", [])
        if any(m.get("card_id") not in by_id for m in matches):
            continue  # the knowledge base changed since this feedback was given
        candidates = [
            SearchResult(by_id[m["card_id"]], float(m["score"]))
            for m in matches
            if not by_id[m["card_id"]].out_of_scope
        ][:3]
        if not candidates:
            continue

        # Rebuild the message the agent saw (follow-up and rewritten search included).
        query = record.get("question", "")
        if record.get("previous_message"):
            query = f"{record['previous_message']} {query}"
        if record.get("search_query"):
            query = f"{query}\n(In English: {record['search_query']})"

        messages = decision_messages(query, candidates)
        messages.append({"role": "assistant", "content": json.dumps(decision_target(record), ensure_ascii=False)})
        examples.append({"messages": messages})
    return examples


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / settings.feedback_path
    if not path.exists():
        print(f"No feedback file at {path}")
        return
    records = load_records(path)
    entries = load_knowledge_base(ROOT / settings.knowledge_base_path)

    queue = build_review_queue(records)
    examples = build_training_examples(records, entries)

    (path.parent / "review_queue.json").write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")
    with (path.parent / "finetune_decisions.jsonl").open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    up = sum(r["rating"] == "up" for r in records)
    print(f"{len(records)} feedback records ({up} up, {len(records) - up} down)")
    print(f"{len(queue)} cases to review      -> {path.parent / 'review_queue.json'}")
    print(f"{len(examples)} training examples -> {path.parent / 'finetune_decisions.jsonl'}")


if __name__ == "__main__":
    main()
