"""Turn evaluation results into numbers.

Not all mistakes are equally bad. For a support assistant:
  - "unsafe":       ANSWERED a question it should have declined (gives wrong info)
  - "missed_clarify": ANSWERED a vague question instead of asking (may guess wrong)
  - "over_clarify": asked a question although the answer was clear (annoying)
  - "over_decline": declined a real support question (annoying, customer must contact support)
  - "wrong_card":   answered, but from the wrong card
"""

from collections import Counter

ACTIONS = ["answer", "clarify", "decline"]


def classify_error(expected: str, predicted: str, correct_card: bool) -> str | None:
    if expected == predicted:
        if predicted == "answer" and not correct_card:
            return "wrong_card"
        return None
    if predicted == "answer" and expected == "decline":
        return "unsafe"
    if predicted == "answer" and expected == "clarify":
        return "missed_clarify"
    if predicted == "clarify":
        return "over_clarify" if expected == "answer" else "clarify_instead_of_decline"
    return "over_decline"


def accepted_actions(row: dict) -> list[str]:
    """expected_action is one action, or a list when several replies are acceptable
    (for example a message with two questions may be answered or clarified)."""
    expected = row["expected_action"]
    return expected if isinstance(expected, list) else [expected]


def row_error(row: dict) -> str | None:
    accepted, predicted = accepted_actions(row), row["predicted_action"]
    correct_card = row.get("predicted_card") in row["expected_cards"]
    if predicted in accepted:
        return classify_error(predicted, predicted, correct_card)
    return classify_error(accepted[0], predicted, correct_card)


def compute_metrics(rows: list[dict]) -> dict:
    """rows: one dict per question with expected_action, expected_cards,
    predicted_action, predicted_card, top3_cards, seconds (and optional category)."""
    total = len(rows)
    correct = 0
    errors: Counter = Counter()
    confusion = {e: {p: 0 for p in ACTIONS} for e in ACTIONS}
    retrieval_at1 = retrieval_at3 = retrieval_total = 0
    by_category: dict[str, list[int]] = {}

    for row in rows:
        expected, predicted = accepted_actions(row)[0], row["predicted_action"]
        confusion[expected][predicted] += 1

        error = row_error(row)
        if error:
            errors[error] += 1
        else:
            correct += 1
        if "category" in row:
            stats = by_category.setdefault(row["category"], [0, 0])
            stats[0] += error is None
            stats[1] += 1

        # Retrieval quality only makes sense for questions with one right card.
        if row["expected_action"] == "answer":
            retrieval_total += 1
            top3 = row.get("top3_cards", [])
            retrieval_at1 += bool(top3) and top3[0] in row["expected_cards"]
            retrieval_at3 += any(card in row["expected_cards"] for card in top3)

    seconds = [row.get("seconds", 0.0) for row in rows]
    return {
        "questions": total,
        "decision_accuracy": round(correct / total, 3) if total else 0.0,
        "errors": dict(errors),
        "confusion": confusion,
        "retrieval_recall_at_1": round(retrieval_at1 / retrieval_total, 3) if retrieval_total else 0.0,
        "retrieval_recall_at_3": round(retrieval_at3 / retrieval_total, 3) if retrieval_total else 0.0,
        "avg_seconds": round(sum(seconds) / total, 2) if total else 0.0,
        "by_category": {cat: f"{ok}/{n}" for cat, (ok, n) in by_category.items()},
    }


def format_report(name: str, metrics: dict) -> str:
    lines = [
        f"## {name}",
        "",
        f"- Questions: {metrics['questions']}",
        f"- Decision accuracy: {metrics['decision_accuracy']:.0%}",
        f"- Retrieval Recall@1: {metrics['retrieval_recall_at_1']:.0%}, "
        f"Recall@3: {metrics['retrieval_recall_at_3']:.0%}",
        f"- Average time per question: {metrics['avg_seconds']}s",
        f"- Errors: {metrics['errors'] or 'none'}",
    ]
    if metrics.get("by_category"):
        lines.append("- Correct by category: " + ", ".join(f"{c} {v}" for c, v in metrics["by_category"].items()))
    lines += [
        "",
        "| expected \\ predicted | " + " | ".join(ACTIONS) + " |",
        "|---|" + "---|" * len(ACTIONS),
    ]
    for expected in ACTIONS:
        counts = " | ".join(str(metrics["confusion"][expected][p]) for p in ACTIONS)
        lines.append(f"| {expected} | {counts} |")
    return "\n".join(lines)
