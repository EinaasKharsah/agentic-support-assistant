import json
from pathlib import Path

from evaluation.metrics import classify_error, compute_metrics


def row(expected, predicted, expected_cards=("Q1",), predicted_card=None, top3=("Q1", "Q2", "Q3")):
    return {
        "expected_action": expected,
        "expected_cards": list(expected_cards),
        "predicted_action": predicted,
        "predicted_card": predicted_card,
        "top3_cards": list(top3),
        "seconds": 1.0,
    }


def test_error_types():
    assert classify_error("decline", "answer", False) == "unsafe"
    assert classify_error("clarify", "answer", True) == "missed_clarify"
    assert classify_error("answer", "clarify", False) == "over_clarify"
    assert classify_error("answer", "decline", False) == "over_decline"
    assert classify_error("answer", "answer", False) == "wrong_card"
    assert classify_error("answer", "answer", True) is None


def test_compute_metrics():
    rows = [
        row("answer", "answer", predicted_card="Q1"),  # correct
        row("answer", "answer", predicted_card="Q2", top3=("Q2", "Q1", "Q3")),  # wrong card
        row("decline", "answer", expected_cards=(), predicted_card="Q1"),  # unsafe
        row("clarify", "clarify", expected_cards=("Q5", "Q6")),  # correct
    ]
    m = compute_metrics(rows)
    assert m["questions"] == 4
    assert m["decision_accuracy"] == 0.5
    assert m["errors"] == {"wrong_card": 1, "unsafe": 1}
    assert m["confusion"]["decline"]["answer"] == 1
    assert m["retrieval_recall_at_1"] == 0.5  # 1 of 2 answer questions had Q1 first
    assert m["retrieval_recall_at_3"] == 1.0


def test_eval_set_is_valid():
    items = json.loads(Path("evaluation/eval_set.json").read_text(encoding="utf-8"))
    ids = [item["id"] for item in items]
    assert len(ids) == len(set(ids))
    for item in items:
        assert item["expected_action"] in {"answer", "clarify", "decline"}
        assert item["split"] in {"dev", "test"}
        if item["expected_action"] == "answer":
            assert len(item["expected_cards"]) == 1
        if item["expected_action"] == "clarify":
            assert len(item["expected_cards"]) >= 2
