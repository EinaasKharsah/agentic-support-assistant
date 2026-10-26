import json
import sys
from pathlib import Path

from app.agent import DECISION_SYSTEM_PROMPT
from app.knowledge_base import load_knowledge_base

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from export_feedback import build_review_queue, build_training_examples, decision_target, load_records  # noqa: E402

ENTRIES = load_knowledge_base("data/knowledge_base.json")

MATCHES = [
    {"card_id": "Q1", "score": 0.72},
    {"card_id": "Q10", "score": 0.56},
    {"card_id": "Q2", "score": 0.61},
    {"card_id": "Q3", "score": 0.58},
]


def record(**changes):
    base = {
        "rating": "up",
        "question": "I forgot my password",
        "previous_message": None,
        "answer": "Click Forgot Password.",
        "action": "answer",
        "source_card": "Q1",
        "options": [],
        "decided_by": "llm",
        "reason": "The customer cannot log in. Second check: clearly about the password.",
        "search_query": None,
        "matches": MATCHES,
    }
    return {**base, **changes}


def test_thumbs_up_answer_becomes_a_training_example():
    [example] = build_training_examples([record()], ENTRIES)
    system, user, assistant = example["messages"]
    assert system["content"] == DECISION_SYSTEM_PROMPT
    assert "Customer message: I forgot my password" in user["content"]
    assert "[Q1]" in user["content"] and "[Q10]" not in user["content"]  # same prompt as the agent
    target = json.loads(assistant["content"])
    assert target["matching_cards"] == ["Q1"]
    assert target["is_support_question"] is True
    assert target["reason"] == "The customer cannot log in."  # notes of later checks removed


def test_clarify_and_decline_targets():
    clarify = decision_target(record(action="clarify", source_card=None, options=["Q6", "Q5"], answer="Cancel or refund?"))
    assert clarify["matching_cards"] == ["Q6", "Q5"] and clarify["clarifying_question"] == "Cancel or refund?"
    decline = decision_target(record(action="decline", source_card=None))
    assert decline["is_support_question"] is False and decline["matching_cards"] == []


def test_follow_up_and_rewrite_are_part_of_the_prompt():
    example = build_training_examples(
        [record(question="refund", previous_message="I want to stop paying", search_query="request a refund")], ENTRIES
    )[0]
    user = example["messages"][1]["content"]
    assert "I want to stop paying refund\n(In English: request a refund)" in user


def test_thumbs_down_and_guardrail_decisions_are_not_training_examples():
    records = [record(rating="down"), record(decided_by="guardrail", action="decline", source_card=None)]
    assert build_training_examples(records, ENTRIES) == []


def test_unknown_cards_are_skipped():
    assert build_training_examples([record(matches=[{"card_id": "Q99", "score": 0.9}])], ENTRIES) == []


def test_thumbs_down_goes_to_the_review_queue():
    [case] = build_review_queue([record(), record(rating="down", comment="wrong card")])
    assert case["id"] == "F001"
    assert case["system_action"] == "answer" and case["system_cards"] == ["Q1"]
    assert case["expected_action"] is None and case["comment"] == "wrong card"


def test_damaged_lines_are_skipped(tmp_path):
    path = tmp_path / "feedback.jsonl"
    path.write_text(json.dumps(record()) + "\nnot json\n" + json.dumps({"rating": "maybe"}) + "\n", encoding="utf-8")
    assert len(load_records(path)) == 1
