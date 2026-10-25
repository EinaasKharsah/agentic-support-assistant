import json

import pytest

from app.knowledge_base import load_knowledge_base


def test_loads_all_ten_entries():
    entries = load_knowledge_base("data/knowledge_base.json")
    assert len(entries) == 10


def test_only_q10_is_out_of_scope():
    entries = load_knowledge_base("data/knowledge_base.json")
    out_of_scope = [e.id for e in entries if e.out_of_scope]
    assert out_of_scope == ["Q10"]


def test_text_contains_question_and_answer():
    q4 = load_knowledge_base("data/knowledge_base.json")[3]
    text = q4.to_text()
    assert "payment methods" in text  # from the question
    assert "Cryptocurrency" in text  # from the answer


def test_missing_field_fails(tmp_path):
    bad = tmp_path / "kb.json"
    bad.write_text(json.dumps([{"id": "Q1", "category": "Account"}]))
    with pytest.raises(ValueError, match="missing fields"):
        load_knowledge_base(bad)


def test_duplicate_ids_fail(tmp_path):
    item = {"id": "Q1", "category": "A", "question": "q", "answer": "a"}
    bad = tmp_path / "kb.json"
    bad.write_text(json.dumps([item, item]))
    with pytest.raises(ValueError, match="unique"):
        load_knowledge_base(bad)
