"""API tests. The real agent is replaced by one with fake models, so no Ollama is needed."""

import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from app import main
from app.agent import SupportAgent
from app.knowledge_base import load_knowledge_base
from app.vector_store import VectorStore, make_client
from tests.fakes import FakeEmbedder, FakeLLM


def make_test_agent(llm: FakeLLM) -> SupportAgent:
    store = VectorStore(make_client(None), FakeEmbedder(), "api_test")
    store.index(load_knowledge_base("data/knowledge_base.json"))
    return SupportAgent(store, llm, min_relevance=0.2)


@pytest.fixture
def client_with(monkeypatch, tmp_path):
    test_settings = dataclasses.replace(main.settings, feedback_path=str(tmp_path / "feedback.jsonl"))
    monkeypatch.setattr(main, "settings", test_settings)

    def _make(llm: FakeLLM) -> TestClient:
        main.app.state.agent = make_test_agent(llm)
        return TestClient(main.app)

    yield _make
    del main.app.state.agent


def decision(cards, support=True, question=""):
    return {"reason": "test", "is_support_question": support, "matching_cards": cards, "clarifying_question": question}


def test_health(client_with):
    with client_with(FakeLLM()) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["cards"] == 10


def test_chat_answer_includes_source_and_matches(client_with):
    llm = FakeLLM(json_reply=decision(["Q1"]), reply="Click Forgot Password.")
    with client_with(llm) as client:
        body = client.post("/chat", json={"message": "How do I reset my password?"}).json()
    assert body["action"] == "answer"
    assert body["message"] == "Click Forgot Password."
    assert body["source"]["id"] == "Q1"
    assert body["source"]["question"] == "How do I reset my password?"
    assert body["decided_by"] == "llm"
    assert len(body["matches"]) == 4  # top 4, so 3 remain if Q10 is removed
    assert body["id"]


def test_chat_clarify_returns_options(client_with):
    llm = FakeLLM(json_reply=decision(["Q6", "Q5"], question="Cancel or refund?"))
    with client_with(llm) as client:
        body = client.post("/chat", json={"message": "How do I cancel my subscription or request a refund?"}).json()
    assert body["action"] == "clarify"
    assert body["message"] == "Cancel or refund?"
    assert [o["card_id"] for o in body["options"]] == ["Q6", "Q5"]
    assert body["source"] is None


def test_chat_follow_up_uses_previous_message(client_with):
    llm = FakeLLM(json_reply=decision(["Q6"]), reply="Go to Settings > Billing.")
    with client_with(llm) as client:
        client.post("/chat", json={"message": "cancel", "previous_message": "I want to stop paying"})
    assert "I want to stop paying cancel" in llm.calls[0][1]["content"]


def test_empty_message_is_rejected(client_with):
    with client_with(FakeLLM()) as client:
        assert client.post("/chat", json={"message": ""}).status_code == 422


def test_feedback_is_saved(client_with, tmp_path):
    with client_with(FakeLLM()) as client:
        response = client.post(
            "/feedback",
            json={"response_id": "abc", "rating": "down", "question": "q", "answer": "a"},
        )
    assert response.json() == {"status": "saved"}
    saved = json.loads((tmp_path / "feedback.jsonl").read_text().strip())
    assert saved["rating"] == "down"
    assert saved["response_id"] == "abc"


def test_invalid_rating_is_rejected(client_with):
    with client_with(FakeLLM()) as client:
        response = client.post(
            "/feedback", json={"response_id": "abc", "rating": "maybe", "question": "q", "answer": "a"}
        )
    assert response.status_code == 422


def test_chat_page_and_static_files_are_served(client_with):
    with client_with(FakeLLM()) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Help center assistant" in page.text
        assert client.get("/static/app.js").status_code == 200


def test_mode_override_uses_rules_for_one_message(client_with):
    llm = FakeLLM(json_reply=decision(["Q1"]), reply="answer text")
    with client_with(llm) as client:
        body = client.post("/chat", json={"message": "How do I reset my password?", "mode": "rules"}).json()
    assert body["decided_by"] == "rules"


def test_index_with_retry_waits_for_qdrant(monkeypatch):
    monkeypatch.setattr(main.time, "sleep", lambda s: None)

    class SlowStore:
        calls = 0

        def index(self, entries):
            SlowStore.calls += 1
            if SlowStore.calls < 3:
                raise ConnectionError("Qdrant is starting")

    main.index_with_retry(SlowStore(), [], attempts=5)
    assert SlowStore.calls == 3
