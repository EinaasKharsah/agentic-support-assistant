import pytest

from app.agent import DECISION_SCHEMA, DEFAULT_DECLINE_MESSAGE, Action, SupportAgent
from app.knowledge_base import load_knowledge_base
from app.llm import LLMError
from app.vector_store import VectorStore, make_client
from tests.fakes import FakeEmbedder, FakeLLM

ENTRIES = load_knowledge_base("data/knowledge_base.json")
Q10_ANSWER = ENTRIES[9].answer


@pytest.fixture(scope="module")
def store():
    store = VectorStore(make_client(None), FakeEmbedder(), "agent_test")
    store.index(ENTRIES)
    return store


def make_agent(store, llm, **kwargs):
    # The fake embedder gives lower scores than the real model, so we use a low threshold.
    kwargs.setdefault("min_relevance", 0.2)
    return SupportAgent(store, llm, **kwargs)


def decision(cards=(), support=True, question="", reason="test"):
    return {
        "reason": reason,
        "is_support_question": support,
        "matching_cards": list(cards),
        "clarifying_question": question,
    }


# ---- guardrail 1: nothing relevant ------------------------------------------

def test_irrelevant_question_is_declined_without_calling_the_llm(store):
    llm = FakeLLM(json_reply=decision(["Q1"]))
    response = make_agent(store, llm).respond("zzz qqq xxx")
    assert response.action == Action.DECLINE
    assert response.decided_by == "guardrail"
    assert response.message == DEFAULT_DECLINE_MESSAGE
    assert llm.calls == []


# ---- LLM decisions ----------------------------------------------------------

def test_llm_answer_uses_the_chosen_card(store):
    llm = FakeLLM(json_reply=decision(["Q1"]), reply="Click Forgot Password.")
    response = make_agent(store, llm).respond("How do I reset my password?")
    assert response.action == Action.ANSWER
    assert response.source == "Q1"
    assert response.message == "Click Forgot Password."
    assert response.decided_by == "llm"


def test_llm_clarify_offers_options_from_real_cards(store):
    llm = FakeLLM(
        json_reply=decision(["Q6", "Q5"], question="Do you want to cancel or get a refund?")
    )
    response = make_agent(store, llm).respond("How do I cancel my subscription or request a refund?")
    assert response.action == Action.CLARIFY
    assert response.message == "Do you want to cancel or get a refund?"
    assert [o.card_id for o in response.options] == ["Q6", "Q5"]
    assert response.options[0].label == "How do I cancel my subscription?"


def test_llm_decline(store):
    llm = FakeLLM(json_reply=decision(support=False, reason="Not a support question."))
    response = make_agent(store, llm).respond("How do I reset my password?")
    assert response.action == Action.DECLINE
    assert response.decided_by == "llm"
    assert response.reason == "Not a support question."
    assert response.message == DEFAULT_DECLINE_MESSAGE


def test_out_of_scope_top_match_is_declined_without_calling_the_llm(store):
    llm = FakeLLM(json_reply=decision(["Q7"]))
    response = make_agent(store, llm).respond("Can you help me write code for my integration project?")
    assert response.action == Action.DECLINE
    assert response.decided_by == "guardrail"
    assert response.message == Q10_ANSWER  # the knowledge base's own decline text
    assert llm.calls == []


def test_out_of_scope_card_is_never_shown_to_the_llm(store):
    llm = FakeLLM(json_reply=decision(["Q7"]), reply="answer")
    agent = make_agent(store, llm, top_k=10)  # top_k=10 so Q10 is surely among the matches
    agent.respond("What third-party tools does the platform integrate with?")
    decision_prompt = llm.calls[0][1]["content"]
    assert "[Q7]" in decision_prompt
    assert "[Q10]" not in decision_prompt


def test_reason_comes_first_in_the_decision_schema():
    # The model writes JSON left to right: it must reason before choosing the action.
    assert list(DECISION_SCHEMA["properties"])[0] == "reason"


# ---- guardrail 2: invalid LLM decisions ---------------------------------------

def test_unknown_card_id_falls_back_to_rules(store):
    llm = FakeLLM(json_reply=decision(["Q99"]), reply="answer text")
    response = make_agent(store, llm).respond("How do I reset my password?")
    assert response.decided_by == "rules"
    assert "no valid card" in response.reason


def test_unknown_ids_are_ignored_and_one_valid_card_means_answer(store):
    llm = FakeLLM(json_reply=decision(["Q99", "Q1"]), reply="answer text")
    response = make_agent(store, llm).respond("How do I reset my password?")
    assert response.action == Action.ANSWER
    assert response.source == "Q1"


def test_invalid_json_shape_falls_back_to_rules(store):
    llm = FakeLLM(json_reply={"reason": "x", "is_support_question": "yes"}, reply="answer text")
    response = make_agent(store, llm).respond("How do I reset my password?")
    assert response.decided_by == "rules"
    assert "not valid" in response.reason


def test_llm_error_falls_back_to_rules(store):
    llm = FakeLLM(error=LLMError("Ollama is down"))
    response = make_agent(store, llm).respond("How do I reset my password?")
    assert response.decided_by == "rules"
    assert response.action == Action.ANSWER
    assert response.source == "Q1"
    assert response.message == ENTRIES[0].answer  # answer also falls back to the card text


# ---- rules mode (baseline) ----------------------------------------------------

def test_rules_mode_never_calls_the_decision_llm(store):
    llm = FakeLLM(reply="answer text")
    response = make_agent(store, llm, mode="rules").respond("How do I reset my password?")
    assert response.decided_by == "rules"
    assert all("decision step" not in m[0]["content"] for m in llm.calls)


def test_rules_clarify_when_top_two_scores_are_close(store):
    llm = FakeLLM(reply="answer text")
    agent = make_agent(store, llm, mode="rules", ambiguity_gap=1.0)  # every gap counts as "close"
    response = agent.respond("How do I reset my password?")
    assert response.action == Action.CLARIFY
    assert len(response.options) == 2


def test_rules_decline_when_best_match_is_out_of_scope(store):
    response = make_agent(store, FakeLLM(), mode="rules").respond(
        "Can you help me write code for my integration project?"
    )
    assert response.action == Action.DECLINE


# ---- follow-up after a clarifying question -------------------------------------

def test_previous_message_is_combined_with_the_follow_up(store):
    llm = FakeLLM(json_reply=decision(["Q6"]), reply="Go to Settings > Billing.")
    response = make_agent(store, llm).respond("cancel my subscription", previous_message="I want to stop paying")
    decision_prompt = llm.calls[0][1]["content"]
    assert "I want to stop paying cancel my subscription" in decision_prompt
    assert response.source == "Q6"


def test_generate_answers_false_returns_card_text_without_llm_answer(store):
    llm = FakeLLM(json_reply=decision(["Q1"]), reply="should not be used")
    response = make_agent(store, llm, generate_answers=False).respond("How do I reset my password?")
    assert response.message == ENTRIES[0].answer
    assert len(llm.calls) == 1  # only the decision call
