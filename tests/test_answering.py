from app.answering import build_messages, generate_answer
from app.knowledge_base import load_knowledge_base
from app.llm import LLMError
from tests.fakes import FakeLLM

Q1 = load_knowledge_base("data/knowledge_base.json")[0]


def test_prompt_contains_only_the_selected_card():
    messages = build_messages("I forgot my password", Q1)
    system = messages[0]["content"]
    assert Q1.answer in system
    assert "ONLY" in system
    assert messages[1] == {"role": "user", "content": "I forgot my password"}


def test_prompt_explains_how_to_handle_facts_not_in_the_article():
    system = build_messages("Can I pay with PayPal?", Q1)[0]["content"]
    assert "does NOT mention" in system
    assert "do not say yes or no" in system


def test_uses_llm_answer():
    llm = FakeLLM(reply="Click Forgot Password on the login page.")
    answer, used_llm = generate_answer("I forgot my password", Q1, llm)
    assert answer == "Click Forgot Password on the login page."
    assert used_llm is True


def test_falls_back_to_card_when_ollama_fails():
    llm = FakeLLM(error=LLMError("Ollama is down"))
    answer, used_llm = generate_answer("I forgot my password", Q1, llm)
    assert answer == Q1.answer
    assert used_llm is False


def test_falls_back_when_llm_returns_empty_text():
    answer, used_llm = generate_answer("I forgot my password", Q1, FakeLLM(reply="   "))
    assert answer == Q1.answer
    assert used_llm is False
