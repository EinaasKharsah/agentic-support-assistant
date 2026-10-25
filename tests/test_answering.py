from app.answering import build_messages, fact_problem, generate_answer
from app.knowledge_base import load_knowledge_base
from app.llm import LLMError
from tests.fakes import FakeLLM

ENTRIES = load_knowledge_base("data/knowledge_base.json")
Q1, Q4, Q5, Q6 = ENTRIES[0], ENTRIES[3], ENTRIES[4], ENTRIES[5]
Q1_ANSWER = "Click Forgot Password on the login page. The link arrives within 2 minutes and expires after 30 minutes."


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
    answer, note = generate_answer("I forgot my password", Q1, FakeLLM(reply=Q1_ANSWER))
    assert answer == Q1_ANSWER
    assert note is None


def test_falls_back_to_card_when_ollama_fails():
    llm = FakeLLM(error=LLMError("Ollama is down"))
    answer, note = generate_answer("I forgot my password", Q1, llm)
    assert answer == Q1.answer
    assert "not available" in note


def test_falls_back_when_llm_returns_empty_text():
    answer, note = generate_answer("I forgot my password", Q1, FakeLLM(reply="   "))
    assert answer == Q1.answer
    assert "empty" in note


# ---- fact check -----------------------------------------------------------------

def test_every_card_passes_its_own_fact_check():
    assert all(fact_problem(entry.answer, entry) is None for entry in ENTRIES)


def test_dropped_limit_falls_back_to_card():
    # A real answer from the chat page: it left out "non-refundable after 48 hours".
    real_answer = (
        "A full refund is available within 14 days of your initial purchase or annual renewal. "
        "To request a refund, contact billing@test.com with your account email and order ID. "
        "Please contact support to confirm your order ID. "
        "Refunds are processed within 5 to 10 business days."
    )
    answer, note = generate_answer("I wanna request a refund", Q5, FakeLLM(reply=real_answer))
    assert answer == Q5.answer
    assert "left out" in note and "48" in note


def test_invented_number_falls_back_to_card():
    reply = Q5.answer.replace("14 days", "30 days")
    assert "added facts" in fact_problem(reply, Q5)


def test_invented_email_falls_back_to_card():
    reply = Q1_ANSWER + " You can also write to help@test.com."
    assert "help@test.com" in fact_problem(reply, Q1)


def test_changed_menu_path_falls_back_to_card():
    reply = "Go to Settings > Billing > Cancel. Access continues until the end of the period. Data is kept for 60 days."
    assert "Settings > Billing > Cancel" in fact_problem(reply, Q6)


def test_rephrased_answer_with_all_facts_passes():
    reply = (
        "You can get a full refund within 14 days of buying or renewing an annual plan. "
        "Monthly charges can't be refunded after 48 hours. Email billing@test.com with your "
        "account email and order ID; refunds take 5 to 10 business days."
    )
    assert fact_problem(reply, Q5) is None


def test_not_mentioned_answer_passes():
    # The PayPal case: no numbers in the card, so the friendly "not mentioned" answer is kept.
    reply = "PayPal isn't mentioned in our help article. We accept major credit and debit cards and bank transfers for annual plans."
    answer, note = generate_answer("Can I pay with PayPal?", Q4, FakeLLM(reply=reply))
    assert answer == reply and note is None
