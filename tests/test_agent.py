import pytest

from app.agent import DECISION_SCHEMA, DEFAULT_DECLINE_MESSAGE, Action, SupportAgent
from app.knowledge_base import load_knowledge_base
from app.llm import LLMError
from app.vector_store import VectorStore, make_client
from tests.fakes import FakeEmbedder, FakeLLM, FakeReranker

ENTRIES = load_knowledge_base("data/knowledge_base.json")


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
    response = make_agent(store, llm, rewrite_weak_queries=False).respond("zzz qqq xxx")
    assert response.action == Action.DECLINE
    assert response.decided_by == "guardrail"
    assert response.message == DEFAULT_DECLINE_MESSAGE
    assert llm.calls == []


# ---- LLM decisions ----------------------------------------------------------

def test_llm_answer_uses_the_chosen_card(store):
    reply = "Click Forgot Password. The link arrives within 2 minutes and expires after 30 minutes."
    llm = FakeLLM(json_reply=decision(["Q1"]), reply=reply)
    response = make_agent(store, llm).respond("How do I reset my password?")
    assert response.action == Action.ANSWER
    assert response.source == "Q1"
    assert response.message == reply
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
    assert response.message == DEFAULT_DECLINE_MESSAGE  # general text, not Q10's coding-help text
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


# ---- second check (reflection) ---------------------------------------------------

def second_check(choice, question="", reason="check"):
    return {"reason": reason, "choice": choice, "clarifying_question": question}


def test_second_check_either_turns_answer_into_clarify(store):
    llm = FakeLLM(json_reply=[decision(["Q1"]), second_check("either", "Password or email?")])
    agent = make_agent(store, llm, min_relevance=0.0, second_check_gap=1.0)  # any runner-up counts as close
    response = agent.respond("How do I reset my password?")
    assert response.action == Action.CLARIFY
    assert response.message == "Password or email?"
    assert len(response.options) == 2
    assert "Second check" in response.reason
    assert "A:" in llm.calls[1][1]["content"] and "B:" in llm.calls[1][1]["content"]


def test_second_check_clear_choice_keeps_answer(store):
    llm = FakeLLM(json_reply=[decision(["Q1"]), second_check("A")], reply="answer text")
    agent = make_agent(store, llm, min_relevance=0.0, second_check_gap=1.0)
    response = agent.respond("How do I reset my password?")
    assert response.action == Action.ANSWER
    assert response.source == "Q1"  # Q1 has the highest score, so it is option A


def test_second_check_is_skipped_when_turned_off(store):
    llm = FakeLLM(json_reply=[decision(["Q1"])], reply="answer text")
    agent = make_agent(store, llm, min_relevance=0.0, second_check_gap=0)
    agent.respond("How do I reset my password?")
    decision_calls = [c for c in llm.calls if "A:" in c[-1]["content"]]
    assert decision_calls == []


def test_second_check_error_keeps_first_decision(store):
    class FailOnSecondCall(FakeLLM):
        def chat_json(self, messages, json_schema):
            if self.calls:
                self.calls.append(messages)
                raise LLMError("timeout")
            return super().chat_json(messages, json_schema)

    llm = FailOnSecondCall(json_reply=decision(["Q1"]), reply="answer text")
    response = make_agent(store, llm, min_relevance=0.0, second_check_gap=1.0).respond("How do I reset my password?")
    assert response.action == Action.ANSWER
    assert response.source == "Q1"


def test_model_sees_three_cards_even_when_q10_is_retrieved(store):
    llm = FakeLLM(json_reply=decision(["Q7"]), reply="answer")
    make_agent(store, llm).respond("What third-party tools does the platform integrate with?")
    prompt = llm.calls[0][1]["content"]
    assert prompt.count("score=") == 3


# ---- search again with a rewritten query ------------------------------------------

def test_weak_search_is_rewritten_and_searched_again(store):
    llm = FakeLLM(
        json_reply=[{"has_meaning": True, "search_query": "How do I reset my password?"}, decision(["Q1"])],
        reply="Click Forgot Password.",
    )
    response = make_agent(store, llm).respond("Ich habe mein Passwort vergessen")
    assert response.search_query == "How do I reset my password?"
    assert response.action == Action.ANSWER
    assert response.source == "Q1"
    assert "In English: How do I reset my password?" in llm.calls[1][1]["content"]


def test_strong_search_is_not_rewritten(store):
    llm = FakeLLM(json_reply=decision(["Q1"]), reply="answer")
    response = make_agent(store, llm).respond("How do I reset my password?")
    assert response.search_query is None
    assert all("search query" not in call[0]["content"] for call in llm.calls)


def test_unchanged_rewrite_is_ignored_and_message_declined(store):
    llm = FakeLLM(json_reply=[{"has_meaning": True, "search_query": "zzz qqq xxx"}])
    response = make_agent(store, llm).respond("zzz qqq xxx")
    assert response.action == Action.DECLINE
    assert response.search_query is None


def test_rewrite_failure_keeps_the_first_search(store):
    class FailingRewrite(FakeLLM):
        def chat_json(self, messages, json_schema):
            raise LLMError("down")

    response = make_agent(store, FailingRewrite()).respond("zzz qqq xxx")
    assert response.action == Action.DECLINE


def test_rules_mode_never_rewrites(store):
    llm = FakeLLM(json_reply=[{"has_meaning": True, "search_query": "How do I reset my password?"}])
    response = make_agent(store, llm, mode="rules").respond("zzz qqq xxx")
    assert response.search_query is None
    assert llm.calls == []


def test_message_without_meaning_is_not_rewritten(store):
    # Regression: an earlier prompt turned "asdfgh qwerty zxcv" into "accounts billing integrations data".
    llm = FakeLLM(json_reply=[{"has_meaning": False, "search_query": "accounts billing integrations data"}])
    response = make_agent(store, llm).respond("zzz qqq xxx")
    assert response.search_query is None
    assert response.action == Action.DECLINE


def test_rewrite_prompt_does_not_list_topics():
    from app.agent import REWRITE_PROMPT

    assert "billing" not in REWRITE_PROMPT.lower()
    assert "integrations" not in REWRITE_PROMPT.lower()


def test_failed_answer_check_shows_card_text_and_explains_why(store):
    llm = FakeLLM(json_reply=decision(["Q1"]), reply="Click Forgot Password.")  # drops "2" and "30" minutes
    response = make_agent(store, llm).respond("How do I reset my password?")
    assert response.message == ENTRIES[0].answer
    assert "Answer check:" in response.reason


# ---- fixes after the held-out evaluation -------------------------------------------

def test_looks_untranslated():
    from app.agent import looks_untranslated

    assert looks_untranslated("¿Cómo cambio el correo?", "Cómo cambiar correo")  # accents
    assert looks_untranslated("Como cambio el correo de mi cuenta", "como cambiar correo cuenta")  # same words
    assert not looks_untranslated("¿Cómo cambio el correo?", "How do I change my email address")
    assert not looks_untranslated("how do i chnage my emial", "How do I change my email address")


def test_untranslated_rewrite_is_translated_on_a_second_try(store):
    llm = FakeLLM(
        json_reply=[
            {"has_meaning": True, "search_query": "Passwort vergessen"},  # still German
            {"search_query": "How do I reset my password?"},
            decision(["Q1"]),
        ],
        reply="Click Forgot Password. The link arrives within 2 minutes and expires after 30 minutes.",
    )
    response = make_agent(store, llm).respond("Ich habe mein Passwort vergessen")
    assert response.search_query == "How do I reset my password?"
    assert "Translate" in llm.calls[1][0]["content"]


def test_failed_translation_keeps_the_first_rewrite(store):
    llm = FakeLLM(json_reply=[{"has_meaning": True, "search_query": "Cómo cambiar correo"}, {"search_query": "Cómo cambiar"}, decision(["Q2"])])
    response = make_agent(store, llm).respond("¿Cómo cambio el correo?")
    assert response.search_query == "Cómo cambiar correo"


def test_internal_names_never_reach_the_customer():
    from app.agent import DEFAULT_CLARIFY_MESSAGE, customer_facing_question

    assert customer_facing_question("Which article are you referring to, A or B?") == DEFAULT_CLARIFY_MESSAGE
    assert customer_facing_question("Do you mean Q5 or Q6?") == DEFAULT_CLARIFY_MESSAGE
    assert customer_facing_question("") == DEFAULT_CLARIFY_MESSAGE
    good = "Do you want to cancel your subscription or request a refund?"
    assert customer_facing_question(good) == good
    assert customer_facing_question("Is this about a refund or a payment method?") == "Is this about a refund or a payment method?"


def test_second_check_question_with_internal_names_is_replaced(store):
    from app.agent import DEFAULT_CLARIFY_MESSAGE

    llm = FakeLLM(json_reply=[decision(["Q1"]), second_check("either", "Which article are you referring to, A or B?")])
    response = make_agent(store, llm, min_relevance=0.0, second_check_gap=1.0).respond("How do I reset my password?")
    assert response.action == Action.CLARIFY
    assert response.message == DEFAULT_CLARIFY_MESSAGE
    assert len(response.options) == 2  # the buttons still show both articles


# ---- re-ranker ---------------------------------------------------------------------

Q1_TITLE = "How do I reset my password?"
Q2_TITLE = "How do I change the email address on my account?"
Q7_TITLE = "What third-party tools does the platform integrate with?"
Q10_TITLE = "Can you help me write code for my integration project?"


def test_llm_sees_the_vector_order_even_with_a_reranker(store):
    # top_k=10: the test search returns every card. The re-ranker prefers Q2, but the LLM still
    # sees Q1 (the vector top) first: the re-ranker's order made it clarify less often.
    reranker = FakeReranker({Q2_TITLE: 8.0, Q1_TITLE: 2.0})
    llm = FakeLLM(json_reply=decision(["Q1"]), reply="answer")
    make_agent(store, llm, reranker=reranker, second_check_gap=0, top_k=10).respond("How do I reset my password?")
    prompt = llm.calls[0][1]["content"]
    first_card = prompt[prompt.index("Cards:"):].split("[", 1)[1][:3]
    assert first_card == "Q1]"


def test_low_reranker_score_declines_without_calling_the_llm(store):
    reranker = FakeReranker({}, default=-9.0)
    llm = FakeLLM(json_reply=decision(["Q1"]))
    response = make_agent(store, llm, reranker=reranker, min_rerank=-2.0).respond("How do I reset my password?")
    assert response.action == Action.DECLINE
    assert response.decided_by == "guardrail"
    assert "re-ranker" in response.reason
    assert llm.calls == []


def test_out_of_scope_card_first_after_reranking_declines(store):
    reranker = FakeReranker({Q10_TITLE: 9.0, Q1_TITLE: 1.0})
    llm = FakeLLM(json_reply=decision(["Q1"]))
    response = make_agent(store, llm, reranker=reranker, min_rerank=-2.0, top_k=10).respond("How do I reset my password?")
    assert response.action == Action.DECLINE
    assert llm.calls == []


def test_out_of_scope_card_first_by_a_small_margin_lets_the_llm_decide(store):
    # A real case: "can i connect to notion?" had Q10 first by 0.6 points, both very low.
    # That is noise, so the LLM decides instead of the re-ranker.
    reranker = FakeReranker({Q10_TITLE: -9.5, Q7_TITLE: -10.1}, default=-12.0)
    llm = FakeLLM(json_reply=decision(["Q7"]), reply="Notion isn't mentioned in our help article.")
    agent = make_agent(store, llm, reranker=reranker, min_rerank=-10.5, rerank_out_of_scope_margin=1.0,
                       second_check_gap=0, top_k=10)
    response = agent.respond("which third-party tools does the platform integrate with? notion?")
    assert response.action == Action.ANSWER
    assert response.source == "Q7"


def test_out_of_scope_card_first_by_a_clear_margin_declines(store):
    reranker = FakeReranker({Q10_TITLE: -6.0, Q7_TITLE: -10.1}, default=-12.0)
    llm = FakeLLM(json_reply=decision(["Q7"]))
    agent = make_agent(store, llm, reranker=reranker, min_rerank=-10.5, rerank_out_of_scope_margin=1.0, top_k=10)
    response = agent.respond("Write a SQL query for my integration database")
    assert response.action == Action.DECLINE
    assert "by 4.10 points" in response.reason
    assert llm.calls == []


def test_reranker_sure_of_one_article_answers_instead_of_clarifying(store):
    # A real case: for "How do I cancel my subscription?" (the title of Q6) the LLM listed two
    # cards, but the re-ranker gave Q6 +6.2 and the other card -8.0.
    q6 = ENTRIES[5]
    reranker = FakeReranker({q6.question: 6.2}, default=-8.0)
    llm = FakeLLM(json_reply=decision(["Q6", "Q5"], question="Cancel or refund?"), reply=q6.answer)
    agent = make_agent(store, llm, reranker=reranker, min_rerank=-10.5, rerank_sure=3.0, top_k=10)
    # (The fake search needs both card titles in the message to offer both cards to the LLM.)
    response = agent.respond("How do I cancel my subscription or request a refund?")
    assert response.action == Action.ANSWER
    assert response.source == "Q6"
    assert "re-ranker is sure" in response.reason


def test_clarifying_question_is_kept_when_the_reranker_is_not_sure(store):
    # As for "I want to stop paying": the best article scored only -1.4, so the question is kept.
    q5, q6 = ENTRIES[4], ENTRIES[5]
    reranker = FakeReranker({q6.question: -1.4, q5.question: -5.5}, default=-9.0)
    llm = FakeLLM(json_reply=decision(["Q6", "Q5"], question="Cancel or refund?"))
    agent = make_agent(store, llm, reranker=reranker, min_rerank=-10.5, rerank_sure=3.0, top_k=10)
    response = agent.respond("How do I cancel my subscription or request a refund?")
    assert response.action == Action.CLARIFY


def test_clarifying_question_is_kept_when_two_articles_are_sure(store):
    q1, q2 = ENTRIES[0], ENTRIES[1]
    reranker = FakeReranker({q1.question: 4.0, q2.question: 3.5}, default=-9.0)
    llm = FakeLLM(json_reply=decision(["Q1", "Q2"], question="Password or email?"))
    agent = make_agent(store, llm, reranker=reranker, min_rerank=-10.5, rerank_sure=3.0, top_k=10)
    response = agent.respond("How do I reset my password and also change my email?")
    assert response.action == Action.CLARIFY


def test_reranker_can_rescue_a_question_where_the_out_of_scope_card_was_first(store):
    # Without the re-ranker, this message is declined because Q10 is the top vector match.
    message = "Can you help me write code for my integration project?"
    assert make_agent(store, FakeLLM()).respond(message).action == Action.DECLINE
    q6 = ENTRIES[5]
    reranker = FakeReranker({q6.question: 6.0, Q10_TITLE: 1.0})
    llm = FakeLLM(json_reply=decision(["Q6"]), reply=q6.answer)
    response = make_agent(store, llm, reranker=reranker, min_rerank=-2.0, second_check_gap=0, top_k=10).respond(message)
    assert response.action == Action.ANSWER
    assert response.source == "Q6"
    assert response.rerank["Q6"] == 6.0


def test_reranker_error_falls_back_to_vector_scores(store):
    reranker = FakeReranker({}, error=RuntimeError("model missing"))
    llm = FakeLLM(json_reply=decision(["Q1"]), reply="answer")
    response = make_agent(store, llm, reranker=reranker, min_rerank=100.0).respond("How do I reset my password?")
    assert response.rerank == {}
    assert response.action != Action.DECLINE  # the impossible threshold was not used


def test_reranker_uses_the_english_rewrite(store):
    reranker = FakeReranker({Q1_TITLE: 5.0})
    llm = FakeLLM(json_reply=[{"has_meaning": True, "search_query": "How do I reset my password?"}, decision(["Q1"])], reply="answer")
    make_agent(store, llm, reranker=reranker).respond("Ich habe mein Passwort vergessen")
    assert reranker.queries == ["How do I reset my password?"]


def test_rules_mode_does_not_use_the_reranker(store):
    reranker = FakeReranker({}, default=-9.0)
    response = make_agent(store, FakeLLM(reply="answer"), reranker=reranker, min_rerank=0.0, mode="rules").respond(
        "How do I reset my password?"
    )
    assert reranker.queries == []
    assert response.action != Action.DECLINE
