"""The agent: decides whether to ANSWER, ask a CLARIFYING question, or DECLINE.

The LLM makes the decision, because it understands meaning. For example,
"I want to stop paying" could mean cancel OR refund; the similarity scores
alone did not show this (gap of only 0.06). Code adds guardrails around it:

  Guardrail 1 (before the LLM): if no card is relevant at all, decline.
                                 No need to ask a small model about the weather.
                                 If the best match is the knowledge base's own
                                 out-of-scope example (Q10), decline as well.
                                 Q10 is never shown to the LLM as a candidate:
                                 in evaluation, seeing it confused the model
                                 ("Q10 is out of scope, so decline") on normal questions.
  Guardrail 2 (after the LLM):  check the decision is valid. The card must be one
                                 of the retrieved cards, and the out-of-scope card
                                 can never be used as an answer.
  Second check (reflection):     if the LLM picked one card but another card scored
                                 almost as high, ask the LLM one focused multiple-choice
                                 question: "A, B, or either?". Small models are much
                                 better at choosing between two options than at
                                 noticing ambiguity on their own.
  Retry search (self-correction): if the first search finds nothing relevant, the LLM
                                 rewrites the message as a short English search query
                                 (fixing typos, translating German, Arabic, ...) and we
                                 search again. Only weak searches pay this extra step.
  Fallback:                      if the LLM fails or returns nonsense, simple
                                 score-based rules decide, so the assistant keeps working.
"""

from dataclasses import dataclass, field, replace
from enum import Enum

from app.answering import generate_answer
from app.llm import LLM, LLMError
from app.vector_store import SearchResult, VectorStore


class Action(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    DECLINE = "decline"


@dataclass(frozen=True)
class Option:
    """A choice offered in a clarifying question. It always maps to a real card."""

    card_id: str
    label: str


@dataclass(frozen=True)
class AgentResponse:
    action: Action
    message: str
    reason: str
    decided_by: str  # "llm", "guardrail" or "rules"
    source: str | None = None  # card id the answer comes from
    options: list[Option] = field(default_factory=list)
    matches: list[SearchResult] = field(default_factory=list)
    search_query: str | None = None  # set when the agent rewrote the message to search again

    @property
    def confidence(self) -> float:
        return self.matches[0].score if self.matches else 0.0


DEFAULT_DECLINE_MESSAGE = (
    "Sorry, I can only help with questions about your account, billing, "
    "integrations, and data and security."
)
DEFAULT_CLARIFY_MESSAGE = "I want to make sure I help with the right thing. Which of these do you mean?"

DECISION_SYSTEM_PROMPT = """You help a customer support assistant understand a customer message.
You get the message and the most similar support articles (cards). Fill in:

1. "reason": think first. What does the customer want? Which cards fit, and why?
2. "is_support_question": true if the message is about the customer's account, billing,
   payments, integrations, or data and security on our platform. This is true EVEN IF the
   cards do not mention the specific detail they ask about (for example a payment method,
   a tool or a country we may not support): then list the closest card.
   false only for things that have nothing to do with our platform
   (for example general coding help, weather, jokes, recommendations).
3. "matching_cards": list EVERY card that could be what the customer means.
   - One card if the message clearly asks for one specific thing.
   - Two or three cards if the message is vague and could mean different things.
   - An empty list if no card fits.
4. "clarifying_question": only if you listed two or more cards, a short question that asks
   which one the customer means. Otherwise an empty string.

Only use card ids from the list below. The score shows how similar a card is.

Examples (not from the list below):
- "How do I update my phone number?" with a card about contact details:
  is_support_question=true, matching_cards=[that card]
- "Something is wrong with my bill" with cards about refunds, cancelling and failed payments:
  is_support_question=true, matching_cards=[all three], clarifying_question="Is this about ...?"
- "Can you recommend a good laptop?": is_support_question=false, matching_cards=[]"""

# The model does the UNDERSTANDING; the code only maps it to an action:
#   not a support question -> decline, one card -> answer, 2-3 cards -> clarify.
# Why: when we asked the 3B model to pick the action itself, it often wrote a
# correct reason ("Q4 is about payment methods") and still chose "decline".
# "reason" stays first so the model thinks before filling in the other fields.
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "is_support_question": {"type": "boolean"},
        "matching_cards": {"type": "array", "items": {"type": "string"}},
        "clarifying_question": {"type": "string"},
    },
    "required": ["reason", "is_support_question", "matching_cards", "clarifying_question"],
}


# Note: this prompt must NOT list the help center topics. In evaluation, an earlier version
# said "a help center about accounts, billing, integrations and data", and the model copied
# those words into every query: random letters became "accounts billing integrations data",
# which matches the cards and made nonsense look relevant.
REWRITE_PROMPT = """Turn the customer's message into a short English search query.

1. "has_meaning": is this a real question or request? false for random letters, keyboard
   mashing, only symbols or punctuation, or an empty message.
2. "search_query": the customer's message in short, plain English.
   - Fix spelling mistakes. Translate it if it is not in English.
   - Use only the meaning of the customer's own words. Do not add topics or words they did not say.
   - Do not answer the question.
   If has_meaning is false, return an empty string."""

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "has_meaning": {"type": "boolean"},
        "search_query": {"type": "string"},
    },
    "required": ["has_meaning", "search_query"],
}

SECOND_CHECK_PROMPT = """A customer wrote a message to support. Two help articles could fit.
Decide if the message clearly asks for one of them, or if it could reasonably mean either.

Answer "A" or "B" only if the customer's words clearly point to that article.
Answer "either" if the customer describes a goal or a problem that both articles could solve,
so a support agent would need to ask which one they mean.

First explain your thinking in "reason". If you answer "either", also write a short
clarifying_question that asks which of the two the customer means."""

SECOND_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "choice": {"type": "string", "enum": ["A", "B", "either"]},
        "clarifying_question": {"type": "string"},
    },
    "required": ["reason", "choice", "clarifying_question"],
}


def format_candidates(matches: list[SearchResult]) -> str:
    lines = []
    for match in matches:
        entry = match.entry
        lines.append(
            f"[{entry.id}] score={match.score:.2f}\n"
            f"  Category: {entry.category}\n"
            f"  Question: {entry.question}\n"
            f"  Answer: {entry.answer}"
        )
    return "\n\n".join(lines)


def decision_messages(query: str, candidates: list[SearchResult]) -> list[dict[str, str]]:
    """The exact messages sent to the LLM for the decision step. Also used to turn
    feedback into training examples, so training and use see the same prompt."""
    return [
        {"role": "system", "content": DECISION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Customer message: {query}\n\nCards:\n{format_candidates(candidates)}"},
    ]


class SupportAgent:
    def __init__(
        self,
        store: VectorStore,
        llm: LLM,
        min_relevance: float = 0.55,
        ambiguity_gap: float = 0.05,
        top_k: int = 4,
        mode: str = "llm",
        generate_answers: bool = True,
        second_check_gap: float = 0.1,
        rewrite_weak_queries: bool = True,
    ):
        self.store = store
        self.llm = llm
        self.min_relevance = min_relevance
        self.ambiguity_gap = ambiguity_gap
        self.top_k = top_k
        self.mode = mode  # "llm" (LLM decides + guardrails) or "rules" (scores only)
        # False = return the card text without asking the LLM to rephrase it.
        # Used by the evaluation to measure decisions quickly.
        self.generate_answers = generate_answers
        # If another card scores within this gap of the chosen one, run the second check.
        # 0 turns the second check off (useful to measure its effect).
        self.second_check_gap = second_check_gap
        # Rewrite the message and search again when the first search finds nothing relevant.
        self.rewrite_weak_queries = rewrite_weak_queries

    # ---- public -----------------------------------------------------------

    def respond(
        self, message: str, previous_message: str | None = None, mode: str | None = None
    ) -> AgentResponse:
        """Handle one customer message.

        previous_message is used after a clarifying question: if the customer
        first wrote "I want to stop paying" and now answers "cancel", we search
        for both together.
        mode overrides the default ("llm" or "rules") for this one message,
        so the chat page can compare both on the same question.
        """
        mode = mode or self.mode
        query = f"{previous_message} {message}" if previous_message else message
        matches = self.store.search(query, top_k=self.top_k)

        # Self-correction: a weak search may be caused by typos or another language.
        search_query = None
        if mode == "llm" and self.rewrite_weak_queries and self._is_weak(matches):
            search_query = self._rewrite(query)
            if search_query:
                matches = self._merge(matches, self.store.search(search_query, top_k=self.top_k))
                query = f"{query}\n(In English: {search_query})"

        response = self._decide(query, matches, mode)
        return replace(response, search_query=search_query) if search_query else response

    def _is_weak(self, matches: list[SearchResult]) -> bool:
        return not matches or matches[0].score < self.min_relevance

    def _rewrite(self, query: str) -> str | None:
        try:
            result = self.llm.chat_json(
                [{"role": "system", "content": REWRITE_PROMPT}, {"role": "user", "content": query}],
                json_schema=REWRITE_SCHEMA,
            )
        except LLMError:
            return None
        if result.get("has_meaning") is not True:
            return None  # random letters or symbols: keep the weak search, so the guardrail declines
        rewritten = str(result.get("search_query", "")).strip()
        # Ignore empty or unchanged rewrites, and very long ones (the model started answering).
        if not rewritten or rewritten.lower() == query.strip().lower() or len(rewritten) > 200:
            return None
        return rewritten

    def _merge(self, first: list[SearchResult], second: list[SearchResult]) -> list[SearchResult]:
        """Keep each card's best score from both searches, best first."""
        best: dict[str, SearchResult] = {}
        for match in first + second:
            if match.entry.id not in best or match.score > best[match.entry.id].score:
                best[match.entry.id] = match
        return sorted(best.values(), key=lambda m: m.score, reverse=True)[: self.top_k]

    def _decide(self, query: str, matches: list[SearchResult], mode: str) -> AgentResponse:
        # Guardrail 1: nothing relevant -> decline without asking the LLM.
        if not matches or matches[0].score < self.min_relevance:
            return self._decline(
                matches, "guardrail", f"No card is relevant enough (best score below {self.min_relevance})."
            )

        # The out-of-scope card (Q10) is an example of what to decline, not an article.
        if matches[0].entry.out_of_scope:
            return self._decline(matches, "guardrail", "The message matches the out-of-scope example card.")
        # We search top_k=4, so after removing Q10 the model still sees 3 cards.
        candidates = [m for m in matches if not m.entry.out_of_scope][:3]

        if mode == "rules":
            return self._rules_decision(query, matches)

        try:
            decision = self.llm.chat_json(decision_messages(query, candidates), json_schema=DECISION_SCHEMA)
        except LLMError:
            return self._rules_decision(query, matches, note="LLM unavailable. ")

        return self._apply_llm_decision(query, matches, candidates, decision)

    # ---- decision handling --------------------------------------------------

    def _apply_llm_decision(
        self, query: str, matches: list[SearchResult], candidates: list[SearchResult], decision: dict
    ) -> AgentResponse:
        """Map the model's understanding to an action. Guardrail 2: ignore invalid output."""
        by_id = {m.entry.id: m.entry for m in candidates}
        reason = str(decision.get("reason", "")).strip() or "No reason given."
        is_support = decision.get("is_support_question")
        cards = decision.get("matching_cards")

        if not isinstance(is_support, bool) or not isinstance(cards, list):
            return self._rules_decision(query, matches, note="LLM decision was not valid. ")

        if not is_support:
            return self._decline(matches, "llm", reason)

        # Keep only real candidate ids, in order, without duplicates (max 3 options).
        valid = [by_id[cid] for cid in dict.fromkeys(str(c).strip() for c in cards) if cid in by_id][:3]

        if len(valid) == 1:
            return self._second_check(query, valid[0], matches, candidates, reason)
        if len(valid) >= 2:
            question = str(decision.get("clarifying_question", "")).strip() or DEFAULT_CLARIFY_MESSAGE
            return self._clarify(valid, matches, "llm", reason, question)

        # A support question, but the model listed no valid card (none fit, or unknown ids).
        return self._rules_decision(query, matches, note="LLM listed no valid card. ")

    def _second_check(
        self, query: str, chosen, matches: list[SearchResult], candidates: list[SearchResult], reason: str
    ) -> AgentResponse:
        """Reflection step: before answering, check that a close runner-up is not just as likely."""
        scores = {m.entry.id: m.score for m in candidates}
        rivals = [
            m.entry for m in candidates
            if m.entry.id != chosen.id and scores[chosen.id] - m.score < self.second_check_gap
            and m.score >= self.min_relevance
        ]
        if self.second_check_gap <= 0 or not rivals:
            return self._answer(query, chosen, matches, "llm", reason)

        # Present the two cards in score order (A = higher score), the same way every time.
        a, b = sorted([chosen, rivals[0]], key=lambda e: scores[e.id], reverse=True)
        try:
            check = self.llm.chat_json(
                [
                    {"role": "system", "content": SECOND_CHECK_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Customer message: {query}\n\n"
                            f"A: {a.question}\n   {a.answer}\n\n"
                            f"B: {b.question}\n   {b.answer}"
                        ),
                    },
                ],
                json_schema=SECOND_CHECK_SCHEMA,
            )
        except LLMError:
            return self._answer(query, chosen, matches, "llm", reason)

        choice = check.get("choice")
        check_reason = str(check.get("reason", "")).strip()
        combined = f"{reason} Second check: {check_reason}".strip()
        if choice == "either":
            question = str(check.get("clarifying_question", "")).strip() or DEFAULT_CLARIFY_MESSAGE
            return self._clarify([a, b], matches, "llm", combined, question)
        if choice in ("A", "B"):
            picked = a if choice == "A" else b
            return self._answer(query, picked, matches, "llm", combined)
        return self._answer(query, chosen, matches, "llm", reason)

    def _rules_decision(self, query: str, matches: list[SearchResult], note: str = "") -> AgentResponse:
        """Simple score-based rules. Used as a fallback, and as a baseline to compare against."""
        best = matches[0]
        if best.entry.out_of_scope:
            return self._decline(matches, "rules", note + "Best match is the out-of-scope card.")

        in_scope = [m for m in matches if not m.entry.out_of_scope]
        if len(in_scope) >= 2 and in_scope[0].score - in_scope[1].score < self.ambiguity_gap:
            return self._clarify(
                [in_scope[0].entry, in_scope[1].entry],
                matches,
                "rules",
                note + "Top two cards have almost the same score.",
                DEFAULT_CLARIFY_MESSAGE,
            )
        return self._answer(query, best.entry, matches, "rules", note + "One card clearly matches best.")

    # ---- building responses -------------------------------------------------

    def _answer(self, query, entry, matches, decided_by, reason) -> AgentResponse:
        if self.generate_answers:
            text, note = generate_answer(query, entry, self.llm)
            if note:
                reason = f"{reason} Answer check: {note}.".strip()
        else:
            text = entry.answer
        return AgentResponse(Action.ANSWER, text, reason, decided_by, source=entry.id, matches=matches)

    def _clarify(self, entries, matches, decided_by, reason, question) -> AgentResponse:
        options = [Option(card_id=e.id, label=e.question) for e in entries]
        return AgentResponse(Action.CLARIFY, question, reason, decided_by, options=options, matches=matches)

    def _decline(self, matches, decided_by, reason) -> AgentResponse:
        # If the message matches the knowledge base's own out-of-scope card (Q10),
        # use its text. Otherwise (e.g. "weather in Berlin") a short general message.
        if matches and matches[0].entry.out_of_scope and matches[0].score >= self.min_relevance:
            message = matches[0].entry.answer
        else:
            message = DEFAULT_DECLINE_MESSAGE
        return AgentResponse(Action.DECLINE, message, reason, decided_by, matches=matches)
