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
  Fallback:                      if the LLM fails or returns nonsense, simple
                                 score-based rules decide, so the assistant keeps working.
"""

from dataclasses import dataclass, field
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
   integrations, or data and security on our platform. false for anything else
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


class SupportAgent:
    def __init__(
        self,
        store: VectorStore,
        llm: LLM,
        min_relevance: float = 0.55,
        ambiguity_gap: float = 0.05,
        top_k: int = 3,
        mode: str = "llm",
        generate_answers: bool = True,
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

    # ---- public -----------------------------------------------------------

    def respond(self, message: str, previous_message: str | None = None) -> AgentResponse:
        """Handle one customer message.

        previous_message is used after a clarifying question: if the customer
        first wrote "I want to stop paying" and now answers "cancel", we search
        for both together.
        """
        query = f"{previous_message} {message}" if previous_message else message
        matches = self.store.search(query, top_k=self.top_k)

        # Guardrail 1: nothing relevant -> decline without asking the LLM.
        if not matches or matches[0].score < self.min_relevance:
            return self._decline(
                matches, "guardrail", f"No card is relevant enough (best score below {self.min_relevance})."
            )

        # The out-of-scope card (Q10) is an example of what to decline, not an article.
        if matches[0].entry.out_of_scope:
            return self._decline(matches, "guardrail", "The message matches the out-of-scope example card.")
        candidates = [m for m in matches if not m.entry.out_of_scope]

        if self.mode == "rules":
            return self._rules_decision(query, matches)

        try:
            decision = self.llm.chat_json(
                [
                    {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Customer message: {query}\n\nCards:\n{format_candidates(candidates)}",
                    },
                ],
                json_schema=DECISION_SCHEMA,
            )
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
            return self._answer(query, valid[0], matches, "llm", reason)
        if len(valid) >= 2:
            question = str(decision.get("clarifying_question", "")).strip() or DEFAULT_CLARIFY_MESSAGE
            return self._clarify(valid, matches, "llm", reason, question)

        # A support question, but the model listed no valid card (none fit, or unknown ids).
        return self._rules_decision(query, matches, note="LLM listed no valid card. ")

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
            text, _used_llm = generate_answer(query, entry, self.llm)
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
