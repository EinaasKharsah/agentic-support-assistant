"""The agent: decides whether to ANSWER, ask a CLARIFYING question, or DECLINE.

The LLM makes the decision, because it understands meaning. For example,
"I want to stop paying" could mean cancel OR refund; the similarity scores
alone did not show this (gap of only 0.06). Code adds guardrails around it:

  Guardrail 1 (before the LLM): if no card is relevant at all, decline.
                                 No need to ask a small model about the weather.
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

DECISION_SYSTEM_PROMPT = """You are the decision step of a customer support assistant.
You get a customer message and the most similar support articles (cards).

First, in "reason", say which card or cards match the message and why.
Then choose exactly ONE action, based on your reason:

- "answer": one card answers the message. Set card_id to that card. This is the most common case.
- "clarify": the message is vague and could mean 2 or 3 different cards.
  Put those card ids in options and write a short clarifying_question.
- "decline": ONLY if the message is not about account, billing, integrations, or data and
  security, or if it asks for what the card marked OUT OF SCOPE describes.

Rules:
- Only use card ids from the list below.
- A support question that matches a normal card must never be declined.
- The score shows how similar a card is (higher is more similar).

Examples (not from the list below):
- "How do I update my phone number?" + a card about contact details -> answer
- "Something is wrong with my bill" + cards about refunds, cancelling and failed payments -> clarify
- "Can you recommend a good laptop?" -> decline"""

# "reason" is FIRST on purpose. The model writes the JSON from left to right,
# so it must think before it decides. When "action" came first, the model
# wrote correct reasons ("this is about a password reset") but had already
# chosen "decline".
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "action": {"type": "string", "enum": ["answer", "clarify", "decline"]},
        "card_id": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
        "clarifying_question": {"type": "string"},
    },
    "required": ["reason", "action", "card_id", "options", "clarifying_question"],
}


def format_candidates(matches: list[SearchResult]) -> str:
    lines = []
    for match in matches:
        entry = match.entry
        flag = "  [OUT OF SCOPE]" if entry.out_of_scope else ""
        lines.append(
            f"[{entry.id}] score={match.score:.2f}{flag}\n"
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
    ):
        self.store = store
        self.llm = llm
        self.min_relevance = min_relevance
        self.ambiguity_gap = ambiguity_gap
        self.top_k = top_k
        self.mode = mode  # "llm" (LLM decides + guardrails) or "rules" (scores only)

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

        if self.mode == "rules":
            return self._rules_decision(query, matches)

        try:
            decision = self.llm.chat_json(
                [
                    {"role": "system", "content": DECISION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Customer message: {query}\n\nCards:\n{format_candidates(matches)}",
                    },
                ],
                json_schema=DECISION_SCHEMA,
            )
        except LLMError:
            return self._rules_decision(query, matches, note="LLM unavailable. ")

        return self._apply_llm_decision(query, matches, decision)

    # ---- decision handling --------------------------------------------------

    def _apply_llm_decision(self, query: str, matches: list[SearchResult], decision: dict) -> AgentResponse:
        """Guardrail 2: only accept decisions that make sense."""
        by_id = {m.entry.id: m.entry for m in matches}
        action = decision.get("action")
        reason = str(decision.get("reason", "")).strip() or "No reason given."
        card_id = str(decision.get("card_id", "")).strip()

        if action == Action.DECLINE.value:
            return self._decline(matches, "llm", reason)

        if action == Action.ANSWER.value and card_id in by_id:
            entry = by_id[card_id]
            if entry.out_of_scope:
                return self._decline(matches, "guardrail", "The best match is the out-of-scope card.")
            return self._answer(query, entry, matches, "llm", reason)

        if action == Action.CLARIFY.value:
            option_ids = [cid for cid in decision.get("options", []) if cid in by_id]
            entries = [by_id[cid] for cid in dict.fromkeys(option_ids) if not by_id[cid].out_of_scope]
            if len(entries) >= 2:
                question = str(decision.get("clarifying_question", "")).strip() or DEFAULT_CLARIFY_MESSAGE
                return self._clarify(entries, matches, "llm", reason, question)

        # The LLM returned something we cannot use (unknown card id, one option, ...).
        return self._rules_decision(query, matches, note="LLM decision was not valid. ")

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
        text, _used_llm = generate_answer(query, entry, self.llm)
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
