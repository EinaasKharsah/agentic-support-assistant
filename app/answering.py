"""Write the final answer for the customer, using ONLY one knowledge base card.

After the LLM writes the answer, code checks its key facts against the card:
  - no invented facts: every number, e-mail address and menu path in the answer
    must appear in the card;
  - no dropped limits: every number in the card (time limits, amounts, versions)
    must appear in the answer. Numbers carry the conditions customers rely on,
    for example "non-refundable after 48 hours".
If the check fails, the card text is shown as written: less friendly, but correct.
"""

import re

from app.knowledge_base import Entry
from app.llm import LLM, LLMError

ANSWER_SYSTEM_PROMPT = """You are a friendly customer support assistant.
Answer the customer's question using ONLY the support article below.

Rules:
- Use only facts from the article. Never add steps, numbers, emails or features that are not in it.
- If the customer asks about something the article does NOT mention (for example a specific
  payment method, tool or country), do not say yes or no. Say it is not mentioned in our help
  article, share what the article does say, and suggest contacting support to confirm.
- Be helpful: when the answer is "no", also say what IS possible according to the article.
- Keep it short, but keep every condition, time limit and exception from the article
  (for example "only within 14 days"). Keep menu paths like "Settings > Billing" exactly as written.

Example (a different, made-up article, only to show the style):
Article: "We ship to Germany and France."
Customer: "Do you ship to Spain?"
Good answer: "Spain isn't mentioned in our help article. We currently ship to Germany and France.
Please contact support to confirm."
Bad answer: "No, we don't ship to Spain."  (the article never says that)

Support article ({entry_id}, {category}):
Question: {question}
Answer: {answer}"""


EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
NUMBER = re.compile(r"\d+(?:\.\d+)?")
MENU_PATH = re.compile(r"[A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*(?: > [A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*)+")


def key_facts(text: str) -> tuple[set[str], set[str]]:
    """Return (numbers, other facts) found in a text. E-mails are removed before
    looking for numbers, so a digit inside an address is not counted twice."""
    emails = {e.lower().rstrip(".") for e in EMAIL.findall(text)}
    without_emails = EMAIL.sub(" ", text)
    numbers = set(NUMBER.findall(without_emails))
    paths = set(MENU_PATH.findall(without_emails))
    return numbers, emails | paths


def fact_problem(answer: str, entry: Entry) -> str | None:
    """Compare the answer's key facts with the card. Return what is wrong, or None."""
    answer_numbers, answer_other = key_facts(answer)
    card_numbers, card_other = key_facts(entry.answer)
    invented = (answer_numbers - card_numbers) | (answer_other - card_other)
    if invented:
        return f"the written answer added facts that are not in the article: {', '.join(sorted(invented))}"
    dropped = card_numbers - answer_numbers
    if dropped:
        return f"the written answer left out limits from the article: {', '.join(sorted(dropped))}"
    return None


def build_messages(question: str, entry: Entry) -> list[dict[str, str]]:
    system = ANSWER_SYSTEM_PROMPT.format(
        entry_id=entry.id,
        category=entry.category,
        question=entry.question,
        answer=entry.answer,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def generate_answer(question: str, entry: Entry, llm: LLM) -> tuple[str, str | None]:
    """Return (answer_text, note).

    note is None when the LLM's answer is used. Otherwise it says why the card text
    is shown instead: Ollama failed, the answer was empty, or the fact check failed.
    The card text is less friendly but always correct, so the assistant keeps working.
    """
    try:
        text = llm.chat(build_messages(question, entry)).strip()
    except LLMError:
        return entry.answer, "the LLM was not available, so the article text is shown"
    if not text:
        return entry.answer, "the LLM returned an empty answer, so the article text is shown"
    problem = fact_problem(text, entry)
    if problem:
        return entry.answer, f"{problem}, so the article text is shown"
    return text, None
