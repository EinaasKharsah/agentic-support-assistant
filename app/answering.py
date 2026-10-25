"""Write the final answer for the customer, using ONLY one knowledge base card."""

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
- Keep it short: 2 to 4 sentences. Keep menu paths like "Settings > Billing" exactly as written.

Example (a different, made-up article, only to show the style):
Article: "We ship to Germany and France."
Customer: "Do you ship to Spain?"
Good answer: "Spain isn't mentioned in our help article. We currently ship to Germany and France.
Please contact support to confirm."
Bad answer: "No, we don't ship to Spain."  (the article never says that)

Support article ({entry_id}, {category}):
Question: {question}
Answer: {answer}"""


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


def generate_answer(question: str, entry: Entry, llm: LLM) -> tuple[str, bool]:
    """Return (answer_text, used_llm).

    If Ollama fails, we fall back to the card text itself. The answer is less
    friendly but still correct, and the assistant keeps working.
    """
    try:
        text = llm.chat(build_messages(question, entry)).strip()
        if text:
            return text, True
    except LLMError:
        pass
    return entry.answer, False
