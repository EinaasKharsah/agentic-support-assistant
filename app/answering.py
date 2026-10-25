"""Write the final answer for the customer, using ONLY one knowledge base card."""

from app.knowledge_base import Entry
from app.llm import LLM, LLMError

ANSWER_SYSTEM_PROMPT = """You are a friendly customer support assistant.
Answer the customer's question using ONLY the support article below.

Rules:
- Use only facts from the article. Never add steps, numbers, emails or features that are not in it.
- If the article does not answer the exact question, say so honestly, share what the article does say, and suggest contacting support.
- Keep it short: 2 to 4 sentences. Keep menu paths like "Settings > Billing" exactly as written.

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
