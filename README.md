# Agentic Customer Support Assistant

An agentic RAG assistant that answers customer support questions from a small knowledge base. It decides whether to answer, ask a clarifying question, or decline.

> Work in progress. Full documentation will follow.

## Project structure

```
app/                     Application code
  config.py              Settings, read from environment variables
  knowledge_base.py      Loads and validates the Q&A entries
  embeddings.py          Turns text into vectors (bge-small via fastembed)
  vector_store.py        Stores vectors in Qdrant and searches them
  llm.py                 Small client for the Ollama chat API
  answering.py           Writes the answer from one card (falls back to the card text)
  agent.py               Decides: answer, ask a clarifying question, or decline
  main.py                REST API (FastAPI): /chat, /feedback, /health
  static/                Chat page (plain HTML, CSS and JavaScript; fonts and icons bundled)
data/
  knowledge_base.json    The 10 Q&A entries
evaluation/
  eval_set.json          41 test questions with the expected decision (dev/test split)
  run_eval.py            Runs the agent on them and reports accuracy and error types
  metrics.py             Accuracy, error types, confusion table, retrieval recall
scripts/
  try_search.py          Try a question and see the similarity scores
  try_answer.py          Search + let the LLM write the answer
  try_agent.py           Ask the full agent and see its decision
tests/                   Unit tests
```

## Run the tests

```bash
pip install -r requirements.txt
pytest
```

Tests with the real embedding model are skipped by default (they download ~130 MB):

```bash
RUN_MODEL_TESTS=1 pytest tests/test_real_retrieval.py
```

## Try the search

```bash
python scripts/try_search.py "Do you take Bitcoin?"
```

## Try an answer (needs Ollama)

```bash
ollama pull qwen2.5:3b
python scripts/try_answer.py "Do you take Bitcoin?"
```

## Try the agent (needs Ollama)

```bash
python scripts/try_agent.py "I want to stop paying"
python scripts/try_agent.py "cancel my subscription" --previous "I want to stop paying"
python scripts/try_agent.py "I want to stop paying" --mode rules   # score-only baseline
```

## Evaluate

```bash
python evaluation/run_eval.py --scores-only --split dev   # similarity scores, no LLM
python evaluation/run_eval.py --mode llm                  # LLM decides + guardrails
python evaluation/run_eval.py --mode rules                # score-only baseline
```

## Run the API (needs Ollama)

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000 for the chat page, http://localhost:8000/docs to try the API, or:

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"I forgot my password\"}"
```
