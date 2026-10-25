# Agentic Customer Support Assistant

An agentic RAG assistant that answers customer support questions from a small knowledge base. It decides whether to answer, ask a clarifying question, or decline.

> Work in progress. Full documentation will follow.

## Project structure

```
app/                     Application code
  config.py              Settings, read from environment variables
  knowledge_base.py      Loads and validates the Q&A entries
data/
  knowledge_base.json    The 10 Q&A entries
tests/                   Unit tests
```

## Run the tests

```bash
pip install -r requirements.txt
pytest
```
