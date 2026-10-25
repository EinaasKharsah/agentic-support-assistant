"""Checks retrieval with the real embedding model.

Skipped by default because it downloads the model (~130 MB).
Run with:  RUN_MODEL_TESTS=1 pytest tests/test_real_retrieval.py
(PowerShell:  $env:RUN_MODEL_TESTS=1; python -m pytest tests/test_real_retrieval.py)
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to run"
)


@pytest.fixture(scope="module")
def store():
    from app.embeddings import FastEmbedEmbedder
    from app.knowledge_base import load_knowledge_base
    from app.vector_store import VectorStore, make_client

    store = VectorStore(make_client(None), FastEmbedEmbedder(), "real_kb")
    store.index(load_knowledge_base("data/knowledge_base.json"))
    return store


@pytest.mark.parametrize(
    "question, expected",
    [
        # Same meaning, different words than the stored question.
        ("I forgot my password", "Q1"),
        ("Do you take Bitcoin?", "Q4"),  # only the ANSWER mentions cryptocurrency
        ("Is my data encrypted?", "Q8"),  # only the ANSWER mentions encryption
        ("Can I connect it to Slack?", "Q7"),
        ("How long do you keep my data after I cancel?", "Q6"),
    ],
)
def test_semantic_search(store, question, expected):
    assert store.search(question)[0].entry.id == expected
