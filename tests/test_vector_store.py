import pytest

from app.knowledge_base import load_knowledge_base
from app.vector_store import VectorStore, make_client
from tests.fakes import FakeEmbedder


@pytest.fixture
def store():
    entries = load_knowledge_base("data/knowledge_base.json")
    store = VectorStore(make_client(None), FakeEmbedder(), "test_kb")
    store.index(entries)
    return store


def test_index_stores_all_entries(store):
    assert store.client.count("test_kb").count == 10


def test_indexing_twice_does_not_duplicate(store):
    store.index(load_knowledge_base("data/knowledge_base.json"))
    assert store.client.count("test_kb").count == 10


def test_search_finds_the_right_entry(store):
    results = store.search("How do I reset my password?")
    assert results[0].entry.id == "Q1"


def test_results_are_sorted_best_first(store):
    results = store.search("refund for my purchase", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert len(results) == 3


def test_result_keeps_out_of_scope_flag(store):
    results = store.search("Can you help me write code for my integration project?")
    assert results[0].entry.id == "Q10"
    assert results[0].entry.out_of_scope is True
