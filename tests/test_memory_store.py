import pytest

from app.vectorstore.base import Namespace, StoredEntry
from app.vectorstore.memory_store import InMemoryVectorStore


@pytest.fixture
def store():
    s = InMemoryVectorStore()
    s.create_namespace(Namespace(name="ns", dimension=2))
    return s


def test_create_and_get_namespace(store):
    ns = store.get_namespace("ns")
    assert ns is not None
    assert ns.dimension == 2


def test_get_missing_namespace_returns_none(store):
    assert store.get_namespace("nope") is None


def test_upsert_then_get_by_key(store):
    store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A"))
    entry = store.get("ns", "a")
    assert entry is not None
    assert entry.value == "A"


def test_upsert_same_key_overwrites(store):
    store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="old"))
    store.upsert("ns", StoredEntry(key="a", embedding=[0.0, 1.0], value="new"))
    entry = store.get("ns", "a")
    assert entry.value == "new"
    assert entry.embedding == [0.0, 1.0]


def test_search_orders_by_similarity_descending(store):
    store.upsert("ns", StoredEntry(key="near", embedding=[1.0, 0.0], value="N"))
    store.upsert("ns", StoredEntry(key="mid", embedding=[1.0, 1.0], value="M"))
    store.upsert("ns", StoredEntry(key="far", embedding=[0.0, 1.0], value="F"))

    results = store.search("ns", [1.0, 0.0], top_k=3)

    assert [r.key for r in results] == ["near", "mid", "far"]
    assert results[0].score == 1.0


def test_search_respects_top_k(store):
    for i in range(5):
        store.upsert("ns", StoredEntry(key=f"k{i}", embedding=[1.0, i / 10], value=i))
    assert len(store.search("ns", [1.0, 0.0], top_k=2)) == 2


def test_namespace_isolation(store):
    store.create_namespace(Namespace(name="other", dimension=2))
    store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A"))
    store.upsert("other", StoredEntry(key="b", embedding=[1.0, 0.0], value="B"))

    results = store.search("other", [1.0, 0.0], top_k=10)

    assert [r.key for r in results] == ["b"]


def test_delete_removes_entry(store):
    store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A"))
    assert store.delete("ns", "a") is True
    assert store.get("ns", "a") is None
    assert store.delete("ns", "a") is False


def test_namespace_round_trips_filter_keys():
    s = InMemoryVectorStore()
    s.create_namespace(Namespace(name="ns", dimension=2, filter_keys=["model"]))
    assert s.get_namespace("ns").filter_keys == ["model"]


def test_namespace_filter_keys_defaults_empty():
    s = InMemoryVectorStore()
    s.create_namespace(Namespace(name="ns", dimension=2))
    assert s.get_namespace("ns").filter_keys == []


def test_search_filters_by_metadata(store):
    store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A", metadata={"model": "x"}))
    store.upsert("ns", StoredEntry(key="b", embedding=[1.0, 0.0], value="B", metadata={"model": "y"}))
    results = store.search("ns", [1.0, 0.0], top_k=10, filter={"model": "x"})
    assert [r.key for r in results] == ["a"]


def test_search_without_filter_returns_all(store):
    store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A", metadata={"model": "x"}))
    store.upsert("ns", StoredEntry(key="b", embedding=[1.0, 0.0], value="B", metadata={"model": "y"}))
    assert len(store.search("ns", [1.0, 0.0], top_k=10)) == 2


def test_search_filter_is_conjunctive(store):
    store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A", metadata={"model": "x", "lang": "en"}))
    store.upsert("ns", StoredEntry(key="b", embedding=[1.0, 0.0], value="B", metadata={"model": "x", "lang": "fr"}))
    results = store.search("ns", [1.0, 0.0], top_k=10, filter={"model": "x", "lang": "en"})
    assert [r.key for r in results] == ["a"]


def test_search_filter_excludes_entry_missing_key(store):
    store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A", metadata={}))
    assert store.search("ns", [1.0, 0.0], top_k=10, filter={"model": "x"}) == []


def test_count_reflects_upserts_and_deletes(store):
    assert store.count("ns") == 0
    store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A"))
    store.upsert("ns", StoredEntry(key="b", embedding=[0.0, 1.0], value="B"))
    assert store.count("ns") == 2
    store.delete("ns", "a")
    assert store.count("ns") == 1
