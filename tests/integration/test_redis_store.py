"""Contract tests for the Redis-backed VectorStore.

Mirrors the in-memory store's contract (so both implementations behave
identically behind the protocol) and adds Redis-specific concerns: that the
returned score is cosine *similarity* (not RediSearch's distance) and that TTL
actually expires entries.

Run with: pytest -m integration   (requires a live redis-stack)
"""

import math
import time

import pytest

from app.vectorstore.base import Namespace, StoredEntry

pytestmark = pytest.mark.integration


def test_create_and_get_namespace(redis_store):
    redis_store.create_namespace(Namespace(name="ns", dimension=2))
    ns = redis_store.get_namespace("ns")
    assert ns is not None
    assert ns.dimension == 2


def test_get_missing_namespace_returns_none(redis_store):
    assert redis_store.get_namespace("nope") is None


def test_upsert_then_get_by_key(redis_store):
    redis_store.create_namespace(Namespace(name="ns", dimension=2))
    redis_store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A"))
    entry = redis_store.get("ns", "a")
    assert entry is not None
    assert entry.value == "A"
    assert entry.embedding == [1.0, 0.0]


def test_upsert_same_key_overwrites(redis_store):
    redis_store.create_namespace(Namespace(name="ns", dimension=2))
    redis_store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="old"))
    redis_store.upsert("ns", StoredEntry(key="a", embedding=[0.0, 1.0], value="new"))
    entry = redis_store.get("ns", "a")
    assert entry.value == "new"


def test_value_and_metadata_survive_round_trip(redis_store):
    redis_store.create_namespace(Namespace(name="ns", dimension=2))
    redis_store.upsert(
        "ns",
        StoredEntry(
            key="a",
            embedding=[1.0, 0.0],
            value={"text": "hello", "n": 3},
            metadata={"source": "unit"},
        ),
    )
    entry = redis_store.get("ns", "a")
    assert entry.value == {"text": "hello", "n": 3}
    assert entry.metadata == {"source": "unit"}


def test_search_returns_cosine_similarity_score(redis_store):
    redis_store.create_namespace(Namespace(name="ns", dimension=2))
    redis_store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A"))
    results = redis_store.search("ns", [1.0, 0.0], top_k=1)
    # Identical vectors -> cosine similarity 1.0 (NOT RediSearch distance 0.0).
    assert math.isclose(results[0].score, 1.0, abs_tol=1e-5)


def test_search_orders_by_similarity_descending(redis_store):
    redis_store.create_namespace(Namespace(name="ns", dimension=2))
    redis_store.upsert("ns", StoredEntry(key="near", embedding=[1.0, 0.0], value="N"))
    redis_store.upsert("ns", StoredEntry(key="mid", embedding=[1.0, 1.0], value="M"))
    redis_store.upsert("ns", StoredEntry(key="far", embedding=[0.0, 1.0], value="F"))

    results = redis_store.search("ns", [1.0, 0.0], top_k=3)

    assert [r.key for r in results] == ["near", "mid", "far"]


def test_search_respects_top_k(redis_store):
    redis_store.create_namespace(Namespace(name="ns", dimension=2))
    for i in range(5):
        redis_store.upsert(
            "ns", StoredEntry(key=f"k{i}", embedding=[1.0, i / 10], value=i)
        )
    assert len(redis_store.search("ns", [1.0, 0.0], top_k=2)) == 2


def test_namespace_isolation(redis_store):
    redis_store.create_namespace(Namespace(name="ns", dimension=2))
    redis_store.create_namespace(Namespace(name="other", dimension=2))
    redis_store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A"))
    redis_store.upsert("other", StoredEntry(key="b", embedding=[1.0, 0.0], value="B"))

    results = redis_store.search("other", [1.0, 0.0], top_k=10)

    assert [r.key for r in results] == ["b"]


def test_delete_removes_entry(redis_store):
    redis_store.create_namespace(Namespace(name="ns", dimension=2))
    redis_store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A"))
    assert redis_store.delete("ns", "a") is True
    assert redis_store.get("ns", "a") is None
    assert redis_store.delete("ns", "a") is False


def test_ttl_expires_entries(redis_store):
    redis_store.create_namespace(Namespace(name="ns", dimension=2, ttl=1))
    redis_store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A"))
    assert redis_store.get("ns", "a") is not None
    time.sleep(1.5)
    assert redis_store.get("ns", "a") is None


def test_namespace_round_trips_filter_keys(redis_store):
    redis_store.create_namespace(
        Namespace(name="ns", dimension=2, filter_keys=["model", "embed_model"])
    )
    assert redis_store.get_namespace("ns").filter_keys == ["model", "embed_model"]


def test_search_filters_by_metadata(redis_store):
    redis_store.create_namespace(Namespace(name="ns", dimension=2, filter_keys=["model"]))
    redis_store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A", metadata={"model": "gpt-oss:120b"}))
    redis_store.upsert("ns", StoredEntry(key="b", embedding=[1.0, 0.0], value="B", metadata={"model": "minimax-m3"}))
    results = redis_store.search("ns", [1.0, 0.0], top_k=10, filter={"model": "gpt-oss:120b"})
    assert [r.key for r in results] == ["a"]


def test_filter_prefilters_before_knn(redis_store):
    # The wanted-model entry is ranked BELOW several other-model entries; a
    # pre-filter must still surface it (a post-filter would drop it).
    redis_store.create_namespace(Namespace(name="ns", dimension=2, filter_keys=["model"]))
    redis_store.upsert("ns", StoredEntry(key="b1", embedding=[1.0, 0.00], value=1, metadata={"model": "B"}))
    redis_store.upsert("ns", StoredEntry(key="b2", embedding=[1.0, 0.01], value=2, metadata={"model": "B"}))
    redis_store.upsert("ns", StoredEntry(key="b3", embedding=[1.0, 0.02], value=3, metadata={"model": "B"}))
    redis_store.upsert("ns", StoredEntry(key="a1", embedding=[0.8, 0.6], value=4, metadata={"model": "A"}))
    results = redis_store.search("ns", [1.0, 0.0], top_k=1, filter={"model": "A"})
    assert [r.key for r in results] == ["a1"]


def test_filter_is_case_sensitive(redis_store):
    redis_store.create_namespace(Namespace(name="ns", dimension=2, filter_keys=["model"]))
    redis_store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A", metadata={"model": "GPT-4"}))
    # A lowercase filter must NOT match an upper-case stored value (parity with in-memory).
    assert redis_store.search("ns", [1.0, 0.0], top_k=10, filter={"model": "gpt-4"}) == []
    # Exact-case filter matches.
    results = redis_store.search("ns", [1.0, 0.0], top_k=10, filter={"model": "GPT-4"})
    assert [r.key for r in results] == ["a"]
