import pytest

from app.services.cache import (
    CacheService,
    DimensionMismatch,
    InvalidFilter,
    NamespaceExists,
    NamespaceNotFound,
)
from app.vectorstore.memory_store import InMemoryVectorStore


@pytest.fixture
def service():
    return CacheService(InMemoryVectorStore())


@pytest.fixture
def svc_ns(service):
    # default_threshold chosen so boundary tests below are exact.
    service.create_namespace("ns", dimension=2, default_threshold=0.6, default_top_k=5)
    return service


def test_create_namespace_round_trip(service):
    ns = service.create_namespace("ns", dimension=3)
    assert ns.dimension == 3
    assert service.get_namespace("ns").dimension == 3


def test_create_duplicate_namespace_raises(svc_ns):
    with pytest.raises(NamespaceExists):
        svc_ns.create_namespace("ns", dimension=2)


def test_put_rejects_dimension_mismatch(svc_ns):
    with pytest.raises(DimensionMismatch):
        svc_ns.put("ns", key="a", embedding=[1.0, 0.0, 0.0], value="A")


def test_query_rejects_dimension_mismatch(svc_ns):
    with pytest.raises(DimensionMismatch):
        svc_ns.query("ns", embedding=[1.0])


def test_put_on_missing_namespace_raises(service):
    with pytest.raises(NamespaceNotFound):
        service.put("nope", key="a", embedding=[1.0, 0.0], value="A")


def test_query_returns_match_above_threshold(svc_ns):
    svc_ns.put("ns", key="a", embedding=[1.0, 0.0], value="A")
    matches = svc_ns.query("ns", embedding=[1.0, 0.0], threshold=0.5)
    assert [m.key for m in matches] == ["a"]
    assert matches[0].score == 1.0


def test_query_excludes_below_threshold(svc_ns):
    # [0.6, 0.8] is a unit vector; cosine with [1,0] == 0.6 exactly.
    svc_ns.put("ns", key="a", embedding=[0.6, 0.8], value="A")
    assert svc_ns.query("ns", embedding=[1.0, 0.0], threshold=0.61) == []


def test_threshold_boundary_is_inclusive(svc_ns):
    svc_ns.put("ns", key="a", embedding=[0.6, 0.8], value="A")
    matches = svc_ns.query("ns", embedding=[1.0, 0.0], threshold=0.6)
    assert [m.key for m in matches] == ["a"]


def test_query_uses_namespace_default_threshold(svc_ns):
    # No threshold passed -> namespace default (0.6) applies.
    svc_ns.put("ns", key="hit", embedding=[1.0, 0.0], value="H")
    svc_ns.put("ns", key="miss", embedding=[0.0, 1.0], value="M")
    matches = svc_ns.query("ns", embedding=[1.0, 0.0])
    assert [m.key for m in matches] == ["hit"]


def test_query_respects_top_k(svc_ns):
    svc_ns.put("ns", key="a", embedding=[1.0, 0.0], value="A")
    svc_ns.put("ns", key="b", embedding=[0.99, 0.01], value="B")
    svc_ns.put("ns", key="c", embedding=[0.98, 0.02], value="C")
    matches = svc_ns.query("ns", embedding=[1.0, 0.0], threshold=0.0, top_k=2)
    assert len(matches) == 2


def test_get_and_delete(svc_ns):
    svc_ns.put("ns", key="a", embedding=[1.0, 0.0], value="A")
    assert svc_ns.get("ns", "a").value == "A"
    assert svc_ns.delete("ns", "a") is True
    assert svc_ns.get("ns", "a") is None


def test_create_namespace_stores_filter_keys(service):
    ns = service.create_namespace("ns", dimension=2, filter_keys=["model"])
    assert ns.filter_keys == ["model"]
    assert service.get_namespace("ns").filter_keys == ["model"]


def test_query_filter_returns_only_matching_model(service):
    service.create_namespace("ns", dimension=2, filter_keys=["model"])
    service.put("ns", key="a", embedding=[1.0, 0.0], value="A", metadata={"model": "x"})
    service.put("ns", key="b", embedding=[1.0, 0.0], value="B", metadata={"model": "y"})
    matches = service.query("ns", embedding=[1.0, 0.0], threshold=0.0, filter={"model": "x"})
    assert [m.key for m in matches] == ["a"]


def test_query_without_filter_returns_all_models(service):
    service.create_namespace("ns", dimension=2, filter_keys=["model"])
    service.put("ns", key="a", embedding=[1.0, 0.0], value="A", metadata={"model": "x"})
    service.put("ns", key="b", embedding=[1.0, 0.0], value="B", metadata={"model": "y"})
    assert len(service.query("ns", embedding=[1.0, 0.0], threshold=0.0)) == 2


def test_query_undeclared_filter_key_raises(service):
    service.create_namespace("ns", dimension=2, filter_keys=["model"])
    with pytest.raises(InvalidFilter):
        service.query("ns", embedding=[1.0, 0.0], filter={"temperature": "0"})


def test_query_records_hit_in_stats(service):
    service.create_namespace("ns", dimension=2)
    service.put("ns", key="a", embedding=[1.0, 0.0], value="A")
    service.query("ns", embedding=[1.0, 0.0], threshold=0.0)
    s = service.stats("ns")
    assert s.hits == 1 and s.misses == 0 and s.queries == 1
    assert s.entries == 1
    assert s.avg_lookup_latency_ms >= 0.0


def test_query_records_miss_in_stats(service):
    service.create_namespace("ns", dimension=2)
    service.put("ns", key="a", embedding=[1.0, 0.0], value="A")
    service.query("ns", embedding=[0.0, 1.0], threshold=0.99)
    s = service.stats("ns")
    assert s.misses == 1 and s.hits == 0


def test_effective_threshold_default_and_override(service):
    service.create_namespace("ns", dimension=2, default_threshold=0.7)
    assert service.effective_threshold("ns", None) == 0.7
    assert service.effective_threshold("ns", 0.9) == 0.9


def test_stats_estimates_savings_with_default_assumptions(service):
    service.create_namespace("ns", dimension=2)
    service.put("ns", key="a", embedding=[1.0, 0.0], value="A")
    service.query("ns", embedding=[1.0, 0.0], threshold=0.0)
    s = service.stats("ns")
    assert s.estimated_latency_saved_ms == 800.0
    assert s.estimated_tokens_saved == 500


def test_stats_missing_namespace_raises(service):
    with pytest.raises(NamespaceNotFound):
        service.stats("nope")
