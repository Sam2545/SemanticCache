from app.metrics.memory_store import InMemoryMetricsStore


def test_snapshot_of_unseen_namespace_is_zero():
    m = InMemoryMetricsStore()
    snap = m.snapshot("ns")
    assert (snap.hits, snap.misses, snap.sim_sum) == (0, 0, 0.0)
    assert (snap.lookup_ms_sum, snap.search_ms_sum) == (0.0, 0.0)


def test_record_hit_accumulates_score_and_latency():
    m = InMemoryMetricsStore()
    m.record_query("ns", hit=True, score=0.9, lookup_ms=3.0, search_ms=1.0)
    snap = m.snapshot("ns")
    assert snap.hits == 1
    assert snap.misses == 0
    assert snap.sim_sum == 0.9
    assert snap.lookup_ms_sum == 3.0
    assert snap.search_ms_sum == 1.0


def test_record_miss_does_not_touch_sim_sum():
    m = InMemoryMetricsStore()
    m.record_query("ns", hit=False, score=None, lookup_ms=2.0, search_ms=0.5)
    snap = m.snapshot("ns")
    assert snap.misses == 1
    assert snap.hits == 0
    assert snap.sim_sum == 0.0
    assert snap.lookup_ms_sum == 2.0


def test_namespaces_are_isolated():
    m = InMemoryMetricsStore()
    m.record_query("a", hit=True, score=0.5, lookup_ms=1.0, search_ms=1.0)
    m.record_query("b", hit=False, score=None, lookup_ms=1.0, search_ms=1.0)
    assert m.snapshot("a").hits == 1
    assert m.snapshot("b").hits == 0
