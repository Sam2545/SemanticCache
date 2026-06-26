from app.metrics.base import MetricsSnapshot
from app.services.stats import StatsAssumptions, compute_stats


def test_compute_stats_basic_math():
    snap = MetricsSnapshot(hits=3, misses=1, sim_sum=2.7, lookup_ms_sum=8.0, search_ms_sum=4.0)
    res = compute_stats(entries=10, snapshot=snap, assumptions=StatsAssumptions())
    assert res.entries == 10
    assert res.queries == 4
    assert res.hits == 3
    assert res.misses == 1
    assert res.hit_rate == 0.75
    assert res.avg_similarity == 0.9          # 2.7 / 3 hits
    assert res.avg_lookup_latency_ms == 2.0   # 8.0 / 4 queries
    assert res.avg_store_search_ms == 1.0     # 4.0 / 4 queries
    assert res.estimated_latency_saved_ms == 2400.0   # 3 * 800
    assert res.estimated_cost_saved_usd == 0.003       # 3 * 0.001
    assert res.estimated_tokens_saved == 1500          # 3 * 500


def test_compute_stats_zero_queries_is_guarded():
    res = compute_stats(entries=0, snapshot=MetricsSnapshot(), assumptions=StatsAssumptions())
    assert res.queries == 0
    assert res.hit_rate == 0.0
    assert res.avg_similarity == 0.0
    assert res.avg_lookup_latency_ms == 0.0
    assert res.avg_store_search_ms == 0.0
    assert res.estimated_latency_saved_ms == 0.0


def test_compute_stats_misses_only_avg_similarity_zero():
    snap = MetricsSnapshot(hits=0, misses=2, sim_sum=0.0, lookup_ms_sum=4.0, search_ms_sum=2.0)
    res = compute_stats(entries=5, snapshot=snap, assumptions=StatsAssumptions())
    assert res.avg_similarity == 0.0           # no hits -> guarded
    assert res.avg_lookup_latency_ms == 2.0    # 4.0 / 2 queries
    assert res.hit_rate == 0.0


def test_compute_stats_uses_custom_assumptions():
    snap = MetricsSnapshot(hits=2, misses=0)
    res = compute_stats(
        entries=2, snapshot=snap,
        assumptions=StatsAssumptions(llm_ms=1000.0, cost_usd=0.01, tokens_per_call=100),
    )
    assert res.estimated_latency_saved_ms == 2000.0
    assert res.estimated_cost_saved_usd == 0.02
    assert res.estimated_tokens_saved == 200
