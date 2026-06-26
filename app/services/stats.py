from __future__ import annotations

from dataclasses import dataclass

from app.metrics.base import MetricsSnapshot


@dataclass
class StatsAssumptions:
    """Configured inputs for estimated-savings metrics."""

    llm_ms: float = 800.0
    cost_usd: float = 0.001
    tokens_per_call: int = 500


@dataclass
class StatsResult:
    entries: int
    queries: int
    hits: int
    misses: int
    hit_rate: float
    avg_similarity: float
    avg_lookup_latency_ms: float
    avg_store_search_ms: float
    estimated_latency_saved_ms: float
    estimated_cost_saved_usd: float
    estimated_tokens_saved: int


def compute_stats(
    entries: int, snapshot: MetricsSnapshot, assumptions: StatsAssumptions
) -> StatsResult:
    hits = snapshot.hits
    misses = snapshot.misses
    queries = hits + misses
    return StatsResult(
        entries=entries,
        queries=queries,
        hits=hits,
        misses=misses,
        hit_rate=hits / queries if queries else 0.0,
        avg_similarity=snapshot.sim_sum / hits if hits else 0.0,
        avg_lookup_latency_ms=snapshot.lookup_ms_sum / queries if queries else 0.0,
        avg_store_search_ms=snapshot.search_ms_sum / queries if queries else 0.0,
        estimated_latency_saved_ms=hits * assumptions.llm_ms,
        estimated_cost_saved_usd=hits * assumptions.cost_usd,
        estimated_tokens_saved=hits * assumptions.tokens_per_call,
    )
