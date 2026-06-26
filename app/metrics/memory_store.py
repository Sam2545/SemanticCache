from __future__ import annotations

from app.metrics.base import MetricsSnapshot


class InMemoryMetricsStore:
    """In-process MetricsStore. Used for tests and local development."""

    def __init__(self) -> None:
        self._data: dict[str, MetricsSnapshot] = {}

    def record_query(
        self,
        namespace: str,
        hit: bool,
        score: float | None,
        lookup_ms: float,
        search_ms: float,
    ) -> None:
        snap = self._data.setdefault(namespace, MetricsSnapshot())
        if hit:
            snap.hits += 1
            if score is not None:
                snap.sim_sum += score
        else:
            snap.misses += 1
        snap.lookup_ms_sum += lookup_ms
        snap.search_ms_sum += search_ms

    def snapshot(self, namespace: str) -> MetricsSnapshot:
        snap = self._data.get(namespace)
        if snap is None:
            return MetricsSnapshot()
        return MetricsSnapshot(
            hits=snap.hits,
            misses=snap.misses,
            sim_sum=snap.sim_sum,
            lookup_ms_sum=snap.lookup_ms_sum,
            search_ms_sum=snap.search_ms_sum,
        )
