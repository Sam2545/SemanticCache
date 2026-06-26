from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class MetricsSnapshot:
    """Raw per-namespace counters; derived metrics are computed elsewhere."""

    hits: int = 0
    misses: int = 0
    sim_sum: float = 0.0
    lookup_ms_sum: float = 0.0
    search_ms_sum: float = 0.0


class MetricsStore(Protocol):
    """Records query outcomes and returns raw counters per namespace."""

    def record_query(
        self,
        namespace: str,
        hit: bool,
        score: float | None,
        lookup_ms: float,
        search_ms: float,
    ) -> None: ...

    def snapshot(self, namespace: str) -> MetricsSnapshot: ...
