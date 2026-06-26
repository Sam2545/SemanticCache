from __future__ import annotations

import time
from typing import Any, Sequence

from app.metrics.base import MetricsStore
from app.metrics.memory_store import InMemoryMetricsStore
from app.services.stats import StatsAssumptions, StatsResult, compute_stats
from app.vectorstore.base import Namespace, ScoredEntry, StoredEntry, VectorStore


class CacheError(Exception):
    """Base class for cache service errors."""


class NamespaceNotFound(CacheError):
    pass


class NamespaceExists(CacheError):
    pass


class DimensionMismatch(CacheError):
    pass


class InvalidFilter(CacheError):
    pass


class CacheService:
    """Business logic for the semantic cache.

    Owns dimension validation, default resolution, and the threshold/top_k
    hit-miss decision. Depends only on the VectorStore interface.
    """

    def __init__(
        self,
        store: VectorStore,
        metrics: MetricsStore | None = None,
        assumptions: StatsAssumptions | None = None,
    ) -> None:
        self._store = store
        self._metrics = metrics or InMemoryMetricsStore()
        self._assumptions = assumptions or StatsAssumptions()

    def create_namespace(
        self,
        name: str,
        dimension: int,
        default_threshold: float | None = None,
        default_top_k: int | None = None,
        ttl: int | None = None,
        filter_keys: list[str] | None = None,
    ) -> Namespace:
        if self._store.get_namespace(name) is not None:
            raise NamespaceExists(name)
        ns = Namespace(name=name, dimension=dimension)
        if default_threshold is not None:
            ns.default_threshold = default_threshold
        if default_top_k is not None:
            ns.default_top_k = default_top_k
        ns.ttl = ttl
        if filter_keys is not None:
            ns.filter_keys = filter_keys
        self._store.create_namespace(ns)
        return ns

    def get_namespace(self, name: str) -> Namespace | None:
        return self._store.get_namespace(name)

    def put(
        self,
        namespace: str,
        key: str,
        embedding: Sequence[float],
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._require_dimension(namespace, embedding)
        self._store.upsert(
            namespace,
            StoredEntry(
                key=key,
                embedding=list(embedding),
                value=value,
                metadata=metadata or {},
            ),
        )

    def query(
        self,
        namespace: str,
        embedding: Sequence[float],
        threshold: float | None = None,
        top_k: int | None = None,
        filter: dict[str, object] | None = None,
    ) -> list[ScoredEntry]:
        start = time.perf_counter()
        ns = self._require_dimension(namespace, embedding)
        flt = filter or {}
        unknown = set(flt) - set(ns.filter_keys)
        if unknown:
            raise InvalidFilter(
                f"filter keys {sorted(unknown)} not declared for namespace "
                f"'{namespace}' (declared: {ns.filter_keys})"
            )
        threshold = ns.default_threshold if threshold is None else threshold
        top_k = ns.default_top_k if top_k is None else top_k

        search_start = time.perf_counter()
        candidates = self._store.search(namespace, embedding, top_k, flt)
        search_ms = (time.perf_counter() - search_start) * 1000.0

        matches = [c for c in candidates if c.score >= threshold]
        hit = len(matches) > 0
        lookup_ms = (time.perf_counter() - start) * 1000.0
        self._metrics.record_query(
            namespace, hit, matches[0].score if hit else None, lookup_ms, search_ms
        )
        return matches

    def effective_threshold(self, namespace: str, threshold: float | None) -> float:
        ns = self._require_namespace(namespace)
        return ns.default_threshold if threshold is None else threshold

    def stats(self, namespace: str) -> StatsResult:
        self._require_namespace(namespace)
        snapshot = self._metrics.snapshot(namespace)
        entries = self._store.count(namespace)
        return compute_stats(entries, snapshot, self._assumptions)

    def get(self, namespace: str, key: str) -> StoredEntry | None:
        self._require_namespace(namespace)
        return self._store.get(namespace, key)

    def delete(self, namespace: str, key: str) -> bool:
        self._require_namespace(namespace)
        return self._store.delete(namespace, key)

    def _require_namespace(self, namespace: str) -> Namespace:
        ns = self._store.get_namespace(namespace)
        if ns is None:
            raise NamespaceNotFound(namespace)
        return ns

    def _require_dimension(
        self, namespace: str, embedding: Sequence[float]
    ) -> Namespace:
        ns = self._require_namespace(namespace)
        if len(embedding) != ns.dimension:
            raise DimensionMismatch(
                f"namespace '{namespace}' expects dimension {ns.dimension}, "
                f"got {len(embedding)}"
            )
        return ns
