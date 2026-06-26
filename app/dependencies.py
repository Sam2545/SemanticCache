from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.metrics.base import MetricsStore
from app.metrics.memory_store import InMemoryMetricsStore
from app.services.cache import CacheService
from app.services.stats import StatsAssumptions
from app.vectorstore.base import VectorStore
from app.vectorstore.memory_store import InMemoryVectorStore


def _build_store() -> VectorStore:
    if settings.backend == "redis":
        from app.vectorstore.redis_store import RedisVectorStore

        return RedisVectorStore(settings.redis_url)
    return InMemoryVectorStore()


def _build_metrics() -> MetricsStore:
    if settings.backend == "redis":
        from app.metrics.redis_store import RedisMetricsStore

        return RedisMetricsStore(settings.redis_url)
    return InMemoryMetricsStore()


@lru_cache
def get_service() -> CacheService:
    """Provide the application's CacheService.

    Backend is chosen by SEMCACHE_BACKEND ("memory" by default, "redis" in the
    container). Routes and the service are storage-agnostic — only this factory
    knows which VectorStore/MetricsStore implementations are in use.
    """
    return CacheService(
        _build_store(),
        _build_metrics(),
        StatsAssumptions(
            llm_ms=settings.assumed_llm_ms,
            cost_usd=settings.assumed_llm_cost_usd,
            tokens_per_call=settings.assumed_tokens_per_call,
        ),
    )
