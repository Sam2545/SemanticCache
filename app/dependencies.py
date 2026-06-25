from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.services.cache import CacheService
from app.vectorstore.base import VectorStore
from app.vectorstore.memory_store import InMemoryVectorStore


def _build_store() -> VectorStore:
    if settings.backend == "redis":
        from app.vectorstore.redis_store import RedisVectorStore

        return RedisVectorStore(settings.redis_url)
    return InMemoryVectorStore()


@lru_cache
def get_service() -> CacheService:
    """Provide the application's CacheService.

    Backend is chosen by SEMCACHE_BACKEND ("memory" by default, "redis" in the
    container). Routes and the service are storage-agnostic — only this factory
    knows which VectorStore implementation is in use.
    """
    return CacheService(_build_store())
