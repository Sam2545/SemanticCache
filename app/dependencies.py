from __future__ import annotations

from functools import lru_cache

from app.services.cache import CacheService
from app.vectorstore.memory_store import InMemoryVectorStore


@lru_cache
def get_service() -> CacheService:
    """Provide the application's CacheService.

    Defaults to the in-process store so the service boots without external
    dependencies. The Redis-backed VectorStore is wired in here once available;
    routes and the service are storage-agnostic and need no changes.
    """
    return CacheService(InMemoryVectorStore())
