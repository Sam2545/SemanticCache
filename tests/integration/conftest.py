import os

import pytest


@pytest.fixture(scope="session")
def redis_url() -> str:
    return os.environ.get("SEMCACHE_REDIS_URL", "redis://localhost:6379")


@pytest.fixture
def redis_store(redis_url):
    """A clean RedisVectorStore, or skip if no redis-stack is reachable.

    The reachability check runs before importing the store so the suite skips
    cleanly (rather than erroring) when Redis is down.
    """
    redis = pytest.importorskip("redis")
    client = redis.Redis.from_url(redis_url)
    try:
        client.ping()
    except redis.exceptions.RedisError:
        pytest.skip("redis-stack not reachable")

    from app.vectorstore.redis_store import RedisVectorStore

    client.flushall()
    yield RedisVectorStore(redis_url)
    client.flushall()
