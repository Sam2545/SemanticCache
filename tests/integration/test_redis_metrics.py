import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def redis_metrics(redis_url):
    redis = pytest.importorskip("redis")
    client = redis.Redis.from_url(redis_url)
    try:
        client.ping()
    except redis.exceptions.RedisError:
        pytest.skip("redis-stack not reachable")
    from app.metrics.redis_store import RedisMetricsStore

    client.flushall()
    yield RedisMetricsStore(redis_url)
    client.flushall()


def test_redis_snapshot_unseen_is_zero(redis_metrics):
    snap = redis_metrics.snapshot("ns")
    assert (snap.hits, snap.misses, snap.sim_sum) == (0, 0, 0.0)


def test_redis_records_hit_and_miss(redis_metrics):
    redis_metrics.record_query("ns", hit=True, score=0.9, lookup_ms=3.0, search_ms=1.0)
    redis_metrics.record_query("ns", hit=False, score=None, lookup_ms=2.0, search_ms=0.5)
    snap = redis_metrics.snapshot("ns")
    assert snap.hits == 1
    assert snap.misses == 1
    assert snap.sim_sum == pytest.approx(0.9)
    assert snap.lookup_ms_sum == pytest.approx(5.0)
    assert snap.search_ms_sum == pytest.approx(1.5)
