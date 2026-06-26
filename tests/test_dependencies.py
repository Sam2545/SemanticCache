from app.dependencies import _build_metrics, get_service
from app.metrics.memory_store import InMemoryMetricsStore
from app.services.cache import CacheService


def test_build_metrics_defaults_to_in_memory():
    assert isinstance(_build_metrics(), InMemoryMetricsStore)


def test_get_service_returns_cache_service():
    get_service.cache_clear()
    assert isinstance(get_service(), CacheService)
