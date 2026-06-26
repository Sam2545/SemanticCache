from __future__ import annotations

import redis

from app.metrics.base import MetricsSnapshot


def _num(mapping: dict, field: str) -> bytes | None:
    return mapping.get(field.encode())


class RedisMetricsStore:
    """MetricsStore backed by a Redis hash per namespace.

    Counters live at semcache:stats:{namespace}; HINCRBY/HINCRBYFLOAT keep them
    consistent across instances and restarts.
    """

    def __init__(self, redis_url: str) -> None:
        self._client = redis.Redis.from_url(redis_url)

    def _key(self, namespace: str) -> str:
        return f"semcache:stats:{namespace}"

    def record_query(
        self,
        namespace: str,
        hit: bool,
        score: float | None,
        lookup_ms: float,
        search_ms: float,
    ) -> None:
        key = self._key(namespace)
        if hit:
            self._client.hincrby(key, "hits", 1)
            if score is not None:
                self._client.hincrbyfloat(key, "sim_sum", score)
        else:
            self._client.hincrby(key, "misses", 1)
        self._client.hincrbyfloat(key, "lookup_ms_sum", lookup_ms)
        self._client.hincrbyfloat(key, "search_ms_sum", search_ms)

    def snapshot(self, namespace: str) -> MetricsSnapshot:
        raw = self._client.hgetall(self._key(namespace))
        return MetricsSnapshot(
            hits=int(_num(raw, "hits") or 0),
            misses=int(_num(raw, "misses") or 0),
            sim_sum=float(_num(raw, "sim_sum") or 0.0),
            lookup_ms_sum=float(_num(raw, "lookup_ms_sum") or 0.0),
            search_ms_sum=float(_num(raw, "search_ms_sum") or 0.0),
        )
