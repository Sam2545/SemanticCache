# Query Observability + Cache Stats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `hit`/`threshold` to the query response and a per-namespace `GET /{namespace}/stats` endpoint reporting hit rate, similarity, measured latency, and estimated latency/cost/tokens saved.

**Architecture:** A `MetricsStore` interface (in-memory + Redis, selected by `SEMCACHE_BACKEND`) records each query outcome. `CacheService` times the query, records metrics, and assembles stats via a pure `compute_stats`. A live `VectorStore.count()` supplies the entry count. Mirrors the existing `VectorStore` pattern; both backends behave identically.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, redis-py (RediSearch), pytest.

## Global Constraints

- Python ≥ 3.12.
- In-memory and Redis backends MUST behave identically behind their protocols.
- No Redis types leak into `app/services/`.
- All stats response fields are **snake_case**.
- Derived metrics are divide-by-zero guarded: `hit_rate`/`avg_lookup_latency_ms`/`avg_store_search_ms` are `0.0` when `queries == 0`; `avg_similarity` is `0.0` when `hits == 0`.
- `avg_similarity` is the mean score of **served hits** only.
- Latency is measured server-side with `time.perf_counter`, in milliseconds.
- Config defaults (env `SEMCACHE_*`): `assumed_llm_ms=800.0`, `assumed_llm_cost_usd=0.001`, `assumed_tokens_per_call=500`.
- Backward-compatible: existing namespaces/queries/constructors keep working.
- Commit messages: a single line under 150 characters, no `Co-Authored-By` trailer.
- Run the fast unit suite with `python -m pytest -m "not integration"`.

---

## File Structure

- `app/vectorstore/base.py` — add `count` to the `VectorStore` protocol.
- `app/vectorstore/memory_store.py` — `count`.
- `app/vectorstore/redis_store.py` — `count` via `FT.INFO num_docs`.
- `app/metrics/__init__.py`, `app/metrics/base.py` — `MetricsStore` protocol + `MetricsSnapshot`.
- `app/metrics/memory_store.py` — `InMemoryMetricsStore`.
- `app/metrics/redis_store.py` — `RedisMetricsStore`.
- `app/services/stats.py` — `StatsAssumptions`, `StatsResult`, pure `compute_stats`.
- `app/services/cache.py` — metrics wiring, query timing/recording, `effective_threshold`, `stats`.
- `app/config.py` — three `assumed_*` settings.
- `app/dependencies.py` — build metrics store + assumptions; inject into `CacheService`.
- `app/models/schemas.py` — `QueryResponse` `hit`/`threshold`; `StatsResponse`.
- `app/routes/cache.py` — query response fields; `GET /{namespace}/stats`.
- Tests: `tests/test_memory_store.py`, `tests/test_metrics_store.py`, `tests/test_stats.py`, `tests/test_cache_service.py`, `tests/test_api.py`, `tests/test_dependencies.py`, `tests/integration/test_redis_metrics.py`.
- `examples/llm_cache_demo.py`, `CLAUDE.md`, `README.md`.

---

### Task 1: `VectorStore.count`

**Files:**
- Modify: `app/vectorstore/base.py`, `app/vectorstore/memory_store.py`
- Test: `tests/test_memory_store.py`

**Interfaces:**
- Produces: `VectorStore.count(namespace: str) -> int` — live count of entries in a namespace.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_memory_store.py`:

```python
def test_count_reflects_upserts_and_deletes(store):
    assert store.count("ns") == 0
    store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A"))
    store.upsert("ns", StoredEntry(key="b", embedding=[0.0, 1.0], value="B"))
    assert store.count("ns") == 2
    store.delete("ns", "a")
    assert store.count("ns") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_memory_store.py::test_count_reflects_upserts_and_deletes -v`
Expected: FAIL — `AttributeError: 'InMemoryVectorStore' object has no attribute 'count'`.

- [ ] **Step 3: Add `count` to the protocol**

In `app/vectorstore/base.py`, inside the `VectorStore` protocol, after `delete`:

```python
    def delete(self, namespace: str, key: str) -> bool: ...

    def count(self, namespace: str) -> int: ...
```

- [ ] **Step 4: Implement in-memory `count`**

In `app/vectorstore/memory_store.py`, add after `delete`:

```python
    def count(self, namespace: str) -> int:
        return len(self._entries[namespace])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_memory_store.py -v`
Expected: PASS (new + existing).

- [ ] **Step 6: Commit**

```bash
git add app/vectorstore/base.py app/vectorstore/memory_store.py tests/test_memory_store.py
git commit -m "Add VectorStore.count for live entry counts (in-memory)"
```

---

### Task 2: `MetricsStore` + `InMemoryMetricsStore`

**Files:**
- Create: `app/metrics/__init__.py`, `app/metrics/base.py`, `app/metrics/memory_store.py`
- Test: `tests/test_metrics_store.py`

**Interfaces:**
- Produces:
  - `MetricsSnapshot` dataclass: `hits: int`, `misses: int`, `sim_sum: float`, `lookup_ms_sum: float`, `search_ms_sum: float` (all default `0`).
  - `MetricsStore` protocol: `record_query(namespace, hit, score, lookup_ms, search_ms) -> None`, `snapshot(namespace) -> MetricsSnapshot`.
  - `InMemoryMetricsStore` implementing it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics_store.py`:

```python
from app.metrics.memory_store import InMemoryMetricsStore


def test_snapshot_of_unseen_namespace_is_zero():
    m = InMemoryMetricsStore()
    snap = m.snapshot("ns")
    assert (snap.hits, snap.misses, snap.sim_sum) == (0, 0, 0.0)
    assert (snap.lookup_ms_sum, snap.search_ms_sum) == (0.0, 0.0)


def test_record_hit_accumulates_score_and_latency():
    m = InMemoryMetricsStore()
    m.record_query("ns", hit=True, score=0.9, lookup_ms=3.0, search_ms=1.0)
    snap = m.snapshot("ns")
    assert snap.hits == 1
    assert snap.misses == 0
    assert snap.sim_sum == 0.9
    assert snap.lookup_ms_sum == 3.0
    assert snap.search_ms_sum == 1.0


def test_record_miss_does_not_touch_sim_sum():
    m = InMemoryMetricsStore()
    m.record_query("ns", hit=False, score=None, lookup_ms=2.0, search_ms=0.5)
    snap = m.snapshot("ns")
    assert snap.misses == 1
    assert snap.hits == 0
    assert snap.sim_sum == 0.0
    assert snap.lookup_ms_sum == 2.0


def test_namespaces_are_isolated():
    m = InMemoryMetricsStore()
    m.record_query("a", hit=True, score=0.5, lookup_ms=1.0, search_ms=1.0)
    m.record_query("b", hit=False, score=None, lookup_ms=1.0, search_ms=1.0)
    assert m.snapshot("a").hits == 1
    assert m.snapshot("b").hits == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.metrics'`.

- [ ] **Step 3: Create the package and protocol**

Create `app/metrics/__init__.py`:

```python
from app.metrics.base import MetricsSnapshot, MetricsStore

__all__ = ["MetricsSnapshot", "MetricsStore"]
```

Create `app/metrics/base.py`:

```python
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
```

- [ ] **Step 4: Implement `InMemoryMetricsStore`**

Create `app/metrics/memory_store.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics_store.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/metrics/ tests/test_metrics_store.py
git commit -m "Add MetricsStore protocol and in-memory implementation"
```

---

### Task 3: `compute_stats` (pure)

**Files:**
- Create: `app/services/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: `MetricsSnapshot` (Task 2).
- Produces:
  - `StatsAssumptions` dataclass: `llm_ms: float = 800.0`, `cost_usd: float = 0.001`, `tokens_per_call: int = 500`.
  - `StatsResult` dataclass: `entries, queries, hits, misses: int`; `hit_rate, avg_similarity, avg_lookup_latency_ms, avg_store_search_ms, estimated_latency_saved_ms, estimated_cost_saved_usd: float`; `estimated_tokens_saved: int`.
  - `compute_stats(entries: int, snapshot: MetricsSnapshot, assumptions: StatsAssumptions) -> StatsResult`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stats.py`:

```python
from app.metrics.base import MetricsSnapshot
from app.services.stats import StatsAssumptions, compute_stats


def test_compute_stats_basic_math():
    snap = MetricsSnapshot(hits=3, misses=1, sim_sum=2.7, lookup_ms_sum=8.0, search_ms_sum=4.0)
    res = compute_stats(entries=10, snapshot=snap, assumptions=StatsAssumptions())
    assert res.entries == 10
    assert res.queries == 4
    assert res.hits == 3
    assert res.misses == 1
    assert res.hit_rate == 0.75
    assert res.avg_similarity == 0.9          # 2.7 / 3 hits
    assert res.avg_lookup_latency_ms == 2.0   # 8.0 / 4 queries
    assert res.avg_store_search_ms == 1.0     # 4.0 / 4 queries
    assert res.estimated_latency_saved_ms == 2400.0   # 3 * 800
    assert res.estimated_cost_saved_usd == 0.003       # 3 * 0.001
    assert res.estimated_tokens_saved == 1500          # 3 * 500


def test_compute_stats_zero_queries_is_guarded():
    res = compute_stats(entries=0, snapshot=MetricsSnapshot(), assumptions=StatsAssumptions())
    assert res.queries == 0
    assert res.hit_rate == 0.0
    assert res.avg_similarity == 0.0
    assert res.avg_lookup_latency_ms == 0.0
    assert res.avg_store_search_ms == 0.0
    assert res.estimated_latency_saved_ms == 0.0


def test_compute_stats_misses_only_avg_similarity_zero():
    snap = MetricsSnapshot(hits=0, misses=2, sim_sum=0.0, lookup_ms_sum=4.0, search_ms_sum=2.0)
    res = compute_stats(entries=5, snapshot=snap, assumptions=StatsAssumptions())
    assert res.avg_similarity == 0.0           # no hits -> guarded
    assert res.avg_lookup_latency_ms == 2.0    # 4.0 / 2 queries
    assert res.hit_rate == 0.0


def test_compute_stats_uses_custom_assumptions():
    snap = MetricsSnapshot(hits=2, misses=0)
    res = compute_stats(
        entries=2, snapshot=snap,
        assumptions=StatsAssumptions(llm_ms=1000.0, cost_usd=0.01, tokens_per_call=100),
    )
    assert res.estimated_latency_saved_ms == 2000.0
    assert res.estimated_cost_saved_usd == 0.02
    assert res.estimated_tokens_saved == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.stats'`.

- [ ] **Step 3: Implement `compute_stats`**

Create `app/services/stats.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from app.metrics.base import MetricsSnapshot


@dataclass
class StatsAssumptions:
    """Configured inputs for estimated-savings metrics."""

    llm_ms: float = 800.0
    cost_usd: float = 0.001
    tokens_per_call: int = 500


@dataclass
class StatsResult:
    entries: int
    queries: int
    hits: int
    misses: int
    hit_rate: float
    avg_similarity: float
    avg_lookup_latency_ms: float
    avg_store_search_ms: float
    estimated_latency_saved_ms: float
    estimated_cost_saved_usd: float
    estimated_tokens_saved: int


def compute_stats(
    entries: int, snapshot: MetricsSnapshot, assumptions: StatsAssumptions
) -> StatsResult:
    hits = snapshot.hits
    misses = snapshot.misses
    queries = hits + misses
    return StatsResult(
        entries=entries,
        queries=queries,
        hits=hits,
        misses=misses,
        hit_rate=hits / queries if queries else 0.0,
        avg_similarity=snapshot.sim_sum / hits if hits else 0.0,
        avg_lookup_latency_ms=snapshot.lookup_ms_sum / queries if queries else 0.0,
        avg_store_search_ms=snapshot.search_ms_sum / queries if queries else 0.0,
        estimated_latency_saved_ms=hits * assumptions.llm_ms,
        estimated_cost_saved_usd=hits * assumptions.cost_usd,
        estimated_tokens_saved=hits * assumptions.tokens_per_call,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_stats.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/services/stats.py tests/test_stats.py
git commit -m "Add pure compute_stats with StatsAssumptions and StatsResult"
```

---

### Task 4: `CacheService` — metrics, timing, `effective_threshold`, `stats`

**Files:**
- Modify: `app/services/cache.py`
- Test: `tests/test_cache_service.py`

**Interfaces:**
- Consumes: `InMemoryMetricsStore`, `MetricsStore` (Task 2); `StatsAssumptions`, `StatsResult`, `compute_stats` (Task 3); `VectorStore.count` (Task 1).
- Produces:
  - `CacheService(store, metrics: MetricsStore | None = None, assumptions: StatsAssumptions | None = None)` — `metrics` defaults to `InMemoryMetricsStore()`, `assumptions` to `StatsAssumptions()`.
  - `CacheService.query` records each outcome to metrics (unchanged return type `list[ScoredEntry]`).
  - `CacheService.effective_threshold(namespace, threshold) -> float`.
  - `CacheService.stats(namespace) -> StatsResult`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cache_service.py`:

```python
def test_query_records_hit_in_stats(service):
    service.create_namespace("ns", dimension=2)
    service.put("ns", key="a", embedding=[1.0, 0.0], value="A")
    service.query("ns", embedding=[1.0, 0.0], threshold=0.0)
    s = service.stats("ns")
    assert s.hits == 1 and s.misses == 0 and s.queries == 1
    assert s.entries == 1
    assert s.avg_lookup_latency_ms >= 0.0


def test_query_records_miss_in_stats(service):
    service.create_namespace("ns", dimension=2)
    service.put("ns", key="a", embedding=[1.0, 0.0], value="A")
    service.query("ns", embedding=[0.0, 1.0], threshold=0.99)
    s = service.stats("ns")
    assert s.misses == 1 and s.hits == 0


def test_effective_threshold_default_and_override(service):
    service.create_namespace("ns", dimension=2, default_threshold=0.7)
    assert service.effective_threshold("ns", None) == 0.7
    assert service.effective_threshold("ns", 0.9) == 0.9


def test_stats_estimates_savings_with_default_assumptions(service):
    service.create_namespace("ns", dimension=2)
    service.put("ns", key="a", embedding=[1.0, 0.0], value="A")
    service.query("ns", embedding=[1.0, 0.0], threshold=0.0)
    s = service.stats("ns")
    assert s.estimated_latency_saved_ms == 800.0
    assert s.estimated_tokens_saved == 500


def test_stats_missing_namespace_raises(service):
    with pytest.raises(NamespaceNotFound):
        service.stats("nope")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cache_service.py::test_query_records_hit_in_stats -v`
Expected: FAIL — `AttributeError: 'CacheService' object has no attribute 'stats'`.

- [ ] **Step 3: Update imports and constructor**

In `app/services/cache.py`, add imports near the top (after the existing imports):

```python
import time

from app.metrics.base import MetricsStore
from app.metrics.memory_store import InMemoryMetricsStore
from app.services.stats import StatsAssumptions, StatsResult, compute_stats
```

Replace `__init__`:

```python
    def __init__(
        self,
        store: VectorStore,
        metrics: MetricsStore | None = None,
        assumptions: StatsAssumptions | None = None,
    ) -> None:
        self._store = store
        self._metrics = metrics or InMemoryMetricsStore()
        self._assumptions = assumptions or StatsAssumptions()
```

- [ ] **Step 4: Time and record in `query`**

In `app/services/cache.py`, replace the `query` method body with:

```python
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
```

- [ ] **Step 5: Add `effective_threshold` and `stats`**

In `app/services/cache.py`, add these methods (e.g. after `query`):

```python
    def effective_threshold(self, namespace: str, threshold: float | None) -> float:
        ns = self._require_namespace(namespace)
        return ns.default_threshold if threshold is None else threshold

    def stats(self, namespace: str) -> StatsResult:
        self._require_namespace(namespace)
        snapshot = self._metrics.snapshot(namespace)
        entries = self._store.count(namespace)
        return compute_stats(entries, snapshot, self._assumptions)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_cache_service.py -v`
Expected: PASS (new + existing).

- [ ] **Step 7: Commit**

```bash
git add app/services/cache.py tests/test_cache_service.py
git commit -m "Record query metrics and add effective_threshold and stats to CacheService"
```

---

### Task 5: API — `hit`/`threshold` on query, `GET /{namespace}/stats`

**Files:**
- Modify: `app/models/schemas.py`, `app/routes/cache.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `CacheService.query`, `CacheService.effective_threshold`, `CacheService.stats` (Task 4); `StatsResult` fields (Task 3).
- Produces: `QueryResponse` with `hit`/`threshold`; `StatsResponse`; route `GET /{namespace}/stats`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_api.py`:

```python
def test_query_response_includes_hit_and_threshold(client):
    client.post("/namespaces", json={"name": "ns", "dimension": 2, "default_threshold": 0.5})
    client.post("/ns/entries", json={"key": "a", "embedding": [1.0, 0.0], "value": "A"})
    hit = client.post("/ns/query", json={"embedding": [1.0, 0.0]}).json()
    assert hit["hit"] is True
    assert hit["threshold"] == 0.5
    miss = client.post("/ns/query", json={"embedding": [0.0, 1.0], "threshold": 0.99}).json()
    assert miss["hit"] is False
    assert miss["threshold"] == 0.99


def test_stats_endpoint_reports_hits_and_misses(client):
    client.post("/namespaces", json={"name": "ns", "dimension": 2, "default_threshold": 0.5})
    client.post("/ns/entries", json={"key": "a", "embedding": [1.0, 0.0], "value": "A"})
    client.post("/ns/query", json={"embedding": [1.0, 0.0]})           # hit
    client.post("/ns/query", json={"embedding": [0.0, 1.0]})           # miss
    r = client.get("/ns/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["entries"] == 1
    assert body["queries"] == 2
    assert body["hits"] == 1
    assert body["misses"] == 1
    assert body["hit_rate"] == 0.5
    assert body["estimated_latency_saved_ms"] == 800.0
    assert "avg_lookup_latency_ms" in body and "avg_store_search_ms" in body


def test_stats_missing_namespace_not_found(client):
    assert client.get("/nope/stats").status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api.py::test_query_response_includes_hit_and_threshold -v`
Expected: FAIL — response has no `hit`/`threshold` keys.

- [ ] **Step 3: Add schema fields and `StatsResponse`**

In `app/models/schemas.py`, replace the `QueryResponse` class with:

```python
class QueryResponse(BaseModel):
    matches: list[Match]
    hit: bool
    threshold: float
```

And add a `StatsResponse` model (e.g. at the end of the file):

```python
class StatsResponse(BaseModel):
    entries: int
    queries: int
    hits: int
    misses: int
    hit_rate: float
    avg_similarity: float
    avg_lookup_latency_ms: float
    avg_store_search_ms: float
    estimated_latency_saved_ms: float
    estimated_cost_saved_usd: float
    estimated_tokens_saved: int
```

(Remove the now-unused `hit` property if present — it is replaced by the real field.)

- [ ] **Step 4: Update the query route and add the stats route**

In `app/routes/cache.py`, add `StatsResponse` to the schema imports, and `from dataclasses import asdict` at the top. Replace the `query` route body's return with:

```python
    matches = service.query(
        namespace=namespace,
        embedding=body.embedding,
        threshold=body.threshold,
        top_k=body.top_k,
        filter=body.filter,
    )
    return QueryResponse(
        matches=[
            Match(key=m.key, score=m.score, value=m.value, metadata=m.metadata)
            for m in matches
        ],
        hit=len(matches) > 0,
        threshold=service.effective_threshold(namespace, body.threshold),
    )
```

Add a new route (e.g. after the `query` route):

```python
@router.get("/{namespace}/stats", response_model=StatsResponse)
def get_stats(
    namespace: str, service: CacheService = Depends(get_service)
) -> StatsResponse:
    return StatsResponse(**asdict(service.stats(namespace)))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_api.py -v`
Expected: PASS (new + existing). The `GET /{namespace}/stats` path does not collide with `GET /{namespace}/entries/{key}`.

- [ ] **Step 6: Commit**

```bash
git add app/models/schemas.py app/routes/cache.py tests/test_api.py
git commit -m "Add hit/threshold to query response and GET /{namespace}/stats endpoint"
```

---

### Task 6: Redis backend — `RedisMetricsStore` + Redis `count`

**Files:**
- Create: `app/metrics/redis_store.py`
- Modify: `app/vectorstore/redis_store.py`
- Test: `tests/integration/test_redis_metrics.py`

**Interfaces:**
- Consumes: `MetricsSnapshot` (Task 2).
- Produces: `RedisMetricsStore(redis_url)` implementing `MetricsStore`; `RedisVectorStore.count`.

**NOTE:** Integration tests require a live redis-stack and CANNOT run in the dev sandbox (no Docker / no `redis-py`). Verify with `python -m py_compile` on the changed files and `python -m pytest -m "not integration"`; the integration tests are written but executed later via `docker compose run --rm api pytest -m integration -v`.

- [ ] **Step 1: Write the failing integration tests**

Create `tests/integration/test_redis_metrics.py`:

```python
import os

import pytest

from app.metrics.base import MetricsSnapshot

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
```

Add to `tests/integration/test_redis_store.py`:

```python
def test_count_reflects_entries(redis_store):
    redis_store.create_namespace(Namespace(name="ns", dimension=2))
    assert redis_store.count("ns") == 0
    redis_store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A"))
    redis_store.upsert("ns", StoredEntry(key="b", embedding=[0.0, 1.0], value="B"))
    assert redis_store.count("ns") == 2
    redis_store.delete("ns", "a")
    assert redis_store.count("ns") == 1
```

- [ ] **Step 2: Run the compile + unit checks (integration cannot run here)**

Run: `python -m py_compile tests/integration/test_redis_metrics.py`
Expected: success. (The integration suite itself runs later in the container.)

- [ ] **Step 3: Implement `RedisMetricsStore`**

Create `app/metrics/redis_store.py`:

```python
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
```

- [ ] **Step 4: Implement Redis `count`**

In `app/vectorstore/redis_store.py`, add after `delete`:

```python
    def count(self, namespace: str) -> int:
        info = self._client.ft(self._index(namespace)).info()
        return int(info["num_docs"])
```

- [ ] **Step 5: Verify compile + unit suite**

Run: `python -m py_compile app/metrics/redis_store.py app/vectorstore/redis_store.py && python -m pytest -m "not integration" -q`
Expected: compile OK; unit suite passes (integration deselected).

- [ ] **Step 6: Commit**

```bash
git add app/metrics/redis_store.py app/vectorstore/redis_store.py tests/integration/
git commit -m "Add Redis metrics store and Redis VectorStore.count (FT.INFO)"
```

---

### Task 7: Config + dependency wiring

**Files:**
- Modify: `app/config.py`, `app/dependencies.py`
- Test: `tests/test_dependencies.py`

**Interfaces:**
- Consumes: `InMemoryMetricsStore` (Task 2), `RedisMetricsStore` (Task 6), `StatsAssumptions` (Task 3), `CacheService` (Task 4).
- Produces: `get_service()` returns a `CacheService` wired with the backend's metrics store and assumptions from settings.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dependencies.py`:

```python
from app.dependencies import _build_metrics, get_service
from app.metrics.memory_store import InMemoryMetricsStore
from app.services.cache import CacheService


def test_build_metrics_defaults_to_in_memory():
    assert isinstance(_build_metrics(), InMemoryMetricsStore)


def test_get_service_returns_cache_service():
    get_service.cache_clear()
    assert isinstance(get_service(), CacheService)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dependencies.py -v`
Expected: FAIL — `ImportError: cannot import name '_build_metrics'`.

- [ ] **Step 3: Add config settings**

In `app/config.py`, inside `Settings`, add after `default_top_k`:

```python
    assumed_llm_ms: float = 800.0
    assumed_llm_cost_usd: float = 0.001
    assumed_tokens_per_call: int = 500
```

- [ ] **Step 4: Wire metrics + assumptions in `dependencies.py`**

In `app/dependencies.py`, add imports and a `_build_metrics`, and update `get_service`:

```python
from app.metrics.base import MetricsStore
from app.metrics.memory_store import InMemoryMetricsStore
from app.services.stats import StatsAssumptions


def _build_metrics() -> MetricsStore:
    if settings.backend == "redis":
        from app.metrics.redis_store import RedisMetricsStore

        return RedisMetricsStore(settings.redis_url)
    return InMemoryMetricsStore()


@lru_cache
def get_service() -> CacheService:
    return CacheService(
        _build_store(),
        _build_metrics(),
        StatsAssumptions(
            llm_ms=settings.assumed_llm_ms,
            cost_usd=settings.assumed_llm_cost_usd,
            tokens_per_call=settings.assumed_tokens_per_call,
        ),
    )
```

(Keep the existing `_build_store` and module docstring; only the `get_service` body and imports change.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dependencies.py tests/test_api.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/config.py app/dependencies.py tests/test_dependencies.py
git commit -m "Wire metrics store and stats assumptions into the service"
```

---

### Task 8: Examples + docs

**Files:**
- Modify: `examples/llm_cache_demo.py`, `CLAUDE.md`, `README.md`

- [ ] **Step 1: Print stats at the end of the demo**

In `examples/llm_cache_demo.py`, add a helper and call it at the end of `main` (after the loop, before `return 0`):

```python
def print_stats() -> None:
    resp = _semcache("GET", f"/{NAMESPACE}/stats")
    if resp.status_code != 200:
        return
    s = resp.json()
    print(
        f"\nstats for '{NAMESPACE}': "
        f"{s['hits']}/{s['queries']} hits ({s['hit_rate']:.0%}), "
        f"avg similarity {s['avg_similarity']:.3f}, "
        f"~{s['estimated_latency_saved_ms']:.0f} ms saved"
    )
```

In `main`, after the `for original, paraphrase in PAIRS:` loop completes:

```python
        print_stats()
```

- [ ] **Step 2: Verify the demo compiles**

Run: `python -m py_compile examples/llm_cache_demo.py`
Expected: success.

- [ ] **Step 3: Document the feature**

In `CLAUDE.md`, under "Data model & semantics" (or a nearby bullet list), add:

```markdown
- The query response includes `hit` (bool) and the effective `threshold`.
  `GET /{namespace}/stats` reports entries, hits/misses, `hit_rate`,
  `avg_similarity`, measured `avg_lookup_latency_ms`/`avg_store_search_ms`, and
  estimated latency/cost/tokens saved. Metrics are Redis-backed via a
  `MetricsStore` (in-memory for dev/tests), selected by `SEMCACHE_BACKEND`.
```

In `README.md`, under "API", add a row:

```markdown
| `GET`  | `/{namespace}/stats` | Cache-effectiveness metrics for the namespace |
```

- [ ] **Step 4: Commit**

```bash
git add examples/llm_cache_demo.py CLAUDE.md README.md
git commit -m "Print cache stats in demo and document the stats endpoint"
```

---

## Final verification

- [ ] `python -m pytest -m "not integration" -v` — all pass.
- [ ] `docker compose run --rm api pytest -m integration -v` — all pass (Redis metrics + count).
- [ ] Push to the feature branch / open PR per finishing-a-development-branch.
