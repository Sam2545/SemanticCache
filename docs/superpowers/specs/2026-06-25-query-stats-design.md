# Query observability + per-namespace cache stats — design

- **Date:** 2026-06-25
- **Status:** Approved (pending spec review)
- **Component:** SemCache core (`app/`), both backends, examples
- **Out of scope:** default-TTL work (separate, still pending)

## Problem

SemCache tracks no metrics, and a query response conveys hit/miss only implicitly
(empty `matches`). We want (1) the query response to state `hit` and the
`threshold` actually applied, and (2) a per-namespace stats endpoint that
demonstrates how the cache saves latency/cost.

## Goals

- Add `hit` and `threshold` to the query response (still 200 for hit and miss).
- A `GET /{namespace}/stats` endpoint reporting entries, hits/misses, hit rate,
  average served similarity, and estimated latency/cost/tokens saved.
- Metrics are **Redis-backed** (persistent, shared across instances), behind a
  `MetricsStore` interface mirroring `VectorStore`, so the in-memory dev/test
  backend behaves identically.
- Backward-compatible; both backends behave identically.

## Design

### 1. Query response: `hit` + `threshold`

`QueryResponse` gains:
- `hit: bool` — `len(matches) > 0`.
- `threshold: float` — the effective threshold applied (request value, else the
  namespace default).

`matches` is unchanged; hit and miss both return 200. The service owns threshold
resolution (a new `CacheService.effective_threshold(namespace, threshold) -> float`,
also used internally by `query`) so the route does not duplicate the defaulting
logic. `query`'s return type is unchanged (avoids churning existing tests); the
route computes `hit` from the returned matches and reads `threshold` from
`effective_threshold`.

### 2. `MetricsStore` abstraction (mirrors `VectorStore`)

```
class MetricsStore(Protocol):
    def record_query(
        self, namespace: str, hit: bool, score: float | None,
        lookup_ms: float, search_ms: float,
    ) -> None: ...
    def snapshot(self, namespace: str) -> MetricsSnapshot: ...
    # MetricsSnapshot: {hits, misses, sim_sum, lookup_ms_sum, search_ms_sum}
```

- `record_query`: increments `hits` or `misses`; on a hit also adds the top
  match's `score` to `sim_sum`; always adds `lookup_ms` to `lookup_ms_sum` and
  `search_ms` to `search_ms_sum`.
- `InMemoryMetricsStore`: per-namespace dict of counters/sums (dev/unit tests).
- `RedisMetricsStore`: `HINCRBY` `hits`/`misses` and `HINCRBYFLOAT`
  `sim_sum`/`lookup_ms_sum`/`search_ms_sum` on `semcache:stats:{namespace}`;
  `snapshot` via `HGETALL` (missing = zeros).
- Selected by the existing `SEMCACHE_BACKEND` switch and wired in
  `app/dependencies.py` (same place the `VectorStore` is chosen).

`CacheService` gains an optional `metrics: MetricsStore | None = None` constructor
arg (defaults to `InMemoryMetricsStore()` so existing constructors keep working).
`CacheService.query` times the store `search` call (`search_ms`) and the whole
query (`lookup_ms`) with `time.perf_counter`, then records the outcome:
`hit = len(matches) > 0`, score = `matches[0].score` if hit else `None`.

### 3. Live entry count

`VectorStore` gains `count(namespace) -> int`:
- in-memory: `len(self._entries[namespace])`.
- Redis: `FT.INFO` `num_docs` for the namespace index.

A live count (reflects TTL expiry and deletes), not a counter.

### 4. Derived metrics — pure computation

A `StatsAssumptions` dataclass holds the configured inputs:
`llm_ms`, `cost_usd`, `tokens_per_call`. `CacheService` holds one (injected,
defaults provided) so tests can pass known values. A pure function computes the
report from raw inputs (testable in isolation):

```
compute_stats(entries, hits, misses, sim_sum,
              lookup_ms_sum, search_ms_sum, assumptions) -> StatsResult
```

All output keys are snake_case:
- `queries = hits + misses`
- `hit_rate = hits / queries` (0.0 if `queries == 0`)
- `avg_similarity = sim_sum / hits` (0.0 if `hits == 0`) — mean score of *served hits*
- `avg_lookup_latency_ms = lookup_ms_sum / queries` (0.0 if `queries == 0`) — mean measured `/query` handling time
- `avg_store_search_ms = search_ms_sum / queries` (0.0 if `queries == 0`) — mean measured vector-store `search` time (the Redis KNN on the Redis backend)
- `estimated_latency_saved_ms = hits * assumptions.llm_ms`
- `estimated_cost_saved_usd = hits * assumptions.cost_usd`
- `estimated_tokens_saved = hits * assumptions.tokens_per_call`

### 5. `GET /{namespace}/stats`

404 if the namespace does not exist. `CacheService.stats(namespace)` resolves the
namespace, reads `metrics.snapshot`, `store.count`, and `compute_stats`. Response:

```json
{
  "entries": 128, "queries": 500, "hits": 412, "misses": 88,
  "hit_rate": 0.824, "avg_similarity": 0.91,
  "avg_lookup_latency_ms": 3.2, "avg_store_search_ms": 1.1,
  "estimated_latency_saved_ms": 329600,
  "estimated_cost_saved_usd": 0.412, "estimated_tokens_saved": 206000
}
```

### 6. Config (env-driven)

- `SEMCACHE_ASSUMED_LLM_MS` (default `800`)
- `SEMCACHE_ASSUMED_LLM_COST_USD` (default `0.001`)
- `SEMCACHE_ASSUMED_TOKENS_PER_CALL` (default `500`)

`app/dependencies.py` builds `StatsAssumptions` from these and injects it.

## Testing

- **Unit (in-memory):**
  - `record_query` increments hits/misses and accumulates `sim_sum`,
    `lookup_ms_sum`, `search_ms_sum`; `snapshot` returns zeros for an unseen
    namespace.
  - `compute_stats` math (including `avg_lookup_latency_ms`/`avg_store_search_ms`),
    with the `queries == 0` and `hits == 0` divide-by-zero guards.
  - `VectorStore.count` reflects upserts and deletes.
  - `query` records the outcome (hit with score; miss with `None`).
  - Query response includes correct `hit` and effective `threshold`
    (default and overridden).
  - `GET /{namespace}/stats` returns the full body; 404 for a missing namespace.
- **Integration (Redis):** `RedisMetricsStore` parity (`HINCRBY`/`HINCRBYFLOAT`,
  `HGETALL` snapshot, zeros when absent) and `count` via `FT.INFO num_docs`.
- **Examples:** `llm_cache_demo.py` prints `GET /{ns}/stats` at the end so a run
  visibly shows hit rate and latency saved.
