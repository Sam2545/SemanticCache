# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SemCache is a generic semantic caching microservice for LLM applications. It caches LLM
responses by semantic similarity (vector search over embeddings) rather than exact-match
keys, so semantically equivalent requests can reuse a prior response.

## Core Rules

- Keep SemCache generic. Do **not** bake RAG, LangChain, LlamaIndex, or agents into the core service.
- RAG and agent workflows belong in `examples/` or `integrations/`, never in the core.
- Core API fields stay generic: `namespace`, `key`, `value`, `metadata`, `threshold`, `top_k`.
- Docker-first development — the service runs and is developed via Docker Compose.
- Keep FastAPI routes thin: parse/validate request, call a service, return response.
- Put business logic in service modules.
- Put Redis / vector-search logic in `vectorstore/` or `cache/` modules.

### Embeddings

- **Embeddings are client-side.** Callers compute embeddings and send the vector with the
  request; the core never imports an embedding/model library and owns no embedding model.
- Validate vector dimension per namespace — a namespace has one fixed dimension; reject
  mismatched vectors.

### Data model & semantics

- A **namespace** is declared explicitly (`POST /namespaces`) with a fixed `dimension`,
  `metric` (cosine), and default `threshold`/`top_k`. Dimension is never inferred from first write.
- An **entry** is `{ key, embedding, value, metadata }`. `key` is the **exact identity**
  (unique per namespace, used for upsert / GET / DELETE); `embedding` is the **similarity axis**
  (what queries search). Two different keys may hold near-identical vectors.
- **Write** keys off `key` (idempotent upsert). **Query** keys off `embedding` and returns up to
  `top_k` entries scoring `>= threshold`, ordered descending. A **hit** = at least one such entry.
- GET/DELETE by `key` are exact, non-similarity operations for inspection and targeted invalidation.

### Vector store

- Services depend on an abstract `VectorStore` interface/protocol, **not** on Redis directly.
  Redis is one implementation behind that interface; no Redis types leak into service modules.
- Similarity is **cosine over normalized vectors**, fixed across the service, so `threshold`
  means the same thing in every namespace.

### Namespaces & lifecycle

- Namespaces are isolated: a lookup in one namespace never returns entries from another.
- Define TTL and eviction behavior explicitly per namespace; entries are not assumed to live forever.

### Config

- All configuration via environment variables (Redis URL, vector dimension, metric,
  default `threshold`/`top_k`). No hardcoded model or connection assumptions in the core.

### Testing

- Public endpoints must have tests.
- Test the **cache decision logic** in the service layer, not just HTTP: hit/miss,
  threshold boundary (just-above and just-below), `top_k` ordering, and namespace isolation.
- Tests supply their own known embedding vectors so similarity outcomes are deterministic —
  no real embedding model in the test path.

## Development Commands

```bash
docker compose up --build   # build and run the service (and dependencies, e.g. Redis)
pytest                      # run the test suite
```

## Architecture

Layering (request flows top to bottom; keep dependencies pointing one direction):

1. **FastAPI routes** (`app/routes/`) — thin HTTP layer. No business logic. Inject the service via `app.dependencies.get_service`.
2. **Service modules** (`app/services/cache.py`) — business logic: dimension validation, default resolution, and the threshold/top_k hit-miss decision. Depend on the `VectorStore` interface, not Redis.
3. **`app/vectorstore/`** — the `VectorStore` protocol (`base.py`) and its implementations. The store does raw KNN `search`; threshold cutoff lives in the service.

The generic core knows nothing about how callers produce embeddings or what they cache.
Callers send embedding vectors; the core deals only with namespaces, keys, values, metadata,
and cosine similarity (`threshold`, `top_k`).

Service-layer exceptions (`NamespaceNotFound`, `NamespaceExists`, `DimensionMismatch`) are
mapped to HTTP status codes by exception handlers in `app/main.py` (404 / 409 / 422).

### Current state

- `InMemoryVectorStore` is the default backend (deterministic, no external deps) and the test/dev double.
- **Not yet built:** the Redis-stack `VectorStore` implementation and TTL/eviction enforcement. Wire Redis in at `app.dependencies.get_service` — routes and the service need no changes.
- Run a single test: `pytest tests/test_cache_service.py::test_threshold_boundary_is_inclusive`.
