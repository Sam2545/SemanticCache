# SemCache

A generic semantic caching microservice for LLM applications. SemCache caches values by
**semantic similarity** (vector search over client-supplied embeddings) rather than
exact-match keys, so semantically equivalent requests can reuse a prior response.

It is deliberately generic: no RAG, agent, or framework logic lives in the core. Callers
compute their own embeddings and send the vectors in.

## Quick start

```bash
docker compose up --build   # FastAPI on :8000 (Redis backend), Redis on :6379
pytest -m "not integration" # fast unit suite (no Redis required)
docker compose run --rm api pytest -m integration   # Redis contract tests
```

Interactive API docs at <http://localhost:8000/docs>.

## Concepts

- **Namespace** — an isolation boundary with a fixed embedding `dimension`, declared up front.
  Carries default `threshold`/`top_k` and `ttl`.
- **Entry** — `{ key, embedding, value, metadata }`. `key` is the exact identity (upsert /
  get / delete); `embedding` is the similarity axis that queries search.
- **Query** — send an `embedding`; get back up to `top_k` entries scoring `>= threshold`
  (cosine, descending). A non-empty result is a cache **hit**.

## API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/namespaces` | Create a namespace (fixed dimension + defaults) |
| `POST` | `/{namespace}/entries` | Upsert an entry by `key` |
| `POST` | `/{namespace}/query` | Similarity lookup by `embedding` |
| `GET`  | `/{namespace}/entries/{key}` | Fetch an entry exactly by `key` |
| `DELETE` | `/{namespace}/entries/{key}` | Invalidate an entry by `key` |

## Status

- Core service, API, and both `VectorStore` backends are implemented and tested:
  the in-process store (default; used for fast unit tests and local dev) and the
  Redis-stack store (`SEMCACHE_BACKEND=redis`, used by the container) with TTL expiry.
- Backend is selected by `SEMCACHE_BACKEND` (`memory` | `redis`); both behave
  identically behind the `VectorStore` protocol.
