# SemCache

A generic semantic caching microservice for LLM applications. SemCache caches values by
**semantic similarity** (vector search over client-supplied embeddings) rather than
exact-match keys, so semantically equivalent requests can reuse a prior response.

It is deliberately generic: no RAG, agent, or framework logic lives in the core. Callers
compute their own embeddings and send the vectors in.

## Quick start

### Run it without cloning

Quick try (in-memory backend, zero infrastructure):

```bash
docker run -p 8000:8000 ghcr.io/sam2545/semcache
```

Full stack (API + Redis), using the published image:

```bash
curl -O https://raw.githubusercontent.com/Sam2545/SemanticCache/main/docker-compose.prod.yml
docker compose -f docker-compose.prod.yml up -d
```

### Run it from a clone

```bash
git clone https://github.com/Sam2545/SemanticCache.git && cd SemanticCache
./start.sh        # builds, waits until healthy, prints the URL  (or: make up)
```

Interactive API docs at <http://localhost:8000/docs>.

### Development

```bash
pytest -m "not integration"                          # fast unit suite (no Redis)
docker compose run --rm api pytest -m integration    # Redis contract tests
make down                                            # stop the stack
```

## Concepts

- **Namespace** — an isolation boundary with a fixed embedding `dimension`, declared up front.
  Carries default `threshold`/`top_k` and `ttl`.
- **Entry** — `{ key, embedding, value, metadata }`. `key` is the exact identity (upsert /
  get / delete); `embedding` is the similarity axis that queries search.
- **Query** — send an `embedding`; get back up to `top_k` entries scoring `>= threshold`
  (cosine, descending). A non-empty result is a cache **hit**.
- **Filter keys** — a namespace can declare `filter_keys` (e.g. `["model"]`).
  Queries pass a matching `filter` so responses from different models are cached
  and served separately. Different *embedding* models belong in separate
  namespaces (their vectors aren't comparable).

## API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/namespaces` | Create a namespace (fixed dimension + defaults) |
| `POST` | `/{namespace}/entries` | Upsert an entry by `key` |
| `POST` | `/{namespace}/query` | Similarity lookup by `embedding` |
| `GET`  | `/{namespace}/entries/{key}` | Fetch an entry exactly by `key` |
| `DELETE` | `/{namespace}/entries/{key}` | Invalidate an entry by `key` |
| `GET`  | `/{namespace}/stats` | Cache-effectiveness metrics for the namespace |

## Status

- Core service, API, and both `VectorStore` backends are implemented and tested:
  the in-process store (default; used for fast unit tests and local dev) and the
  Redis-stack store (`SEMCACHE_BACKEND=redis`, used by the container) with TTL expiry.
- Backend is selected by `SEMCACHE_BACKEND` (`memory` | `redis`); both behave
  identically behind the `VectorStore` protocol.
