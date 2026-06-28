# SemCache

A generic semantic caching microservice for LLM applications. SemCache caches values by
**semantic similarity** (vector search over client-supplied embeddings) rather than
exact-match keys, so semantically equivalent requests can reuse a prior response.

It is deliberately generic: no RAG, agent, or framework logic lives in the core. Callers
compute their own embeddings and send the vectors in.

## Quick start

There are **three ways to run SemCache** — all serve the same API on
<http://localhost:8000> (interactive docs at `/docs`). Pick one:

| # | Option | Needs | Best for |
|---|--------|-------|----------|
| 1 | Published image (no clone) | Docker | Fastest try — one `docker run`, no source |
| 2 | Docker from a clone | Docker + git | Full Redis-backed stack; local development |
| 3 | Docker-free (Python) | Python 3.12 | No Docker at all; isolated virtual environment |

### Option 1 — Published image (no clone)

Quick try (in-memory backend, zero infrastructure):

```bash
docker run -p 8000:8000 ghcr.io/sam2545/semcache
```

Full stack with Redis, still no clone (downloads only the compose file):

```bash
curl -O https://raw.githubusercontent.com/Sam2545/SemanticCache/main/docker-compose.prod.yml
docker compose -f docker-compose.prod.yml up -d
```

### Option 2 — Docker from a clone

Builds from source and runs the full API + Redis stack:

```bash
git clone https://github.com/Sam2545/SemanticCache.git && cd SemanticCache
./start.sh        # builds, waits until healthy, prints the URL  (or: make up)
```

### Option 3 — Docker-free (Python)

No Docker — a one-command script that creates its own virtual environment:

```bash
git clone https://github.com/Sam2545/SemanticCache.git && cd SemanticCache
./start-local.sh  # makes .venv, installs SemCache, serves on :8000  (or: make serve)
```

Or manage your own venv, without cloning:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install git+https://github.com/Sam2545/SemanticCache.git
semcache
```

Options 1 (`docker run`) and 3 use the in-memory backend (single-process, no Redis
persistence or TTL) — the zero-infra way to try it. Options 1 (compose) and 2 run
the Redis-backed stack.

Once it's running, open the interactive API docs at <http://localhost:8000/docs>.

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

## Usage

A full walkthrough against a running instance (`http://localhost:8000`). SemCache
never computes embeddings — your application does, and sends the vectors in. The
vectors below are tiny (3-dimensional) and hand-picked so the example is runnable
and deterministic; real apps send vectors from an embedding model (often hundreds
or thousands of dimensions).

### 1. Create a namespace

A namespace fixes the embedding `dimension` and the default similarity `threshold`.

```bash
curl -X POST http://localhost:8000/namespaces \
  -H 'content-type: application/json' \
  -d '{"name": "faqs", "dimension": 3, "default_threshold": 0.8}'
```

### 2. Add a vector embedding

Store an entry: `key` is its exact identity (re-using a key upserts in place),
`embedding` is the vector queries search against, `value` is what you get back,
and `metadata` is anything you want to carry along.

```bash
curl -X POST http://localhost:8000/faqs/entries \
  -H 'content-type: application/json' \
  -d '{
        "key": "capital-of-france",
        "embedding": [0.10, 0.20, 0.97],
        "value": "The capital of France is Paris.",
        "metadata": {"source": "geo-facts"}
      }'
```

Add a second, unrelated entry so search has something to discriminate against:

```bash
curl -X POST http://localhost:8000/faqs/entries \
  -H 'content-type: application/json' \
  -d '{
        "key": "speed-of-light",
        "embedding": [0.95, 0.15, 0.05],
        "value": "Light travels at about 299,792 km/s.",
        "metadata": {"source": "physics-facts"}
      }'
```

### 3. Query / search by similarity

Send a query vector. SemCache returns up to `top_k` entries scoring `>= threshold`
(cosine, highest first). A near-identical vector to the France entry hits:

```bash
curl -X POST http://localhost:8000/faqs/query \
  -H 'content-type: application/json' \
  -d '{"embedding": [0.11, 0.19, 0.98]}'
```

```json
{
  "matches": [
    {
      "key": "capital-of-france",
      "score": 0.9999,
      "value": "The capital of France is Paris.",
      "metadata": {"source": "geo-facts"}
    }
  ],
  "hit": true,
  "threshold": 0.8
}
```

A query vector that isn't close to anything scores below the threshold and misses
(`matches` empty, `hit` false):

```bash
curl -X POST http://localhost:8000/faqs/query \
  -H 'content-type: application/json' \
  -d '{"embedding": [0.20, 0.95, 0.10]}'
```

```json
{ "matches": [], "hit": false, "threshold": 0.8 }
```

Per-request overrides are supported: `"top_k": 3` to return more matches, or
`"threshold": 0.6` to loosen the cutoff for a single query.

### In Python (the client SDK)

The bundled `client` package wraps the embed → search → hit/miss → store loop so
you never write it by hand. Install it with `pip install -e ".[client]"`.

#### Provide an embedding function

SemCache never computes embeddings — you give the client an `embed` callable that
turns a string into a vector (`Callable[[str], list[float]]`). It's the only place
a model is involved, which is what keeps the cache generic. The vector length must
match the namespace `dimension`, and you must use the **same** model for writes and
queries. Pick whatever embedder you like:

```python
# OpenAI
from openai import OpenAI
oai = OpenAI()
def embed(text: str) -> list[float]:
    return oai.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding

# Local, no API (sentence-transformers)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
def embed(text: str) -> list[float]:
    return model.encode(text).tolist()

# Ollama (local)
import ollama
def embed(text: str) -> list[float]:
    return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]
```

#### Add a vector embedding, then search

`store()` embeds the text and saves an entry; `lookup()` embeds a query and returns
the closest stored value (or `None` on a miss). The namespace is auto-created on the
first call, sized to your embedder's output:

```python
from client import SemCache

cache = SemCache("http://localhost:8000", namespace="faqs", embed=embed)

# add a vector embedding (embed runs under the hood; key defaults to a hash of the text)
cache.store("What is the capital of France?", "The capital of France is Paris.")

# search by similarity
cache.lookup("Remind me, France's capital?")   # -> "The capital of France is Paris."  (hit)
cache.lookup("How do I sort a list in Python?") # -> None                                (miss)
```

#### Or wrap an LLM call so it caches itself

```python
@cache.cached
def answer(question: str) -> str:
    return call_my_llm(question)          # only runs on a cache miss

answer("What is the capital of France?")  # miss → calls the LLM, caches the answer
answer("Remind me, France's capital?")    # semantically similar → served from cache
```

## Status

- Core service, API, and both `VectorStore` backends are implemented and tested:
  the in-process store (default; used for fast unit tests and local dev) and the
  Redis-stack store (`SEMCACHE_BACKEND=redis`, used by the container) with TTL expiry.
- Backend is selected by `SEMCACHE_BACKEND` (`memory` | `redis`); both behave
  identically behind the `VectorStore` protocol.
