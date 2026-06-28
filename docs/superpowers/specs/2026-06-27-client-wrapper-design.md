# SemCache Client Wrapper & LangChain Adapter — Design

**Date:** 2026-06-27
**Status:** Approved

## Goal

Give developers a reusable wrapper so they can use SemCache without hand-writing
the embed → query → hit/miss → store loop on every call. Two layers: a generic
Python client SDK, and a thin LangChain cache adapter built on top of it. The
core service (`app/`) is not touched and gains no framework dependencies.

## Constraints (from CLAUDE.md)

- The core stays generic: no LangChain/RAG/agent code in `app/`. Framework
  coupling lives outside the core (`integrations/`); the generic client is a
  plain HTTP client and lives in `client/`.
- Embeddings stay client-side: the wrapper never imports an embedding model;
  the caller supplies an `embed` callable.
- Tests supply their own known embedding vectors; no real embedding model in the
  test path. Similarity outcomes stay deterministic.
- Commit messages: a single line under 150 characters, no `Co-Authored-By` trailer.

## Architecture & Layout

```
client/                         # generic SDK — pure HTTP client, no LLM/framework deps
  __init__.py
  semcache.py                   # SemCache class: sync + async, fail-open
  errors.py                     # SemCacheError
integrations/
  langchain_semcache.py         # LangChainSemCache(BaseCache) — thin adapter over the client
tests/
  test_client.py                # client vs in-process FastAPI app (httpx ASGITransport)
  test_langchain_adapter.py     # adapter mapping, importorskip("langchain_core")
  integration/
    test_langchain_rag.py       # real RAG pipeline → adapter → client → app → Redis
```

Key decisions:

- **Generic client in `client/`, not `integrations/`** — it is only an HTTP
  client for the service (no RAG, LangChain, or agents), so it does not violate
  the core rule. LangChain-specific code is isolated in `integrations/`.
- **Transport: `httpx`** — one dependency provides both a sync `Client` and an
  async `AsyncClient`. Already present as a dev dependency.
- **Optional dependencies** — `pyproject.toml` gains extras:
  `client = ["httpx"]` and `langchain = ["langchain-core"]`. The core service
  pulls in neither.
- **Embeddings client-supplied** — `embed` (and `aembed` for the async path) are
  constructor arguments. The client never imports an embedding model.

### Data flow for a wrapped call — `cached(llm)("...")`

```
text ──embed()──▶ vector ──POST /{ns}/query──▶ hit?
   ├─ hit  → return cached value          (no LLM call)
   └─ miss → llm(text) → POST /{ns}/entries → return result
on any SemCache HTTP/connection error at either step → log warning, fall through to llm()
```

## Generic Client API (`client/semcache.py`)

Two tiers: explicit primitives, and a convenience wrapper built on them. Every
method has an `a`-prefixed async twin sharing the same logic.

```python
class SemCache:
    def __init__(
        self,
        base_url: str,
        namespace: str,
        embed: Callable[[str], Sequence[float]],          # async twin: aembed for acached
        *,
        aembed: Callable[[str], Awaitable[Sequence[float]]] | None = None,
        model: str | None = None,        # tags metadata={"model": ...} + filters queries
        threshold: float | None = None,  # per-call override of namespace default
        top_k: int | None = None,
        fail_open: bool = True,           # False → raise SemCacheError instead of falling through
        timeout: float = 120.0,
        http_client: "httpx.Client | None" = None,    # test seam: inject sync client
        http_aclient: "httpx.AsyncClient | None" = None,  # test seam: inject async client
    ): ...

    # --- namespace setup (explicit; optional — auto-create handles the common case) ---
    def ensure_namespace(self, dimension: int, **defaults) -> None      # POST /namespaces, 409 = ok
    async def aensure_namespace(self, dimension: int, **defaults) -> None

    # --- primitives ---
    def lookup(self, text: str) -> str | None      # embed + POST /query; top match's value or None
    def store(self, text: str, value: str, *, key: str | None = None) -> None  # POST /entries
    async def alookup(self, text: str) -> str | None
    async def astore(self, text: str, value: str, *, key: str | None = None) -> None

    # --- convenience wrapper ---
    def cached(self, llm: Callable[[str], str]) -> Callable[[str], str]
    def acached(self, llm: Callable[[str], Awaitable[str]]) -> Callable[[str], Awaitable[str]]

    # --- model-aware view (used by the LangChain adapter and available standalone) ---
    def with_model(self, model: str) -> "SemCache"   # cheap copy sharing the HTTP client
```

Semantics:

- **`cached(llm)`** returns a wrapped `str -> str`: `lookup` → on hit return value
  (no LLM); on miss call `llm(text)`, `store`, return. Usable as a decorator
  (`@cache.cached`) or inline. `acached` mirrors it with `await`.
- **Key (exact identity):** defaults to `sha256(text).hexdigest()` — deterministic,
  fixed-length, Redis-safe, idempotent for repeated text. Overridable via
  `store(..., key=)`. The key is never consulted during similarity queries; it
  only governs upsert identity and GET/DELETE. Different prompt text → different
  key → separate entries that still find each other by vector similarity.
- **Model-aware:** when `model` is set, `store` tags `metadata={"model": model}`
  and `lookup` sends `filter={"model": model}`, so two models never share entries.
  Requires the namespace to declare `filter_keys=["model"]` (auto-create and
  `ensure_namespace` pass this through when `model` is set).
- **Auto-create on first use:** the first `lookup`/`store` measures the embedding
  dimension `D` and issues `POST /namespaces` with that `D` (plus
  `filter_keys=["model"]` when `model` is set, and any provided default
  threshold/top_k). 201 = created, 409 = already exists — both fine. A one-time
  guard on the shared session (see `with_model`) prevents re-posting on every
  call, including across model-scoped views. This is compatible with
  the core rule that the *service* never infers dimension from a write: the
  *client* declares a concrete measured `D`; the service still requires an
  explicit `POST /namespaces`. `ensure_namespace(dimension)` remains public for
  callers who want to pre-declare (e.g. set threshold up front).
- **Fail-open:** every HTTP/connection error inside `lookup`/`store` (and the
  auto-create POST) is caught, logged at WARNING, and treated as a miss / no-op,
  so a SemCache outage never breaks the app. With `fail_open=False` they raise
  `SemCacheError` instead.

## LangChain Adapter (`integrations/langchain_semcache.py`)

Implements LangChain's `BaseCache`; `set_llm_cache(...)` then routes every LLM
call through SemCache. The adapter owns only LangChain ↔ SemCache translation and
delegates all caching logic to the generic client.

```python
from langchain_core.caches import BaseCache
from langchain_core.outputs import Generation
RETURN_VAL_TYPE = Sequence[Generation]

class LangChainSemCache(BaseCache):
    def __init__(self, client: SemCache): self._client = client

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        value = self._client.with_model(llm_string).lookup(prompt)
        return [Generation(text=value)] if value is not None else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        self._client.with_model(llm_string).store(prompt, return_val[0].text)

    async def alookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None: ...
    async def aupdate(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None: ...

    def clear(self, **kwargs) -> None: ...   # optional; no-op for now
```

Mapping decisions:

- **`llm_string` → `model` filter.** LangChain's `llm_string` is a stable
  serialization of the model and its params; routing it through `with_model`
  makes different models/configs cache separately for free — the model-aware
  feature driven by LangChain's own identity string.
- **`prompt` → cached text.** Embedded for the vector, sha256'd for the key.
- **`Generation` wrap/unwrap.** SemCache stores a plain string; the adapter wraps
  it as `[Generation(text=value)]` outbound and reads `return_val[0].text`
  inbound (first generation in the common single-generation path).
- **Fail-open carries through:** a SemCache outage makes `lookup` return `None`
  (a LangChain cache miss), so LangChain just calls the model.

`with_model(llm_string)` returns a model-scoped view via a shallow copy that
overrides only the model. The copy shares one internal session object holding
the httpx clients and the auto-create guard, so model-scoped views reuse
connections and never re-POST the namespace. The LangChain adapter creates a
view on every `lookup`/`update`; sharing the session keeps that to one
namespace POST for the whole process.

## Testing

All deterministic: no real LLM, no real embedding model, known vectors only.

### `tests/test_client.py` — generic client against the real core, in-process

Drive the FastAPI `app` in-process — no socket, in-memory backend. Exercises
client + routes + service + in-memory store end-to-end. The sync path injects a
`fastapi.testclient.TestClient` (a sync `httpx.Client` over the ASGI app); the
async path injects `AsyncClient(transport=ASGITransport(app=app))` (sync
`httpx.Client` cannot drive `ASGITransport`).

```python
from fastapi.testclient import TestClient
client = SemCache(base_url="http://test", namespace="t", embed=embed,
                  http_client=TestClient(app, base_url="http://test"))
```

Cases:

- **auto-create on first use** — first call POSTs `/namespaces` with the measured
  dimension; a 409 on a second client is swallowed.
- **hit path** — `cached(llm)` on a stored paraphrase returns the cached value and
  `llm` is never called (assert call count 0).
- **miss path** — calls `llm` exactly once, stores, and a subsequent near-vector
  query hits.
- **idempotent key** — storing the same text twice yields one entry (`sha256`
  dedup), checked via `GET /{ns}/stats` entry count.
- **threshold boundary** — just-above hits, just-below misses.
- **model isolation** — two clients with different `model` values do not see each
  other's entries.
- **fail-open** — a dead transport / 500 still returns `llm()`'s result and logs a
  warning; with `fail_open=False` it raises `SemCacheError`.
- **async twins** — `acached` hit/miss via `AsyncClient` + `ASGITransport`.

### `tests/test_langchain_adapter.py` — adapter mapping in isolation

`importorskip("langchain_core")`; drive the adapter with a fake `SemCache` that
records calls and returns canned values. Asserts translation only (caching logic
is covered by `test_client.py`):

- `lookup` hit → `[Generation(text=...)]`; prompt passed through verbatim.
- `lookup` miss → `None`.
- `update` → unwraps `Generation` and calls `store(prompt, return_val[0].text)`.
- `llm_string` routed into `with_model` → model isolation.
- async `alookup`/`aupdate` mirror the sync ones.

### `tests/integration/test_langchain_rag.py` — real RAG pipeline

`@pytest.mark.integration` + `importorskip("langchain_core")`. Wires the real
path end-to-end: a LangChain RAG chain → `LangChainSemCache` → real `SemCache`
client → real FastAPI app → real Redis vector search. Stubbed only where project
convention already stubs: a deterministic `embed` (France prompts map to vectors
with cosine ≥ threshold) and a counting fake LLM. The app is reached via
a `fastapi.testclient.TestClient(app)` injected as the sync `http_client`, with
`SEMCACHE_BACKEND=redis` so the real service code and `RedisVectorStore` run
without a socket. (LangChain's `set_llm_cache` drives the adapter's sync path,
so the sync client is the one that matters here.)

Pipeline (real LangChain Runnables, minimal but genuine RAG shape):

```python
retriever = RunnableLambda(lambda q: FRANCE_DOCS)          # tiny fixed corpus
prompt    = PromptTemplate.from_template(
    "Context: {context}\n\nQuestion: {question}\nAnswer:")
llm       = CountingFakeLLM(responses=["Paris"])           # records call_count
set_llm_cache(LangChainSemCache(SemCache(base_url="http://test", namespace="rag",
                                         embed=stub_embed,
                                         http_client=TestClient(app, base_url="http://test"))))
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
```

Cases:

1. **Cold call (miss):** `chain.invoke("What is the capital of France?")` → lookup
   misses → fake LLM runs (`call_count == 1`) → update stores it.
2. **Paraphrase (semantic hit):** `chain.invoke("Which city is France's
   capital?")` → same context, slightly different rendered prompt → embedding
   ≥ threshold of the stored one → hit, `call_count still == 1`, same `"Paris"`.
   This is the assertion that proves semantic caching works in a RAG pipeline.
3. **Off-topic miss (control):** `chain.invoke("How do I sort a list in
   Python?")` → embedding below threshold → miss → `call_count == 2`.
4. **Stats check:** `GET /rag/stats` reports `hits == 1` after the three calls.

Dependency note: this needs `langchain_core` in the integration/container
environment, so the implementation adds it to the `langchain` extra and ensures
the integration image installs `.[langchain]`. The command stays
`docker compose run --rm api pytest -m integration`.

## Out of Scope (YAGNI)

- No real embedding model or real LLM in any test.
- No streaming-response caching (LangChain caches whole generations).
- No multi-generation caching beyond the first generation.
- No retry/backoff layer — fail-open covers the resilience requirement.
