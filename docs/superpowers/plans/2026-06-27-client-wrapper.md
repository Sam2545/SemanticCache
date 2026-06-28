# SemCache Client Wrapper & LangChain Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a generic Python client SDK (`client/`) that wraps the embed→query→hit/miss→store loop, plus a thin LangChain cache adapter (`integrations/`) built on it, without touching the core service.

**Architecture:** `client/semcache.py` holds a `SemCache` class that talks HTTP (via `httpx`) to the running service; the caller supplies an `embed` callable so no embedding model is imported. `integrations/langchain_semcache.py` implements LangChain's `BaseCache` by delegating to `SemCache`. Tests drive the real FastAPI `app` in-process (no socket) using an injected `fastapi.testclient.TestClient` (sync) or `AsyncClient(transport=ASGITransport(app))` (async).

**Tech Stack:** Python 3.12, httpx, FastAPI/Starlette TestClient, pytest, langchain-core (optional extra), Redis (integration test only).

## Global Constraints

- Core service (`app/`) is never modified or imported by the client; the client is a pure HTTP client. (CLAUDE.md)
- LangChain code lives only in `integrations/`; never in `app/` or `client/`. (CLAUDE.md)
- Embeddings are client-supplied via an `embed`/`aembed` callable; no embedding model is imported anywhere. (CLAUDE.md)
- Tests supply their own known embedding vectors; no real embedding model or real LLM in any test. (CLAUDE.md)
- Similarity is cosine; namespace default threshold is `0.8`, default top_k `5`. (`app/vectorstore/base.py`)
- Cache is fail-open by default: SemCache HTTP/connection errors are logged at WARNING and treated as miss/no-op; `fail_open=False` raises `SemCacheError`. Embedding-callable and LLM-callable errors are NOT swallowed.
- Default entry key is `sha256(text).hexdigest()`; overridable via `store(..., key=)`.
- `langchain-core` is an optional dependency; its tests use `pytest.importorskip("langchain_core")`.
- Commit messages: a single line under 150 characters, no `Co-Authored-By` trailer. (CLAUDE.md)

## HTTP contract (from `app/routes/cache.py`, `app/models/schemas.py`)

- `POST /namespaces` body `{name, dimension, default_threshold?, default_top_k?, ttl?, filter_keys?}` → `201` on create, `409` if it already exists.
- `POST /{ns}/entries` body `{key, embedding, value, metadata}` → `201`.
- `POST /{ns}/query` body `{embedding, threshold?, top_k?, filter?}` → `200` `{matches: [{key, score, value, metadata}], hit, threshold}`.
- `GET /{ns}/stats` → `200` `{entries, queries, hits, misses, hit_rate, ...}`.

## File Structure

- `client/__init__.py` — exports `SemCache`, `SemCacheError`.
- `client/errors.py` — `SemCacheError(Exception)`.
- `client/semcache.py` — `SemCache` class + internal `_Session`.
- `integrations/__init__.py` — empty package marker.
- `integrations/langchain_semcache.py` — `LangChainSemCache(BaseCache)`.
- `tests/test_client.py` — sync + async client tests against in-process app.
- `tests/test_langchain_adapter.py` — adapter mapping unit tests (fake client).
- `tests/integration/test_langchain_rag.py` — real RAG pipeline → adapter → client → app → Redis.
- `pyproject.toml` — add `client`/`langchain` extras, add `langchain-core` to `dev`, widen packaged dirs.
- `Dockerfile` — copy `client/` and `integrations/` into the image.

---

### Task 1: Package scaffolding, errors, packaging

**Files:**
- Create: `client/__init__.py`, `client/errors.py`, `integrations/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/test_client.py`

**Interfaces:**
- Produces: `from client import SemCacheError`; `class SemCacheError(Exception)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_client.py`:

```python
def test_semcache_error_is_exception():
    from client import SemCacheError

    assert issubclass(SemCacheError, Exception)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_client.py::test_semcache_error_is_exception -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'client'`

- [ ] **Step 3: Create the package files**

`client/errors.py`:

```python
from __future__ import annotations


class SemCacheError(Exception):
    """Raised when a SemCache operation fails and fail_open is disabled."""
```

`client/__init__.py`:

```python
from __future__ import annotations

from client.errors import SemCacheError
from client.semcache import SemCache

__all__ = ["SemCache", "SemCacheError"]
```

`integrations/__init__.py`:

```python
```

(empty file — package marker)

- [ ] **Step 4: Stub `client/semcache.py` so the import in `__init__` resolves**

Create `client/semcache.py` with a placeholder that Task 2 replaces:

```python
from __future__ import annotations


class SemCache:  # implemented in Task 2
    pass
```

- [ ] **Step 5: Widen packaging and add extras in `pyproject.toml`**

Change the `[project.optional-dependencies]` block and the packages-find include. Replace:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",
]
```

with:

```toml
[project.optional-dependencies]
client = [
    "httpx>=0.27",
]
langchain = [
    "langchain-core>=0.2",
]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",
    "langchain-core>=0.2",
]
```

And replace:

```toml
[tool.setuptools.packages.find]
include = ["app*"]
```

with:

```toml
[tool.setuptools.packages.find]
include = ["app*", "client*", "integrations*"]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_client.py::test_semcache_error_is_exception -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add client/ integrations/ pyproject.toml tests/test_client.py
git commit -m "Scaffold client package, SemCacheError, and packaging extras"
```

---

### Task 2: Sync `SemCache` core — primitives, auto-create, fail-open

**Files:**
- Modify: `client/semcache.py` (replace the Task 1 stub)
- Test: `tests/test_client.py`

**Interfaces:**
- Consumes: `client.errors.SemCacheError`; the HTTP contract above.
- Produces:
  - `SemCache(base_url, namespace, embed, *, aembed=None, model=None, threshold=None, top_k=None, fail_open=True, timeout=120.0, http_client=None, http_aclient=None)`
  - `ensure_namespace(dimension, *, default_threshold=None, default_top_k=None) -> None`
  - `lookup(text) -> str | None`
  - `store(text, value, *, key=None) -> None`
  - internal `_query(vector) -> str | None`, `_put(key, vector, value) -> None`, `_ensure(dimension)`, `_key(text)`, `_filter()`, `_metadata()`, `_filter_keys()`, `_fail(what, exc)`; attribute `self._session` (a `_Session`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_client.py` (top-of-file imports and a fixture, then the tests):

```python
import httpx
import pytest
from fastapi.testclient import TestClient

# known vectors → deterministic cosine similarity (namespace default threshold = 0.8)
VECS = {
    "capital of France?": [1.0, 0.0, 0.0],
    "France's capital, remind me?": [0.99, 0.01, 0.0],   # ~1.0 cosine vs the first
    "sort a list in Python?": [0.0, 1.0, 0.0],           # 0.0 cosine vs the first
    "cosine 0.8": [0.8, 0.6, 0.0],                        # exactly 0.8 cosine vs [1,0,0]
}


def embed(text):
    return VECS[text]


@pytest.fixture
def app_client():
    from app.dependencies import get_service

    get_service.cache_clear()                 # isolate the in-memory store per test
    from app.main import app

    with TestClient(app, base_url="http://test") as client:
        yield client
    get_service.cache_clear()


def make(app_client, **kwargs):
    from client import SemCache

    return SemCache(
        base_url="http://test",
        namespace=kwargs.pop("namespace", "t"),
        embed=embed,
        http_client=app_client,
        **kwargs,
    )


def test_auto_create_then_store_and_lookup_hit(app_client):
    cache = make(app_client)
    assert cache.lookup("capital of France?") is None          # miss, auto-creates ns
    cache.store("capital of France?", "Paris")
    assert cache.lookup("France's capital, remind me?") == "Paris"   # semantic hit


def test_lookup_miss_below_threshold(app_client):
    cache = make(app_client)
    cache.store("capital of France?", "Paris")
    assert cache.lookup("sort a list in Python?") is None


def test_idempotent_key_dedups(app_client):
    cache = make(app_client)
    cache.store("capital of France?", "Paris")
    cache.store("capital of France?", "Paris")                 # same text → same sha256 key
    stats = app_client.get("/t/stats").json()
    assert stats["entries"] == 1


def test_threshold_boundary_inclusive(app_client):
    at = make(app_client, namespace="at", threshold=0.8)
    at.store("capital of France?", "Paris")
    assert at.lookup("cosine 0.8") == "Paris"                  # 0.8 >= 0.8 → hit

    above = make(app_client, namespace="ab", threshold=0.81)
    above.store("capital of France?", "Paris")
    assert above.lookup("cosine 0.8") is None                  # 0.8 < 0.81 → miss


def test_fail_open_returns_miss_on_server_error():
    from client import SemCache

    def server_error(request):
        return httpx.Response(500, text="boom")

    dead = httpx.Client(transport=httpx.MockTransport(server_error), base_url="http://test")
    cache = SemCache("http://test", "t", embed, http_client=dead)
    assert cache.lookup("capital of France?") is None          # swallowed → miss
    cache.store("capital of France?", "Paris")                 # swallowed → no raise


def test_fail_closed_raises_on_server_error():
    from client import SemCache, SemCacheError

    def server_error(request):
        return httpx.Response(500, text="boom")

    dead = httpx.Client(transport=httpx.MockTransport(server_error), base_url="http://test")
    cache = SemCache("http://test", "t", embed, fail_open=False, http_client=dead)
    with pytest.raises(SemCacheError):
        cache.lookup("capital of France?")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_client.py -v`
Expected: the new tests FAIL (the `SemCache` stub has no `lookup`/`store`); `test_semcache_error_is_exception` still passes.

- [ ] **Step 3: Implement `client/semcache.py`**

Replace the entire file:

```python
from __future__ import annotations

import functools
import hashlib
import logging
from typing import Any, Awaitable, Callable, Sequence

import httpx

from client.errors import SemCacheError

logger = logging.getLogger("semcache")

Embed = Callable[[str], Sequence[float]]
AEmbed = Callable[[str], Awaitable[Sequence[float]]]


class _Session:
    """Shared HTTP clients and a one-time namespace-ensure guard.

    A SemCache and all of its with_model views share one _Session, so they
    reuse connections and POST the namespace at most once per process.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float,
        http_client: httpx.Client | None,
        http_aclient: httpx.AsyncClient | None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = http_client
        self._aclient = http_aclient
        self.ensured = False

    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client

    def aclient(self) -> httpx.AsyncClient:
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._aclient


class SemCache:
    """HTTP client wrapper around the SemCache service.

    Wraps the embed -> query -> hit/miss -> store loop. Embeddings are supplied
    by the caller (embed/aembed); this client imports no embedding model.
    """

    def __init__(
        self,
        base_url: str,
        namespace: str,
        embed: Embed,
        *,
        aembed: AEmbed | None = None,
        model: str | None = None,
        threshold: float | None = None,
        top_k: int | None = None,
        fail_open: bool = True,
        timeout: float = 120.0,
        http_client: httpx.Client | None = None,
        http_aclient: httpx.AsyncClient | None = None,
    ) -> None:
        self.namespace = namespace
        self._embed = embed
        self._aembed = aembed
        self._model = model
        self._threshold = threshold
        self._top_k = top_k
        self._fail_open = fail_open
        self._session = _Session(base_url, timeout, http_client, http_aclient)

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _filter(self) -> dict[str, str] | None:
        return {"model": self._model} if self._model is not None else None

    def _metadata(self) -> dict[str, str]:
        return {"model": self._model} if self._model is not None else {}

    def _filter_keys(self) -> list[str]:
        return ["model"] if self._model is not None else []

    def _fail(self, what: str, exc: Exception) -> None:
        if self._fail_open:
            logger.warning("SemCache %s failed (%s); continuing", what, exc)
            return None
        raise SemCacheError(f"SemCache {what} failed: {exc}") from exc

    def _query_body(self, vector: list[float]) -> dict[str, Any]:
        body: dict[str, Any] = {"embedding": vector}
        if self._threshold is not None:
            body["threshold"] = self._threshold
        if self._top_k is not None:
            body["top_k"] = self._top_k
        flt = self._filter()
        if flt is not None:
            body["filter"] = flt
        return body

    def _create_body(
        self, dimension: int, default_threshold: float | None, default_top_k: int | None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "name": self.namespace,
            "dimension": dimension,
            "filter_keys": self._filter_keys(),
        }
        if default_threshold is not None:
            body["default_threshold"] = default_threshold
        if default_top_k is not None:
            body["default_top_k"] = default_top_k
        return body

    # --- namespace creation ---------------------------------------------

    def ensure_namespace(
        self,
        dimension: int,
        *,
        default_threshold: float | None = None,
        default_top_k: int | None = None,
    ) -> None:
        self._create(dimension, default_threshold, default_top_k)

    def _create(
        self, dimension: int, default_threshold: float | None, default_top_k: int | None
    ) -> None:
        body = self._create_body(dimension, default_threshold, default_top_k)
        try:
            resp = self._session.client().post("/namespaces", json=body)
            if resp.status_code not in (201, 409):
                raise SemCacheError(f"create namespace {resp.status_code}: {resp.text}")
        except (httpx.HTTPError, SemCacheError) as exc:
            self._fail("ensure_namespace", exc)
            return
        self._session.ensured = True

    def _ensure(self, dimension: int) -> None:
        if self._session.ensured:
            return
        self._create(dimension, None, None)

    # --- sync primitives -------------------------------------------------

    def _query(self, vector: Sequence[float]) -> str | None:
        vector = list(vector)
        self._ensure(len(vector))
        try:
            resp = self._session.client().post(
                f"/{self.namespace}/query", json=self._query_body(vector)
            )
            if resp.status_code != 200:
                raise SemCacheError(f"query {resp.status_code}: {resp.text}")
            matches = resp.json().get("matches", [])
        except (httpx.HTTPError, SemCacheError) as exc:
            return self._fail("lookup", exc)
        return matches[0]["value"] if matches else None

    def _put(self, key: str, vector: Sequence[float], value: Any) -> None:
        vector = list(vector)
        self._ensure(len(vector))
        body = {"key": key, "embedding": vector, "value": value, "metadata": self._metadata()}
        try:
            resp = self._session.client().post(f"/{self.namespace}/entries", json=body)
            if resp.status_code != 201:
                raise SemCacheError(f"store {resp.status_code}: {resp.text}")
        except (httpx.HTTPError, SemCacheError) as exc:
            self._fail("store", exc)

    def lookup(self, text: str) -> str | None:
        return self._query(self._embed(text))

    def store(self, text: str, value: Any, *, key: str | None = None) -> None:
        self._put(key or self._key(text), self._embed(text), value)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_client.py -v`
Expected: all Task 1 + Task 2 tests PASS.

- [ ] **Step 5: Run the full fast suite to check for regressions**

Run: `pytest -m "not integration" -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add client/semcache.py tests/test_client.py
git commit -m "Add sync SemCache client: primitives, auto-create, fail-open"
```

---

### Task 3: `with_model` view and the `cached` wrapper (sync)

**Files:**
- Modify: `client/semcache.py`
- Test: `tests/test_client.py`

**Interfaces:**
- Consumes: Task 2 `SemCache`, `_query`, `_put`, `_key`, `self._session`.
- Produces:
  - `with_model(model) -> SemCache` (shallow copy sharing `self._session`, only `_model` changed)
  - `cached(llm: Callable[[str], str]) -> Callable[[str], str]`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_client.py`:

```python
def test_cached_hit_does_not_call_llm(app_client):
    cache = make(app_client)
    calls = {"n": 0}

    def llm(text):
        calls["n"] += 1
        return "Paris"

    wrapped = cache.cached(llm)
    assert wrapped("capital of France?") == "Paris"            # miss → calls llm
    assert wrapped("France's capital, remind me?") == "Paris"  # hit → no llm
    assert calls["n"] == 1


def test_cached_miss_calls_llm_once_and_stores(app_client):
    cache = make(app_client)
    calls = {"n": 0}

    def llm(text):
        calls["n"] += 1
        return "Paris"

    cache.cached(llm)("capital of France?")
    assert calls["n"] == 1
    assert cache.lookup("capital of France?") == "Paris"       # was stored


def test_model_isolation(app_client):
    a = make(app_client, namespace="m", model="model-a")
    b = make(app_client, namespace="m", model="model-b")
    a.store("capital of France?", "Paris-A")
    assert a.lookup("France's capital, remind me?") == "Paris-A"
    assert b.lookup("France's capital, remind me?") is None    # different model → isolated


def test_with_model_shares_ensure_guard(app_client):
    cache = make(app_client, namespace="share", model="m1")
    view = cache.with_model("m2")
    cache.store("capital of France?", "Paris")
    assert view._session is cache._session                     # shared session object
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_client.py -k "cached or model" -v`
Expected: FAIL with `AttributeError: 'SemCache' object has no attribute 'cached'`.

- [ ] **Step 3: Implement `with_model` and `cached`**

Add `import copy` to the imports at the top of `client/semcache.py` (place it with the stdlib imports):

```python
import copy
```

Append these methods to the `SemCache` class (after `store`):

```python
    # --- model-scoped view ----------------------------------------------

    def with_model(self, model: str) -> "SemCache":
        view = copy.copy(self)            # shares self._session (clients + ensure guard)
        view._model = model
        return view

    # --- convenience wrapper --------------------------------------------

    def cached(self, llm: Callable[[str], str]) -> Callable[[str], str]:
        @functools.wraps(llm)
        def wrapped(text: str) -> str:
            vector = list(self._embed(text))    # embed once, reuse for query + store
            hit = self._query(vector)
            if hit is not None:
                return hit
            result = llm(text)
            self._put(self._key(text), vector, result)
            return result

        return wrapped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_client.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add client/semcache.py tests/test_client.py
git commit -m "Add with_model view and cached() wrapper to SemCache client"
```

---

### Task 4: Async twins (`aembed`, `alookup`, `astore`, `aensure_namespace`, `acached`)

**Files:**
- Modify: `client/semcache.py`
- Test: `tests/test_client.py`

**Interfaces:**
- Consumes: Task 2/3 internals and `self._session.aclient()`.
- Produces:
  - `aensure_namespace(dimension, *, default_threshold=None, default_top_k=None)`
  - `alookup(text) -> str | None`, `astore(text, value, *, key=None)`
  - `acached(llm: Callable[[str], Awaitable[str]]) -> Callable[[str], Awaitable[str]]`
  - internal `_aquery(vector)`, `_aput(key, vector, value)`, `_aensure(dimension)`, `_aembed_call(text)`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_client.py` (uses `asyncio.run` so no async pytest plugin is needed):

```python
import asyncio


def _async_app_client():
    from app.dependencies import get_service

    get_service.cache_clear()
    from app.main import app

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def aembed(text):
    return VECS[text]


def test_acached_hit_does_not_call_llm():
    async def scenario():
        from client import SemCache

        calls = {"n": 0}

        async def llm(text):
            calls["n"] += 1
            return "Paris"

        async with _async_app_client() as ac:
            cache = SemCache(
                "http://test", "at", embed, aembed=aembed, http_aclient=ac
            )
            wrapped = cache.acached(llm)
            assert await wrapped("capital of France?") == "Paris"
            assert await wrapped("France's capital, remind me?") == "Paris"
            assert calls["n"] == 1

    asyncio.run(scenario())


def test_alookup_after_astore():
    async def scenario():
        from client import SemCache

        async with _async_app_client() as ac:
            cache = SemCache(
                "http://test", "at2", embed, aembed=aembed, http_aclient=ac
            )
            assert await cache.alookup("capital of France?") is None
            await cache.astore("capital of France?", "Paris")
            assert await cache.alookup("France's capital, remind me?") == "Paris"

    asyncio.run(scenario())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_client.py -k "acached or alookup" -v`
Expected: FAIL with `AttributeError: 'SemCache' object has no attribute 'acached'`.

- [ ] **Step 3: Implement the async methods**

Append to the `SemCache` class:

```python
    # --- async creation --------------------------------------------------

    async def aensure_namespace(
        self,
        dimension: int,
        *,
        default_threshold: float | None = None,
        default_top_k: int | None = None,
    ) -> None:
        await self._acreate(dimension, default_threshold, default_top_k)

    async def _acreate(
        self, dimension: int, default_threshold: float | None, default_top_k: int | None
    ) -> None:
        body = self._create_body(dimension, default_threshold, default_top_k)
        try:
            resp = await self._session.aclient().post("/namespaces", json=body)
            if resp.status_code not in (201, 409):
                raise SemCacheError(f"create namespace {resp.status_code}: {resp.text}")
        except (httpx.HTTPError, SemCacheError) as exc:
            self._fail("ensure_namespace", exc)
            return
        self._session.ensured = True

    async def _aensure(self, dimension: int) -> None:
        if self._session.ensured:
            return
        await self._acreate(dimension, None, None)

    async def _aembed_call(self, text: str) -> list[float]:
        if self._aembed is None:
            raise SemCacheError("async path requires aembed= in the SemCache constructor")
        return list(await self._aembed(text))

    # --- async primitives ------------------------------------------------

    async def _aquery(self, vector: Sequence[float]) -> str | None:
        vector = list(vector)
        await self._aensure(len(vector))
        try:
            resp = await self._session.aclient().post(
                f"/{self.namespace}/query", json=self._query_body(vector)
            )
            if resp.status_code != 200:
                raise SemCacheError(f"query {resp.status_code}: {resp.text}")
            matches = resp.json().get("matches", [])
        except (httpx.HTTPError, SemCacheError) as exc:
            return self._fail("lookup", exc)
        return matches[0]["value"] if matches else None

    async def _aput(self, key: str, vector: Sequence[float], value: Any) -> None:
        vector = list(vector)
        await self._aensure(len(vector))
        body = {"key": key, "embedding": vector, "value": value, "metadata": self._metadata()}
        try:
            resp = await self._session.aclient().post(f"/{self.namespace}/entries", json=body)
            if resp.status_code != 201:
                raise SemCacheError(f"store {resp.status_code}: {resp.text}")
        except (httpx.HTTPError, SemCacheError) as exc:
            self._fail("store", exc)

    async def alookup(self, text: str) -> str | None:
        return await self._aquery(await self._aembed_call(text))

    async def astore(self, text: str, value: Any, *, key: str | None = None) -> None:
        await self._aput(key or self._key(text), await self._aembed_call(text), value)

    def acached(
        self, llm: Callable[[str], Awaitable[str]]
    ) -> Callable[[str], Awaitable[str]]:
        @functools.wraps(llm)
        async def wrapped(text: str) -> str:
            vector = await self._aembed_call(text)
            hit = await self._aquery(vector)
            if hit is not None:
                return hit
            result = await llm(text)
            await self._aput(self._key(text), vector, result)
            return result

        return wrapped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_client.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full fast suite**

Run: `pytest -m "not integration" -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add client/semcache.py tests/test_client.py
git commit -m "Add async twins to SemCache client: alookup, astore, acached"
```

---

### Task 5: LangChain adapter + mapping unit tests

**Files:**
- Create: `integrations/langchain_semcache.py`
- Test: `tests/test_langchain_adapter.py`

**Interfaces:**
- Consumes: `client.semcache.SemCache` (`with_model`, `lookup`, `store`, `alookup`, `astore`).
- Produces: `LangChainSemCache(BaseCache)` with `lookup`, `update`, `alookup`, `aupdate`, `clear`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_langchain_adapter.py`:

```python
import asyncio

import pytest

pytest.importorskip("langchain_core")

from langchain_core.outputs import Generation  # noqa: E402

from integrations.langchain_semcache import LangChainSemCache  # noqa: E402


class FakeClient:
    """Records calls and returns canned values; stands in for SemCache."""

    def __init__(self, lookup_returns=None):
        self._lookup_returns = lookup_returns
        self.model = None
        self.stored = []
        self.looked_up = []

    def with_model(self, model):
        self.model = model
        return self

    def lookup(self, text):
        self.looked_up.append(text)
        return self._lookup_returns

    def store(self, text, value):
        self.stored.append((text, value))

    async def alookup(self, text):
        self.looked_up.append(text)
        return self._lookup_returns

    async def astore(self, text, value):
        self.stored.append((text, value))


def test_lookup_hit_wraps_in_generation():
    fake = FakeClient(lookup_returns="Paris")
    cache = LangChainSemCache(fake)
    result = cache.lookup("capital of France?", "openai:gpt-4:temp=0")
    assert result == [Generation(text="Paris")]
    assert fake.looked_up == ["capital of France?"]


def test_lookup_miss_returns_none():
    cache = LangChainSemCache(FakeClient(lookup_returns=None))
    assert cache.lookup("anything", "llm-x") is None


def test_update_unwraps_generation_and_stores():
    fake = FakeClient()
    cache = LangChainSemCache(fake)
    cache.update("capital of France?", "llm-x", [Generation(text="Paris")])
    assert fake.stored == [("capital of France?", "Paris")]


def test_llm_string_routed_to_model():
    fake = FakeClient(lookup_returns=None)
    LangChainSemCache(fake).lookup("q", "openai:gpt-4:temp=0")
    assert fake.model == "openai:gpt-4:temp=0"


def test_async_lookup_and_update():
    fake = FakeClient(lookup_returns="Paris")
    cache = LangChainSemCache(fake)

    async def scenario():
        assert await cache.alookup("q", "llm-x") == [Generation(text="Paris")]
        await cache.aupdate("q", "llm-x", [Generation(text="Paris")])

    asyncio.run(scenario())
    assert fake.stored == [("q", "Paris")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_langchain_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'integrations.langchain_semcache'` (or skip if `langchain_core` is not installed — install it with `pip install -e ".[dev]"`).

- [ ] **Step 3: Implement the adapter**

Create `integrations/langchain_semcache.py`:

```python
from __future__ import annotations

from typing import Any, Sequence

from langchain_core.caches import BaseCache
from langchain_core.outputs import Generation

from client.semcache import SemCache

RETURN_VAL_TYPE = Sequence[Generation]


class LangChainSemCache(BaseCache):
    """LangChain LLM cache backed by SemCache semantic similarity.

    Translates between LangChain's (prompt, llm_string, [Generation]) contract
    and the generic SemCache client. All caching logic lives in the client;
    this adapter only maps types and routes llm_string to the model filter so
    different models cache separately.
    """

    def __init__(self, client: SemCache) -> None:
        self._client = client

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        value = self._client.with_model(llm_string).lookup(prompt)
        return [Generation(text=value)] if value is not None else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        self._client.with_model(llm_string).store(prompt, return_val[0].text)

    async def alookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        value = await self._client.with_model(llm_string).alookup(prompt)
        return [Generation(text=value)] if value is not None else None

    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        await self._client.with_model(llm_string).astore(prompt, return_val[0].text)

    def clear(self, **kwargs: Any) -> None:
        """No-op: SemCache entries are managed per-namespace by the service."""
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_langchain_adapter.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add integrations/langchain_semcache.py tests/test_langchain_adapter.py
git commit -m "Add LangChain BaseCache adapter over SemCache client"
```

---

### Task 6: RAG integration test + container dependencies

**Files:**
- Create: `tests/integration/test_langchain_rag.py`
- Modify: `Dockerfile`
- Test: `tests/integration/test_langchain_rag.py` (run in the container)

**Interfaces:**
- Consumes: `app.main.app`, `app.config.settings`, `app.dependencies.get_service`, `integrations.langchain_semcache.LangChainSemCache`, `client.semcache.SemCache`, the `redis_url` fixture from `tests/integration/conftest.py`.

- [ ] **Step 1: Copy `client/` and `integrations/` into the image**

In `Dockerfile`, after `COPY app ./app`, add two lines so the packages exist in the image (the `.[dev]` install already adds `langchain-core` via Task 1). Replace:

```dockerfile
COPY pyproject.toml ./
COPY app ./app
COPY tests ./tests
RUN pip install --no-cache-dir -e ".[dev]"
```

with:

```dockerfile
COPY pyproject.toml ./
COPY app ./app
COPY client ./client
COPY integrations ./integrations
COPY tests ./tests
RUN pip install --no-cache-dir -e ".[dev]"
```

- [ ] **Step 2: Write the integration test**

Create `tests/integration/test_langchain_rag.py`:

```python
import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("langchain_core")

from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.language_models.llms import LLM  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402
from langchain_core.runnables import RunnableLambda, RunnablePassthrough  # noqa: E402

from integrations.langchain_semcache import LangChainSemCache  # noqa: E402

FRANCE_DOCS = "France is a country in Europe. Its capital city is Paris."


def stub_embed(text):
    t = text.lower()
    if "france" in t or "french" in t:
        return [1.0, 0.0, 0.0]
    if "python" in t or "sort" in t:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


class CountingLLM(LLM):
    answer: str = "Paris"
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "counting"

    def _call(self, prompt, stop=None, run_manager=None, **kwargs) -> str:
        self.calls += 1
        return self.answer


@pytest.fixture
def rag(redis_url):
    redis = pytest.importorskip("redis")
    client = redis.Redis.from_url(redis_url)
    try:
        client.ping()
    except redis.exceptions.RedisError:
        pytest.skip("redis-stack not reachable")
    client.flushall()

    from app.config import settings
    from app.dependencies import get_service

    settings.backend = "redis"
    settings.redis_url = redis_url
    get_service.cache_clear()

    from app.main import app
    from client import SemCache
    from langchain_core.globals import set_llm_cache

    test_client = TestClient(app, base_url="http://test")
    set_llm_cache(
        LangChainSemCache(
            SemCache("http://test", "rag", stub_embed, http_client=test_client)
        )
    )
    yield test_client

    set_llm_cache(None)
    get_service.cache_clear()
    client.flushall()


def build_chain(llm):
    retriever = RunnableLambda(lambda q: FRANCE_DOCS)
    prompt = PromptTemplate.from_template(
        "Context: {context}\n\nQuestion: {question}\nAnswer:"
    )
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm


def test_rag_semantic_cache_hit_skips_llm(rag):
    llm = CountingLLM()
    chain = build_chain(llm)

    assert chain.invoke("What is the capital of France?") == "Paris"   # cold miss
    assert llm.calls == 1

    assert chain.invoke("Which city is France's capital?") == "Paris"  # semantic hit
    assert llm.calls == 1                                              # llm NOT called again

    assert chain.invoke("How do I sort a list in Python?") == "Paris"  # off-topic miss
    assert llm.calls == 2

    stats = rag.get("/rag/stats").json()
    assert stats["hits"] == 1
```

- [ ] **Step 3: Run the integration test in the container**

Run: `docker compose run --rm api pytest tests/integration/test_langchain_rag.py -v`
Expected: PASS (the semantic-hit assertion `llm.calls == 1` after the paraphrase proves caching).

Note: if Docker is unavailable in your environment, verify the file compiles with `python -m py_compile tests/integration/test_langchain_rag.py` and run the fast suite; the container run is the real gate.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_langchain_rag.py Dockerfile
git commit -m "Add LangChain RAG integration test and package the client in the image"
```

---

## Final verification

- [ ] Run the full fast suite: `pytest -m "not integration" -q` — all pass.
- [ ] Run the integration suite in the container: `docker compose run --rm api pytest -m integration -v` — all pass.
- [ ] Confirm `app/` was not modified by any task (`git diff --stat <base> -- app/` is empty).
