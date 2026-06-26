# Model-aware Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a query scope cache matches to entries sharing declared metadata (e.g. generation `model`), so responses from different models are cached and served separately.

**Architecture:** A namespace declares immutable `filter_keys`. Queries pass a `filter` (conjunctive exact-match on those keys), applied as a pre-filter inside each `VectorStore` so KNN ranks within the matching subset. In-memory uses dict equality; Redis uses a hybrid `(@flt_<key>:{val})=>[KNN ...]` query with a TAG field per declared key. The core stays generic — it filters declared metadata keys and never learns "model."

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, redis-py (RediSearch), numpy, pytest.

## Global Constraints

- Python ≥ 3.12.
- The in-memory and Redis backends MUST behave identically behind the `VectorStore` protocol. Filter values are compared as **strings** on both backends.
- No Redis types leak into `app/services/` — filtering is expressed through the `VectorStore` interface.
- Filter matching is **conjunctive exact-match** on the keys given; an entry missing a filtered key is excluded; extra entry metadata is ignored; an empty/absent filter matches everything (backward-compatible).
- A query filter key not in the namespace's `filter_keys` raises `InvalidFilter` → HTTP 422.
- `filter_keys` is declared at namespace creation and is immutable (like `dimension`).
- Tests supply their own known embedding vectors (deterministic; no real model in the test path).
- Commit messages: a single line under 150 characters, no `Co-Authored-By` trailer.

---

## File Structure

- `app/vectorstore/base.py` — add `filter_keys` to `Namespace`; add `filter` param to `VectorStore.search`.
- `app/vectorstore/memory_store.py` — pre-filter in `search`.
- `app/vectorstore/redis_store.py` — persist `filter_keys`, index TAG fields, write `flt_<key>` hash fields, hybrid `search`.
- `app/services/cache.py` — `InvalidFilter`; `filter_keys` on create; `filter` validation + delegation on query.
- `app/models/schemas.py` — `filter_keys` on namespace request/response; `filter` on query request.
- `app/routes/cache.py` — pass `filter_keys` and `filter` through.
- `app/main.py` — map `InvalidFilter` → 422.
- `tests/test_memory_store.py`, `tests/test_cache_service.py`, `tests/test_api.py`, `tests/integration/test_redis_store.py` — coverage.
- `examples/llm_cache_demo.py` — declare/tag/filter by `model`.
- `CLAUDE.md`, `README.md` — document the feature.

---

### Task 1: `Namespace.filter_keys`

**Files:**
- Modify: `app/vectorstore/base.py`
- Test: `tests/test_memory_store.py`

**Interfaces:**
- Produces: `Namespace(..., filter_keys: list[str] = [])` — a new field consumed by every later task.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_memory_store.py`:

```python
def test_namespace_round_trips_filter_keys():
    s = InMemoryVectorStore()
    s.create_namespace(Namespace(name="ns", dimension=2, filter_keys=["model"]))
    assert s.get_namespace("ns").filter_keys == ["model"]


def test_namespace_filter_keys_defaults_empty():
    s = InMemoryVectorStore()
    s.create_namespace(Namespace(name="ns", dimension=2))
    assert s.get_namespace("ns").filter_keys == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_memory_store.py::test_namespace_round_trips_filter_keys -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'filter_keys'`.

- [ ] **Step 3: Add the field**

In `app/vectorstore/base.py`, in the `Namespace` dataclass, add after `ttl`:

```python
    ttl: int | None = None
    filter_keys: list[str] = field(default_factory=list)
```

(`field` is already imported in this module.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_memory_store.py -v`
Expected: PASS (new + existing).

- [ ] **Step 5: Commit**

```bash
git add app/vectorstore/base.py tests/test_memory_store.py
git commit -m "Add filter_keys field to Namespace"
```

---

### Task 2: In-memory `search` pre-filter

**Files:**
- Modify: `app/vectorstore/base.py` (protocol signature)
- Modify: `app/vectorstore/memory_store.py`
- Test: `tests/test_memory_store.py`

**Interfaces:**
- Produces: `VectorStore.search(namespace, embedding, top_k, filter: dict[str, object] | None = None) -> list[ScoredEntry]`. `filter` is a pre-filter (conjunctive string equality; missing key excludes the entry).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_memory_store.py`:

```python
def test_search_filters_by_metadata(store):
    store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A", metadata={"model": "x"}))
    store.upsert("ns", StoredEntry(key="b", embedding=[1.0, 0.0], value="B", metadata={"model": "y"}))
    results = store.search("ns", [1.0, 0.0], top_k=10, filter={"model": "x"})
    assert [r.key for r in results] == ["a"]


def test_search_without_filter_returns_all(store):
    store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A", metadata={"model": "x"}))
    store.upsert("ns", StoredEntry(key="b", embedding=[1.0, 0.0], value="B", metadata={"model": "y"}))
    assert len(store.search("ns", [1.0, 0.0], top_k=10)) == 2


def test_search_filter_is_conjunctive(store):
    store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A", metadata={"model": "x", "lang": "en"}))
    store.upsert("ns", StoredEntry(key="b", embedding=[1.0, 0.0], value="B", metadata={"model": "x", "lang": "fr"}))
    results = store.search("ns", [1.0, 0.0], top_k=10, filter={"model": "x", "lang": "en"})
    assert [r.key for r in results] == ["a"]


def test_search_filter_excludes_entry_missing_key(store):
    store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A", metadata={}))
    assert store.search("ns", [1.0, 0.0], top_k=10, filter={"model": "x"}) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_memory_store.py::test_search_filters_by_metadata -v`
Expected: FAIL — `TypeError: search() got an unexpected keyword argument 'filter'`.

- [ ] **Step 3: Update the protocol signature**

In `app/vectorstore/base.py`, change the `VectorStore.search` line to:

```python
    def search(
        self,
        namespace: str,
        embedding: Sequence[float],
        top_k: int,
        filter: dict[str, object] | None = None,
    ) -> list[ScoredEntry]: ...
```

- [ ] **Step 4: Implement the in-memory filter**

In `app/vectorstore/memory_store.py`, replace the `search` method with:

```python
    def search(
        self,
        namespace: str,
        embedding: Sequence[float],
        top_k: int,
        filter: dict[str, object] | None = None,
    ) -> list[ScoredEntry]:
        flt = filter or {}
        scored = [
            ScoredEntry(
                key=e.key,
                score=cosine_similarity(embedding, e.embedding),
                value=e.value,
                metadata=e.metadata,
            )
            for e in self._entries[namespace].values()
            if all(k in e.metadata and str(e.metadata[k]) == str(v) for k, v in flt.items())
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_memory_store.py -v`
Expected: PASS (new + existing).

- [ ] **Step 6: Commit**

```bash
git add app/vectorstore/base.py app/vectorstore/memory_store.py tests/test_memory_store.py
git commit -m "Add metadata pre-filter to VectorStore.search (in-memory)"
```

---

### Task 3: Service layer — `filter_keys`, `filter`, `InvalidFilter`

**Files:**
- Modify: `app/services/cache.py`
- Test: `tests/test_cache_service.py`

**Interfaces:**
- Consumes: `Namespace.filter_keys`; `store.search(..., filter=...)`.
- Produces:
  - `CacheService.create_namespace(name, dimension, default_threshold=None, default_top_k=None, ttl=None, filter_keys: list[str] | None = None) -> Namespace`
  - `CacheService.query(namespace, embedding, threshold=None, top_k=None, filter: dict | None = None) -> list[ScoredEntry]`
  - `class InvalidFilter(CacheError)`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cache_service.py` (extend the existing import to include `InvalidFilter`):

```python
from app.services.cache import (
    CacheService,
    DimensionMismatch,
    InvalidFilter,
    NamespaceExists,
    NamespaceNotFound,
)


def test_create_namespace_stores_filter_keys(service):
    ns = service.create_namespace("ns", dimension=2, filter_keys=["model"])
    assert ns.filter_keys == ["model"]
    assert service.get_namespace("ns").filter_keys == ["model"]


def test_query_filter_returns_only_matching_model(service):
    service.create_namespace("ns", dimension=2, filter_keys=["model"])
    service.put("ns", key="a", embedding=[1.0, 0.0], value="A", metadata={"model": "x"})
    service.put("ns", key="b", embedding=[1.0, 0.0], value="B", metadata={"model": "y"})
    matches = service.query("ns", embedding=[1.0, 0.0], threshold=0.0, filter={"model": "x"})
    assert [m.key for m in matches] == ["a"]


def test_query_without_filter_returns_all_models(service):
    service.create_namespace("ns", dimension=2, filter_keys=["model"])
    service.put("ns", key="a", embedding=[1.0, 0.0], value="A", metadata={"model": "x"})
    service.put("ns", key="b", embedding=[1.0, 0.0], value="B", metadata={"model": "y"})
    assert len(service.query("ns", embedding=[1.0, 0.0], threshold=0.0)) == 2


def test_query_undeclared_filter_key_raises(service):
    service.create_namespace("ns", dimension=2, filter_keys=["model"])
    with pytest.raises(InvalidFilter):
        service.query("ns", embedding=[1.0, 0.0], filter={"temperature": "0"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cache_service.py::test_query_undeclared_filter_key_raises -v`
Expected: FAIL — `ImportError: cannot import name 'InvalidFilter'`.

- [ ] **Step 3: Add the exception**

In `app/services/cache.py`, after the `DimensionMismatch` class:

```python
class InvalidFilter(CacheError):
    pass
```

- [ ] **Step 4: Thread `filter_keys` through `create_namespace`**

In `app/services/cache.py`, update `create_namespace` signature and body:

```python
    def create_namespace(
        self,
        name: str,
        dimension: int,
        default_threshold: float | None = None,
        default_top_k: int | None = None,
        ttl: int | None = None,
        filter_keys: list[str] | None = None,
    ) -> Namespace:
        if self._store.get_namespace(name) is not None:
            raise NamespaceExists(name)
        ns = Namespace(name=name, dimension=dimension)
        if default_threshold is not None:
            ns.default_threshold = default_threshold
        if default_top_k is not None:
            ns.default_top_k = default_top_k
        ns.ttl = ttl
        if filter_keys is not None:
            ns.filter_keys = filter_keys
        self._store.create_namespace(ns)
        return ns
```

- [ ] **Step 5: Add filter validation + delegation to `query`**

In `app/services/cache.py`, replace `query` with:

```python
    def query(
        self,
        namespace: str,
        embedding: Sequence[float],
        threshold: float | None = None,
        top_k: int | None = None,
        filter: dict[str, object] | None = None,
    ) -> list[ScoredEntry]:
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
        candidates = self._store.search(namespace, embedding, top_k, flt)
        return [c for c in candidates if c.score >= threshold]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_cache_service.py -v`
Expected: PASS (new + existing).

- [ ] **Step 7: Commit**

```bash
git add app/services/cache.py tests/test_cache_service.py
git commit -m "Add filter_keys and query filter with InvalidFilter to CacheService"
```

---

### Task 4: API layer — schemas, routes, 422 handler

**Files:**
- Modify: `app/models/schemas.py`
- Modify: `app/routes/cache.py`
- Modify: `app/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `CacheService.create_namespace(filter_keys=...)`, `CacheService.query(filter=...)`, `InvalidFilter`.
- Produces: request/response fields `filter_keys` (namespace) and `filter` (query); HTTP 422 for `InvalidFilter`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_api.py`:

```python
def test_create_namespace_with_filter_keys(client):
    r = client.post("/namespaces", json={"name": "ns", "dimension": 2, "filter_keys": ["model"]})
    assert r.status_code == 201
    assert r.json()["filter_keys"] == ["model"]


def test_query_with_filter_scopes_to_model(client):
    client.post("/namespaces", json={"name": "ns", "dimension": 2, "default_threshold": 0.0, "filter_keys": ["model"]})
    client.post("/ns/entries", json={"key": "a", "embedding": [1.0, 0.0], "value": "A", "metadata": {"model": "x"}})
    client.post("/ns/entries", json={"key": "b", "embedding": [1.0, 0.0], "value": "B", "metadata": {"model": "y"}})
    r = client.post("/ns/query", json={"embedding": [1.0, 0.0], "filter": {"model": "x"}})
    assert r.status_code == 200
    assert [m["key"] for m in r.json()["matches"]] == ["a"]


def test_query_undeclared_filter_key_unprocessable(client):
    client.post("/namespaces", json={"name": "ns", "dimension": 2, "filter_keys": ["model"]})
    r = client.post("/ns/query", json={"embedding": [1.0, 0.0], "filter": {"temperature": "0"}})
    assert r.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api.py::test_create_namespace_with_filter_keys -v`
Expected: FAIL — response has no `filter_keys` key (KeyError) / 422 not returned.

- [ ] **Step 3: Add schema fields**

In `app/models/schemas.py`:

`CreateNamespaceRequest` — add:
```python
    filter_keys: list[str] = Field(default_factory=list)
```

`NamespaceResponse` — add:
```python
    filter_keys: list[str] = Field(default_factory=list)
```

`QueryRequest` — add:
```python
    filter: dict[str, str | int | float | bool] | None = None
```

- [ ] **Step 4: Pass fields through the routes**

In `app/routes/cache.py`, in `create_namespace`, pass `filter_keys` and include it in the response:

```python
    ns = service.create_namespace(
        name=body.name,
        dimension=body.dimension,
        default_threshold=body.default_threshold,
        default_top_k=body.default_top_k,
        ttl=body.ttl,
        filter_keys=body.filter_keys,
    )
    return NamespaceResponse(
        name=ns.name,
        dimension=ns.dimension,
        metric=ns.metric,
        default_threshold=ns.default_threshold,
        default_top_k=ns.default_top_k,
        ttl=ns.ttl,
        filter_keys=ns.filter_keys,
    )
```

In `query`, pass the filter:

```python
    matches = service.query(
        namespace=namespace,
        embedding=body.embedding,
        threshold=body.threshold,
        top_k=body.top_k,
        filter=body.filter,
    )
```

- [ ] **Step 5: Map `InvalidFilter` to 422**

In `app/main.py`, extend the import and add a handler:

```python
from app.services.cache import (
    DimensionMismatch,
    InvalidFilter,
    NamespaceExists,
    NamespaceNotFound,
)


@app.exception_handler(InvalidFilter)
def _invalid_filter(request: Request, exc: InvalidFilter) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_api.py -v`
Expected: PASS (new + existing).

- [ ] **Step 7: Commit**

```bash
git add app/models/schemas.py app/routes/cache.py app/main.py tests/test_api.py
git commit -m "Expose filter_keys and query filter in the HTTP API (422 on undeclared key)"
```

---

### Task 5: Redis backend — TAG fields, hash fields, hybrid search

**Files:**
- Modify: `app/vectorstore/redis_store.py`
- Test: `tests/integration/test_redis_store.py`

**Interfaces:**
- Consumes: `Namespace.filter_keys`; `search(..., filter=...)`.
- Produces: Redis backend honoring the same filter contract; persists `filter_keys`; indexes `flt_<key>` TAG fields; pre-filters KNN.

- [ ] **Step 1: Write the failing integration tests**

Add to `tests/integration/test_redis_store.py`:

```python
def test_namespace_round_trips_filter_keys(redis_store):
    redis_store.create_namespace(
        Namespace(name="ns", dimension=2, filter_keys=["model", "embed_model"])
    )
    assert redis_store.get_namespace("ns").filter_keys == ["model", "embed_model"]


def test_search_filters_by_metadata(redis_store):
    redis_store.create_namespace(Namespace(name="ns", dimension=2, filter_keys=["model"]))
    redis_store.upsert("ns", StoredEntry(key="a", embedding=[1.0, 0.0], value="A", metadata={"model": "gpt-oss:120b"}))
    redis_store.upsert("ns", StoredEntry(key="b", embedding=[1.0, 0.0], value="B", metadata={"model": "minimax-m3"}))
    results = redis_store.search("ns", [1.0, 0.0], top_k=10, filter={"model": "gpt-oss:120b"})
    assert [r.key for r in results] == ["a"]


def test_filter_prefilters_before_knn(redis_store):
    # The wanted-model entry is ranked BELOW several other-model entries; a
    # pre-filter must still surface it (a post-filter would drop it).
    redis_store.create_namespace(Namespace(name="ns", dimension=2, filter_keys=["model"]))
    redis_store.upsert("ns", StoredEntry(key="b1", embedding=[1.0, 0.00], value=1, metadata={"model": "B"}))
    redis_store.upsert("ns", StoredEntry(key="b2", embedding=[1.0, 0.01], value=2, metadata={"model": "B"}))
    redis_store.upsert("ns", StoredEntry(key="b3", embedding=[1.0, 0.02], value=3, metadata={"model": "B"}))
    redis_store.upsert("ns", StoredEntry(key="a1", embedding=[0.8, 0.6], value=4, metadata={"model": "A"}))
    results = redis_store.search("ns", [1.0, 0.0], top_k=1, filter={"model": "A"})
    assert [r.key for r in results] == ["a1"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run (needs redis-stack): `docker compose run --rm api pytest tests/integration/test_redis_store.py::test_search_filters_by_metadata -v`
Expected: FAIL — `search()` rejects `filter`, or `get_namespace` has no `filter_keys`.

- [ ] **Step 3: Add the TAG-escape helper and imports**

In `app/vectorstore/redis_store.py`, add near the top (after the existing imports):

```python
import re

_TAG_SPECIAL = re.compile(r"([,.<>{}\[\]\"':;!@#$%^&*()\-+=~|/\\ ])")


def _escape_tag(value: str) -> str:
    """Escape RediSearch TAG special characters for exact-match queries."""
    return _TAG_SPECIAL.sub(r"\\\1", value)
```

- [ ] **Step 4: Persist and read back `filter_keys`**

In `create_namespace`, add `filter_keys` to the metadata mapping:

```python
        self._client.hset(
            self._ns_key(namespace.name),
            mapping={
                "dimension": namespace.dimension,
                "metric": namespace.metric,
                "default_threshold": namespace.default_threshold,
                "default_top_k": namespace.default_top_k,
                "ttl": "" if namespace.ttl is None else namespace.ttl,
                "filter_keys": json.dumps(namespace.filter_keys),
            },
        )
        self._ensure_index(namespace)
```

In `get_namespace`, add `filter_keys` to the returned `Namespace`:

```python
        return Namespace(
            name=name,
            dimension=int(d["dimension"]),
            metric=d["metric"],
            default_threshold=float(d["default_threshold"]),
            default_top_k=int(d["default_top_k"]),
            ttl=int(d["ttl"]) if d["ttl"] != "" else None,
            filter_keys=json.loads(d["filter_keys"]) if "filter_keys" in d else [],
        )
```

- [ ] **Step 5: Index TAG fields for declared keys**

In `_ensure_index`, add the TAG fields to the schema tuple (before `VectorField`):

```python
        schema = (
            TagField("key"),
            TextField("value"),
            TextField("metadata"),
            *(TagField(f"flt_{k}") for k in namespace.filter_keys),
            VectorField(
                "embedding",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": namespace.dimension,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        )
```

- [ ] **Step 6: Materialize `flt_<key>` hash fields on upsert**

Replace `upsert` with (fetch namespace first, then build the mapping):

```python
    def upsert(self, namespace: str, entry: StoredEntry) -> None:
        ns = self.get_namespace(namespace)
        doc_key = self._doc_key(namespace, entry.key)
        mapping: dict[str, object] = {
            "key": entry.key,
            "embedding": np.asarray(entry.embedding, dtype=np.float32).tobytes(),
            "value": json.dumps(entry.value),
            "metadata": json.dumps(entry.metadata),
        }
        if ns:
            for k in ns.filter_keys:
                if k in entry.metadata:
                    mapping[f"flt_{k}"] = str(entry.metadata[k])
        self._client.hset(doc_key, mapping=mapping)
        if ns and ns.ttl:
            self._client.expire(doc_key, ns.ttl)
```

- [ ] **Step 7: Build the hybrid filtered KNN query**

Replace `search` with:

```python
    def search(
        self,
        namespace: str,
        embedding: Sequence[float],
        top_k: int,
        filter: dict[str, object] | None = None,
    ) -> list[ScoredEntry]:
        flt = filter or {}
        blob = np.asarray(embedding, dtype=np.float32).tobytes()
        if flt:
            clause = " ".join(
                f"@flt_{k}:{{{_escape_tag(str(v))}}}" for k, v in flt.items()
            )
            prefix = f"({clause})"
        else:
            prefix = "*"
        query = (
            Query(f"{prefix}=>[KNN {top_k} @embedding $vec AS distance]")
            .sort_by("distance")
            .return_fields("key", "value", "metadata", "distance")
            .dialect(2)
        )
        result = self._client.ft(self._index(namespace)).search(
            query, query_params={"vec": blob}
        )
        return [
            ScoredEntry(
                key=_decode(doc.key),
                score=1.0 - float(_decode(doc.distance)),
                value=json.loads(_decode(doc.value)),
                metadata=json.loads(_decode(doc.metadata)),
            )
            for doc in result.docs
        ]
```

- [ ] **Step 8: Run integration tests to verify they pass**

Run: `docker compose run --rm api pytest tests/integration/test_redis_store.py -v`
Expected: PASS (new + existing).

- [ ] **Step 9: Commit**

```bash
git add app/vectorstore/redis_store.py tests/integration/test_redis_store.py
git commit -m "Add model-aware filtering to Redis backend (TAG pre-filter hybrid KNN)"
```

---

### Task 6: Examples + docs

**Files:**
- Modify: `examples/llm_cache_demo.py`
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the new `filter_keys` and `filter` API fields.

- [ ] **Step 1: Declare, tag, and filter by model in the demo**

In `examples/llm_cache_demo.py`:

In `ensure_namespace`, add `filter_keys` to the create payload:

```python
        json={
            "name": NAMESPACE,
            "dimension": dimension,
            "default_threshold": THRESHOLD,
            "filter_keys": ["model"],
        },
```

In `query_cache`, send the model filter:

```python
def query_cache(embedding: list[float]) -> list[dict]:
    resp = _semcache(
        "POST",
        f"/{NAMESPACE}/query",
        json={"embedding": embedding, "filter": {"model": CHAT_MODEL}},
    )
    if resp.status_code != 200:
        raise DemoError(f"query failed ({resp.status_code}): {resp.text}")
    return resp.json()["matches"]
```

In `store`, tag the entry with the model:

```python
def store(key: str, embedding: list[float], answer: str) -> None:
    resp = _semcache(
        "POST",
        f"/{NAMESPACE}/entries",
        json={
            "key": key,
            "embedding": embedding,
            "value": answer,
            "metadata": {"model": CHAT_MODEL},
        },
    )
    if resp.status_code != 201:
        raise DemoError(f"store failed ({resp.status_code}): {resp.text}")
```

- [ ] **Step 2: Verify the demo still compiles**

Run: `python -m py_compile examples/llm_cache_demo.py`
Expected: no output (success).

- [ ] **Step 3: Document the feature**

In `CLAUDE.md`, under "Data model & semantics", add a bullet:

```markdown
- A namespace may declare immutable `filter_keys`. A query may pass a `filter`
  (conjunctive exact-match on those keys) to scope matches — e.g. keep responses
  from different generation models separate. Undeclared filter key → 422.
  Filtering is a pre-filter inside the `VectorStore` (correct `top_k`).
```

In `README.md`, under "Concepts", add:

```markdown
- **Filter keys** — a namespace can declare `filter_keys` (e.g. `["model"]`).
  Queries pass a matching `filter` so responses from different models are cached
  and served separately. Different *embedding* models belong in separate
  namespaces (their vectors aren't comparable).
```

- [ ] **Step 4: Commit**

```bash
git add examples/llm_cache_demo.py CLAUDE.md README.md
git commit -m "Tag demo entries by model and document filter_keys feature"
```

---

## Final verification

- [ ] Run the full unit suite: `python -m pytest -m "not integration" -v` — all pass.
- [ ] Run the integration suite: `docker compose run --rm api pytest -m integration -v` — all pass.
- [ ] Push: `git push origin main`.
