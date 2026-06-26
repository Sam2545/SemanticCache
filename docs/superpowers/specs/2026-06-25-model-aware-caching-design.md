# Model-aware caching (declared-metadata filtering) — design

- **Date:** 2026-06-25
- **Status:** Approved (pending spec review)
- **Component:** SemCache core (`app/`), both `VectorStore` backends, examples

## Problem

SemCache currently returns the most similar cached entry within a namespace,
keyed purely on cosine similarity. There is no way to keep responses generated
by *different models* separate: a query can receive an answer produced by a
different model than the caller intends. We want responses from different
generation models cached and served separately.

## Goals

- Let a caller scope a query so it only matches entries that share specified
  metadata (e.g. the generation `model`).
- Keep SemCache **generic**: the core never learns the concept of "model." It
  filters on caller-declared metadata keys; `model` is just the first such key.
- Preserve correctness: filtering must not drop a valid match ranked beyond
  `top_k` (i.e. pre-filter, not post-filter).
- Keep the in-memory and Redis backends behaving identically behind the
  `VectorStore` protocol.
- Fully backward-compatible: existing namespaces and queries are unaffected.

## Non-goals

- Prompt/template version and generation-param partitioning are **out of scope**
  for this change. The same `filter_keys` mechanism supports them later with no
  code change (just declare the key), so no special handling is added now.
- Embedding-model partitioning is **not** an API field — see "Embedding model"
  below; it is handled by using separate namespaces.

## Design

### Concept

A namespace may declare a set of **filter keys** — metadata keys that become
filterable. A query may supply a **filter** (a map of `key -> value`) and is
matched against entries by **conjunctive exact equality** on those keys.

- Entries place the values in their existing `metadata` (e.g.
  `metadata: {"model": "gpt-oss:120b"}`).
- A query with no filter matches everything (today's behavior).
- An entry that is missing a filtered key is treated as not-equal (excluded).
- An entry may carry metadata keys beyond those filtered; extras are ignored.

### API changes (all backward-compatible)

1. **`POST /namespaces`** gains optional `filter_keys: list[str]` (default `[]`).
   Declared at creation and **immutable** thereafter — same model as `dimension`.
   ```json
   { "name": "llm", "dimension": 768, "filter_keys": ["model"] }
   ```

2. **`POST /{ns}/entries`** — unchanged. The caller puts partition attributes in
   `metadata`. Writing a value for a declared filter key is what makes the entry
   selectable by that key.

3. **`POST /{ns}/query`** gains optional `filter: dict[str, str|int|float|bool]`
   (default none / empty).
   - Conjunctive exact-match on the given keys; remaining behavior (threshold,
     `top_k`, ordering) is unchanged and applied **after** filtering.
   - A filter key **not** in the namespace's `filter_keys` → **422**
     (`InvalidFilter`). We refuse rather than silently ignore, because silently
     ignoring would reintroduce the exact cross-model bug this feature prevents,
     and because Redis cannot honor an un-indexed key — failing keeps both
     backends consistent.

### Data model

- `Namespace` gains `filter_keys: list[str] = []`.
- `Query` request gains `filter: dict[str, Any] | None`.
- Entry metadata is unchanged (already an arbitrary dict).

### VectorStore interface

`search` gains a filter argument; the filter is applied as a **pre-filter** so
KNN ranks within the matching subset:

```
search(namespace, embedding, top_k, filter: dict[str, str]) -> list[ScoredEntry]
```

- **In-memory:** keep entries whose `metadata[key] == value` for every key in
  `filter` (missing key ⇒ excluded), then cosine-rank, then `top_k`.
- **Redis:** hybrid query
  `(@flt_model:{<escaped value>} ...)=>[KNN <top_k> @embedding $vec AS distance]`.
  - At `create_namespace`, add a `TagField("flt_<key>")` to the index schema for
    each declared filter key.
  - At `upsert`, for each declared filter key present in the entry metadata,
    write a hash field `flt_<key> = str(value)` alongside the existing JSON
    `metadata` blob (the blob remains the source of truth on read).
  - Filter values are matched as **strings**; both sides stringify. TAG values
    must be escaped for RediSearch special characters (`:`, `-`, space, etc.) in
    the query — model names like `gpt-oss:120b` contain them.

Values are compared as strings on both backends so behavior is identical.

### Service layer

- `CacheService.create_namespace(..., filter_keys: list[str] | None = None)`.
- `CacheService.query(..., filter: dict | None = None)`:
  1. Resolve namespace; validate every filter key ∈ `namespace.filter_keys`,
     else raise `InvalidFilter`.
  2. Delegate to `store.search(..., filter=filter or {})`.
  3. Apply the threshold cutoff as today.
- New exception `InvalidFilter(CacheError)` mapped to **422** in `app/main.py`
  (next to the existing `DimensionMismatch` handler).

### Embedding model (handled by namespaces, not the API)

Different embedding models produce **incomparable** vectors (and usually
different dimensions, which dimension validation already rejects within one
namespace). Embedding model therefore defines *which vector space you are in* —
a namespace-level fact, not a row attribute. Guidance: **use a separate
namespace per embedder** (e.g. encode the embedder in the namespace name). The
generic mechanism still *allows* declaring `embed_model` as a filter key for
defense-in-depth, but separate namespaces are the recommended, correct boundary.

### Backward compatibility

- `filter_keys` defaults to `[]`; `filter` defaults to empty. Existing
  namespaces, writes, and queries behave exactly as before.

## Testing

- **Unit (in-memory):**
  - filter returns only entries with the matching value; no filter returns all.
  - conjunctive multi-key match.
  - undeclared filter key → `InvalidFilter`.
  - entry missing a filtered key is excluded.
  - threshold still applies after filtering.
- **Integration (Redis):** same contract, plus the **critical correctness
  case** — a wanted-model entry ranked *below* several other-model entries must
  still be returned (proves pre-filter, not post-filter). Verify TAG indexing of
  declared keys and escaping of values containing `:`/`-`.
- **API:** `POST /namespaces` with `filter_keys`; `POST /{ns}/query` with
  `filter` (hit, miss, 422 on undeclared key).
- **Examples:** update `llm_cache_demo.py` to declare `filter_keys:["model"]`,
  tag entries with `metadata:{"model": CHAT_MODEL}`, and query with
  `filter:{"model": CHAT_MODEL}`, so two chat models cache separately.
