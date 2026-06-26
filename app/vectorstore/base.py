from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass
class Namespace:
    """A declared isolation boundary with a fixed embedding dimension."""

    name: str
    dimension: int
    metric: str = "cosine"
    default_threshold: float = 0.8
    default_top_k: int = 5
    ttl: int | None = None
    filter_keys: list[str] = field(default_factory=list)


@dataclass
class StoredEntry:
    """An entry as written/stored. `key` is the exact identity."""

    key: str
    embedding: list[float]
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredEntry:
    """A search result: an entry plus its similarity `score` to the query."""

    key: str
    score: float
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    """Storage interface the service depends on. Redis is one implementation.

    The store performs raw KNN search (returning up to `top_k` candidates by
    similarity). Threshold/hit-miss decisions are the service's responsibility,
    not the store's.
    """

    def create_namespace(self, namespace: Namespace) -> None: ...

    def get_namespace(self, name: str) -> Namespace | None: ...

    def upsert(self, namespace: str, entry: StoredEntry) -> None: ...

    def search(
        self,
        namespace: str,
        embedding: Sequence[float],
        top_k: int,
        filter: dict[str, object] | None = None,
    ) -> list[ScoredEntry]: ...

    def get(self, namespace: str, key: str) -> StoredEntry | None: ...

    def delete(self, namespace: str, key: str) -> bool: ...

    def count(self, namespace: str) -> int: ...
