from __future__ import annotations

from typing import Sequence

from app.vectorstore.base import Namespace, ScoredEntry, StoredEntry
from app.vectorstore.similarity import cosine_similarity


class InMemoryVectorStore:
    """In-process VectorStore. Used for tests and local development.

    Implements the VectorStore protocol with brute-force cosine search.
    """

    def __init__(self) -> None:
        self._namespaces: dict[str, Namespace] = {}
        self._entries: dict[str, dict[str, StoredEntry]] = {}

    def create_namespace(self, namespace: Namespace) -> None:
        self._namespaces[namespace.name] = namespace
        self._entries.setdefault(namespace.name, {})

    def get_namespace(self, name: str) -> Namespace | None:
        return self._namespaces.get(name)

    def upsert(self, namespace: str, entry: StoredEntry) -> None:
        self._entries[namespace][entry.key] = entry

    def search(
        self, namespace: str, embedding: Sequence[float], top_k: int
    ) -> list[ScoredEntry]:
        scored = [
            ScoredEntry(
                key=e.key,
                score=cosine_similarity(embedding, e.embedding),
                value=e.value,
                metadata=e.metadata,
            )
            for e in self._entries[namespace].values()
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def get(self, namespace: str, key: str) -> StoredEntry | None:
        return self._entries[namespace].get(key)

    def delete(self, namespace: str, key: str) -> bool:
        return self._entries[namespace].pop(key, None) is not None
