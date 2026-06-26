from __future__ import annotations

import json
import re
from typing import Any, Sequence

import numpy as np
import redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.query import Query

try:
    # redis-py >= 6.x
    from redis.commands.search.index_definition import IndexDefinition, IndexType
except ImportError:  # pragma: no cover - older redis-py
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from app.vectorstore.base import Namespace, ScoredEntry, StoredEntry

_TAG_SPECIAL = re.compile(r"([,.<>{}\[\]\"':;!@#$%^&*()\-+=~|/\\ ])")


def _escape_tag(value: str) -> str:
    """Escape RediSearch TAG special characters for exact-match queries."""
    return _TAG_SPECIAL.sub(r"\\\1", value)


def _decode(v: Any) -> Any:
    return v.decode() if isinstance(v, (bytes, bytearray)) else v


class RedisVectorStore:
    """VectorStore backed by redis-stack (RediSearch vector index per namespace).

    Implements the VectorStore protocol. RediSearch returns cosine *distance*
    (0 = identical); this store converts it to cosine *similarity* (1 = identical)
    so scores match the in-memory backend and the service's threshold semantics
    hold regardless of which store is in use.

    Key layout:
      semcache:ns:{name}          - namespace metadata (hash)
      semcache:doc:{name}:{key}   - entry (hash: key, embedding, value, metadata)
      semcache:idx:{name}         - RediSearch index over that namespace's entries
    """

    def __init__(self, redis_url: str) -> None:
        self._client = redis.Redis.from_url(redis_url)

    # --- key helpers -----------------------------------------------------

    def _ns_key(self, name: str) -> str:
        return f"semcache:ns:{name}"

    def _index(self, name: str) -> str:
        return f"semcache:idx:{name}"

    def _prefix(self, name: str) -> str:
        return f"semcache:doc:{name}:"

    def _doc_key(self, name: str, key: str) -> str:
        return f"{self._prefix(name)}{key}"

    # --- VectorStore protocol -------------------------------------------

    def create_namespace(self, namespace: Namespace) -> None:
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

    def get_namespace(self, name: str) -> Namespace | None:
        raw = self._client.hgetall(self._ns_key(name))
        if not raw:
            return None
        d = {_decode(k): _decode(v) for k, v in raw.items()}
        return Namespace(
            name=name,
            dimension=int(d["dimension"]),
            metric=d["metric"],
            default_threshold=float(d["default_threshold"]),
            default_top_k=int(d["default_top_k"]),
            ttl=int(d["ttl"]) if d["ttl"] != "" else None,
            filter_keys=json.loads(d["filter_keys"]) if "filter_keys" in d else [],
        )

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

    def get(self, namespace: str, key: str) -> StoredEntry | None:
        raw = self._client.hgetall(self._doc_key(namespace, key))
        if not raw:
            return None
        d = {_decode(k): v for k, v in raw.items()}
        return StoredEntry(
            key=_decode(d["key"]),
            embedding=np.frombuffer(d["embedding"], dtype=np.float32).tolist(),
            value=json.loads(_decode(d["value"])),
            metadata=json.loads(_decode(d["metadata"])),
        )

    def delete(self, namespace: str, key: str) -> bool:
        return self._client.delete(self._doc_key(namespace, key)) > 0

    # --- internals -------------------------------------------------------

    def _ensure_index(self, namespace: Namespace) -> None:
        index = self._index(namespace.name)
        try:
            self._client.ft(index).info()
            return  # already exists
        except redis.ResponseError:
            pass
        schema = (
            TagField("key"),
            TextField("value"),
            TextField("metadata"),
            *(TagField(f"flt_{k}", case_sensitive=True) for k in namespace.filter_keys),
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
        self._client.ft(index).create_index(
            schema,
            definition=IndexDefinition(
                prefix=[self._prefix(namespace.name)], index_type=IndexType.HASH
            ),
        )
