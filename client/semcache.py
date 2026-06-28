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
