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
