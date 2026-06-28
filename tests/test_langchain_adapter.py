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
