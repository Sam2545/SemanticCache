import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("langchain_core")

from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.language_models.llms import LLM  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402
from langchain_core.runnables import RunnableLambda, RunnablePassthrough  # noqa: E402

from integrations.langchain_semcache import LangChainSemCache  # noqa: E402

FRANCE_DOCS = "France is a country in Europe. Its capital city is Paris."
PYTHON_DOCS = "Python lists are sorted with the built-in sorted() function."


def retrieve(query):
    """Tiny query-dependent retriever — returns docs relevant to the question."""
    q = query.lower()
    if "python" in q or "sort" in q:
        return PYTHON_DOCS
    return FRANCE_DOCS


def stub_embed(text):
    # Embeds the full rendered prompt (context + question). The retriever makes
    # the context topic-specific, so France prompts and Python prompts separate.
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

    from langchain_core.globals import set_llm_cache

    from app.main import app
    from client import SemCache

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
    retriever = RunnableLambda(retrieve)
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
