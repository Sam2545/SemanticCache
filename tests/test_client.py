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


def test_semcache_error_is_exception():
    from client import SemCacheError

    assert issubclass(SemCacheError, Exception)


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
