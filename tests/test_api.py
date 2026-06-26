import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_service
from app.main import app
from app.services.cache import CacheService
from app.vectorstore.memory_store import InMemoryVectorStore


@pytest.fixture
def client():
    service = CacheService(InMemoryVectorStore())
    app.dependency_overrides[get_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


def _create_ns(client, name="ns", dimension=2, **kw):
    return client.post(
        "/namespaces",
        json={"name": name, "dimension": dimension, **kw},
    )


def test_create_namespace(client):
    r = _create_ns(client, default_threshold=0.6)
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "ns"
    assert body["dimension"] == 2
    assert body["metric"] == "cosine"


def test_create_duplicate_namespace_conflicts(client):
    _create_ns(client)
    assert _create_ns(client).status_code == 409


def test_write_entry(client):
    _create_ns(client)
    r = client.post(
        "/ns/entries",
        json={"key": "a", "embedding": [1.0, 0.0], "value": "A"},
    )
    assert r.status_code == 201


def test_write_dimension_mismatch_unprocessable(client):
    _create_ns(client)
    r = client.post(
        "/ns/entries",
        json={"key": "a", "embedding": [1.0, 0.0, 0.0], "value": "A"},
    )
    assert r.status_code == 422


def test_write_to_missing_namespace_not_found(client):
    r = client.post(
        "/ns/entries",
        json={"key": "a", "embedding": [1.0, 0.0], "value": "A"},
    )
    assert r.status_code == 404


def test_query_hit(client):
    _create_ns(client, default_threshold=0.6)
    client.post("/ns/entries", json={"key": "a", "embedding": [1.0, 0.0], "value": "A"})
    r = client.post("/ns/query", json={"embedding": [1.0, 0.0]})
    assert r.status_code == 200
    matches = r.json()["matches"]
    assert [m["key"] for m in matches] == ["a"]
    assert matches[0]["value"] == "A"


def test_query_miss_returns_empty(client):
    _create_ns(client, default_threshold=0.6)
    client.post("/ns/entries", json={"key": "a", "embedding": [0.0, 1.0], "value": "A"})
    r = client.post("/ns/query", json={"embedding": [1.0, 0.0]})
    assert r.status_code == 200
    assert r.json()["matches"] == []


def test_get_entry_by_key(client):
    _create_ns(client)
    client.post("/ns/entries", json={"key": "a", "embedding": [1.0, 0.0], "value": "A"})
    r = client.get("/ns/entries/a")
    assert r.status_code == 200
    assert r.json()["value"] == "A"


def test_get_missing_entry_not_found(client):
    _create_ns(client)
    assert client.get("/ns/entries/nope").status_code == 404


def test_delete_entry(client):
    _create_ns(client)
    client.post("/ns/entries", json={"key": "a", "embedding": [1.0, 0.0], "value": "A"})
    assert client.delete("/ns/entries/a").status_code == 204
    assert client.get("/ns/entries/a").status_code == 404


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
