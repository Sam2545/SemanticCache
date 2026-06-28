import pytest

from app import cli
from app.main import app


@pytest.fixture
def captured(monkeypatch):
    calls = {}

    def fake_run(application, host, port):
        calls["app"] = application
        calls["host"] = host
        calls["port"] = port

    monkeypatch.setattr("app.cli.uvicorn.run", fake_run)
    return calls


def test_defaults(captured, monkeypatch):
    monkeypatch.delenv("SEMCACHE_HOST", raising=False)
    monkeypatch.delenv("SEMCACHE_PORT", raising=False)
    cli.main([])
    assert captured["app"] is app
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000


def test_port_flag_overrides(captured, monkeypatch):
    monkeypatch.delenv("SEMCACHE_PORT", raising=False)
    cli.main(["--port", "9000"])
    assert captured["port"] == 9000


def test_env_port_fallback(captured, monkeypatch):
    monkeypatch.setenv("SEMCACHE_PORT", "9999")
    cli.main([])
    assert captured["port"] == 9999


def test_flag_beats_env(captured, monkeypatch):
    monkeypatch.setenv("SEMCACHE_PORT", "9999")
    cli.main(["--port", "7000"])
    assert captured["port"] == 7000
