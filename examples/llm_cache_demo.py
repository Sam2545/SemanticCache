"""Semantic LLM-response cache demo for SemCache.

Shows the point of a semantic cache: a *paraphrased* question reuses a cached
LLM answer instead of paying for a second generation.

This is a client of the running service — it imports nothing from `app/`. It
talks HTTP to:
  - SemCache                          (SEMCACHE_URL, default :8000)
  - a chat provider for answers       (CHAT_BASE_URL / CHAT_API_KEY / CHAT_MODEL)
  - an embeddings provider for vectors (EMBED_BASE_URL / EMBED_API_KEY / EMBED_MODEL)

Chat and embeddings are configured separately on purpose: they often come from
different services. Ollama Cloud (ollama.com), for example, serves chat models
but no embedding models — so the defaults below pair Ollama Cloud for chat with
a local Ollama for embeddings. Both speak the OpenAI-compatible API.

Configure via environment variables or examples/.env (see examples/.env.example).
Never hardcode keys.

Run:
    pip install -r examples/requirements.txt
    # edit examples/.env with your CHAT_API_KEY (your ollama.com key)
    python examples/llm_cache_demo.py
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import requests


def _load_dotenv() -> None:
    """Load .env files (repo root and examples/) into the environment.

    Values from a .env file take precedence so the file you edit is the source
    of truth, even if a stale variable is exported in your shell.
    """
    here = Path(__file__).resolve().parent
    for path in (here.parent / ".env", here / ".env"):
        if not path.exists():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line = line.removeprefix("export ").strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ[key.strip()] = val.strip().strip('"').strip("'")


_load_dotenv()


def _env(name: str, *fallbacks: str, default: str = "") -> str:
    for key in (name, *fallbacks):
        val = os.environ.get(key)
        if val:
            return val
    return default


SEMCACHE_URL = _env("SEMCACHE_URL", default="http://localhost:8000")
NAMESPACE = _env("SEMCACHE_NAMESPACE", default="llm-demo")
THRESHOLD = float(_env("SEMCACHE_THRESHOLD", default="0.82"))

# Chat / answer generation — defaults to Ollama Cloud.
CHAT_BASE_URL = _env("CHAT_BASE_URL", "LLM_BASE_URL", default="https://ollama.com/v1")
CHAT_API_KEY = _env("CHAT_API_KEY", "OLLAMA_API_KEY", "LLM_API_KEY")
CHAT_MODEL = _env("CHAT_MODEL", default="gpt-oss:120b")

# Embeddings — defaults to a local Ollama (Ollama Cloud has no embedding models).
EMBED_BASE_URL = _env("EMBED_BASE_URL", default="http://localhost:11434/v1")
EMBED_API_KEY = _env("EMBED_API_KEY", default="ollama")
EMBED_MODEL = _env("EMBED_MODEL", default="nomic-embed-text")

TIMEOUT = 120


class DemoError(RuntimeError):
    """A user-actionable failure (bad config, service down, API error)."""


# --- OpenAI-compatible provider calls ------------------------------------


def _post(base_url: str, api_key: str, path: str, payload: dict, *, what: str) -> dict:
    if not api_key:
        raise DemoError(
            f"No API key for {what} ({base_url}). Set it in examples/.env."
        )
    try:
        resp = requests.post(
            f"{base_url}{path}",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=TIMEOUT,
        )
    except requests.ConnectionError as exc:
        hint = ""
        if "localhost" in base_url or "127.0.0.1" in base_url:
            hint = (
                "\n  This is a local endpoint — is Ollama running, and have you "
                f"run `ollama pull {payload.get('model', '')}`?"
            )
        raise DemoError(f"Cannot reach {what} at {base_url}: {exc}{hint}") from exc
    if resp.status_code != 200:
        hint = ""
        if resp.status_code == 401:
            hint = (
                f"\n  auth rejected by {base_url} — key length "
                f"{len(api_key)} chars. Check the key for {what} in examples/.env."
            )
        raise DemoError(
            f"{what} request to {path} failed ({resp.status_code}): "
            f"{resp.text}{hint}"
        )
    return resp.json()


def embed(text: str) -> list[float]:
    data = _post(
        EMBED_BASE_URL,
        EMBED_API_KEY,
        "/embeddings",
        {"model": EMBED_MODEL, "input": text},
        what="embeddings",
    )
    return data["data"][0]["embedding"]


def generate(question: str) -> str:
    data = _post(
        CHAT_BASE_URL,
        CHAT_API_KEY,
        "/chat/completions",
        {"model": CHAT_MODEL, "messages": [{"role": "user", "content": question}]},
        what="chat",
    )
    return data["choices"][0]["message"]["content"].strip()


# --- SemCache (the service under demo) -----------------------------------


def _semcache(method: str, path: str, **kwargs) -> requests.Response:
    try:
        return requests.request(
            method, f"{SEMCACHE_URL}{path}", timeout=TIMEOUT, **kwargs
        )
    except requests.ConnectionError as exc:
        raise DemoError(
            f"Cannot reach SemCache at {SEMCACHE_URL}. Is it running "
            f"(docker compose up)?\n    {exc}"
        ) from exc


def ensure_namespace(dimension: int) -> None:
    resp = _semcache(
        "POST",
        "/namespaces",
        json={
            "name": NAMESPACE,
            "dimension": dimension,
            "default_threshold": THRESHOLD,
            "filter_keys": ["model"],
        },
    )
    if resp.status_code == 201:
        print(
            f"created namespace '{NAMESPACE}' "
            f"(dim={dimension}, threshold={THRESHOLD})"
        )
    elif resp.status_code == 409:
        print(f"namespace '{NAMESPACE}' already exists — reusing it")
    else:
        raise DemoError(
            f"create namespace failed ({resp.status_code}): {resp.text}"
        )


def query_cache(embedding: list[float]) -> list[dict]:
    resp = _semcache(
        "POST",
        f"/{NAMESPACE}/query",
        json={"embedding": embedding, "filter": {"model": CHAT_MODEL}},
    )
    if resp.status_code != 200:
        raise DemoError(f"query failed ({resp.status_code}): {resp.text}")
    return resp.json()["matches"]


def store(key: str, embedding: list[float], answer: str) -> None:
    resp = _semcache(
        "POST",
        f"/{NAMESPACE}/entries",
        json={
            "key": key,
            "embedding": embedding,
            "value": answer,
            "metadata": {"model": CHAT_MODEL},
        },
    )
    if resp.status_code != 201:
        raise DemoError(f"store failed ({resp.status_code}): {resp.text}")


# --- the cache loop -------------------------------------------------------


def ask(question: str) -> str:
    """Return an answer, serving from cache when a similar question was seen."""
    vector = embed(question)
    matches = query_cache(vector)
    if matches:
        top = matches[0]
        print(f"  ✔ CACHE HIT (score={top['score']:.3f}) — no LLM call")
        return top["value"]
    print("  … miss — calling the LLM and caching the answer")
    answer = generate(question)
    store(question, vector, answer)
    return answer


PAIRS = [
    (
        "What is the capital of France?",
        "Remind me, which city is France's capital?",
    ),
    (
        "How do I reverse a list in Python?",
        "What's the Pythonic way to flip a list around?",
    ),
]


def print_stats() -> None:
    resp = _semcache("GET", f"/{NAMESPACE}/stats")
    if resp.status_code != 200:
        return
    s = resp.json()
    print(
        f"\nstats for '{NAMESPACE}': "
        f"{s['hits']}/{s['queries']} hits ({s['hit_rate']:.0%}), "
        f"avg similarity {s['avg_similarity']:.3f}, "
        f"~{s['estimated_latency_saved_ms']:.0f} ms saved"
    )


def main() -> int:
    print(f"chat:       {CHAT_MODEL} @ {CHAT_BASE_URL}")
    print(f"embeddings: {EMBED_MODEL} @ {EMBED_BASE_URL}")
    try:
        # Auto-detect the embedding dimension so any model/provider works.
        probe = embed("dimension probe")
        ensure_namespace(len(probe))

        for original, paraphrase in PAIRS:
            print(f"\nQ: {original}")
            print(textwrap.indent(ask(original), "   "))
            print(f"\nQ (paraphrase): {paraphrase}")
            print(textwrap.indent(ask(paraphrase), "   "))
            print("-" * 70)
        print_stats()
    except DemoError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
