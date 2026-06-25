# SemCache examples

Client demos of the SemCache service. These import nothing from `app/` — they
talk to the running service over HTTP, the way a real LLM application would.

## `llm_cache_demo.py` — semantic LLM-response cache

Demonstrates the core value of a semantic cache: a **paraphrased** question
reuses a cached LLM answer instead of triggering a second (paid) generation.

Flow per question: embed it → `POST /{ns}/query` → on a hit, return the cached
answer with no LLM call; on a miss, call the chat model, then `POST /{ns}/entries`
to cache the answer keyed by the question.

**Chat and embeddings are configured separately**, because they often come from
different providers. The defaults pair **Ollama Cloud for chat** with a **local
Ollama for embeddings** — Ollama Cloud serves chat models but no embedding
models, so the vectors come from a local Ollama. Both sides speak the
OpenAI-compatible API, so either can be repointed at any compatible endpoint.

### Prerequisites

1. **SemCache running:**
   ```bash
   docker compose up --build
   ```
2. **Chat: an Ollama Cloud key** — create one at
   <https://ollama.com/settings/keys>, put it in `examples/.env` as `CHAT_API_KEY`.
3. **Embeddings: a local Ollama** with an embedding model pulled:
   ```bash
   ollama pull nomic-embed-text
   ```
   (Don't want to run a local Ollama? Repoint `EMBED_*` at any OpenAI-compatible
   embeddings endpoint instead.)

### Configure

Copy `.env.example` to `.env` (gitignored) and fill in `CHAT_API_KEY`:

```bash
cp examples/.env.example examples/.env
# edit examples/.env -> CHAT_API_KEY=<your ollama.com key>
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `CHAT_BASE_URL` | `https://ollama.com/v1` | Chat provider (Ollama Cloud) |
| `CHAT_API_KEY` | — | Your ollama.com key |
| `CHAT_MODEL` | `gpt-oss:120b` | Cloud chat model (see ollama.com/search?c=cloud) |
| `EMBED_BASE_URL` | `http://localhost:11434/v1` | Embeddings provider (local Ollama) |
| `EMBED_API_KEY` | `ollama` | Ignored by local Ollama |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model (dimension auto-detected) |
| `SEMCACHE_URL` | `http://localhost:8000` | SemCache service |
| `SEMCACHE_THRESHOLD` | `0.82` | Cosine similarity needed for a cache hit |

### Run

```bash
pip install -r examples/requirements.txt
python examples/llm_cache_demo.py
```

Each original question misses (chat model called + cached); each paraphrase
**hits** the cache with no LLM call.

> Security: keys belong only in your environment or an untracked `.env` file.
> If a key is ever pasted into a chat or shared, rotate it.
