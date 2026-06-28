# Docker-Free One-Command Startup — Design

**Date:** 2026-06-28
**Status:** Approved

## Goal

Give SemCache a Docker-free startup: a `semcache` console script that launches the
API with plain Python, and a `./start-local.sh` wrapper that bootstraps an isolated
virtual environment, installs the package into it, and runs the server — one command,
no Docker, no system-Python pollution.

## Constraints

- No external accounts; no PyPI publishing (console script + git-install only).
- `uvicorn` is already a core dependency — add nothing new to run the server.
- Default backend stays in-memory (no Redis, no Docker). `SEMCACHE_BACKEND=redis`
  is still honored via existing config if the user has Redis.
- `app/cli.py` is a process entry point only — no business logic (routes stay thin,
  logic in services).
- Default host `127.0.0.1` (loopback-only), default port `8000`.
- Commit messages: a single line under 150 characters, no `Co-Authored-By` trailer.

## Components

### `pyproject.toml` — console entry point

```toml
[project.scripts]
semcache = "app.cli:main"
```

After `pip install`, this creates a `semcache` command on PATH.

### `app/cli.py` (new) — launcher

A thin launcher: parse `--host`/`--port` (with env fallbacks), print the URL, run
uvicorn. Parsing is factored from launching so it is unit-testable without starting
a real server.

```python
from __future__ import annotations

import argparse
import os

import uvicorn

from app.config import settings
from app.main import app


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="semcache", description="Run the SemCache API server.")
    parser.add_argument("--host", default=os.environ.get("SEMCACHE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SEMCACHE_PORT", "8000")))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    print(f"Starting SemCache at http://{args.host}:{args.port}  (docs: /docs)")
    print(f"Backend: {settings.backend}  ·  set SEMCACHE_BACKEND=redis for the Redis store")
    uvicorn.run(app, host=args.host, port=args.port)
```

- CLI flags override env; env overrides the built-in defaults.
- Passes the `app` object directly to `uvicorn.run` (no reload/workers — out of scope).

### `start-local.sh` (new, repo root, executable) — venv bootstrap + run

```bash
#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
.venv/bin/pip install -e . --quiet
exec .venv/bin/semcache "$@"
```

- Creates `.venv/` if missing, reuses it on later runs.
- `pip install -e .` is cheap when already satisfied; editable means source changes
  are picked up without reinstalling.
- `exec` so Ctrl-C reaches uvicorn directly; `"$@"` forwards `--host`/`--port`.

### `.gitignore` — ignore the venv

Add `.venv/`.

### `Makefile` — add a target

Add `serve` for parity with `up`:

```makefile
serve:
	./start-local.sh
```

### `README.md` — "Run it without Docker (Python)"

Add a subsection under Quick start:

```bash
# from a clone — one command, sets up its own venv:
./start-local.sh

# no clone — your own venv:
python3 -m venv .venv && source .venv/bin/activate
pip install git+https://github.com/Sam2545/SemanticCache.git
semcache
```

Note it is single-process and in-memory (no Redis persistence/TTL) — the zero-infra
way to try SemCache.

## Testing

- **Unit tests** (`tests/test_cli.py`), monkeypatching `app.cli.uvicorn.run` to capture
  args without starting a server:
  - defaults → `run(app, host="127.0.0.1", port=8000)` and the `app` passed is `app.main.app`
  - `--port 9000` → port `9000` (flag wins)
  - `SEMCACHE_PORT=9999` env → port `9999` (env fallback)
- **Script syntax:** `bash -n start-local.sh` (and `shellcheck` if available).
- **Manual gate:** `./start-local.sh` creates the venv, installs, and serves; `semcache`
  resolves on PATH. The console-script wiring and venv creation exist only after a real
  run/install, so they are not unit-tested.

## Out of Scope (YAGNI)

- No PyPI publishing.
- No `--reload` / `--workers` flags.
- No automatic Redis management from the Python path.
