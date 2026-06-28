# Docker-Free One-Command Startup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `semcache` console script and a `./start-local.sh` wrapper that bootstraps a venv and runs the API with plain Python — no Docker.

**Architecture:** A `[project.scripts]` entry point maps `semcache` to `app/cli.py:main`, which parses host/port and calls `uvicorn.run(app, ...)`. `start-local.sh` creates `.venv`, `pip install -e .`s into it, and `exec`s the console script. Parsing is factored from launching so it is unit-testable without a real server.

**Tech Stack:** Python 3.12, argparse, uvicorn (already a core dependency), pytest, venv.

## Global Constraints

- No PyPI publishing; console script + git-install only.
- Add no new runtime dependency (`uvicorn` already present).
- Default backend stays in-memory; `SEMCACHE_BACKEND=redis` honored via existing config.
- `app/cli.py` is a process entry point only — no business logic.
- Default host `127.0.0.1`, default port `8000`; CLI flag overrides env overrides default.
- Commit messages: a single line under 150 characters, no `Co-Authored-By` trailer.

## File Structure

- `app/cli.py` — launcher: `_parse_args(argv)` + `main(argv=None)` (new).
- `pyproject.toml` — add `[project.scripts] semcache = "app.cli:main"` (modified).
- `tests/test_cli.py` — unit tests for arg parsing + uvicorn invocation (new).
- `start-local.sh` — venv bootstrap + run (new, executable).
- `.gitignore` — add `.venv/` (modified).
- `Makefile` — add `serve` target (modified).
- `README.md` — add "Run it without Docker (Python)" subsection (modified).

---

### Task 1: `semcache` CLI launcher + entry point

**Files:**
- Create: `app/cli.py`, `tests/test_cli.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `app.main.app` (the FastAPI app), `app.config.settings`, `uvicorn.run`.
- Produces: `app.cli.main(argv: list[str] | None = None) -> None`, `app.cli._parse_args(argv)`;
  console script `semcache = "app.cli:main"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.cli'`.

- [ ] **Step 3: Implement `app/cli.py`**

```python
from __future__ import annotations

import argparse
import os

import uvicorn

from app.config import settings
from app.main import app


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="semcache", description="Run the SemCache API server."
    )
    parser.add_argument("--host", default=os.environ.get("SEMCACHE_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("SEMCACHE_PORT", "8000"))
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    print(f"Starting SemCache at http://{args.host}:{args.port}  (docs: /docs)")
    print(
        f"Backend: {settings.backend}  ·  set SEMCACHE_BACKEND=redis for the Redis store"
    )
    uvicorn.run(app, host=args.host, port=args.port)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: all 4 PASS.

- [ ] **Step 5: Add the console-script entry point in `pyproject.toml`**

After the `[project.optional-dependencies]` block (and before `[tool.pytest.ini_options]`),
add:

```toml
[project.scripts]
semcache = "app.cli:main"
```

- [ ] **Step 6: Verify pyproject still parses**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('ok')"`
Expected: prints `ok`.

- [ ] **Step 7: Run the full fast suite**

Run: `pytest -m "not integration" -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add app/cli.py tests/test_cli.py pyproject.toml
git commit -m "Add semcache console script to run the API without Docker"
```

---

### Task 2: `start-local.sh` venv bootstrap + .gitignore + Makefile + README

**Files:**
- Create: `start-local.sh` (executable)
- Modify: `.gitignore`, `Makefile`, `README.md`

**Interfaces:**
- Consumes: the `semcache` console script from Task 1 (installed into `.venv`).

- [ ] **Step 1: Create `start-local.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
.venv/bin/pip install -e . --quiet
exec .venv/bin/semcache "$@"
```

- [ ] **Step 2: Make it executable and syntax-check**

Run: `chmod +x start-local.sh && bash -n start-local.sh && echo "syntax ok"`
Expected: prints `syntax ok`.

Run (if available): `shellcheck start-local.sh`
Expected: no warnings. (Skip if `shellcheck` is not installed — `bash -n` is the gate.)

- [ ] **Step 3: Ignore the venv**

Append `.venv/` to `.gitignore` (the file already exists). After editing, the file
must contain a line `.venv/`.

- [ ] **Step 4: Add a `serve` target to the `Makefile`**

Update the `.PHONY` line and add the target. Change:

```makefile
.PHONY: up down logs test
```

to:

```makefile
.PHONY: up down logs test serve
```

and add (recipes use a TAB, not spaces):

```makefile
serve:
	./start-local.sh
```

- [ ] **Step 5: Verify the Makefile target parses**

Run: `make -n serve`
Expected: prints `./start-local.sh` with no "missing separator" error.

- [ ] **Step 6: Add the README subsection**

In `README.md`, immediately after the "### Run it from a clone" code block (the one
ending with the `./start.sh` line) and before "Interactive API docs at …", insert:

````markdown
### Run it without Docker (Python)

No Docker — a one-command script that creates its own virtual environment:

```bash
./start-local.sh        # makes .venv, installs SemCache, serves on :8000
```

Or manage your own venv (works without cloning):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install git+https://github.com/Sam2545/SemanticCache.git
semcache
```

Single-process and in-memory (no Redis persistence/TTL) — the zero-infra way to try it.

````

- [ ] **Step 7: Verify README + gitignore edits**

Run:
```bash
grep -n "start-local.sh" README.md && grep -qx ".venv/" .gitignore && echo "gitignore ok"
```
Expected: at least one README match and `gitignore ok`.

- [ ] **Step 8: Commit**

```bash
git add start-local.sh .gitignore Makefile README.md
git commit -m "Add start-local.sh venv bootstrap, serve target, and README"
```

---

## Final verification

- [ ] `pytest -m "not integration" -q` — all pass (includes `tests/test_cli.py`).
- [ ] `bash -n start-local.sh` — syntax ok.
- [ ] `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` — parses.
- [ ] Manual (real run): `./start-local.sh` creates `.venv`, installs, and serves at
  `http://127.0.0.1:8000/docs`; `--port 9000` is honored; Ctrl-C stops it cleanly.
