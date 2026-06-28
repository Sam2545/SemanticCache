# One-Command Startup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let anyone run SemCache either without cloning (published GHCR image) or with one command from a checkout (`./start.sh`).

**Architecture:** A healthcheck on the `api` service lets `docker compose up --wait` block until ready; `start.sh` wraps that with a Docker-running guard and a friendly URL. A pull-based `docker-compose.prod.yml` and a GitHub Actions workflow publish/consume a prebuilt image on GHCR. README leads with the no-clone path.

**Tech Stack:** Docker + Compose v2, GitHub Actions, GHCR, FastAPI (`/health`), pytest.

## Global Constraints

- Docker-only; assume Docker + Compose v2. No non-Docker path.
- Single Docker image (no separate slim prod image); the same image still runs `docker compose run --rm api pytest -m integration`.
- Image name: `ghcr.io/sam2545/semcache` (lowercase).
- README order: no-clone options first, then clone-and-run.
- `app/` is unchanged (the `/health` route already exists); only a test is added.
- Commit messages: a single line under 150 characters, no `Co-Authored-By` trailer.
- Docker is unavailable in the dev sandbox: validate YAML with a Python parse and shell with `bash -n`; `docker compose config` and a real `./start.sh` run are the human/CI gates.

## File Structure

- `tests/test_health.py` — unit test for the `/health` readiness contract (new).
- `docker-compose.yml` — add a healthcheck to `api` (modified).
- `docker-compose.prod.yml` — pull-based API + Redis stack, no build/volumes (new).
- `start.sh` — one-command clone-and-run wrapper (new, executable).
- `Makefile` — `up`/`down`/`logs`/`test` command palette (new).
- `.github/workflows/publish.yml` — build & push image to GHCR (new).
- `README.md` — Quick start rewritten, no-clone first (modified).

---

### Task 1: `/health` readiness test + dev compose healthcheck

**Files:**
- Create: `tests/test_health.py`
- Modify: `docker-compose.yml`

**Interfaces:**
- Consumes: the existing `GET /health` route in `app/main.py` (returns `{"status": "ok"}`).
- Produces: a healthcheck on the `api` service that `docker compose up --wait` (used by `start.sh` in Task 2) depends on.

- [ ] **Step 1: Write the readiness test**

Create `tests/test_health.py`:

```python
from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_ok():
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_health.py -v`
Expected: PASS. (The `/health` route already exists; this test locks the readiness
contract that the healthcheck, `start.sh`, and `--wait` all depend on, so it must
not silently change.)

- [ ] **Step 3: Add a healthcheck to the `api` service**

In `docker-compose.yml`, add the `healthcheck` block to the `api` service (after
the `volumes:` block, still under `api:`). The check uses Python (already in the
image — no `curl` dependency):

```yaml
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 3s
      timeout: 3s
      retries: 10
      start_period: 2s
```

- [ ] **Step 4: Validate the compose YAML parses**

Run: `python -c "import yaml; yaml.safe_load(open('docker-compose.yml')); print('ok')"`
Expected: prints `ok`.

(If Docker is available, also run `docker compose config >/dev/null && echo ok`.)

- [ ] **Step 5: Commit**

```bash
git add tests/test_health.py docker-compose.yml
git commit -m "Add /health readiness test and api healthcheck for --wait"
```

---

### Task 2: `start.sh` one-command wrapper + `Makefile`

**Files:**
- Create: `start.sh` (executable), `Makefile`

**Interfaces:**
- Consumes: the `api` healthcheck from Task 1 (so `--wait` resolves).
- Produces: `./start.sh` and `make up`/`make down`/`make logs`/`make test`.

- [ ] **Step 1: Create `start.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

if ! docker info >/dev/null 2>&1; then
  echo "Docker isn't running — start Docker Desktop and re-run ./start.sh" >&2
  exit 1
fi

echo "Building and starting SemCache (this can take a minute the first time)…"
if ! docker compose up -d --build --wait; then
  echo "SemCache did not become healthy. Check logs with: docker compose logs" >&2
  exit 1
fi

echo "✅ SemCache is ready at http://localhost:8000/docs"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x start.sh`

- [ ] **Step 3: Syntax-check the script**

Run: `bash -n start.sh && echo "syntax ok"`
Expected: prints `syntax ok`.

Run (if available): `shellcheck start.sh`
Expected: no warnings. (If `shellcheck` is not installed, skip — `bash -n` is the gate.)

- [ ] **Step 4: Create `Makefile`**

Note: Makefile recipes must be indented with a TAB, not spaces.

```makefile
.PHONY: up down logs test

up:
	./start.sh

down:
	docker compose down

logs:
	docker compose logs -f

test:
	pytest -m "not integration"
```

- [ ] **Step 5: Verify the Makefile targets parse**

Run: `make -n up && make -n down && make -n logs && make -n test`
Expected: prints the commands each target would run, with no "missing separator" error.

- [ ] **Step 6: Commit**

```bash
git add start.sh Makefile
git commit -m "Add start.sh one-command startup and Makefile command palette"
```

---

### Task 3: Pull-based prod compose + GHCR publish workflow

**Files:**
- Create: `docker-compose.prod.yml`, `.github/workflows/publish.yml`

**Interfaces:**
- Consumes: the published image `ghcr.io/sam2545/semcache` (produced by the workflow).
- Produces: a no-clone full-stack compose and a CI job that builds/pushes the image.

- [ ] **Step 1: Create `docker-compose.prod.yml`**

```yaml
services:
  api:
    image: ghcr.io/sam2545/semcache:latest
    ports:
      - "8000:8000"
    environment:
      SEMCACHE_BACKEND: redis
      SEMCACHE_REDIS_URL: redis://redis:6379
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 3s
      timeout: 3s
      retries: 10
      start_period: 2s

  redis:
    image: redis/redis-stack-server:latest
    ports:
      - "6379:6379"
```

- [ ] **Step 2: Create `.github/workflows/publish.yml`**

```yaml
name: publish image
on:
  push:
    branches: [main]
    tags: ["v*"]
permissions:
  contents: read
  packages: write
jobs:
  build-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/metadata-action@v5
        id: meta
        with:
          images: ghcr.io/sam2545/semcache
          tags: |
            type=raw,value=latest,enable={{is_default_branch}}
            type=ref,event=tag
            type=sha
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

- [ ] **Step 3: Validate both YAML files parse**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('docker-compose.prod.yml')); print('compose ok')"
python -c "import yaml; yaml.safe_load(open('.github/workflows/publish.yml')); print('workflow ok')"
```
Expected: prints `compose ok` then `workflow ok`.

(If Docker is available, also run `docker compose -f docker-compose.prod.yml config >/dev/null && echo ok`.)

- [ ] **Step 4: Commit**

```bash
git add docker-compose.prod.yml .github/workflows/publish.yml
git commit -m "Add pull-based prod compose and GHCR image publish workflow"
```

---

### Task 4: README Quick start rewrite

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: `docker run` image (Task 3), `docker-compose.prod.yml` (Task 3), `./start.sh` (Task 2).

- [ ] **Step 1: Replace the Quick start section**

In `README.md`, replace the current Quick start block:

```markdown
## Quick start

```bash
docker compose up --build   # FastAPI on :8000 (Redis backend), Redis on :6379
pytest -m "not integration" # fast unit suite (no Redis required)
docker compose run --rm api pytest -m integration   # Redis contract tests
```

Interactive API docs at <http://localhost:8000/docs>.
```

with:

```markdown
## Quick start

### Run it without cloning

Quick try (in-memory backend, zero infrastructure):

```bash
docker run -p 8000:8000 ghcr.io/sam2545/semcache
```

Full stack (API + Redis), using the published image:

```bash
curl -O https://raw.githubusercontent.com/Sam2545/SemanticCache/main/docker-compose.prod.yml
docker compose -f docker-compose.prod.yml up -d
```

### Run it from a clone

```bash
git clone https://github.com/Sam2545/SemanticCache.git && cd SemanticCache
./start.sh        # builds, waits until healthy, prints the URL  (or: make up)
```

Interactive API docs at <http://localhost:8000/docs>.

### Development

```bash
pytest -m "not integration"                          # fast unit suite (no Redis)
docker compose run --rm api pytest -m integration    # Redis contract tests
make down                                            # stop the stack
```
```

- [ ] **Step 2: Verify the section renders and links resolve**

Run: `grep -n "ghcr.io/sam2545/semcache" README.md`
Expected: at least the `docker run` line matches (no stray casing/typo in the image name).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Rewrite README Quick start: no-clone first, then clone-and-run"
```

---

## Final verification

- [ ] `pytest -m "not integration" -q` — all pass (includes the new `/health` test).
- [ ] `bash -n start.sh` — syntax ok.
- [ ] All YAML parses (Task 1/3 Python checks succeed).
- [ ] On a machine with Docker: `./start.sh` brings the stack up healthy and `http://localhost:8000/docs` responds; `docker compose -f docker-compose.prod.yml config` validates.
- [ ] One-time GHCR setup (manual, documented in the PR): after the first workflow run, make the `semcache` package public in the repo's package settings so `docker run`/`docker pull` works without auth.
