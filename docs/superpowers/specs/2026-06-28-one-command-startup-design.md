# One-Command Startup for SemCache — Design

**Date:** 2026-06-28
**Status:** Approved

## Goal

Make SemCache trivial to run for anyone: a published prebuilt image people can
run without cloning, and a single frictionless command (`./start.sh`) for those
who clone the repo. Docker is the only assumed prerequisite.

## Constraints

- Docker-only: assume Docker + Compose v2. No non-Docker path in scope.
- Keep a single Docker image (no separate slim prod image), so the same image
  continues to run `docker compose run --rm api pytest -m integration`.
- Core service code (`app/`) is unchanged except for adding a `/health` test and,
  if needed, nothing else — the `/health` route already exists.
- Commit messages: a single line under 150 characters, no `Co-Authored-By` trailer.

## Distribution (README order: no-clone first, then clone-and-run)

### No-clone — run without cloning the repo

- **Quick try (zero infra, in-memory backend):**
  `docker run -p 8000:8000 ghcr.io/sam2545/semcache`
  Works out of the box because the default `SEMCACHE_BACKEND` is `memory`.
- **Full stack (API + Redis):** a `docker-compose.prod.yml` that *pulls* the
  published image (`image:` not `build:`, no volume mounts):
  `docker compose -f docker-compose.prod.yml up -d`.

### Clone-and-run — one command from a checkout

`./start.sh`:
1. Verify Docker is running (`docker info`); if not, print
   `Docker isn't running — start Docker Desktop and re-run ./start.sh` and exit 1.
2. `docker compose up -d --build --wait` — `--wait` blocks until the `api`
   service is healthy (requires a healthcheck; see below).
3. Print `✅ SemCache is ready at http://localhost:8000/docs`.
4. If `--wait` fails (non-zero), print a pointer to `docker compose logs` and exit 1.

A thin `Makefile` provides a command palette: `make up` (calls `./start.sh`),
`make down` (`docker compose down`), `make logs` (`docker compose logs -f`),
`make test` (`pytest -m "not integration"`). The script is the source of truth.

## Components

### `docker-compose.yml` (dev, build-from-source) — modified

Add a healthcheck to the `api` service so `--wait` works. The check runs inside
the container using Python (already installed; no `curl` dependency):

```yaml
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 3s
      timeout: 3s
      retries: 10
      start_period: 2s
```

### `docker-compose.prod.yml` (new, pull published image)

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

No volume mounts — the image is self-contained.

### `start.sh` (new, repo root, executable)

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

### `Makefile` (new)

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

### `.github/workflows/publish.yml` (new)

Build and push to GHCR on push to `main` and on `v*` tags, using the built-in
`GITHUB_TOKEN`. Tags: `latest` and the git ref. One-time manual step (documented):
ensure the repo grants package write and the package is public.

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

### `README.md` — modified

Rewrite Quick start: no-clone options first (`docker run`, then
`docker-compose.prod.yml`), then clone-and-run (`./start.sh` / `make up`).

## Testing

Startup scripts and CI workflows are not unit-testable; the gates are:

- **Unit test** `GET /health` returns 200 with `{"status": "ok"}` — the readiness
  contract that `start.sh`, the healthcheck, and `--wait` depend on. Add it if it
  does not already exist.
- **Compose validation:** `docker compose config` and
  `docker compose -f docker-compose.prod.yml config` parse without error.
- **Lint:** `shellcheck start.sh` if available.
- **Real run:** `./start.sh` brings the stack up healthy and the docs URL responds.

These last three are manual/CI checks, explicitly not unit tests.

## Out of Scope (YAGNI)

- No non-Docker / pip console-script path.
- No multi-stage slim production image.
- No Kubernetes/Helm or cloud-specific deploy manifests.
- No multi-arch build matrix (single linux/amd64 is fine to start).
