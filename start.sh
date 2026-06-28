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
