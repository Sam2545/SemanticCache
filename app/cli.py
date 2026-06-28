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
