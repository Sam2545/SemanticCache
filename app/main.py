from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routes.cache import router
from app.services.cache import (
    DimensionMismatch,
    NamespaceExists,
    NamespaceNotFound,
)

app = FastAPI(title="SemCache", version="0.1.0")
app.include_router(router)


@app.exception_handler(NamespaceNotFound)
def _namespace_not_found(request: Request, exc: NamespaceNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(NamespaceExists)
def _namespace_exists(request: Request, exc: NamespaceExists) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(DimensionMismatch)
def _dimension_mismatch(request: Request, exc: DimensionMismatch) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
