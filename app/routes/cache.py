from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.dependencies import get_service
from app.models.schemas import (
    CreateNamespaceRequest,
    EntryResponse,
    Match,
    NamespaceResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
    WriteEntryRequest,
)
from app.services.cache import CacheService

router = APIRouter()


@router.post(
    "/namespaces",
    response_model=NamespaceResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_namespace(
    body: CreateNamespaceRequest, service: CacheService = Depends(get_service)
) -> NamespaceResponse:
    ns = service.create_namespace(
        name=body.name,
        dimension=body.dimension,
        default_threshold=body.default_threshold,
        default_top_k=body.default_top_k,
        ttl=body.ttl,
        filter_keys=body.filter_keys,
    )
    return NamespaceResponse(
        name=ns.name,
        dimension=ns.dimension,
        metric=ns.metric,
        default_threshold=ns.default_threshold,
        default_top_k=ns.default_top_k,
        ttl=ns.ttl,
        filter_keys=ns.filter_keys,
    )


@router.post(
    "/{namespace}/entries",
    status_code=status.HTTP_201_CREATED,
    response_model=EntryResponse,
)
def write_entry(
    namespace: str,
    body: WriteEntryRequest,
    service: CacheService = Depends(get_service),
) -> EntryResponse:
    service.put(
        namespace=namespace,
        key=body.key,
        embedding=body.embedding,
        value=body.value,
        metadata=body.metadata,
    )
    return EntryResponse(key=body.key, value=body.value, metadata=body.metadata)


@router.post("/{namespace}/query", response_model=QueryResponse)
def query(
    namespace: str,
    body: QueryRequest,
    service: CacheService = Depends(get_service),
) -> QueryResponse:
    matches = service.query(
        namespace=namespace,
        embedding=body.embedding,
        threshold=body.threshold,
        top_k=body.top_k,
        filter=body.filter,
    )
    return QueryResponse(
        matches=[
            Match(key=m.key, score=m.score, value=m.value, metadata=m.metadata)
            for m in matches
        ],
        hit=len(matches) > 0,
        threshold=service.effective_threshold(namespace, body.threshold),
    )


@router.get("/{namespace}/stats", response_model=StatsResponse)
def get_stats(
    namespace: str, service: CacheService = Depends(get_service)
) -> StatsResponse:
    return StatsResponse(**asdict(service.stats(namespace)))


@router.get("/{namespace}/entries/{key}", response_model=EntryResponse)
def get_entry(
    namespace: str, key: str, service: CacheService = Depends(get_service)
) -> EntryResponse:
    entry = service.get(namespace, key)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"entry '{key}' not found in namespace '{namespace}'",
        )
    return EntryResponse(key=entry.key, value=entry.value, metadata=entry.metadata)


@router.delete("/{namespace}/entries/{key}", status_code=status.HTTP_204_NO_CONTENT)
def delete_entry(
    namespace: str, key: str, service: CacheService = Depends(get_service)
) -> Response:
    if not service.delete(namespace, key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"entry '{key}' not found in namespace '{namespace}'",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
