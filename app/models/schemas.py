from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateNamespaceRequest(BaseModel):
    name: str
    dimension: int = Field(gt=0)
    default_threshold: float | None = None
    default_top_k: int | None = Field(default=None, gt=0)
    ttl: int | None = Field(default=None, gt=0)
    filter_keys: list[str] = Field(default_factory=list)


class NamespaceResponse(BaseModel):
    name: str
    dimension: int
    metric: str
    default_threshold: float
    default_top_k: int
    ttl: int | None = None
    filter_keys: list[str] = Field(default_factory=list)


class WriteEntryRequest(BaseModel):
    key: str
    embedding: list[float]
    value: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntryResponse(BaseModel):
    key: str
    value: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    embedding: list[float]
    threshold: float | None = None
    top_k: int | None = Field(default=None, gt=0)
    filter: dict[str, str | int | float | bool] | None = None


class Match(BaseModel):
    key: str
    score: float
    value: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    matches: list[Match]

    @property
    def hit(self) -> bool:
        return len(self.matches) > 0
