"""Shared types for the pipeline. Keep loose — modules own their own internal shapes."""

from typing import Any, Literal
from pydantic import BaseModel, Field

Source = Literal["hybrid", "kg", "both"]


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None


class TaggedCandidate(BaseModel):
    """Candidate after hybrid+kg merge, tagged with its origin."""
    id: str
    score: float | None = None
    source: Source
    hybrid: dict[str, Any] | None = None
    kg: dict[str, Any] | None = None


class PipelineResult(BaseModel):
    """Final payload returned to the frontend."""
    query: str
    intent: dict[str, Any]
    hybrid_raw: dict[str, Any]
    kg_raw: dict[str, Any]
    tagged_candidates: list[dict[str, Any]]
    reranked: dict[str, Any]
    explanations: dict[str, Any]
    trace: dict[str, Any] = Field(default_factory=dict)
