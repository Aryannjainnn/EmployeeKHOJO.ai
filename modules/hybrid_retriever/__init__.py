"""Adapter: bridges the teammate's HybridRetriever to the orchestrator contract.

Teammate API   : HybridRetriever.retrieve(intent_json) -> {hybrid:{results}, lexical:..., semantic:...}
Orchestrator   : retrieve(intent_dict) -> {"candidates": [{"id","score",...}], ...}

The HybridRetriever needs the FULL parsed intent (skills, negated_skills, exp band,
etc.) to apply hard filters — so this adapter accepts the canonical intent dict
emitted by `modules.intent_processor.process()`, not just the queries list.

Index dir resolution order: $INDEX_DIR → $DATA_DIR → ./data
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Expose package dir on sys.path so internal `from retriever import ...` resolves
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

_retriever = None  # cached HybridRetriever instance — heavy index load on first call


def _resolve_index_dir() -> Path:
    raw = os.getenv("INDEX_DIR") or os.getenv("DATA_DIR") or "./data"
    return Path(raw).resolve()


def _get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever
    from .retriever import HybridRetriever
    index_dir = _resolve_index_dir()
    top_k = int(os.getenv("HYBRID_TOP_K", "30"))
    logger.info("HybridRetriever loading from %s (top_k=%d)", index_dir, top_k)
    _retriever = HybridRetriever(index_dir=str(index_dir), top_k=top_k)
    return _retriever


def _flatten(raw: dict) -> dict:
    """teammate's nested {hybrid|lexical|semantic} → orchestrator's flat candidates list."""
    hybrid_block = raw.get("hybrid") or {}
    candidates: list[dict[str, Any]] = []
    for r in hybrid_block.get("results", []):
        scores = r.get("scores") or {}
        cid = str(r.get("candidate_id")) if r.get("candidate_id") is not None else None
        candidates.append({
            "id":    cid,
            "candidate_id": cid,   # preserve for downstream reranker contract
            "score": scores.get("primary"),
            "rank":              r.get("rank"),
            "name":              r.get("name"),
            "years_of_experience": r.get("years_of_experience"),
            "potential_roles":   r.get("potential_roles"),
            "core_skills":       r.get("core_skills"),
            "secondary_skills":  r.get("secondary_skills"),
            "soft_skills":       r.get("soft_skills"),
            "skill_summary":     r.get("skill_summary"),
            "scores":            scores,
            "explanation":       r.get("explanation"),
        })
    return {
        "candidates":      candidates,
        "lexical":         raw.get("lexical"),
        "semantic":        raw.get("semantic"),
        "meta":            hybrid_block.get("meta"),
        "query_breakdown": hybrid_block.get("query_breakdown"),
    }


def retrieve(intent: dict | list) -> dict:
    """
    Run hybrid (BM25 + dense) retrieval and return the orchestrator-shaped dict:
        {"candidates": [{"id","score",...}], "lexical":..., "semantic":..., "meta":..., "query_breakdown":...}

    `intent` should be the canonical intent dict from `modules.intent_processor.process()`.
    A bare list is also accepted (treated as queries with a minimal intent stub) for
    backward compatibility with simple callers.
    """
    if isinstance(intent, list):
        # Backward-compat path: caller passed just the queries list
        intent = {"queries": intent, "corrected": intent[0] if intent else "", "parsed": {}, "intent": {}}
    hr = _get_retriever()
    raw = hr.retrieve(intent)
    return _flatten(raw)


__all__ = ["retrieve"]
