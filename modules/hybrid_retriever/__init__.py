"""Adapter: bridges the teammate's HybridRetriever to the orchestrator contract.

Teammate API   : HybridRetriever.retrieve(intent_json) -> {hybrid:{results}, ...}
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

# Expose package dir on sys.path so internal imports resolve
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

_retriever = None  # cached HybridRetriever instance


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
    """
    Teammate's nested {hybrid|lexical|semantic} → orchestrator's flat candidates list.

    Each candidate in the output has BOTH:
      - "id"           (orchestrator/_tag_and_combine contract)
      - "candidate_id" (reranker contract)
    so every downstream consumer can use whichever key it expects.
    """
    hybrid_block = raw.get("hybrid") or {}
    candidates: list[dict[str, Any]] = []

    for r in hybrid_block.get("results", []):
        scores = r.get("scores") or {}

        # candidate_id comes from the retriever; normalise to string
        cid = str(r.get("candidate_id")) if r.get("candidate_id") is not None else None

        candidates.append({
            # Both key names — orchestrator uses "id", reranker uses "candidate_id"
            "id":                  cid,
            "candidate_id":        cid,
            "score":               scores.get("primary") or scores.get("rrf"),
            "rank":                r.get("rank"),
            "name":                r.get("name"),
            "years_of_experience": r.get("years_of_experience"),
            "potential_roles":     r.get("potential_roles"),
            "core_skills":         r.get("core_skills"),
            "secondary_skills":    r.get("secondary_skills"),
            "soft_skills":         r.get("soft_skills"),
            "skill_summary":       r.get("skill_summary"),
            "scores":              scores,
            "explanation":         r.get("explanation"),
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
    Run hybrid (BM25 + dense) retrieval and return orchestrator-shaped dict:

        {
            "candidates": [{"id", "candidate_id", "score", ...}],
            "lexical":    ...,
            "semantic":   ...,
            "meta":       ...,
            "query_breakdown": ...,
        }

    `intent` may be:
      - The canonical intent dict from `modules.intent_processor.process()`
        (preferred — carries parsed entities for hard filtering).
      - A bare list of query strings (backward-compat; wrapped into a minimal
        intent stub so the retriever's query loop still works).
    """
    if isinstance(intent, list):
        # Backward-compat: caller passed just the queries list.
        # Wrap into a minimal intent dict so the retriever's internal logic
        # (strategy_map, parsed entities, etc.) still functions gracefully.
        intent = {
            "queries":             intent,
            "corrected":           intent[0] if intent else "",
            "original":            intent[0] if intent else "",
            "parsed":              {},
            "entities":            {},
            "intent":              {},
            "strategy_map":        {q: "original" for q in intent},
            "exclusion_filters":   {},
            "kg_expanded_queries": [],
            "kg_ready":            False,
        }

    hr  = _get_retriever()
    raw = hr.retrieve(intent)
    return _flatten(raw)


__all__ = ["retrieve"]