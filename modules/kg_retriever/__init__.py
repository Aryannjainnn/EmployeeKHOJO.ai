"""Adapter: bridges KGRetriever to the orchestrator contract.

Internal API  : KGRetriever.retrieve(intent_dict) -> list[dict]
Orchestrator  : retrieve(intent_dict) -> {"candidates": [{"id","candidate_id","score",...}], "raw_results": [...]}

The KGRetriever instance (Neo4j driver + CSV load) is cached across calls — the
first call is heavy, subsequent calls reuse the connection.

Env vars consumed:
    NEO4J_URI, NEO4J_USERNAME (or NEO4J_USER), NEO4J_PASSWORD
    PROFILES_CSV   — full path to profiles.csv  (optional)
    DATA_DIR / INDEX_DIR — directory containing profiles.csv (fallback)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_retriever = None


def _resolve_csv_path() -> Path:
    explicit = os.getenv("PROFILES_CSV")
    if explicit:
        return Path(explicit).resolve()
    data_dir = os.getenv("DATA_DIR") or os.getenv("INDEX_DIR") or "./data"
    return (Path(data_dir) / "profiles.csv").resolve()


def _get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever
    from .retrieve import KGRetriever

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PASSWORD")
    if not (uri and user and pwd):
        raise RuntimeError(
            "kg_retriever: missing Neo4j credentials — set "
            "NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD in .env"
        )

    csv_path = _resolve_csv_path()
    if not csv_path.exists():
        raise RuntimeError(f"kg_retriever: profiles.csv not found at {csv_path}")

    logger.info("KGRetriever loading — uri=%s csv=%s", uri, csv_path)
    _retriever = KGRetriever(uri, user, pwd, str(csv_path))
    return _retriever


def _to_candidate(r: dict) -> dict:
    """Normalise a raw KGRetriever result to the orchestrator candidate shape.

    Preserves both ``id`` (orchestrator contract) and ``candidate_id``
    (reranker contract) so downstream code can use either.
    """
    cid = str(r.get("candidate_id"))
    return {
        "id":                   cid,
        "candidate_id":         cid,
        "score":                r.get("score"),
        "name":                 r.get("name", ""),
        "graph_score":          r.get("graph_score"),
        "experience_score":     r.get("experience_score"),
        "core_skills":          r.get("core_skills", []),
        "secondary_skills":     r.get("secondary_skills", []),
        "soft_skills":          r.get("soft_skills", []),
        "skill_summary":        r.get("skill_summary", ""),
        "years_of_experience":  r.get("years_of_experience"),
        "potential_roles":      r.get("potential_roles", []),
        "matched_terms":        r.get("matched_terms", []),
        "match_reasons":        r.get("match_reasons", []),
    }


def retrieve(intent: dict | list) -> dict:
    """Run KG retrieval and return orchestrator-shaped output.

    Parameters
    ----------
    intent : dict | list
        Full canonical intent dict from ``modules.intent_processor.process()``
        (preferred). A bare list of query strings is accepted as a backward-
        compat convenience — it is wrapped into a minimal intent stub.

    Returns
    -------
    dict
        ``{"candidates": [...], "raw_results": [...]}``
        where each candidate has at minimum ``id``, ``candidate_id`` and
        ``score``, plus the KG-enriched metadata (match_reasons, core_skills …).
    """
    if isinstance(intent, list):
        intent = {
            "queries":             list(intent),
            "corrected":           intent[0] if intent else "",
            "parsed":              {},
            "intent":              {},
            "strategy_map":        {},
            "kg_expanded_queries": list(intent),
        }

    kg = _get_retriever()
    raw_list = kg.retrieve(intent)
    candidates = [_to_candidate(r) for r in raw_list]
    return {"candidates": candidates, "raw_results": raw_list}


__all__ = ["retrieve"]
