"""Adapter: exposes `process(query) -> dict` over whichever intent pipeline is
available.

Resolution order (first one that works wins):
  1. LLMIntentPipeline from llm_intent_pipeline.py
     — requires GROQ_API_KEY (or the env var for your chosen provider)
     — also has its own built-in fallback to IntentQueryPipeline on API failure
  2. IntentQueryPipeline from intent_pipeline.py
     — pure local, uses BART-large-mnli + SBERT + SymSpell, no API key needed

Both paths are normalised to the canonical orchestrator shape:
    {original, corrected, intent, parsed, queries, strategy_map,
     exclusion_filters, kg_expanded_queries, kg_ready}
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Any

logger = logging.getLogger(__name__)

# llm_intent_pipeline.py uses `from intent_pipeline import IntentQueryPipeline`
# (a bare, non-relative import) inside its fallback path. Make that resolve by
# putting this folder on sys.path. Keeps the teammate's file untouched.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

_pipeline = None
_pipeline_kind: str | None = None  # "llm" or "local"


# ── Provider → env-var map (mirrors llm_intent_pipeline.PROVIDERS) ──────────
_PROVIDER_ENV = {
    "groq":       "GROQ_API_KEY",
    "cerebras":   "CEREBRAS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "sambanova":  "SAMBANOVA_API_KEY",
}


def _try_llm_pipeline():
    """Return an LLMIntentPipeline instance or None if not usable."""
    provider = os.getenv("INTENT_LLM_PROVIDER", "groq").lower()
    env_var = _PROVIDER_ENV.get(provider)
    if not env_var or not os.getenv(env_var):
        logger.info("LLM intent pipeline skipped — no %s in env", env_var)
        return None
    try:
        from .llm_intent_pipeline import LLMIntentPipeline
    except ImportError as e:
        logger.info("llm_intent_pipeline.py not present (%s) — falling back to local", e)
        return None
    try:
        return LLMIntentPipeline(provider=provider)
    except Exception as e:
        logger.warning("LLMIntentPipeline init failed (%s) — falling back to local", e)
        return None


def _try_local_pipeline():
    from .intent_pipeline import IntentQueryPipeline
    return IntentQueryPipeline()


def _get_pipeline():
    global _pipeline, _pipeline_kind
    if _pipeline is not None:
        return _pipeline
    llm = _try_llm_pipeline()
    if llm is not None:
        _pipeline = llm
        _pipeline_kind = "llm"
        logger.info("Intent pipeline: LLM")
        return _pipeline
    _pipeline = _try_local_pipeline()
    _pipeline_kind = "local"
    logger.info("Intent pipeline: LOCAL (no LLM provider configured)")
    return _pipeline


# ── Output normalisation ────────────────────────────────────────────────────

def _normalise_llm(raw: dict, original_query: str) -> dict:
    """LLMIntentResult.to_dict() → canonical orchestrator shape."""
    ent = raw.get("entities", {}) or {}
    intent = raw.get("intent", {}) or {}
    return {
        "original": original_query,
        "corrected": raw.get("corrected_query", original_query),
        "intent": {
            "primary_intent": intent.get("primary", "unknown"),
            "confidence": intent.get("confidence", 0.0),
            "modifiers": intent.get("modifiers", []),
            "top3_scores": intent.get("top3_scores", {}),
            "reasoning": intent.get("reasoning", ""),
        },
        "parsed": {
            "skills": ent.get("skills", []),
            "negated_skills": ent.get("negated_skills", []),
            "role": ent.get("role"),
            "experience_band": ent.get("experience_band"),
            "experience_years": ent.get("experience_years"),
            "domain": ent.get("domain"),
            "location": ent.get("location"),
            "negated_location": ent.get("negated_location"),
        },
        "queries": raw.get("expanded_queries", [raw.get("corrected_query", original_query)]),
        "strategy_map": raw.get("query_strategies", {}),
        "exclusion_filters": raw.get("exclusion_filters", {}),
        "kg_expanded_queries": [],
        "kg_ready": False,
        "_source": "llm",
    }


def _normalise_local(raw: dict) -> dict:
    """Local IntentQueryPipeline already returns the canonical shape."""
    raw = dict(raw)
    raw["_source"] = "local"
    return raw


# ── Public API ──────────────────────────────────────────────────────────────

def process(query: str) -> dict:
    pipe = _get_pipeline()
    result = pipe.run(query)
    raw = result.to_dict()
    if _pipeline_kind == "llm":
        return _normalise_llm(raw, query)
    return _normalise_local(raw)


__all__ = ["process"]
