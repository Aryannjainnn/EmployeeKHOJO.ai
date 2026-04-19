"""
modules/explainability/explain.py

On-demand, single-candidate explainer using Groq (llama-3.1-8b-instant).
Called by the /explain endpoint in main.py when the user clicks the
"Why Selected" button in the UI.

Input  : one candidate row_data dict (the exact shape NexusSearch.jsx
         sends — i.e. the frontend card's full data object).
Output : a streaming generator of text chunks (for SSE) or a plain string
         (for non-streaming callers).

Score fields used (from reranker output, mapped through _to_frontend_shape):
    row_data.rrf_score          ← final_score   (0-1 fused)
    row_data.explanation.bm25_raw        ← hybrid_score  (BM25 + dense RRF)
    row_data.explanation.semantic_cosine ← kg_score      (KG graph score)
    row_data.explanation.hybrid_score    ← final_score   (same as rrf_score)
    row_data.explanation.score_breakdown.BM25      ← hybrid fraction
    row_data.explanation.score_breakdown.Semantic  ← kg fraction
    row_data.explanation.keyword_highlights        ← matched_terms list
    row_data.explanation.skill_overlap             ← skills matching query
    row_data.explanation.intent_alignment          ← intent string
    row_data.explanation.detail_bullets            ← match_reasons bullets
    row_data.skills                                ← core_skills list
    row_data.title                                 ← potential_roles[0]
"""

from __future__ import annotations

import os
import logging
from typing import Generator, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider config — mirrors llm_intent_pipeline.py convention
# ---------------------------------------------------------------------------
_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model":    "llama-3.1-8b-instant",
        "env_var":  "GROQ_API_KEY",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "model":    "llama3.1-8b",
        "env_var":  "CEREBRAS_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model":    "meta-llama/llama-3.1-8b-instruct:free",
        "env_var":  "OPENROUTER_API_KEY",
    },
}

_SYSTEM_PROMPT = """\
You are a senior talent advisor writing a concise hiring recommendation.
Based on the candidate signal data provided, write exactly 2 short paragraphs:

Paragraph 1 — MATCH QUALITY:
  In 2-3 sentences explain how strongly this candidate matched the search.
  Reference the hybrid retrieval signal and knowledge-graph alignment naturally
  (no raw numbers, no field names). Phrase it like a recruiter would.

Paragraph 2 — RECOMMENDATION:
  In 1-2 sentences give a direct hiring recommendation. State any notable
  strength or caveat. End with a clear action: "recommend for interview",
  "strong shortlist", etc.

Rules:
  - Never mention field names (bm25, rrf, cosine, kg_score, alpha, beta …)
  - Never write raw numbers from scores
  - No bullet points — flowing paragraphs only
  - Do not start with "This candidate"
  - Total length: 80-120 words
  - Be specific, concrete, and direct\
"""


def _build_user_prompt(row: dict[str, Any]) -> str:
    """
    Build the LLM user prompt from the frontend card's row_data dict.
    Uses the exact field names that _to_frontend_shape produces.
    """
    expl   = row.get("explanation") or {}
    sb     = expl.get("score_breakdown") or {}
    skills = row.get("skills") or []
    title  = row.get("title") or "Unknown Role"
    
    # Score signals — map frontend names back to meaningful tiers
    hybrid_frac = float(sb.get("BM25", 0))       # hybrid retrieval contribution
    kg_frac     = float(sb.get("Semantic", 0))    # KG graph contribution
    rrf         = float(row.get("rrf_score") or expl.get("hybrid_score") or 0)

    def _tier(frac: float) -> str:
        if frac >= 0.65: return "very strong"
        if frac >= 0.45: return "strong"
        if frac >= 0.25: return "moderate"
        return "weak"

    def _rrf_tier(score: float) -> str:
        if score >= 0.80: return "top-ranked"
        if score >= 0.50: return "highly ranked"
        if score >= 0.25: return "mid-ranked"
        return "lower-ranked"

    # Keywords and overlap
    kw_list = [k.get("term") or k if isinstance(k, dict) else str(k)
               for k in (expl.get("keyword_highlights") or [])]
    overlap  = expl.get("skill_overlap") or []
    bullets  = expl.get("detail_bullets") or []
    intent   = expl.get("intent_alignment") or ""
    source   = row.get("source") or expl.get("intent_alignment") or ""

    lines = [
        f"Role/Title: {title}",
        f"Skills on profile: {', '.join(skills[:8]) if skills else 'not listed'}",
        "",
        "=== Match Signals ===",
        f"Overall ranking: {_rrf_tier(rrf)} (combined fusion score)",
        f"Keyword/Hybrid retrieval: {_tier(hybrid_frac)} signal"
        f" — lexical and dense vector search contribution",
        f"Knowledge Graph alignment: {_tier(kg_frac)} signal"
        f" — graph traversal across skill/role/domain nodes",
        f"Retrieved via: {source}",
    ]

    if kw_list:
        lines.append(f"Matched search terms: {', '.join(kw_list[:8])}")
    if overlap:
        lines.append(f"Skills overlapping with query: {', '.join(overlap[:6])}")
    if intent:
        lines.append(f"Intent context: {intent[:120]}")
    if bullets:
        lines.append("Notable signals:")
        for b in bullets[:4]:
            lines.append(f"  - {b}")

    lines.append("\nWrite the 2-paragraph hiring recommendation now.")
    return "\n".join(lines)


def _get_client():
    """Return an OpenAI-compatible client for the fastest available provider."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai  — required for the explainer")

    provider_name = os.getenv("INTENT_LLM_PROVIDER", "groq").lower()
    cfg = _PROVIDERS.get(provider_name) or _PROVIDERS["groq"]

    api_key = os.getenv(cfg["env_var"]) or os.getenv("GROQ_API_KEY")
    if not api_key:
        # Try all providers in order
        for name, c in _PROVIDERS.items():
            key = os.getenv(c["env_var"])
            if key:
                cfg = c
                api_key = key
                logger.info("Explainer: using provider %s", name)
                break

    if not api_key:
        raise RuntimeError(
            "No LLM API key found. Set GROQ_API_KEY (free at console.groq.com)"
        )

    return OpenAI(api_key=api_key, base_url=cfg["base_url"]), cfg["model"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_stream(row: dict[str, Any]) -> Generator[str, None, None]:
    """
    Stream explanation text chunks for a single candidate row.
    Yields raw text strings as the LLM produces them.
    Use this for SSE/streaming HTTP responses.
    """
    client, model = _get_client()
    prompt = _build_user_prompt(row)

    try:
        stream = client.chat.completions.create(
            model=model,
            temperature=0.3,
            max_tokens=220,
            stream=True,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
    except Exception as exc:
        logger.error("Explainer stream error: %s", exc)
        yield _fallback_explanation(row)


def explain_single(row: dict[str, Any]) -> str:
    """
    Non-streaming version — returns the full explanation string.
    Used by the orchestrator chain (explain module contract).
    """
    return "".join(explain_stream(row))


def explain(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Orchestrator chain contract: explain(payload) -> dict.
    payload keys: query, intent, reranked, hybrid, kg, tagged_candidates.

    Returns per-candidate explanation stubs — real text is generated
    on-demand when the user clicks "Why Selected" in the UI.
    """
    reranked = payload.get("reranked") or {}
    results  = reranked.get("results") or []

    # Return lightweight stubs so the chain doesn't block on LLM calls.
    # The frontend fetches real explanations one-at-a-time via /explain.
    per_candidate = {}
    for r in results:
        cid = str(r.get("candidate_id"))
        per_candidate[cid] = {
            "stub": True,
            "summary": (
                f"Candidate {cid}: final score {r.get('final_score', 0):.3f} "
                f"(source: {r.get('source', '?')}). "
                "Click 'Why Selected' to generate a full AI explanation."
            ),
        }

    return {
        "per_candidate": per_candidate,
        "process_summary": "On-demand explanations — click Why Selected per card.",
        "intent_summary":  str(payload.get("intent", {}).get("intent", {})
                               .get("primary_intent", "")),
        "_stub": True,
    }


# ---------------------------------------------------------------------------
# Fallback (no LLM available)
# ---------------------------------------------------------------------------

def _fallback_explanation(row: dict[str, Any]) -> str:
    expl   = row.get("explanation") or {}
    sb     = expl.get("score_breakdown") or {}
    skills = row.get("skills") or []
    title  = row.get("title") or "this candidate"
    overlap = expl.get("skill_overlap") or []

    h = float(sb.get("BM25", 0))
    k = float(sb.get("Semantic", 0))
    dominant = "keyword and lexical" if h >= k else "knowledge-graph and semantic"

    skill_str = ", ".join(overlap[:4]) if overlap else ", ".join(skills[:4])
    return (
        f"{title} was retrieved with {dominant} signals leading the match. "
        f"{'Relevant skills include ' + skill_str + '.' if skill_str else ''} "
        f"The combined retrieval score places this candidate in the result set "
        f"based on alignment with the search intent across both retrieval channels.\n\n"
        f"Review the keyword matches and skill overlap for a manual fit assessment. "
        f"Recommend for initial screening."
    )