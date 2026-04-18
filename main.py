"""FastAPI entry — serves the NexusSearch React UI and the retrieval API.

Endpoints
---------
GET  /                  → static index.html (NexusSearch UI, loaded via Babel)
GET  /search?q=...      → runs the orchestrator pipeline, returns frontend-shape
POST /search  {query}   → same, JSON body
POST /explain {row_data}→ stub explanation endpoint (returns canned summary)
GET  /health            → liveness probe
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from orchestrator import run as run_pipeline
from orchestrator.schemas import SearchRequest

load_dotenv()

BASE = Path(__file__).parent
FRONTEND_DIR = BASE / "frontend"

app = FastAPI(title="Nexus — Talent Intelligence")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline output → frontend shape
# ─────────────────────────────────────────────────────────────────────────────

def _as_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [t.strip() for t in v.split(",") if t.strip()]
    return [v]


def _to_frontend_shape(out: dict) -> dict:
    """Transform orchestrator output into the shape NexusSearch.jsx expects.

    Frontend (per result) needs:
        id, rank, title, industry, location, rrf_score, preview, skills,
        explanation: {summary, detail_bullets, score_breakdown{BM25,Semantic},
                      bm25_raw, semantic_cosine, hybrid_score,
                      keyword_highlights[{term}], skill_overlap,
                      transparency_notes, intent_alignment}

    We carry two signals through the pipeline — hybrid (lexical+dense) and KG
    (graph traversal). We map them onto the card's BM25/Semantic slots so the
    existing UI renders without changes: BM25 ← hybrid signal, Semantic ← KG.
    """
    reranked   = out.get("reranked") or {}
    rr_results = reranked.get("results") or []
    hybrid_raw = out.get("hybrid_raw") or {}
    kg_raw     = out.get("kg_raw") or {}

    hybrid_by_id = {str(c.get("id")): c for c in hybrid_raw.get("candidates", [])}
    kg_by_id     = {str(c.get("id")): c for c in kg_raw.get("candidates", [])}

    intent_dict  = out.get("intent") or {}
    intent_block = intent_dict.get("intent") or {}
    parsed       = intent_dict.get("parsed") or {}
    q_skills     = [s.lower() for s in parsed.get("skills", [])]

    frontend_results = []
    for i, r in enumerate(rr_results, start=1):
        cid = str(r.get("candidate_id"))
        h = hybrid_by_id.get(cid, {})
        k = kg_by_id.get(cid, {})

        # -- display fields -------------------------------------------------
        name  = h.get("name") or k.get("name") or f"Candidate {cid}"
        roles = _as_list(k.get("potential_roles") or h.get("potential_roles"))
        title = roles[0] if roles else name

        skills = r.get("core_skills") or k.get("core_skills") or h.get("core_skills") or []
        skills = _as_list(skills)

        preview = (
            k.get("skill_summary")
            or h.get("skill_summary")
            or ""
        )

        # -- score breakdown ------------------------------------------------
        final   = float(r.get("final_score")  or 0.0)
        h_score = float(r.get("hybrid_score") or 0.0)
        k_score = float(r.get("kg_score")     or 0.0)
        total   = (h_score + k_score) or 1.0
        h_frac  = h_score / total
        k_frac  = k_score / total

        # -- keyword chips --------------------------------------------------
        matched = _as_list(r.get("matched_terms") or k.get("matched_terms"))
        kw_highlights = [{"term": t} for t in matched[:8]]

        # -- skill overlap --------------------------------------------------
        overlap = [
            s for s in skills
            if any(qs and (qs in s.lower() or s.lower() in qs) for qs in q_skills)
        ]

        # -- reasons → bullets ---------------------------------------------
        reasons = k.get("match_reasons") or []
        bullets = [
            f"Matched '{rr.get('query_term')}' → {rr.get('matched_node')} "
            f"via {rr.get('relationship')} ({rr.get('hops')}-hop, +{rr.get('score_delta'):.3f})"
            for rr in reasons[:6]
            if rr.get("matched_node") is not None
        ]
        alpha = r.get("alpha")
        beta  = r.get("beta")
        delta = r.get("modifier_delta")
        if alpha is not None and beta is not None:
            bullets.append(
                f"Fusion: α={alpha:.2f}·hybrid + β={beta:.2f}·kg"
                + (f"  (modifier Δ={delta:+.3f})" if delta else "")
            )

        frontend_results.append({
            "id":         cid,
            "rank":       i,
            "title":      title,
            "industry":   "—",
            "location":   "—",
            "rrf_score":  final,
            "preview":    (preview or "")[:500],
            "skills":     skills[:10],
            "explanation": {
                "summary":
                    f"{name}: final score {final:.3f} "
                    f"(source: {r.get('source')}). "
                    + (f"{len(matched)} graph terms matched. " if matched else "")
                    + (f"{len(overlap)} core-skill overlaps with query. " if overlap else ""),
                "detail_bullets":     bullets,
                "score_breakdown":    {"BM25": round(h_frac, 3), "Semantic": round(k_frac, 3)},
                "bm25_raw":           round(h_score, 4),
                "semantic_cosine":    round(k_score, 4),
                "hybrid_score":       round(final, 4),
                "keyword_highlights": kw_highlights,
                "skill_overlap":      overlap[:8],
                "transparency_notes": (
                    [hybrid_raw["_error"]] if hybrid_raw.get("_error") else []
                ) + (
                    [kg_raw["_error"]] if kg_raw.get("_error") else []
                ),
                "intent_alignment":
                    f"Intent: {intent_block.get('primary_intent', 'unknown')} "
                    f"(confidence {intent_block.get('confidence', 0):.2f}). "
                    f"Retrieved via: {r.get('source')}.",
            },
        })

    trace = out.get("trace") or {}

    return {
        "results":           frontend_results,
        "intent":            {
            "primary":    intent_block.get("primary_intent", "unknown"),
            "confidence": intent_block.get("confidence"),
            "modifiers":  intent_block.get("modifiers", []),
            "corrected":  intent_dict.get("corrected"),
        },
        "total_candidates":  reranked.get("total_count", len(frontend_results)),
        "timing_ms":         trace.get("total_ms"),
        "expanded_queries":  (intent_dict.get("queries") or [])[:6],
        "trace":             trace,
    }


# ─────────────────────────────────────────────────────────────────────────────
# API routes
# ─────────────────────────────────────────────────────────────────────────────

def _run_and_shape(q: str) -> dict:
    q = (q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is required")
    try:
        result = run_pipeline(q)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pipeline error: {e!s}")
    return _to_frontend_shape(result)


@app.get("/search")
def search_get(
    q: str = Query("", description="search query"),
    k: int = Query(10, ge=1, le=100),
    mode: str = Query("hybrid"),
) -> JSONResponse:
    return JSONResponse(content=_run_and_shape(q))


@app.post("/search")
def search_post(req: SearchRequest) -> JSONResponse:
    return JSONResponse(content=_run_and_shape(req.query))


@app.post("/explain")
def explain_endpoint(payload: dict = Body(...)) -> JSONResponse:
    """Stub: returns the canned summary already baked into the row.

    Swap this with an LLM call when the explainability module is ready.
    """
    row = (payload or {}).get("row_data") or {}
    expl = row.get("explanation") or {}
    text = (
        expl.get("summary")
        or expl.get("intent_alignment")
        or "No explanation available for this candidate."
    )
    return JSONResponse(content={"explanation": text})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
# Static frontend — must be last so API routes take priority
# ─────────────────────────────────────────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
