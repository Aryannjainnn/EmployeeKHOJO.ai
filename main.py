"""FastAPI entry — serves the NexusSearch React UI and the retrieval API.

Endpoints
---------
GET  /                  → static index.html (NexusSearch UI, loaded via Babel)
GET  /search?q=...      → runs the orchestrator pipeline, returns frontend-shape
POST /search  {query}   → same, JSON body
POST /explain {row_data}→ streams AI explanation via SSE (text/event-stream)
GET  /health            → liveness probe
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
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

    Score field mapping (reranker → frontend):
        hybrid_score  → bm25_raw        (BM25 + dense hybrid signal)
        kg_score      → semantic_cosine (Knowledge Graph graph score)
        final_score   → hybrid_score    (fused final score, also rrf_score)
        alpha         → score_breakdown.BM25      (hybrid weight fraction)
        beta          → score_breakdown.Semantic   (KG weight fraction)
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

        # -- score breakdown (reranker fields → frontend slots) -------------
        final_score   = float(r.get("final_score")  or 0.0)
        hybrid_score  = float(r.get("hybrid_score") or 0.0)   # BM25+dense RRF
        kg_score      = float(r.get("kg_score")     or 0.0)   # KG graph score
        alpha         = float(r.get("alpha")        or 0.5)   # hybrid weight
        beta          = float(r.get("beta")         or 0.5)   # KG weight
        modifier_delta = float(r.get("modifier_delta") or 0.0)

        # score_breakdown percentages for the bar chart
        # alpha/beta already sum to 1 from the reranker
        bm25_pct     = round(alpha * 100)
        semantic_pct = round(beta  * 100)

        # -- keyword chips --------------------------------------------------
        matched = _as_list(r.get("matched_terms") or k.get("matched_terms"))
        kw_highlights = [{"term": t} for t in matched[:8]]

        # -- skill overlap --------------------------------------------------
        overlap = [
            s for s in skills
            if any(qs and (qs in s.lower() or s.lower() in qs) for qs in q_skills)
        ]

        # -- match_reasons → bullets (from KG) -----------------------------
        reasons = k.get("match_reasons") or []
        bullets = [
            f"Matched '{rr.get('query_term')}' → {rr.get('matched_node')} "
            f"via {rr.get('relationship')} ({rr.get('hops')}-hop, +{rr.get('score_delta'):.3f})"
            for rr in reasons[:6]
            if rr.get("matched_node") is not None
        ]
        # Add fusion info bullet
        bullets.append(
            f"Fusion: α={alpha:.2f}·hybrid + β={beta:.2f}·kg"
            + (f"  (modifier Δ={modifier_delta:+.3f})" if modifier_delta else "")
        )

        frontend_results.append({
            "id":        cid,
            "rank":      i,
            "title":     title,
            "industry":  "—",
            "location":  "—",
            "rrf_score": final_score,          # final fused score
            "source":    r.get("source", "both"),
            "preview":   (preview or "")[:500],
            "skills":    skills[:10],
            "explanation": {
                # Human-readable summary (used as fallback before LLM runs)
                "summary": (
                    f"{name}: final score {final_score:.3f} "
                    f"(source: {r.get('source')}). "
                    + (f"{len(matched)} search terms matched via graph. " if matched else "")
                    + (f"{len(overlap)} core-skill overlaps with query. " if overlap else "")
                ),
                "detail_bullets": bullets,

                # score_breakdown drives the three progress bars in the UI
                # BM25 slot  → hybrid retrieval weight  (alpha)
                # Semantic slot → KG weight             (beta)
                "score_breakdown": {
                    "BM25":     round(alpha, 3),
                    "Semantic": round(beta,  3),
                },

                # Raw score values shown in the Signal Analysis tab
                # bm25_raw        ← hybrid_score  (BM25+dense signal)
                # semantic_cosine ← kg_score       (knowledge-graph score)
                # hybrid_score    ← final_score    (fused output)
                "bm25_raw":        round(hybrid_score, 4),
                "semantic_cosine": round(kg_score,     4),
                "hybrid_score":    round(final_score,  4),

                # modifier delta (experience match bonus / negation penalty)
                "modifier_delta":  round(modifier_delta, 4),

                "keyword_highlights": kw_highlights,
                "skill_overlap":      overlap[:8],
                "transparency_notes": (
                    [hybrid_raw["_error"]] if hybrid_raw.get("_error") else []
                ) + (
                    [kg_raw["_error"]] if kg_raw.get("_error") else []
                ),
                "intent_alignment": (
                    f"Intent: {intent_block.get('primary_intent', 'unknown')} "
                    f"(confidence {intent_block.get('confidence', 0):.2f}). "
                    f"Retrieved via: {r.get('source')}."
                ),
            },
        })

    trace = out.get("trace") or {}

    return {
        "results":          frontend_results,
        "intent": {
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
async def explain_endpoint(payload: dict = Body(...)) -> StreamingResponse:
    """
    Stream an AI explanation for a single candidate card.

    The frontend sends the full row_data object (the card's data).
    We stream back SSE chunks so the text appears word-by-word.

    SSE format:
        data: <text chunk>\\n\\n
        data: [DONE]\\n\\n
    """
    row = (payload or {}).get("row_data") or {}

    async def _generate():
        try:
            from modules.explainability.explain import explain_stream
            import asyncio

            # explain_stream is a sync generator — run it in a thread pool
            loop = asyncio.get_event_loop()

            def _sync_gen():
                return list(explain_stream(row))

            chunks = await loop.run_in_executor(None, _sync_gen)
            for chunk in chunks:
                # SSE format
                safe = chunk.replace("\n", " ")
                yield f"data: {safe}\n\n"

        except Exception as exc:
            # Send the fallback explanation as a single chunk
            from modules.explainability.explain import _fallback_explanation
            text = _fallback_explanation(row)
            yield f"data: {text}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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