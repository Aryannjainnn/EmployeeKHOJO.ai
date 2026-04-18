"""LangChain pipeline — wires the 5 modules end to end.

Flow:
    query
     -> intent_processor.process(query)                 # returns JSON with many queries
     -> RunnableParallel:
            hybrid_retriever.retrieve(queries)
            kg_retriever.retrieve(queries_or_kg_queries)
     -> tag_and_combine  (attaches source: hybrid|kg|both)
     -> reranker.rerank(combined)
     -> explainability.explain(reranked + context)
     -> PipelineResult

Parallelism: LangChain's RunnableParallel runs its branches concurrently
(ThreadPoolExecutor) on .invoke, so hybrid + kg execute simultaneously.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.runnables import RunnableLambda, RunnableParallel

from modules.intent_processor import process as intent_process
from modules.hybrid_retriever import retrieve as hybrid_retrieve
from modules.kg_retriever import retrieve as kg_retrieve
from modules.reranker import rerank as rerank_module
from modules.explainability import explain as explain_module


# ─── Stage 1: intent ─────────────────────────────────────────
def _run_intent(payload: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    intent = intent_process(payload["query"])
    return {
        "query": payload["query"],
        "intent": intent,
        "_trace": {"intent_ms": round((time.time() - t0) * 1000, 1)},
    }


# ─── Stage 2a: hybrid (parallel branch) ──────────────────────
def _run_hybrid(payload: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    # Pass the full canonical intent dict — the hybrid retriever uses parsed
    # entities (negated_skills, experience_band, ...) for hard filtering, not
    # just the expanded queries.
    result = hybrid_retrieve(payload["intent"])
    result.setdefault("_timing_ms", round((time.time() - t0) * 1000, 1))
    return result


# ─── Stage 2b: kg (parallel branch) ──────────────────────────
def _run_kg(payload: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    # Prefer KG-specific expanded queries if the intent module produced them
    kg_qs = payload["intent"].get("kg_expanded_queries") or payload["intent"].get("queries", [])
    result = kg_retrieve(kg_qs)
    result.setdefault("_timing_ms", round((time.time() - t0) * 1000, 1))
    return result


# Pass-through for carrying context into the parallel block
_passthrough_query = RunnableLambda(lambda p: p["query"])
_passthrough_intent = RunnableLambda(lambda p: p["intent"])
_passthrough_trace = RunnableLambda(lambda p: p.get("_trace", {}))


# ─── Stage 3: tag + combine ──────────────────────────────────
def _tag_and_combine(payload: dict[str, Any]) -> dict[str, Any]:
    """Merge hybrid and kg candidate lists; tag each with its source.

    Expected candidate shape from each retriever (minimum contract):
        {"candidates": [{"id": "<str>", "score": <float>, ...}, ...]}

    Any extra fields are preserved under the per-source sub-dict so the
    reranker/explainability can use them.
    """
    hybrid = payload["hybrid"] or {}
    kg = payload["kg"] or {}
    hybrid_cands = hybrid.get("candidates", [])
    kg_cands = kg.get("candidates", [])

    by_id: dict[str, dict[str, Any]] = {}

    for c in hybrid_cands:
        cid = str(c.get("id"))
        by_id[cid] = {
            "id": cid,
            "score": c.get("score"),
            "source": "hybrid",
            "hybrid": c,
            "kg": None,
        }

    for c in kg_cands:
        cid = str(c.get("id"))
        if cid in by_id:
            by_id[cid]["source"] = "both"
            by_id[cid]["kg"] = c
        else:
            by_id[cid] = {
                "id": cid,
                "score": c.get("score"),
                "source": "kg",
                "hybrid": None,
                "kg": c,
            }

    tagged = list(by_id.values())

    trace = dict(payload.get("trace", {}))
    trace["hybrid_ms"] = hybrid.get("_timing_ms")
    trace["kg_ms"] = kg.get("_timing_ms")
    trace["hybrid_n"] = len(hybrid_cands)
    trace["kg_n"] = len(kg_cands)
    trace["overlap_n"] = sum(1 for c in tagged if c["source"] == "both")

    return {
        "query": payload["query"],
        "intent": payload["intent"],
        "hybrid_raw": hybrid,
        "kg_raw": kg,
        "tagged_candidates": tagged,
        "trace": trace,
    }


# ─── Stage 4: rerank ─────────────────────────────────────────
def _run_rerank(payload: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    reranked = rerank_module({
        "query": payload["query"],
        "intent": payload["intent"],
        "hybrid": payload["hybrid_raw"],
        "kg": payload["kg_raw"],
        "tagged_candidates": payload["tagged_candidates"],
    })
    payload["reranked"] = reranked
    payload["trace"]["rerank_ms"] = round((time.time() - t0) * 1000, 1)
    return payload


# ─── Stage 5: explainability ─────────────────────────────────
def _run_explain(payload: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    explanations = explain_module({
        "query": payload["query"],
        "intent": payload["intent"],
        "reranked": payload["reranked"],
        "hybrid": payload["hybrid_raw"],
        "kg": payload["kg_raw"],
        "tagged_candidates": payload["tagged_candidates"],
    })
    payload["explanations"] = explanations
    payload["trace"]["explain_ms"] = round((time.time() - t0) * 1000, 1)
    return payload


# ─── Assembled pipeline ──────────────────────────────────────
pipeline = (
    RunnableLambda(_run_intent)
    | RunnableParallel(
        hybrid=RunnableLambda(_run_hybrid),
        kg=RunnableLambda(_run_kg),
        query=_passthrough_query,
        intent=_passthrough_intent,
        trace=_passthrough_trace,
    )
    | RunnableLambda(_tag_and_combine)
    | RunnableLambda(_run_rerank)
    | RunnableLambda(_run_explain)
)


def run(query: str) -> dict[str, Any]:
    """Sync entry point used by FastAPI. Returns the final payload dict."""
    t0 = time.time()
    out = pipeline.invoke({"query": query})
    out["trace"]["total_ms"] = round((time.time() - t0) * 1000, 1)
    return out


async def arun(query: str) -> dict[str, Any]:
    """Async entry point — use ainvoke for true non-blocking parallel branches."""
    t0 = time.time()
    out = await pipeline.ainvoke({"query": query})
    out["trace"]["total_ms"] = round((time.time() - t0) * 1000, 1)
    return out
