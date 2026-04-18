"""
test_integration.py — End-to-end smoke test for the orchestrator pipeline.

Run from project root:
    python test_integration.py
    python test_integration.py --query "senior python developer with aws"

What it checks, in order:
    1. Imports resolve                    (modules + orchestrator)
    2. Env / files present                (NEO4J_*, profiles.csv, index_dir)
    3. intent_processor.process(query)    → canonical intent dict
    4. hybrid_retriever.retrieve(intent)  → {"candidates": [...]}
    5. kg_retriever.retrieve(intent)      → {"candidates": [...]}
    6. reranker.rerank(intent, H, K)      → {"total_count", "results"}
    7. orchestrator.run(query)            → full pipeline payload
    8. Schema sanity on the final payload
    9. Latency breakdown printed from payload["trace"]

Exit code: 0 on success, 1 on any hard failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ── Pretty-print helpers ──────────────────────────────────────────────────
def hdr(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")

def line(label: str, value: Any, ok: bool | None = None) -> None:
    mark = "    " if ok is None else (" OK " if ok else "FAIL")
    print(f"  [{mark}] {label:<40} {value}")

RESULTS: list[tuple[str, bool, str]] = []

def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))


# ── Load .env ─────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    for cand in [Path.cwd() / ".env", Path(__file__).parent / ".env"]:
        if cand.exists():
            load_dotenv(cand)
            break
    else:
        load_dotenv()
except ImportError:
    pass


# ── 1. Imports ────────────────────────────────────────────────────────────
hdr("1. Imports")

intent_process = hybrid_retrieve = kg_retrieve = rerank_module = run_pipeline = None

try:
    from modules.intent_processor import process as intent_process
    line("modules.intent_processor.process", "OK", True)
    record("import: intent_processor", True)
except Exception as e:
    line("modules.intent_processor.process", f"FAIL — {e}", False)
    record("import: intent_processor", False, str(e))

try:
    from modules.hybrid_retriever import retrieve as hybrid_retrieve
    line("modules.hybrid_retriever.retrieve", "OK", True)
    record("import: hybrid_retriever", True)
except Exception as e:
    line("modules.hybrid_retriever.retrieve", f"FAIL — {e}", False)
    record("import: hybrid_retriever", False, str(e))

try:
    from modules.kg_retriever import retrieve as kg_retrieve
    line("modules.kg_retriever.retrieve", "OK", True)
    record("import: kg_retriever", True)
except Exception as e:
    line("modules.kg_retriever.retrieve", f"FAIL — {e}", False)
    record("import: kg_retriever", False, str(e))

try:
    from modules.reranker import rerank as rerank_module
    line("modules.reranker.rerank", "OK", True)
    record("import: reranker", True)
except Exception as e:
    line("modules.reranker.rerank", f"FAIL — {e}", False)
    record("import: reranker", False, str(e))

try:
    from orchestrator import run as run_pipeline
    line("orchestrator.run", "OK", True)
    record("import: orchestrator", True)
except Exception as e:
    line("orchestrator.run", f"FAIL — {e}", False)
    record("import: orchestrator", False, str(e))


# ── 2. Env & files ────────────────────────────────────────────────────────
hdr("2. Env & files")

env_ok = True
for var in ("NEO4J_URI", "NEO4J_PASSWORD"):
    val = os.getenv(var)
    ok = bool(val)
    display = (val[:20] + "...") if val and len(val) > 20 else (val or "NOT SET")
    line(var, display, ok)
    record(f"env: {var}", ok)
    env_ok &= ok

# NEO4J_USERNAME or NEO4J_USER
user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
line("NEO4J_USERNAME/USER", user or "NOT SET", bool(user))
record("env: NEO4J_USERNAME", bool(user))
env_ok &= bool(user)

data_dir = Path(os.getenv("DATA_DIR") or os.getenv("INDEX_DIR") or "./data")
csv_path = Path(os.getenv("PROFILES_CSV") or data_dir / "profiles.csv")
line("profiles.csv", f"{csv_path} {'(exists)' if csv_path.exists() else '(MISSING)'}", csv_path.exists())
record("file: profiles.csv", csv_path.exists())

for fname in ("bm25.pkl", "dense.pkl", "dense_matrix.npy"):
    p = data_dir / fname
    line(f"index/{fname}", f"{p} {'(exists)' if p.exists() else '(MISSING)'}", p.exists())
    record(f"file: {fname}", p.exists())


# ── Parse args ────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--query", default="senior python developer with machine learning")
ap.add_argument("--skip-pipeline", action="store_true",
                help="run only the per-module checks, not the full orchestrator")
args = ap.parse_args()
Q = args.query


# ── 3. Intent ─────────────────────────────────────────────────────────────
hdr(f"3. intent_processor.process({Q!r})")

intent = None
if intent_process is None:
    line("intent", "SKIPPED — import failed", False)
    record("stage: intent", False, "import failed")
else:
    try:
        t0 = time.perf_counter()
        intent = intent_process(Q)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        line("process()", f"OK  ({ms}ms)  _source={intent.get('_source')}", True)
        line("  corrected", intent.get("corrected"))
        line("  parsed.skills", intent.get("parsed", {}).get("skills"))
        line("  parsed.role", intent.get("parsed", {}).get("role"))
        line("  parsed.experience_band", intent.get("parsed", {}).get("experience_band"))
        line("  intent.primary_intent", intent.get("intent", {}).get("primary_intent"))
        line("  #queries", len(intent.get("queries", [])))
        record("stage: intent", True, f"{ms}ms")
    except Exception as e:
        line("process()", f"FAIL — {e}", False)
        traceback.print_exc()
        record("stage: intent", False, str(e))


# ── 4. Hybrid ─────────────────────────────────────────────────────────────
hdr("4. hybrid_retriever.retrieve(intent)")

hybrid_out = None
if hybrid_retrieve is None or intent is None:
    line("hybrid", "SKIPPED — upstream missing", False)
    record("stage: hybrid", False, "upstream missing")
else:
    try:
        t0 = time.perf_counter()
        hybrid_out = hybrid_retrieve(intent)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        cands = hybrid_out.get("candidates", [])
        line("retrieve()", f"OK  ({ms}ms)  candidates={len(cands)}", True)
        if cands:
            top = cands[0]
            line("  candidate[0].id", top.get("id"))
            line("  candidate[0].candidate_id", top.get("candidate_id"))
            line("  candidate[0].score", top.get("score"))
            missing = {"id", "candidate_id", "score"} - set(top.keys())
            line("  has id+candidate_id+score", "yes" if not missing else f"MISSING={missing}", not missing)
        record("stage: hybrid", True, f"n={len(cands)} {ms}ms")
    except Exception as e:
        line("retrieve()", f"FAIL — {e}", False)
        traceback.print_exc()
        record("stage: hybrid", False, str(e))


# ── 5. KG ─────────────────────────────────────────────────────────────────
hdr("5. kg_retriever.retrieve(intent)")

kg_out = None
if kg_retrieve is None or intent is None:
    line("kg", "SKIPPED — upstream missing", False)
    record("stage: kg", False, "upstream missing")
else:
    try:
        t0 = time.perf_counter()
        kg_out = kg_retrieve(intent)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        cands = kg_out.get("candidates", [])
        line("retrieve()", f"OK  ({ms}ms)  candidates={len(cands)}", True)
        if cands:
            top = cands[0]
            line("  candidate[0].id", top.get("id"))
            line("  candidate[0].candidate_id", top.get("candidate_id"))
            line("  candidate[0].score", top.get("score"))
            line("  candidate[0].match_reasons", f"{len(top.get('match_reasons', []))} reasons")
        record("stage: kg", True, f"n={len(cands)} {ms}ms")
    except Exception as e:
        line("retrieve()", f"FAIL — {e}", False)
        traceback.print_exc()
        record("stage: kg", False, str(e))


# ── 6. Rerank (standalone) ────────────────────────────────────────────────
hdr("6. reranker.rerank(intent, hybrid_cands, kg_cands)")

reranked = None
if rerank_module is None or intent is None:
    line("rerank", "SKIPPED — upstream missing", False)
    record("stage: rerank", False, "upstream missing")
else:
    try:
        h_cands = (hybrid_out or {}).get("candidates", [])
        k_cands = (kg_out or {}).get("candidates", [])
        t0 = time.perf_counter()
        reranked = rerank_module(intent, h_cands, k_cands)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        total = reranked.get("total_count")
        results = reranked.get("results", [])
        line("rerank()", f"OK  ({ms}ms)  total_count={total}", True)
        if results:
            top = results[0]
            line("  top.candidate_id", top.get("candidate_id"))
            line("  top.final_score", top.get("final_score"))
            line("  top.alpha/beta", f"α={top.get('alpha')} β={top.get('beta')}")
            line("  top.source", top.get("source"))
        record("stage: rerank", True, f"total={total} {ms}ms")
    except Exception as e:
        line("rerank()", f"FAIL — {e}", False)
        traceback.print_exc()
        record("stage: rerank", False, str(e))


# ── 7. Full pipeline ──────────────────────────────────────────────────────
if not args.skip_pipeline:
    hdr(f"7. orchestrator.run({Q!r}) — full pipeline")

    if run_pipeline is None:
        line("run()", "SKIPPED — import failed", False)
        record("stage: pipeline", False, "import failed")
    else:
        try:
            t0 = time.perf_counter()
            out = run_pipeline(Q)
            ms = round((time.perf_counter() - t0) * 1000, 1)
            line("run()", f"OK  ({ms}ms)", True)
            for key in ("query", "intent", "hybrid_raw", "kg_raw",
                        "tagged_candidates", "reranked", "explanations", "trace"):
                present = key in out
                line(f"  payload['{key}']", "present" if present else "MISSING", present)
                record(f"payload: {key}", present)

            tc = out.get("tagged_candidates", [])
            rr = (out.get("reranked") or {}).get("results", [])
            line("  tagged_candidates #", len(tc))
            line("  reranked.results #", len(rr))

            trace = out.get("trace", {})
            hdr("Latency breakdown (ms)")
            for k in ("intent_ms", "hybrid_ms", "kg_ms", "rerank_ms", "explain_ms", "total_ms"):
                print(f"    {k:<12} {trace.get(k)}")
            print(f"    hybrid_n     {trace.get('hybrid_n')}")
            print(f"    kg_n         {trace.get('kg_n')}")
            print(f"    overlap_n    {trace.get('overlap_n')}")

            record("stage: pipeline", True, f"{ms}ms, reranked={len(rr)}")

            if rr:
                hdr("Top 3 reranked candidates")
                for r in rr[:3]:
                    print(f"    [{r['source']:<11}]  id={r['candidate_id']:<8}  "
                          f"F={r['final_score']:.4f}  "
                          f"H={r['hybrid_score']:.3f}  K={r['kg_score']:.3f}  "
                          f"α={r['alpha']:.2f} β={r['beta']:.2f}  Δ={r['modifier_delta']:+.3f}")
        except Exception as e:
            line("run()", f"FAIL — {e}", False)
            traceback.print_exc()
            record("stage: pipeline", False, str(e))


# ── SUMMARY ───────────────────────────────────────────────────────────────
hdr("SUMMARY")

passed = [r for r in RESULTS if r[1]]
failed = [r for r in RESULTS if not r[1]]

for name, ok, detail in RESULTS:
    mark = " PASS " if ok else " FAIL "
    print(f"  [{mark}] {name:<36} {detail}")

print()
print(f"  Passed : {len(passed)}/{len(RESULTS)}")
print(f"  Failed : {len(failed)}")

if failed:
    print("\n  Hard failures:")
    for name, _, detail in failed:
        print(f"    * {name} -> {detail}")

sys.exit(0 if not failed else 1)
