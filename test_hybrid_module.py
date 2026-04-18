"""
Diagnostic test script for the hybrid_retriever module.

Run from project root:
    python test_hybrid_retriever.py

Checks (in order):
    1. Environment       — index_dir, .env loaded
    2. File presence     — adapter, retriever.py, run_retrieval.py
    3. Dependencies      — numpy, faiss, sentence_transformers, transformers
    4. Index artefacts   — bm25.pkl, dense.pkl, dense_matrix.npy, metadata.pkl
    5. Direct retriever  — instantiate HybridRetriever and run one intent_json
    6. Adapter           — call retrieve(queries) [orchestrator contract]
    7. Schema validation — orchestrator expects {"candidates": [{"id","score",...}]}
    8. End-to-end        — chain intent_processor.process() → hybrid_retriever.retrieve()

Exit code: 0 if everything PASSED, 1 otherwise.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from pathlib import Path

# ── Pretty-print helpers ─────────────────────────────────────────────────────
def hdr(title: str) -> None:
    print(f"\n{'═' * 78}\n  {title}\n{'═' * 78}")

def line(label: str, value, ok: bool | None = None) -> None:
    mark = "    " if ok is None else (" OK " if ok else "FAIL")
    print(f"  [{mark}] {label:<32} {value}")

RESULTS: list[tuple[str, bool, str]] = []

def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))


# ── 1. Environment ───────────────────────────────────────────────────────────
hdr("1. Environment")

try:
    from dotenv import load_dotenv
    load_dotenv()
    line(".env loaded", "yes", True)
except Exception as e:
    line(".env loaded", f"NO ({e})", False)

ROOT      = Path(__file__).parent
PKG       = ROOT / "modules" / "hybrid_retriever"
INDEX_DIR = Path(os.getenv("INDEX_DIR") or os.getenv("DATA_DIR") or "./data")

line("INDEX_DIR (resolved)", INDEX_DIR.resolve(), INDEX_DIR.exists())
record("env: INDEX_DIR exists", INDEX_DIR.exists(), str(INDEX_DIR.resolve()))


# ── 2. File presence ─────────────────────────────────────────────────────────
hdr("2. File presence (module)")

files = {
    "adapter":       PKG / "__init__.py",
    "retriever":     PKG / "retriever.py",
    "run_retrieval": PKG / "run_retrieval.py",
}
present = {}
for label, p in files.items():
    exists = p.exists()
    present[label] = exists
    line(label, p.relative_to(ROOT) if exists else f"MISSING ({p})", exists)
    record(f"file: {label}", exists)


# ── 3. Dependencies ──────────────────────────────────────────────────────────
hdr("3. Dependencies")

deps = ["numpy", "faiss", "sentence_transformers", "transformers", "torch"]
deps_ok = {}
for m in deps:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        deps_ok[m] = True
        line(m, f"v{v}", True)
    except Exception as e:
        deps_ok[m] = False
        line(m, f"MISSING ({type(e).__name__})", False)
    record(f"dep: {m}", deps_ok[m])


# ── 4. Index artefacts ───────────────────────────────────────────────────────
hdr("4. Index artefacts")

REQUIRED_INDEX = ["bm25.pkl", "dense.pkl", "dense_matrix.npy", "metadata.pkl"]
OPTIONAL_INDEX = ["skills.pkl"]
artefacts_ok = True
for name in REQUIRED_INDEX:
    p = INDEX_DIR / name
    ok = p.exists()
    if not ok:
        artefacts_ok = False
    size = f"{p.stat().st_size:,} bytes" if ok else "MISSING"
    line(name, size, ok)
    record(f"index: {name}", ok)
for name in OPTIONAL_INDEX:
    p = INDEX_DIR / name
    ok = p.exists()
    line(f"{name} (optional)", f"{p.stat().st_size:,} bytes" if ok else "absent (ok)", True)


# ── 5. Direct retriever instantiation + run ──────────────────────────────────
hdr("5. Direct HybridRetriever (teammate API)")

direct_out = None
direct_ok = False
SAMPLE_INTENT = {
    "original":  "senior python developer fintech 5 years no java",
    "corrected": "senior python developer fintech 5 years no java",
    "intent": {
        "primary_intent": "multi_skill",
        "primary":        "multi_skill",      # support both key names
        "confidence":     0.9,
        "modifiers":      ["experience_filter", "domain_search"],
        "top3_scores":    {"multi_skill": 0.9},
    },
    "parsed": {
        "skills":           ["python"],
        "negated_skills":   ["java"],
        "role":             "python developer",
        "experience_band":  "senior",
        "experience_years": "5",
        "domain":           "fintech",
        "location":         None,
        "negated_location": None,
    },
    "entities": {  # alternative key name the retriever also accepts
        "skills":           ["python"],
        "negated_skills":   ["java"],
        "role":             "python developer",
        "experience_band":  "senior",
        "experience_years": "5",
        "domain":           "fintech",
    },
    "queries": [
        "senior python developer fintech 5 years",
        "python developer fintech",
        "fintech python developer 5 years experience",
        "senior python engineer in fintech",
    ],
    "strategy_map": {
        "senior python developer fintech 5 years":          "original",
        "python developer fintech":                         "synonym",
        "fintech python developer 5 years experience":      "template",
        "senior python engineer in fintech":                "role_variant",
    },
    "exclusion_filters": {"must_not_skills": ["java"]},
    "kg_expanded_queries": [],
    "kg_ready": False,
}

if not present["retriever"]:
    line("skipped", "retriever.py missing", False)
    record("direct: retriever import", False)
elif not artefacts_ok:
    line("skipped", "required index files missing", False)
    record("direct: retriever import", False, "no index")
else:
    try:
        # Add package dir to sys.path so internal `from retriever import ...` works
        if str(PKG) not in sys.path:
            sys.path.insert(0, str(PKG))
        from modules.hybrid_retriever.retriever import HybridRetriever
        line("import HybridRetriever", "OK", True)
        hr = HybridRetriever(index_dir=str(INDEX_DIR), top_k=10)
        line("instantiate", f"loaded {hr.n_docs} docs from {INDEX_DIR}", True)

        direct_out = hr.retrieve(SAMPLE_INTENT)
        n_hyb = len(direct_out.get("hybrid", {}).get("results", []))
        n_lex = len(direct_out.get("lexical", {}).get("results", []))
        n_sem = len(direct_out.get("semantic", {}).get("results", []))
        ms    = direct_out.get("hybrid", {}).get("meta", {}).get("retrieval_time_ms")
        direct_ok = n_hyb > 0
        line("retrieve(intent_json)", f"hybrid={n_hyb} lexical={n_lex} semantic={n_sem} ({ms}ms)", direct_ok)
        record("direct: retrieve runs", True)
        record("direct: returns hybrid results", direct_ok, f"n={n_hyb}")
    except Exception as e:
        line("FAIL", f"{type(e).__name__}: {e}", False)
        traceback.print_exc()
        record("direct: retrieve runs", False, f"{type(e).__name__}: {e}")


# ── 6. Teammate output schema (sanity) ───────────────────────────────────────
hdr("6. Teammate output schema (per retriever.py)")

if direct_out:
    for mode in ("hybrid", "lexical", "semantic"):
        block = direct_out.get(mode) or {}
        ok = "results" in block and "meta" in block
        line(f"{mode}.results / .meta", "present" if ok else "MISSING", ok)
        record(f"teammate-schema: {mode}", ok)
    if direct_out.get("hybrid", {}).get("results"):
        sample = direct_out["hybrid"]["results"][0]
        keys = set(sample.keys())
        expected = {"rank", "candidate_id", "name", "scores", "explanation"}
        missing = expected - keys
        line("hybrid.results[0] keys", "all present" if not missing else f"missing={missing}", not missing)
        record("teammate-schema: result keys", not missing)
else:
    line("skipped", "no direct output", False)


# ── 7. Adapter end-to-end (orchestrator contract) ────────────────────────────
hdr("7. Adapter retrieve(queries) — orchestrator contract")

adapter_out = None
adapter_ok = False
try:
    from modules.hybrid_retriever import retrieve as hybrid_retrieve
    line("import retrieve()", "OK", True)
    queries = SAMPLE_INTENT["queries"]
    adapter_out = hybrid_retrieve(queries)
    adapter_ok = True
    line("retrieve(queries)", f"returned dict with keys={sorted(adapter_out.keys())}", True)
    record("adapter: retrieve runs", True)
except NotImplementedError as e:
    line("FAIL", "adapter still stubbed (NotImplementedError) — not wired to teammate code", False)
    record("adapter: retrieve runs", False, "stub NotImplementedError")
    print(f"        → {e}")
except Exception as e:
    line("FAIL", f"{type(e).__name__}: {e}", False)
    traceback.print_exc()
    record("adapter: retrieve runs", False, f"{type(e).__name__}: {e}")


# ── 8. Orchestrator-shape schema check on adapter output ─────────────────────
hdr("8. Orchestrator schema check (adapter output)")

if adapter_out:
    cands = adapter_out.get("candidates")
    has_cands = isinstance(cands, list)
    line('top-level "candidates" list', f"{type(cands).__name__}", has_cands)
    record("orchestrator-schema: candidates list", has_cands)
    if has_cands and cands:
        sample = cands[0]
        keys = set(sample.keys())
        # _tag_and_combine reads .id and .score
        has_id    = "id" in keys
        has_score = "score" in keys
        line("candidate.id present", str(sample.get("id"))[:30] if has_id else "MISSING", has_id)
        line("candidate.score present", sample.get("score") if has_score else "MISSING", has_score)
        record("orchestrator-schema: candidate.id", has_id)
        record("orchestrator-schema: candidate.score", has_score)
        line(f"# candidates", len(cands), len(cands) > 0)
    elif has_cands:
        line("# candidates", "0 (empty)", False)
        record("orchestrator-schema: candidates non-empty", False)
else:
    line("skipped", "adapter not working — fix Section 7 first", False)
    record("orchestrator-schema: candidates list", False, "no adapter output")


# ── 9. End-to-end via the real intent_processor output ───────────────────────
hdr("9. End-to-end: intent_processor → hybrid_retriever (adapter)")

try:
    from modules.intent_processor import process as intent_process
    intent_out = intent_process("senior python developer fintech 5 years no java")
    line("intent_processor.process()", f"_source={intent_out.get('_source')} #queries={len(intent_out.get('queries', []))}", True)

    if adapter_ok:
        e2e_out = hybrid_retrieve(intent_out["queries"])
        n = len(e2e_out.get("candidates", []))
        line("hybrid_retrieve(intent.queries)", f"{n} candidates", n > 0)
        record("e2e: intent → hybrid", n > 0, f"#candidates={n}")
    else:
        line("skipped", "adapter not wired", False)
        record("e2e: intent → hybrid", False, "adapter stub")
except Exception as e:
    line("FAIL", f"{type(e).__name__}: {e}", False)
    traceback.print_exc()
    record("e2e: intent → hybrid", False, str(e))


# ── 10. Sample output ────────────────────────────────────────────────────────
hdr("10. Sample output")

if adapter_out:
    print(json.dumps(adapter_out, indent=2, default=str)[:1500])
elif direct_out:
    print("[adapter not wired — showing direct teammate output instead]")
    print(json.dumps(direct_out.get("hybrid", {}).get("results", [{}])[0], indent=2, default=str)[:1500])
else:
    print("  (no output to show)")


# ── Summary ──────────────────────────────────────────────────────────────────
hdr("SUMMARY")

passed = sum(1 for _, ok, _ in RESULTS if ok)
total  = len(RESULTS)
for name, ok, detail in RESULTS:
    mark = " PASS " if ok else " FAIL "
    print(f"  [{mark}] {name:<42} {detail}")

print(f"\n  → {passed}/{total} checks passed")
sys.exit(0 if passed == total else 1)

