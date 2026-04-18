"""
Diagnostic test script for the intent_processor module.

Run from project root:
    python test_intent_processor.py

Checks (in order):
    1. Environment       — API keys, provider, .env loaded
    2. File presence     — adapter, llm_pipeline, local_pipeline, vocab
    3. Dependencies      — openai, sentence_transformers, transformers, symspellpy, torch
    4. Direct LLM path   — instantiate LLMIntentPipeline and run one query
    5. Direct local path — instantiate IntentQueryPipeline and run one query
    6. Adapter           — call process() and check it picked the expected path
    7. Schema validation — assert canonical keys present in output
    8. Behaviour         — a few queries with expected intents / negation

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

# Track results so we can summarise at the end
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

PROVIDER = (os.getenv("INTENT_LLM_PROVIDER") or "groq").lower()
PROVIDER_ENV = {
    "groq": "GROQ_API_KEY", "cerebras": "CEREBRAS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY", "sambanova": "SAMBANOVA_API_KEY",
}
api_var = PROVIDER_ENV.get(PROVIDER)
api_set = bool(api_var and os.getenv(api_var))
line("INTENT_LLM_PROVIDER", PROVIDER, True)
line(f"{api_var} set" if api_var else "API var", api_set, api_set)
record("env: provider known", api_var is not None, f"provider={PROVIDER}")
record("env: API key present", api_set, api_var or "no var")


# ── 2. File presence ─────────────────────────────────────────────────────────
hdr("2. File presence")

ROOT = Path(__file__).parent
PKG = ROOT / "modules" / "intent_processor"
files = {
    "adapter":          PKG / "__init__.py",
    "llm_pipeline":     PKG / "llm_intent_pipeline.py",
    "local_pipeline":   PKG / "intent_pipeline.py",
    "vocab":            PKG / "vocab.py",
}
present = {}
for label, p in files.items():
    exists = p.exists()
    present[label] = exists
    line(label, p.relative_to(ROOT), exists)
    record(f"file: {label}", exists, str(p.relative_to(ROOT)))


# ── 3. Dependencies ──────────────────────────────────────────────────────────
hdr("3. Dependencies")

deps = ["openai", "symspellpy", "sentence_transformers", "transformers", "torch"]
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


# ── 4. Direct LLM path ───────────────────────────────────────────────────────
hdr("4. Direct LLM path")

llm_ok = False
llm_intent = None
if not present["llm_pipeline"]:
    line("skipped", "llm_intent_pipeline.py missing", False)
    record("llm: direct run", False, "file missing")
elif not deps_ok.get("openai"):
    line("skipped", "openai package missing", False)
    record("llm: direct run", False, "openai missing")
elif not api_set:
    line("skipped", f"{api_var} not set", False)
    record("llm: direct run", False, "no api key")
else:
    try:
        from modules.intent_processor.llm_intent_pipeline import LLMIntentPipeline
        line("import LLMIntentPipeline", "OK", True)
        p = LLMIntentPipeline(provider=PROVIDER)
        line("instantiate", f"model={p.model}", True)
        r = p.run("senior python developer fintech 5 years no java")
        llm_intent = r.primary_intent.value
        llm_ok = True
        line("run query", f"intent={llm_intent} conf={r.confidence:.2f} #q={len(r.expanded_queries)}", True)
        record("llm: direct run", True, f"intent={llm_intent}")
    except Exception as e:
        line("FAIL", f"{type(e).__name__}: {e}", False)
        traceback.print_exc()
        record("llm: direct run", False, f"{type(e).__name__}: {e}")


# ── 5. Direct local path ─────────────────────────────────────────────────────
hdr("5. Direct local path")

local_ok = False
local_intent = None
if not present["local_pipeline"]:
    line("skipped", "intent_pipeline.py missing", False)
    record("local: direct run", False, "file missing")
elif not deps_ok.get("sentence_transformers") or not deps_ok.get("transformers"):
    line("skipped", "sentence_transformers/transformers missing", False)
    record("local: direct run", False, "deps missing")
else:
    try:
        # The adapter normally puts the package dir on sys.path so the local
        # pipeline can do `from vocab import ...` and `from intent_pipeline import ...`.
        # Replicate that here for direct testing.
        if str(PKG) not in sys.path:
            sys.path.insert(0, str(PKG))

        from modules.intent_processor.intent_pipeline import IntentQueryPipeline
        line("import IntentQueryPipeline", "OK", True)
        p = IntentQueryPipeline()
        line("instantiate", "OK (BART+SBERT loaded)", True)
        r = p.run("senior python developer fintech 5 years no java")
        local_intent = r.intent.primary_intent.value
        local_ok = True
        line("run query", f"intent={local_intent} conf={r.intent.confidence:.2f} #q={len(r.queries)}", True)
        record("local: direct run", True, f"intent={local_intent}")
    except Exception as e:
        line("FAIL", f"{type(e).__name__}: {e}", False)
        traceback.print_exc()
        record("local: direct run", False, f"{type(e).__name__}: {e}")


# ── 6. Adapter end-to-end ────────────────────────────────────────────────────
hdr("6. Adapter end-to-end (process)")

adapter_ok = False
adapter_source = None
adapter_out = None
try:
    from modules.intent_processor import process
    line("import process()", "OK", True)
    adapter_out = process("senior python developer fintech 5 years no java")
    adapter_source = adapter_out.get("_source")
    adapter_ok = True
    line("run", f"_source={adapter_source}", True)
    record("adapter: import + run", True, f"source={adapter_source}")
except Exception as e:
    line("FAIL", f"{type(e).__name__}: {e}", False)
    traceback.print_exc()
    record("adapter: import + run", False, f"{type(e).__name__}: {e}")

# Check that the adapter picked LLM if it was supposed to
expected_source = "llm" if llm_ok else "local"
if adapter_ok:
    correct_path = adapter_source == expected_source
    line(f"path expected={expected_source}", f"got={adapter_source}", correct_path)
    record("adapter: picked expected path", correct_path,
           f"expected={expected_source} got={adapter_source}")


# ── 7. Schema validation ─────────────────────────────────────────────────────
hdr("7. Canonical schema check")

REQUIRED_TOP = {
    "original", "corrected", "intent", "parsed", "queries",
    "strategy_map", "exclusion_filters", "kg_expanded_queries", "kg_ready", "_source",
}
REQUIRED_INTENT = {"primary_intent", "confidence", "modifiers", "top3_scores"}
REQUIRED_PARSED = {
    "skills", "negated_skills", "role", "experience_band",
    "experience_years", "domain", "location", "negated_location",
}

if not adapter_out:
    line("skipped", "no adapter output", False)
    record("schema: top-level keys", False)
else:
    missing_top = REQUIRED_TOP - set(adapter_out.keys())
    line("top-level keys", "all present" if not missing_top else f"missing={missing_top}", not missing_top)
    record("schema: top-level keys", not missing_top, f"missing={missing_top}")

    missing_intent = REQUIRED_INTENT - set(adapter_out.get("intent", {}).keys())
    line("intent keys", "all present" if not missing_intent else f"missing={missing_intent}", not missing_intent)
    record("schema: intent keys", not missing_intent, f"missing={missing_intent}")

    missing_parsed = REQUIRED_PARSED - set(adapter_out.get("parsed", {}).keys())
    line("parsed keys", "all present" if not missing_parsed else f"missing={missing_parsed}", not missing_parsed)
    record("schema: parsed keys", not missing_parsed, f"missing={missing_parsed}")

    qs_ok = isinstance(adapter_out.get("queries"), list) and len(adapter_out["queries"]) > 0
    line("queries non-empty list", f"len={len(adapter_out.get('queries', []))}", qs_ok)
    record("schema: queries non-empty", qs_ok)


# ── 8. Behaviour spot-checks ─────────────────────────────────────────────────
hdr("8. Behaviour spot-checks")

CASES = [
    # (query, expected_intent_or_None, expect_negation, expected_skill_in_negated)
    ("senior react and node developer",                "multi_skill",      False, None),
    ("python developer no java experience",            "role_search",      True,  "java"),
    ("Pyhton develoer fintch 5 yeras",                 "experience_filter",False, None),
    ("best machine learning engineers for our team",   "ranking",          False, None),
]

if adapter_ok:
    for q, expected, expect_neg, neg_skill in CASES:
        try:
            out = process(q)
            intent = out.get("intent", {}).get("primary_intent")
            negated = out.get("parsed", {}).get("negated_skills", [])
            ok_intent = (expected is None) or (intent == expected)
            ok_neg = (bool(negated) == expect_neg) and (neg_skill is None or neg_skill in negated)
            ok = ok_intent and ok_neg
            detail = f"intent={intent} negated={negated}"
            line(f"'{q[:42]:<42}'", detail, ok)
            record(f"behaviour: {q[:30]}", ok, detail)
        except Exception as e:
            line(f"'{q[:42]:<42}'", f"ERROR {type(e).__name__}: {e}", False)
            record(f"behaviour: {q[:30]}", False, str(e))
else:
    line("skipped", "adapter not working", False)


# ── 9. Sample full output ────────────────────────────────────────────────────
hdr("9. Sample output (first query, truncated)")

if adapter_out:
    print(json.dumps(adapter_out, indent=2, default=str)[:1200])
else:
    print("  (no adapter output to show)")


# ── Summary ──────────────────────────────────────────────────────────────────
hdr("SUMMARY")

passed = sum(1 for _, ok, _ in RESULTS if ok)
total  = len(RESULTS)
for name, ok, detail in RESULTS:
    mark = " PASS " if ok else " FAIL "
    print(f"  [{mark}] {name:<40} {detail}")

print(f"\n  → {passed}/{total} checks passed")
sys.exit(0 if passed == total else 1)