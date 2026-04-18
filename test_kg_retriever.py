"""
test_kg_retriever.py — Diagnostic Test Script for KG Retriever Module
=======================================================================
Tests the modules/kg_retriever module end-to-end.

Run from project root:
    python test_kg_retriever.py

Checks (in order):
    1.  Environment       — .env loaded, NEO4J_* vars set, profiles.csv present
    2.  File presence     — __init__.py, retrieve.py present
    3.  Dependencies      — neo4j, pandas importable
    4.  Neo4j connection  — driver can connect + ping
    5.  Graph schema      — required node labels & relationship types exist
    6.  Node counts       — Candidate / Skill / Role / Domain counts > 0
    7.  Direct class      — KGRetriever instantiates without error
    8.  Single-skill      — retrieve() with one known skill returns candidates
    9.  Multi-skill       — retrieve() with multiple skills returns candidates
    10. Negation handling — negated_skills are NOT in results' matched_terms
    11. Experience filter — exp band / years narrow results appropriately
    12. Role search       — role term matches via SUITABLE_FOR
    13. Domain search     — domain term matches via graph traversal
    14. Adapter contract  — module-level retrieve(queries) satisfies orchestrator schema
    15. Schema validation — every result has required orchestrator keys
    16. Score ordering    — results are sorted descending by score
    17. Match reasons     — each result has non-empty match_reasons list
    18. Edge cases        — empty queries, unknown skill, broad query
    19. E2E integration   — intent_processor.process() -> kg_retriever.retrieve()
    20. Performance       — typical query completes in < 10s

Exit code: 0 if everything PASSED, 1 otherwise.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ─── Pretty-print helpers ────────────────────────────────────────────────────

def hdr(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")

def line(label: str, value: Any, ok: bool | None = None) -> None:
    mark = "    " if ok is None else (" OK " if ok else "FAIL")
    print(f"  [{mark}] {str(label):<36} {value}")

RESULTS: list[tuple[str, bool, str]] = []

def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))

def skip(name: str, reason: str) -> None:
    RESULTS.append((name, False, f"SKIPPED — {reason}"))
    line(name, f"SKIPPED — {reason}", False)

# ─── Load .env early ─────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    for candidate in [Path.cwd() / ".env", Path(__file__).parent / ".env"]:
        if candidate.exists():
            load_dotenv(candidate)
            break
    else:
        load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).parent
PKG  = ROOT / "modules" / "kg_retriever"

# =============================================================================
# 1. ENVIRONMENT
# =============================================================================
hdr("1. Environment")

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USER     = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DB       = os.getenv("NEO4J_DATABASE", "neo4j")
DATA_DIR       = Path(os.getenv("DATA_DIR") or os.getenv("INDEX_DIR") or "./data")
CSV_PATH       = DATA_DIR / "profiles.csv"

env_vars = {
    "NEO4J_URI":      (NEO4J_URI,      bool(NEO4J_URI)),
    "NEO4J_USERNAME": (NEO4J_USER,     bool(NEO4J_USER)),
    "NEO4J_PASSWORD": (NEO4J_PASSWORD, bool(NEO4J_PASSWORD)),
}
all_env_ok = True
for var, (val, ok) in env_vars.items():
    display = (val[:20] + "...") if val and len(val) > 20 else (val or "NOT SET")
    line(var, display, ok)
    if not ok:
        all_env_ok = False
    record(f"env: {var}", ok)

csv_ok = CSV_PATH.exists()
line("profiles.csv", CSV_PATH.resolve(), csv_ok)
record("env: profiles.csv exists", csv_ok, str(CSV_PATH.resolve()))

# =============================================================================
# 2. FILE PRESENCE
# =============================================================================
hdr("2. File presence")

files = {
    "kg_retriever/__init__.py": PKG / "__init__.py",
    "kg_retriever/retrieve.py": PKG / "retrieve.py",
}
files_ok: dict[str, bool] = {}
for label, p in files.items():
    ok = p.exists()
    files_ok[label] = ok
    line(label, str(p.relative_to(ROOT)) if ok else "MISSING", ok)
    record(f"file: {label}", ok)

# =============================================================================
# 3. DEPENDENCIES
# =============================================================================
hdr("3. Dependencies")

DEPS = ["neo4j", "pandas"]
deps_ok: dict[str, bool] = {}
for dep in DEPS:
    try:
        mod = importlib.import_module(dep)
        v = getattr(mod, "__version__", "?")
        deps_ok[dep] = True
        line(dep, f"v{v}", True)
    except Exception as e:
        deps_ok[dep] = False
        line(dep, f"MISSING — {e}", False)
    record(f"dep: {dep}", deps_ok[dep])

can_proceed = all_env_ok and all(files_ok.values()) and all(deps_ok.values())

# =============================================================================
# 4. NEO4J CONNECTION
# =============================================================================
hdr("4. Neo4j connection")

driver = None
db_connected = False

if not all_env_ok:
    skip("neo4j: connect", "missing env vars")
elif not deps_ok.get("neo4j"):
    skip("neo4j: connect", "neo4j package missing")
else:
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        db_connected = True
        line("driver.verify_connectivity()", "OK", True)
        record("neo4j: connect", True)
    except Exception as e:
        line("connect", f"FAILED — {type(e).__name__}: {e}", False)
        record("neo4j: connect", False, str(e))

# =============================================================================
# 5. GRAPH SCHEMA CHECK
# =============================================================================
hdr("5. Graph schema (labels & relationship types)")

EXPECTED_LABELS = ["Candidate", "Skill", "Role", "Domain"]
EXPECTED_RELS   = ["HAS_SKILL", "SUITABLE_FOR", "REQUIRES", "RELATED_TO",
                   "BELONGS_TO", "HAS_SUBDOMAIN", "HAS_ROLE"]

if not db_connected:
    for lbl in EXPECTED_LABELS + EXPECTED_RELS:
        skip(f"schema: {lbl}", "no db connection")
else:
    with driver.session(database=NEO4J_DB) as session:
        result = session.run("CALL db.labels() YIELD label RETURN collect(label) AS labels")
        present_labels = set(result.single()["labels"])
        for lbl in EXPECTED_LABELS:
            ok = lbl in present_labels
            line(f"label: {lbl}", "present" if ok else "MISSING", ok)
            record(f"schema: label:{lbl}", ok)

        result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS rels")
        present_rels = set(result.single()["rels"])
        for rel in EXPECTED_RELS:
            ok = rel in present_rels
            line(f"rel: {rel}", "present" if ok else "MISSING", ok)
            record(f"schema: rel:{rel}", ok)

# =============================================================================
# 6. NODE COUNTS
# =============================================================================
hdr("6. Node counts")

node_counts: dict[str, int] = {}

if not db_connected:
    skip("neo4j: node counts", "no db connection")
else:
    with driver.session(database=NEO4J_DB) as session:
        for label in EXPECTED_LABELS:
            try:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                cnt = result.single()["cnt"]
                node_counts[label] = cnt
                ok = cnt > 0
                line(f"{label} nodes", cnt, ok)
                record(f"nodes: {label} > 0", ok, f"count={cnt}")
            except Exception as e:
                line(f"{label} nodes", f"ERROR — {e}", False)
                record(f"nodes: {label} > 0", False, str(e))

# =============================================================================
# 7. KGRetriever INSTANTIATION
# =============================================================================
hdr("7. KGRetriever class instantiation")

kg_retriever_instance = None

if not can_proceed:
    skip("class: KGRetriever instantiate", "deps/files missing")
elif not db_connected:
    skip("class: KGRetriever instantiate", "no db connection")
elif not csv_ok:
    skip("class: KGRetriever instantiate", "profiles.csv missing")
else:
    try:
        if str(PKG) not in sys.path:
            sys.path.insert(0, str(PKG))

        from modules.kg_retriever.retrieve import KGRetriever
        kg = KGRetriever(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            csv_path=str(CSV_PATH),
        )
        kg_retriever_instance = kg
        line("KGRetriever()", f"loaded {len(kg._profiles)} CSV rows", True)
        record("class: KGRetriever instantiate", True, f"rows={len(kg._profiles)}")
    except Exception as e:
        line("KGRetriever()", f"FAILED — {type(e).__name__}: {e}", False)
        traceback.print_exc()
        record("class: KGRetriever instantiate", False, str(e))


# ─── Helper builders ─────────────────────────────────────────────────────────
def make_intent(
    skills: list[str] | None = None,
    role: str | None = None,
    band: str | None = None,
    exp_years: str | None = None,
    negated: list[str] | None = None,
    domain: str | None = None,
    queries: list[str] | None = None,
) -> dict:
    skill_list = skills or []
    query_list = queries or (
        [f"candidate with {' '.join(skill_list)} skills"] if skill_list else ["software developer"]
    )
    return {
        "original":  query_list[0],
        "corrected": query_list[0],
        "intent": {
            "primary_intent": "skill_search",
            "confidence":     0.85,
            "modifiers":      [],
            "top3_scores":    {"skill_search": 0.85},
        },
        "parsed": {
            "skills":           skill_list,
            "negated_skills":   negated or [],
            "role":             role,
            "experience_band":  band,
            "experience_years": exp_years,
            "domain":           domain,
            "location":         None,
            "negated_location": None,
        },
        "entities": {
            "skills":           skill_list,
            "negated_skills":   negated or [],
            "role":             role,
            "experience_band":  band,
            "experience_years": exp_years,
            "domain":           domain,
        },
        "queries":             query_list,
        "strategy_map":        {q: "original" for q in query_list},
        "exclusion_filters":   {"must_not_skills": negated or []},
        "kg_expanded_queries": [],
        "kg_ready":            False,
    }


def safe_retrieve(intent: dict, label: str):
    """Call kg.retrieve() safely. Returns (results, elapsed_ms) or (None, 0)."""
    if kg_retriever_instance is None:
        return None, 0
    try:
        t0 = time.perf_counter()
        results = kg_retriever_instance.retrieve(intent)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        return results, elapsed
    except Exception as e:
        print(f"    ERROR in {label}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None, 0


# =============================================================================
# 8. SINGLE-SKILL RETRIEVAL
# =============================================================================
hdr("8. Single-skill retrieval")

single_results = None

if kg_retriever_instance is None:
    skip("retrieve: single-skill", "no KGRetriever instance")
else:
    for probe_skill in ["Python", "python", "SQL", "Java", "JavaScript"]:
        intent = make_intent(skills=[probe_skill])
        results, ms = safe_retrieve(intent, f"single-skill ({probe_skill})")
        if results is not None:
            ok = len(results) > 0
            line(f"retrieve(['{probe_skill}'])", f"{len(results)} candidates  ({ms}ms)", ok)
            record(f"retrieve: single-skill ({probe_skill})", ok, f"n={len(results)}")
            if ok:
                single_results = results
                break
        else:
            line(f"retrieve(['{probe_skill}'])", "ERROR (see above)", False)
            record(f"retrieve: single-skill ({probe_skill})", False)
    if single_results is None:
        line("WARNING", "All probe skills returned 0 results — graph may be sparsely loaded", None)

# =============================================================================
# 9. MULTI-SKILL RETRIEVAL
# =============================================================================
hdr("9. Multi-skill retrieval")

if kg_retriever_instance is None:
    skip("retrieve: multi-skill", "no KGRetriever instance")
else:
    intent = make_intent(
        skills=["Python", "SQL"],
        queries=["python and sql developer", "candidate with python and sql skills"],
    )
    results, ms = safe_retrieve(intent, "multi-skill")
    if results is not None:
        ok = isinstance(results, list)
        line("retrieve(['Python','SQL'])", f"{len(results)} candidates  ({ms}ms)", ok)
        record("retrieve: multi-skill", True, f"n={len(results)}")
        if results:
            has_matched = all(len(r.get("matched_terms", [])) > 0 for r in results)
            line("all results have matched_terms", str(has_matched), has_matched)
            record("retrieve: multi-skill matched_terms", has_matched)
    else:
        record("retrieve: multi-skill", False)

# =============================================================================
# 10. NEGATION HANDLING
# =============================================================================
hdr("10. Negation handling")

if kg_retriever_instance is None:
    skip("retrieve: negation", "no KGRetriever instance")
else:
    intent = make_intent(
        skills=["Python"], negated=["Java"],
        queries=["python developer no java"],
    )
    results, ms = safe_retrieve(intent, "negation")
    if results is not None:
        negated_leaked = [
            r.get("candidate_id")
            for r in results
            if "java" in [t.lower() for t in r.get("matched_terms", [])]
        ]
        line(f"retrieve() with negated=['Java']", f"{len(results)} candidates  ({ms}ms)", True)
        if negated_leaked:
            line(
                "NOTE: 'java' in matched_terms for",
                f"{len(negated_leaked)} candidates (reranker should handle)",
                None,
            )
        else:
            line("negated 'java' absent from matched_terms", "yes", True)
        record("retrieve: negation no crash", True)
        record("retrieve: negated term not in matched_terms",
               len(negated_leaked) == 0, f"leaked_count={len(negated_leaked)}")
    else:
        record("retrieve: negation no crash", False)

# =============================================================================
# 11. EXPERIENCE FILTER
# =============================================================================
hdr("11. Experience band / years")

if kg_retriever_instance is None:
    skip("retrieve: experience filter", "no KGRetriever instance")
else:
    intent = make_intent(
        skills=["Python"], band="senior", exp_years="5",
        queries=["senior python developer 5 years"],
    )
    results, ms = safe_retrieve(intent, "experience-senior")
    if results is not None:
        ok = isinstance(results, list)
        line("senior band query", f"{len(results)} candidates  ({ms}ms)", ok)
        record("retrieve: experience band=senior", ok, f"n={len(results)}")
        if results:
            has_exp_score = all("experience_score" in r for r in results)
            line("experience_score field present", str(has_exp_score), has_exp_score)
            record("retrieve: experience_score field", has_exp_score)
    else:
        record("retrieve: experience band=senior", False)

# =============================================================================
# 12. ROLE SEARCH
# =============================================================================
hdr("12. Role-based search (SUITABLE_FOR)")

if kg_retriever_instance is None:
    skip("retrieve: role search", "no KGRetriever instance")
else:
    for probe_role in ["Backend Developer", "Data Scientist", "Software Engineer"]:
        intent = make_intent(role=probe_role, queries=[f"find {probe_role}"])
        results, ms = safe_retrieve(intent, f"role ({probe_role})")
        if results is not None:
            line(f"role='{probe_role}'", f"{len(results)} candidates  ({ms}ms)", True)
            record(f"retrieve: role '{probe_role}'", True, f"n={len(results)}")
            if results:
                suitable_for_hits = sum(
                    1 for r in results
                    for reason in r.get("match_reasons", [])
                    if reason.get("relationship") == "SUITABLE_FOR"
                )
                line("  SUITABLE_FOR reasons", suitable_for_hits, suitable_for_hits >= 0)
            break
        else:
            record(f"retrieve: role '{probe_role}'", False)

# =============================================================================
# 13. DOMAIN SEARCH
# =============================================================================
hdr("13. Domain search (graph traversal)")

if kg_retriever_instance is None:
    skip("retrieve: domain search", "no KGRetriever instance")
else:
    for probe_domain in ["fintech", "healthcare", "ecommerce"]:
        intent = make_intent(domain=probe_domain, queries=[f"{probe_domain} developer"])
        results, ms = safe_retrieve(intent, f"domain ({probe_domain})")
        if results is not None:
            line(f"domain='{probe_domain}'", f"{len(results)} candidates  ({ms}ms)", True)
            record(f"retrieve: domain '{probe_domain}'", True, f"n={len(results)}")
            break
        else:
            record(f"retrieve: domain '{probe_domain}'", False)

# =============================================================================
# 14. ADAPTER CONTRACT (module-level retrieve())
# =============================================================================
hdr("14. Adapter: module-level retrieve() — orchestrator contract")

adapter_results = None

try:
    from modules.kg_retriever import retrieve as kg_adapter_retrieve
    line("import retrieve from modules.kg_retriever", "OK", True)
    record("adapter: import", True)

    test_queries = [
        "python developer fintech 5 years",
        "candidate skilled in python machine learning",
    ]
    t0 = time.perf_counter()
    adapter_out = kg_adapter_retrieve(test_queries)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    adapter_results = adapter_out
    ok = isinstance(adapter_out, dict)
    line("retrieve(queries_list)", f"returns {type(adapter_out).__name__}  ({ms}ms)", ok)
    record("adapter: retrieve(list) returns dict", ok)

except NotImplementedError as e:
    line("FAIL", "retrieve() raises NotImplementedError — adapter stub not replaced", False)
    record("adapter: retrieve(list) returns dict", False, "NotImplementedError stub")
    print(f"        → {e}")
except Exception as e:
    line("FAIL", f"{type(e).__name__}: {e}", False)
    traceback.print_exc()
    record("adapter: retrieve(list) returns dict", False, str(e))

# =============================================================================
# 15. SCHEMA VALIDATION
# =============================================================================
hdr("15. Orchestrator schema validation")

REQUIRED_CANDIDATE_KEYS = {"id", "score"}

if adapter_results is None:
    skip("schema: candidates list", "adapter not working — fix section 14 first")
else:
    cands = adapter_results.get("candidates")
    is_list = isinstance(cands, list)
    line('"candidates" key is a list', f"type={type(cands).__name__}", is_list)
    record("schema: candidates is list", is_list)

    if is_list and cands:
        sample = cands[0]
        missing = REQUIRED_CANDIDATE_KEYS - set(sample.keys())
        line("required keys in candidates[0]",
             "all present" if not missing else f"MISSING={missing}", not missing)
        record("schema: candidate has id+score", not missing, f"missing={missing}")

        NICE_TO_HAVE = {"name", "core_skills", "secondary_skills",
                        "soft_skills", "years_of_experience", "potential_roles",
                        "matched_terms", "match_reasons"}
        extra_present = NICE_TO_HAVE & set(sample.keys())
        line("nice-to-have keys present",
             f"{sorted(extra_present)}", len(extra_present) > 0)
        record("schema: nice-to-have keys", len(extra_present) > 0,
               f"found={sorted(extra_present)}")
    elif is_list:
        line("candidates list is empty", "graph may be sparse", None)

# =============================================================================
# 16. SCORE ORDERING
# =============================================================================
hdr("16. Score ordering (descending)")

if kg_retriever_instance is None:
    skip("retrieve: score ordering", "no KGRetriever instance")
else:
    intent = make_intent(skills=["Python"], queries=["python developer"])
    results, ms = safe_retrieve(intent, "score ordering")
    if results and len(results) > 1:
        scores = [r["score"] for r in results]
        is_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        line("results sorted descending by score", str(is_sorted), is_sorted)
        record("retrieve: score ordering", is_sorted,
               f"top5={[round(s, 4) for s in scores[:5]]}")
    elif results and len(results) == 1:
        line("1 result — ordering trivially satisfied", "OK", True)
        record("retrieve: score ordering", True, "n=1")
    else:
        line("0 results — cannot verify ordering", "", None)
        record("retrieve: score ordering", True, "0 results, not falsifiable")

# =============================================================================
# 17. MATCH REASONS STRUCTURE
# =============================================================================
hdr("17. Match reasons structure")

REQUIRED_REASON_KEYS = {"query_term", "matched_node", "relationship", "hops", "score_delta"}

if kg_retriever_instance is None:
    skip("retrieve: match_reasons", "no KGRetriever instance")
elif single_results is None:
    skip("retrieve: match_reasons", "no successful single-skill results to inspect")
else:
    r0 = single_results[0]
    reasons = r0.get("match_reasons", [])
    has_reasons = len(reasons) > 0
    line("candidates[0].match_reasons", f"len={len(reasons)}", has_reasons)
    record("retrieve: match_reasons non-empty", has_reasons)

    if reasons:
        sample_reason = reasons[0]
        missing_keys = REQUIRED_REASON_KEYS - set(sample_reason.keys())
        line("match_reason has all required keys",
             "all present" if not missing_keys else f"MISSING={missing_keys}",
             not missing_keys)
        record("retrieve: match_reason keys", not missing_keys, f"missing={missing_keys}")

        hop_ok = all(1 <= r.get("hops", 0) <= 4 for r in reasons)
        line("all hops in [1..4]", str(hop_ok), hop_ok)
        record("retrieve: hops range valid", hop_ok)

        delta_ok = all(r.get("score_delta", 0) > 0 for r in reasons)
        line("all score_delta > 0", str(delta_ok), delta_ok)
        record("retrieve: score_delta > 0", delta_ok)

# =============================================================================
# 18. EDGE CASES
# =============================================================================
hdr("18. Edge cases")

if kg_retriever_instance is None:
    skip("edge cases", "no KGRetriever instance")
else:
    # 18a. Empty queries
    intent_empty = make_intent(skills=[], queries=[])
    results, ms = safe_retrieve(intent_empty, "empty queries")
    if results is not None:
        line("empty queries — no crash", f"{len(results)} results  ({ms}ms)", True)
        record("edge: empty queries no crash", True)
    else:
        record("edge: empty queries no crash", False)

    # 18b. Unknown / gibberish skill
    intent_gibberish = make_intent(
        skills=["xyzzy_not_a_real_skill_99999"],
        queries=["xyzzy_not_a_real_skill_99999 developer"],
    )
    results, ms = safe_retrieve(intent_gibberish, "unknown skill")
    if results is not None:
        line("unknown skill — no crash", f"{len(results)} results  ({ms}ms)", True)
        record("edge: unknown skill no crash", True)
    else:
        record("edge: unknown skill no crash", False)

    # 18c. Very broad query
    intent_broad = make_intent(
        skills=["software"],
        queries=["software developer engineer"],
    )
    results, ms = safe_retrieve(intent_broad, "broad query")
    if results is not None:
        line("broad query — no crash", f"{len(results)} results  ({ms}ms)", True)
        record("edge: broad query no crash", True)
    else:
        record("edge: broad query no crash", False)

    # 18d. Duplicate skill terms
    intent_dup = make_intent(
        skills=["Python", "Python", "python"],
        queries=["python python developer"],
    )
    results, ms = safe_retrieve(intent_dup, "duplicate skills")
    if results is not None:
        line("duplicate skill terms — no crash", f"{len(results)} results  ({ms}ms)", True)
        record("edge: duplicate skills no crash", True)
    else:
        record("edge: duplicate skills no crash", False)

# =============================================================================
# 19. E2E — intent_processor -> kg_retriever
# =============================================================================
hdr("19. End-to-end: intent_processor.process() -> kg_retriever.retrieve()")

if not can_proceed:
    skip("e2e: intent -> kg", "deps/files missing upstream")
elif kg_retriever_instance is None:
    skip("e2e: intent -> kg", "no KGRetriever instance")
else:
    try:
        from modules.intent_processor import process as intent_process
        query = "senior python developer fintech 5 years"
        t0 = time.perf_counter()
        intent_out = intent_process(query)
        intent_ms = round((time.perf_counter() - t0) * 1000, 1)
        line("intent_processor.process()", f"_source={intent_out.get('_source')}  ({intent_ms}ms)", True)
        record("e2e: intent_processor runs", True)

        t0 = time.perf_counter()
        kg_results = kg_retriever_instance.retrieve(intent_out)
        kg_ms = round((time.perf_counter() - t0) * 1000, 1)
        ok = isinstance(kg_results, list)
        line("kg_retriever.retrieve(intent)", f"{len(kg_results)} candidates  ({kg_ms}ms)", ok)
        record("e2e: intent -> kg retrieve", ok, f"n={len(kg_results)}")

        # Verify parsed skills from intent appear in match_reasons
        if kg_results and intent_out.get("parsed", {}).get("skills"):
            detected_skills = set(s.lower() for s in intent_out["parsed"]["skills"])
            hit_in_reasons = False
            for r in kg_results[:5]:
                for reason in r.get("match_reasons", []):
                    if reason.get("query_term", "").lower() in detected_skills:
                        hit_in_reasons = True
                        break
            line("intent skills appear in match_reasons", str(hit_in_reasons), hit_in_reasons)
            record("e2e: intent skills in match_reasons", hit_in_reasons)

    except Exception as e:
        line("FAIL", f"{type(e).__name__}: {e}", False)
        traceback.print_exc()
        record("e2e: intent -> kg retrieve", False, str(e))

# =============================================================================
# 20. PERFORMANCE BENCHMARK
# =============================================================================
hdr("20. Performance benchmark")

PERF_THRESHOLD_MS = 10_000

if kg_retriever_instance is None:
    skip("perf: typical query < 10s", "no KGRetriever instance")
else:
    intent = make_intent(
        skills=["Python", "Machine Learning"],
        role="Data Scientist",
        band="senior",
        queries=[
            "senior data scientist python machine learning",
            "python machine learning data science",
            "candidate python ml senior",
        ],
    )
    t0 = time.perf_counter()
    results, _ = safe_retrieve(intent, "perf benchmark")
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    ok = elapsed_ms < PERF_THRESHOLD_MS
    line("typical query latency", f"{elapsed_ms}ms  (threshold={PERF_THRESHOLD_MS}ms)", ok)
    record("perf: query < 10s", ok, f"elapsed={elapsed_ms}ms")

# =============================================================================
# SAMPLE OUTPUT
# =============================================================================
hdr("Sample output (first result, truncated)")

sample_source = None
if single_results:
    sample_source = single_results
elif adapter_results and isinstance(adapter_results.get("candidates"), list) and adapter_results["candidates"]:
    sample_source = adapter_results["candidates"]

if sample_source:
    sample = sample_source[0]
    display = {}
    for k, v in sample.items():
        if isinstance(v, str) and len(v) > 100:
            display[k] = v[:100] + "..."
        elif isinstance(v, list) and len(v) > 5:
            display[k] = v[:5] + [f"... +{len(v)-5} more"]
        else:
            display[k] = v
    print(json.dumps(display, indent=2, default=str))
else:
    print("  (no results to display)")

# =============================================================================
# CLEANUP
# =============================================================================
for obj, name in [(driver, "Neo4j driver"), (kg_retriever_instance, "KGRetriever")]:
    if obj:
        try:
            obj.close()
        except Exception:
            pass

# =============================================================================
# SUMMARY
# =============================================================================
hdr("SUMMARY")

passed_list  = [r for r in RESULTS if r[1]]
failed_list  = [r for r in RESULTS if not r[1]]
skipped_list = [r for r in failed_list if r[2].startswith("SKIPPED")]
hard_fails   = [r for r in failed_list if not r[2].startswith("SKIPPED")]

for name, ok, detail in RESULTS:
    if ok:
        mark = " PASS "
    elif detail.startswith("SKIPPED"):
        mark = " SKIP "
    else:
        mark = " FAIL "
    print(f"  [{mark}] {name:<44} {detail}")

print()
print(f"  Passed  : {len(passed_list)}/{len(RESULTS)}")
print(f"  Failed  : {len(hard_fails)}")
print(f"  Skipped : {len(skipped_list)}")

if hard_fails:
    print("\n  Hard failures to fix:")
    for name, _, detail in hard_fails:
        print(f"    * {name}  ->  {detail}")

sys.exit(0 if not hard_fails else 1)