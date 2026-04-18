"""
explainability/row_explainer.py

Reads a JSON file of candidate retrieval results (scores, KG triplets, keywords, etc.)
and produces rich, hiring-manager-friendly explanations using a local Llama model
(via Ollama).

Each explanation answers: "Why should we hire / shortlist this candidate?"
— translating every technical signal into plain, persuasive language.

Requirements:
    pip install ollama pydantic

Ollama setup (one-time):
    1. Install Ollama → https://ollama.com/download
    2. Pull a model:
         ollama pull llama3.2          # fast, good quality (~2GB)
         ollama pull llama3.1:8b       # larger, better reasoning (~5GB)
    3. Ollama runs automatically when you call it — no manual start needed.

Usage (from your pipeline):
    from explainability.row_explainer import explain_rows, explain_single

    results = explain_rows(my_list_of_candidate_dicts)
    for r in results:
        print(r["title"], "→", r["explanation"])

CLI usage:
    python row_explainer.py candidates.json --output explanations.json
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from pydantic import BaseModel

try:
    import ollama
except ImportError:
    raise ImportError(
        "Ollama Python client not found.\n"
        "Run: pip install ollama\n"
        "Then install Ollama itself from https://ollama.com/download"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

MODEL   = "llama-3.1-8b-instant"   # swap to "llama3.1:8b" for deeper reasoning
TIMEOUT = 90              # seconds per candidate


# ═══════════════════════════════════════════════════════════════════════════════
# Score interpretation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _interpret_bm25(val: float) -> tuple[str, str]:
    """Return (tier, plain-English phrase) for a BM25 score."""
    if val >= 10:
        return "very strong", f"extremely high keyword overlap (score {val:.1f}) — almost every key term in the job query appears in this candidate's profile"
    if val >= 7:
        return "strong",      f"strong keyword match (score {val:.1f}) — most of the required skills and role terms appear explicitly in the profile"
    if val >= 4:
        return "moderate",    f"moderate keyword match (score {val:.1f}) — several relevant terms were found, though some required skills are implicit rather than stated"
    if val >= 2:
        return "weak",        f"limited keyword overlap (score {val:.1f}) — few direct skill or role terms matched; the candidate may use different terminology"
    return "very weak",       f"minimal keyword match (score {val:.1f}) — the profile shares very little literal vocabulary with the job query"


def _interpret_semantic(val: float) -> tuple[str, str]:
    """Return (tier, plain-English phrase) for a 0-1 semantic cosine score."""
    if val >= 0.85:
        return "very strong", f"outstanding conceptual alignment ({val:.2f}) — the candidate's overall experience and background are exceptionally close in meaning to what was searched for"
    if val >= 0.70:
        return "strong",      f"strong semantic fit ({val:.2f}) — the meaning and context of the candidate's background closely mirrors the job requirements, even beyond exact keywords"
    if val >= 0.50:
        return "moderate",    f"reasonable conceptual overlap ({val:.2f}) — the candidate's domain and skills are broadly aligned with the search intent"
    if val >= 0.30:
        return "weak",        f"partial semantic match ({val:.2f}) — some thematic overlap exists but the candidate's focus area may differ from what was searched"
    return "very weak",       f"low conceptual similarity ({val:.2f}) — the candidate's background does not closely align with the search intent in meaning"


def _interpret_cross_encoder(val: float) -> tuple[str, str]:
    """Return (tier, plain-English phrase) for a cross-encoder / reranker score."""
    if val >= 6:
        return "very strong", f"AI double-check confirms excellent fit (score {val:.1f}) — a second-pass AI model independently ranked this candidate as highly relevant"
    if val >= 3.5:
        return "strong",      f"AI validation supports the match (score {val:.1f}) — the re-ranking model confirms this candidate is a solid fit for the role"
    if val >= 1.5:
        return "moderate",    f"moderate AI confidence (score {val:.1f}) — the re-ranking model sees a reasonable fit but with some reservations"
    return "weak",            f"low AI confidence (score {val:.1f}) — the re-ranking model was not strongly convinced; worth reviewing manually"


def _interpret_rrf(val: float) -> tuple[str, str]:
    """Return (tier, plain-English phrase) for an RRF fusion score."""
    if val >= 0.05:
        return "top tier",    f"top-tier combined ranking score ({val:.4f}) — this candidate ranked highly on both keyword and semantic searches simultaneously"
    if val >= 0.025:
        return "strong",      f"strong combined ranking ({val:.4f}) — performed well across multiple retrieval methods"
    if val >= 0.01:
        return "moderate",    f"decent combined ranking ({val:.4f}) — appeared in results from at least one retrieval method with reasonable strength"
    return "lower",           f"lower combined ranking ({val:.4f}) — ranked toward the middle or bottom of the candidate pool overall"


def _interpret_kg(row: dict) -> str | None:
    """
    Interpret knowledge-graph signals into a single human sentence.
    Handles various KG field naming conventions.
    """
    connected   = row.get("kg_connected") or row.get("knowledge_graph_connected") or row.get("kg_match")
    strength    = row.get("kg_strength")  or row.get("knowledge_graph_strength")  or ""
    confidence  = row.get("kg_confidence") or row.get("knowledge_graph_confidence") or 0.0
    path        = row.get("kg_path")      or row.get("knowledge_graph_path")       or row.get("kg_triplets") or []

    if not connected:
        return None

    # Decode hop / strength
    strength_str = str(strength).lower()
    if "direct" in strength_str or "one" in strength_str or strength_str == "1":
        hop_desc = "directly"
    elif "two" in strength_str or "2" in strength_str:
        hop_desc = "through one intermediate concept"
    elif "three" in strength_str or "3" in strength_str:
        hop_desc = "through two intermediate concepts"
    else:
        hop_desc = "through the knowledge graph"

    # Build path narrative
    path_narrative = ""
    if path:
        if isinstance(path, list) and len(path) >= 1:
            # Could be a list of triplet strings or a list of nodes
            first = path[0]
            if isinstance(first, (list, tuple)) and len(first) == 3:
                # Triplet format: [subject, relation, object]
                steps = " → ".join(f'"{t[0]}" {t[1]} "{t[2]}"' for t in path[:3])
                path_narrative = f" (path: {steps})"
            elif isinstance(first, str):
                path_narrative = f" (via: {' → '.join(str(p) for p in path[:4])})"

    conf_desc = ""
    if isinstance(confidence, float) and confidence > 0:
        if confidence >= 0.8:
            conf_desc = " with high confidence"
        elif confidence >= 0.5:
            conf_desc = " with moderate confidence"
        else:
            conf_desc = " with some uncertainty"

    return (
        f"The knowledge graph connects this candidate to the job requirements {hop_desc}"
        f"{path_narrative}{conf_desc}, "
        f"revealing relevant domain connections that pure keyword search would miss."
    )


def _collect_scores(row: dict) -> dict[str, Any]:
    """
    Extract and interpret all score fields into a structured dict ready for the prompt.
    Returns a dict with keys: bm25, semantic, rrf, cross_encoder, kg_sentence, rank_info.
    """
    out = {}

    # BM25
    bm25_val = row.get("bm25_score") or row.get("bm25") or row.get("lexical_score")
    if bm25_val is not None:
        tier, phrase = _interpret_bm25(float(bm25_val))
        out["bm25"] = {"value": bm25_val, "tier": tier, "phrase": phrase}

    # Semantic
    sem_val = row.get("semantic_score") or row.get("cosine_similarity") or row.get("embedding_score")
    if sem_val is not None:
        tier, phrase = _interpret_semantic(float(sem_val))
        out["semantic"] = {"value": sem_val, "tier": tier, "phrase": phrase}

    # RRF / hybrid
    rrf_val = row.get("rrf_score") or row.get("hybrid_score") or row.get("final_score")
    if rrf_val is not None:
        tier, phrase = _interpret_rrf(float(rrf_val))
        out["rrf"] = {"value": rrf_val, "tier": tier, "phrase": phrase}

    # Cross-encoder / reranker
    ce_val = (row.get("cross_encoder_score") or row.get("rerank_score")
              or row.get("reranker_score")    or row.get("ce_score"))
    if ce_val is not None:
        tier, phrase = _interpret_cross_encoder(float(ce_val))
        out["cross_encoder"] = {"value": ce_val, "tier": tier, "phrase": phrase}

    # Rank positions
    bm25_rank = row.get("bm25_rank") or row.get("lexical_rank")
    sem_rank  = row.get("semantic_rank") or row.get("embedding_rank")
    if bm25_rank or sem_rank:
        parts = []
        if bm25_rank:  parts.append(f"ranked #{bm25_rank} in keyword search")
        if sem_rank:   parts.append(f"ranked #{sem_rank} in semantic search")
        out["rank_info"] = "; ".join(parts)

    # Knowledge graph
    kg = _interpret_kg(row)
    if kg:
        out["kg_sentence"] = kg

    return out


def _collect_candidate_profile(row: dict) -> dict[str, Any]:
    """
    Extract candidate profile fields: skills, keywords, role, location, summary, etc.
    Handles many common field name conventions.
    """
    profile = {}

    for key in ["title", "job_title", "role", "position", "name", "candidate_name", "doc_id", "id"]:
        if row.get(key):
            profile["title"] = row[key]
            break

    for key in ["skills", "skill_list", "matched_skills", "skill_overlap", "tags"]:
        val = row.get(key)
        if val:
            if isinstance(val, list):
                profile["skills"] = val[:10]
            else:
                profile["skills"] = [s.strip() for s in str(val).split(",")][:10]
            break

    for key in ["keywords", "keyword_highlights", "matched_keywords", "query_keywords"]:
        val = row.get(key)
        if val:
            if isinstance(val, list):
                # Handle list of strings or list of dicts like [{"term": "python"}]
                terms = []
                for item in val[:8]:
                    if isinstance(item, dict):
                        terms.append(item.get("term") or item.get("keyword") or str(item))
                    else:
                        terms.append(str(item))
                profile["keywords"] = terms
            break

    for key in ["summary", "description", "preview", "excerpt", "bio", "profile_summary"]:
        if row.get(key):
            profile["summary"] = str(row[key])[:400]
            break

    for key in ["location", "city", "region", "place"]:
        if row.get(key):
            profile["location"] = row[key]
            break

    for key in ["experience", "years_experience", "experience_years", "exp"]:
        if row.get(key):
            profile["experience"] = row[key]
            break

    for key in ["industry", "domain", "sector", "vertical"]:
        if row.get(key):
            profile["industry"] = row[key]
            break

    for key in ["intent_alignment", "query_intent", "intent"]:
        if row.get(key):
            profile["intent"] = str(row[key]).replace("_", " ")
            break

    return profile


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt construction
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a senior talent acquisition advisor writing hiring recommendations
for a recruiter or hiring manager. Your task is to explain WHY a specific candidate
should be seriously considered for a role, based on data from an AI-powered search system.

TONE: Confident, professional, and persuasive. Write as if you are advocating for this
candidate to a decision-maker who will act on your recommendation.

STRUCTURE: Write exactly 3 focused paragraphs:

  Paragraph 1 — THE MATCH QUALITY:
    Explain how strongly and in what ways this candidate matches the job requirements.
    Use the score interpretations provided (do NOT mention raw numbers or field names).
    Phrases like "strong keyword alignment", "deep semantic fit", "independently validated
    by a second AI model" are good. Be specific about what the scores mean in hiring terms.

  Paragraph 2 — CANDIDATE STRENGTHS:
    Highlight the candidate's skills, background, and experience in concrete hiring language.
    Mention specific skills and keywords found. Reference the knowledge graph connections
    if present — explain them as "related domain expertise" or "adjacent skill connections".
    If a summary or description is given, weave in concrete details.

  Paragraph 3 — HIRING RECOMMENDATION:
    Give a clear, direct recommendation sentence. State why this candidate stands out
    relative to the hiring need. Mention any caveats naturally (e.g. if a score was
    only moderate, note it as "worth a conversation to explore X further").
    End with an action — "recommend for interview", "strong shortlist candidate", etc.

RULES:
  - Never mention field names, variable names, or metric names (bm25, rrf, cosine, etc.)
  - Never write raw numbers from scores
  - Never use bullet points — flowing paragraphs only
  - Do not start with "This candidate" — start with their name/title or a strong verb
  - Be specific and concrete, not vague and generic
  - Total length: 120-180 words
"""


def _build_user_prompt(
    profile: dict[str, Any],
    scores: dict[str, Any],
    row_idx: int,
    total: int,
) -> str:
    lines = [f"=== Candidate {row_idx + 1} of {total} ===\n"]

    # Profile info
    if profile.get("title"):
        lines.append(f"Role/Title: {profile['title']}")
    if profile.get("location"):
        lines.append(f"Location: {profile['location']}")
    if profile.get("industry"):
        lines.append(f"Industry: {profile['industry']}")
    if profile.get("experience"):
        lines.append(f"Experience: {profile['experience']}")
    if profile.get("skills"):
        lines.append(f"Skills: {', '.join(str(s) for s in profile['skills'])}")
    if profile.get("keywords"):
        lines.append(f"Query keywords matched: {', '.join(profile['keywords'])}")
    if profile.get("intent"):
        lines.append(f"Search intent: {profile['intent']}")
    if profile.get("summary"):
        lines.append(f"\nProfile excerpt:\n{profile['summary']}")

    lines.append("\n=== Match Signals ===\n")

    # Score interpretations
    if scores.get("bm25"):
        s = scores["bm25"]
        lines.append(f"Keyword Match: {s['tier'].upper()} — {s['phrase']}")

    if scores.get("semantic"):
        s = scores["semantic"]
        lines.append(f"Semantic Fit: {s['tier'].upper()} — {s['phrase']}")

    if scores.get("cross_encoder"):
        s = scores["cross_encoder"]
        lines.append(f"AI Re-ranking: {s['tier'].upper()} — {s['phrase']}")

    if scores.get("rrf"):
        s = scores["rrf"]
        lines.append(f"Combined Ranking: {s['tier'].upper()} — {s['phrase']}")

    if scores.get("rank_info"):
        lines.append(f"Rank Positions: {scores['rank_info']}")

    if scores.get("kg_sentence"):
        lines.append(f"Knowledge Graph: {scores['kg_sentence']}")

    lines.append(
        "\nNow write the 3-paragraph hiring recommendation as instructed. "
        "Be specific, persuasive, and actionable."
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Core explainer functions
# ═══════════════════════════════════════════════════════════════════════════════

def explain_single(
    row: BaseModel | dict,
    row_idx: int = 0,
    total: int = 1,
    model: str = MODEL,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Generate a hiring-focused explanation for one candidate row.

    Parameters
    ----------
    row      : Pydantic model instance or plain dict of candidate data
    row_idx  : 0-based index in the batch (for progress display)
    total    : total number of rows being processed
    model    : Ollama model name
    verbose  : print progress to stdout

    Returns
    -------
    {
        "row_index":   int,
        "title":       str,      # candidate name/role
        "raw":         dict,     # original field values
        "scores":      dict,     # interpreted score signals
        "explanation": str,      # 3-paragraph hiring recommendation
        "error":       str|None
    }
    """
    # Normalise to dict
    row_dict = row.model_dump() if isinstance(row, BaseModel) else dict(row)

    # Extract profile and scores
    profile = _collect_candidate_profile(row_dict)
    scores  = _collect_scores(row_dict)

    title = profile.get("title", f"Candidate {row_idx + 1}")

    if verbose:
        print(f"  [{row_idx + 1}/{total}] Generating recommendation for: {title} … ", end="", flush=True)

    user_prompt = _build_user_prompt(profile, scores, row_idx, total)

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": 0.4, "num_predict": 450},
        )
        explanation = response["message"]["content"].strip()
        if verbose:
            print("done ✓")

    except Exception as exc:
        if verbose:
            print(f"ERROR — {exc}")
        return {
            "row_index":   row_idx,
            "title":       title,
            "raw":         row_dict,
            "scores":      scores,
            "explanation": None,
            "error":       str(exc),
        }

    return {
        "row_index":   row_idx,
        "title":       title,
        "raw":         row_dict,
        "scores":      scores,
        "explanation": explanation,
        "error":       None,
    }


def explain_rows(
    rows: list[BaseModel | dict],
    model: str = MODEL,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Generate hiring explanations for a list of candidate rows.

    Parameters
    ----------
    rows    : list of Pydantic model instances or plain dicts
    model   : Ollama model name (default: llama3.2)
    verbose : print progress to stdout

    Returns
    -------
    List of explanation dicts (same order as input).
    Each dict has keys: row_index, title, raw, scores, explanation, error.
    """
    if not rows:
        return []

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  Nexus Candidate Explainer — {len(rows)} candidate(s) · model: {model}")
        print(f"{'═'*60}\n")

    results = []
    for i, row in enumerate(rows):
        result = explain_single(row, row_idx=i, total=len(rows), model=model, verbose=verbose)
        results.append(result)

    if verbose:
        ok  = sum(1 for r in results if r["error"] is None)
        bad = len(results) - ok
        print(f"\n{'─'*60}")
        print(f"  Complete: {ok} explained, {bad} failed.")
        print(f"{'─'*60}\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Integration helper
# ═══════════════════════════════════════════════════════════════════════════════

def from_pydantic_list(
    model_list: list[BaseModel],
    model_name: str = MODEL,
    output_json: str | None = None,
) -> list[dict]:
    """
    One-call convenience wrapper for use inside your pipeline.

    Example
    -------
    from pydantic import BaseModel
    from explainability.row_explainer import from_pydantic_list

    class CandidateResult(BaseModel):
        title: str
        bm25_score: float
        semantic_score: float
        rrf_score: float
        cross_encoder_score: float
        skills: list[str]
        kg_connected: bool
        kg_path: list

    results = [CandidateResult(...), ...]
    explanations = from_pydantic_list(results, output_json="explanations.json")
    # explanations[0]["explanation"]  → 3-paragraph hiring recommendation
    """
    explained = explain_rows(model_list, model=model_name)
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(explained, f, indent=2, ensure_ascii=False)
        print(f"  Saved → {output_json}")
    return explained


# ═══════════════════════════════════════════════════════════════════════════════
# Pretty printer
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(results: list[dict[str, Any]], show_scores: bool = False) -> None:
    """Pretty-print explanation results to the terminal."""
    sep  = "═" * 68
    sep2 = "─" * 68

    for r in results:
        print(sep)
        status = "✓" if not r["error"] else "✗"
        print(f"  {status}  #{r['row_index'] + 1}  {r['title']}")
        print(sep2)

        if show_scores and r.get("scores"):
            print("  SIGNALS:")
            for sig_key, sig_val in r["scores"].items():
                if sig_key == "kg_sentence":
                    print(f"    • KG: {sig_val}")
                elif sig_key == "rank_info":
                    print(f"    • {sig_val}")
                elif isinstance(sig_val, dict):
                    print(f"    • {sig_key.upper()} [{sig_val['tier']}]")
            print()

        if r["error"]:
            print(f"  ERROR: {r['error']}")
        else:
            # Word-wrap at 72 chars
            paragraphs = r["explanation"].split("\n\n")
            for para in paragraphs:
                words = para.split()
                line, out_lines = [], []
                for w in words:
                    if sum(len(x) + 1 for x in line) + len(w) > 70:
                        out_lines.append("  " + " ".join(line))
                        line = [w]
                    else:
                        line.append(w)
                if line:
                    out_lines.append("  " + " ".join(line))
                print("\n".join(out_lines))
                print()

    print(sep)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _load_json_file(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "candidates", "rows", "items", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    raise ValueError(f"Cannot parse JSON: expected list or dict, got {type(data)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generate hiring-focused explanations for candidate retrieval results.\n"
            "Translates BM25, semantic, RRF, cross-encoder scores and KG triplets\n"
            "into plain-English hiring recommendations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input", nargs="?",
        help="Path to JSON file of candidate results. Omit to run the built-in demo.",
    )
    parser.add_argument(
        "--model", default=MODEL,
        help=f"Ollama model name (default: {MODEL})",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save explanations as JSON (e.g. explanations.json)",
    )
    parser.add_argument(
        "--show-scores", action="store_true",
        help="Also print interpreted score signals alongside each explanation",
    )
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    if args.input:
        print(f"Loading {args.input} …")
        rows = _load_json_file(args.input)
        print(f"Found {len(rows)} candidate(s).\n")
    else:
        print("No input file — running built-in demo with 3 sample candidates.\n")
        rows = [
            {
                "doc_id": "JD-042",
                "title": "Machine Learning Engineer",
                "location": "Pune, India",
                "industry": "FinTech",
                "skills": ["Python", "TensorFlow", "scikit-learn", "MLOps", "SQL", "Docker"],
                "keywords": [{"term": "machine learning"}, {"term": "python"}, {"term": "deep learning"}],
                "summary": (
                    "5 years building production ML pipelines at a leading payments firm. "
                    "Led migration from batch to real-time inference, reducing latency by 60%. "
                    "Published internal research on fraud detection using graph neural networks."
                ),
                "bm25_score": 9.66,
                "semantic_score": 0.83,
                "rrf_score": 0.0312,
                "cross_encoder_score": 6.2,
                "bm25_rank": 2,
                "semantic_rank": 1,
                "kg_connected": True,
                "kg_strength": "direct",
                "kg_confidence": 0.91,
                "kg_path": [
                    ["Python", "is_core_skill_of", "Machine Learning"],
                    ["Machine Learning", "is_required_by", "ML Engineer"],
                ],
                "intent_alignment": "role_search",
            },
            {
                "doc_id": "JD-091",
                "title": "Data Scientist / AI-ML Expert",
                "location": "Bengaluru, India",
                "industry": "E-Commerce",
                "skills": ["Python", "R", "NLP", "Spark", "Hadoop", "A/B Testing", "Statistics"],
                "keywords": [{"term": "data science"}, {"term": "NLP"}, {"term": "python"}, {"term": "analytics"}],
                "summary": (
                    "7 years in data science across e-commerce and recommendation systems. "
                    "Designed NLP-based product tagging pipeline processing 2M SKUs daily. "
                    "Strong background in statistical modelling and business intelligence."
                ),
                "bm25_score": 8.81,
                "semantic_score": 0.76,
                "rrf_score": 0.0287,
                "cross_encoder_score": 4.1,
                "bm25_rank": 3,
                "semantic_rank": 2,
                "kg_connected": True,
                "kg_strength": "two-hop",
                "kg_confidence": 0.72,
                "kg_path": [
                    ["NLP", "is_subfield_of", "Machine Learning"],
                    ["Machine Learning", "is_required_by", "ML Engineer"],
                ],
                "intent_alignment": "skill_filter",
            },
            {
                "doc_id": "JD-224",
                "title": "Python Backend Engineer",
                "location": "Remote",
                "industry": "SaaS",
                "skills": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker"],
                "keywords": [{"term": "python"}, {"term": "backend"}],
                "summary": (
                    "4 years building high-throughput REST APIs in Python. "
                    "No direct ML experience but comfortable with data pipelines and SQL."
                ),
                "bm25_score": 3.2,
                "semantic_score": 0.41,
                "rrf_score": 0.0091,
                "cross_encoder_score": 1.1,
                "bm25_rank": 9,
                "semantic_rank": 8,
                "kg_connected": False,
                "intent_alignment": "role_search",
            },
        ]

    # ── Run ────────────────────────────────────────────────────────────────────
    results = explain_rows(rows, model=args.model)
    print_results(results, show_scores=args.show_scores)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nExplanations saved → {args.output}")