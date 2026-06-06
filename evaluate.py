#!/usr/bin/env python3
"""
evaluate_real.py — Production Evaluation Script for EmployeeKHOJO.ai
======================================================================
Evaluates the full retrieval pipeline (System API vs BM25 vs Semantic baseline)
using LLM-generated ground truth judgments and standard IR + RAGAS metrics.

Architecture:
  1. Ground Truth Generation  — Ollama (qwen2.5:72b or best available) rates
                                each candidate in BM25 top-100 on 0–3 scale.
  2. Three retrieval modes    — System API (/search), local BM25, local TF-IDF+SVD
  3. IR Metrics               — P@K, R@K, MAP, nDCG@K for K ∈ {5, 10, 20}
  4. RAGAS Metrics            — Context Precision/Recall, Answer Relevance, Faithfulness

Usage:
    python evaluate_real.py                     # full run
    python evaluate_real.py --regen-gt          # regenerate ground truth
    python evaluate_real.py --skip-ragas        # skip RAGAS (faster)
    python evaluate_real.py --queries Q01 Q05   # only specific queries
    python evaluate_real.py --gt-model qwen2.5:7b  # override LLM model
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── Optional imports (fail gracefully) ────────────────────────────────────────
try:
    import ollama as _ollama
    OLLAMA_OK = True
except ImportError:
    OLLAMA_OK = False

try:
    from rank_bm25 import BM25Okapi
    BM25_OK = True
except ImportError:
    BM25_OK = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize as sk_norm
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from rich.console import Console
    from rich.table import Table, Column
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich import print as rprint
    RICH_OK = True
    console = Console()
except ImportError:
    RICH_OK = False
    import re as _re
    class _FallbackConsole:
        def print(self, msg="", **kw): print(_re.sub(r"\[/?[^\]]*\]", "", str(msg)))
        def rule(self, title="", **kw): print(f"\n{'─'*30} {title} {'─'*30}")
    console = _FallbackConsole()


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS & TEST QUERIES
# ═══════════════════════════════════════════════════════════════════════════

CSV_PATH        = Path("data/profiles.csv")
GT_CACHE_PATH   = Path("eval_gt_cache.json")
RESULTS_PATH    = Path("eval_results.json")
REPORT_PATH     = Path("eval_report.md")
API_BASE_URL    = "http://localhost:8000"
API_SLEEP_S     = 0.5     # seconds between /search calls to avoid overwhelming intent pipeline
K_VALUES        = [5, 10, 20]
RELEVANCE_THRESHOLD = 2   # score >= 2 counts as "relevant" for binary P/R/MAP
GT_POOL_SIZE    = 100     # BM25 pre-filter size for ground truth candidate pool
RAGAS_TOP_N     = 5       # top-N candidate summaries used as RAGAS contexts
API_K           = 30      # results requested from /search endpoint

# Model priority for ground-truth generation
MODEL_PRIORITY = [
    "qwen2.5:72b",
    "qwen2.5:32b",
    "llama3.3:70b",
    "qwen2.5:7b",
]

TEST_QUERIES = [
    # Single skill
    {"qid": "Q01", "query": "Python developer", "intent": "skill_search"},
    {"qid": "Q02", "query": "Java backend developer", "intent": "role_search"},
    # Multi-skill
    {"qid": "Q03", "query": "Python and React full stack developer", "intent": "multi_skill"},
    {"qid": "Q04", "query": "React Node.js MongoDB MERN stack", "intent": "multi_skill"},
    {"qid": "Q05", "query": "Python machine learning pandas scikit-learn", "intent": "multi_skill"},
    # Experience filtered
    {"qid": "Q06", "query": "senior Python developer 5 years experience", "intent": "experience_filter"},
    {"qid": "Q07", "query": "junior fresher Java Python HTML entry level", "intent": "experience_filter"},
    {"qid": "Q08", "query": "Python developer 1 year experience", "intent": "experience_filter"},
    # Domain
    {"qid": "Q09", "query": "regulatory affairs FDA compliance specialist", "intent": "domain_search"},
    {"qid": "Q10", "query": "brand marketing cosmetics industry specialist", "intent": "domain_search"},
    {"qid": "Q11", "query": "legal research Westlaw compliance advisor", "intent": "domain_search"},
    {"qid": "Q12", "query": "supply chain procurement SAP ERP specialist", "intent": "domain_search"},
    {"qid": "Q13", "query": "Azure cloud microservices API development C#", "intent": "domain_search"},
    # Role
    {"qid": "Q14", "query": "Technical Lead solutions architect leadership", "intent": "role_search"},
    {"qid": "Q15", "query": "data analyst Power BI business intelligence SQL", "intent": "role_search"},
    {"qid": "Q16", "query": "DevOps engineer Docker CI/CD cloud infrastructure", "intent": "role_search"},
    # Negation
    {"qid": "Q17", "query": "Python developer no Java experience", "intent": "skill_search"},
    {"qid": "Q18", "query": "full stack developer not entry level", "intent": "multi_skill"},
    # Semantic / soft
    {"qid": "Q19", "query": "leadership communication stakeholder management", "intent": "skill_search"},
    {"qid": "Q20", "query": "problem solving analytical thinking research", "intent": "skill_search"},
    # Compound
    {"qid": "Q21", "query": "senior full stack developer React Node AWS fintech 3 years", "intent": "multi_skill"},
    {"qid": "Q22", "query": "regulatory affairs manager cosmetics FDA brand marketing", "intent": "domain_search"},
    {"qid": "Q23", "query": "machine learning engineer deep learning NLP Python TensorFlow", "intent": "multi_skill"},
    {"qid": "Q24", "query": "business process analyst Lean Six Sigma ERP process improvement", "intent": "domain_search"},
    {"qid": "Q25", "query": "software engineer data structures algorithms competitive programming", "intent": "skill_search"},
]


# ═══════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """Simple lowercase tokenizer — strips punctuation, splits on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def _profile_text(row) -> str:
    """Build combined searchable text for a profile row."""
    parts = []
    for col in ["core_skills", "secondary_skills", "soft_skills", "potential_roles", "skill_summary"]:
        val = str(row.get(col, "") or "")
        if val and val.lower() not in ("nan", "none", ""):
            parts.append(val)
    return " ".join(parts)


def _norm_id(v) -> str:
    """Normalise a candidate ID to a clean string (strip trailing .0 etc.)."""
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _select_model(override: str | None = None) -> str | None:
    """Pick the best available Ollama model from MODEL_PRIORITY."""
    if not OLLAMA_OK:
        return None
    try:
        available = {m.model for m in _ollama.list().models}
        if override:
            if override in available:
                return override
            # Try partial match (e.g. "qwen2.5:72b" vs "qwen2.5:72b-instruct-q4_K_M")
            for m in available:
                if override in m:
                    return m
            console.print(f"[yellow]⚠  Requested model '{override}' not found. Falling back.[/yellow]")
        for candidate in MODEL_PRIORITY:
            if candidate in available:
                return candidate
            for m in available:
                if candidate.split(":")[0] in m:
                    return m
        # Nothing matched — take whatever is available
        if available:
            return next(iter(available))
    except Exception as exc:
        console.print(f"[red]Ollama list failed: {exc}[/red]")
    return None


def _call_ollama(model: str, user_prompt: str, system_prompt: str | None = None,
                 temperature: float = 0.0, max_retries: int = 3) -> dict | None:
    """
    Call Ollama and parse JSON response.
    Returns parsed dict or None on failure.
    """
    if not OLLAMA_OK:
        return None
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    for attempt in range(max_retries):
        try:
            resp = _ollama.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature, "num_gpu": 99},
            )
            raw = resp["message"]["content"]
            # Strip markdown fences if present
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            # Find first {...} block
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
        except Exception as exc:
            if attempt == max_retries - 1:
                console.print(f"[red]Ollama error ({model}): {exc}[/red]")
            time.sleep(2 ** attempt)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  GROUND TRUTH GENERATION
# ═══════════════════════════════════════════════════════════════════════════

class GroundTruthGenerator:
    """
    Generates relevance judgments for each (query, candidate) pair using Ollama.

    Cache format (eval_gt_cache.json):
        { "Q01": { "180648": {"score": 0, "reason": "..."}, ... }, ... }

    Ground truth is sacred — never re-generated unless --regen-gt is passed.
    BM25 pre-filter narrows the full corpus to a pool of top-100 candidates
    before calling the LLM, reducing total LLM calls from 1782 × 25 to 100 × 25.
    """

    SYSTEM_PROMPT = (
        "You are a senior technical recruiter evaluating candidate relevance. "
        "Rate how well the candidate matches the recruiter's search query on a scale of 0–3. "
        "0 = not relevant (completely different domain or skills). "
        "1 = somewhat relevant (minor or tangential overlap). "
        "2 = relevant (matches key requirements; worth a closer look). "
        "3 = highly relevant (strong match to all or most requirements). "
        "Respond with pure JSON only — no markdown, no explanation outside the JSON object."
    )

    def __init__(self, df: pd.DataFrame, model: str):
        self.df = df
        self.model = model
        self._corpus_texts: list[str] = []
        self._corpus_ids: list[str] = []
        self._bm25: Any = None
        self._build_bm25()

    def _build_bm25(self):
        """Build a BM25 index over the full profile corpus."""
        if not BM25_OK:
            raise RuntimeError("rank_bm25 is required for ground truth generation. pip install rank-bm25")
        texts, ids = [], []
        for _, row in self.df.iterrows():
            cid = _norm_id(row["id"])
            text = _profile_text(row)
            ids.append(cid)
            texts.append(text)
        self._corpus_texts = texts
        self._corpus_ids = ids
        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def prefilter(self, query: str, pool_size: int = GT_POOL_SIZE) -> list[str]:
        """Return the top `pool_size` candidate IDs by BM25 score for `query`."""
        q_tokens = _tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[::-1][:pool_size]
        return [self._corpus_ids[i] for i in top_idx if scores[i] > 0]

    def _rate_candidate(self, query: str, row: pd.Series) -> dict:
        """Call Ollama to rate one candidate's relevance to `query`. Returns {score, reason}."""
        core = str(row.get("core_skills", "") or "")
        secondary = str(row.get("secondary_skills", "") or "")
        soft = str(row.get("soft_skills", "") or "")
        roles = str(row.get("potential_roles", "") or "")
        summary = str(row.get("skill_summary", "") or "")[:600]
        yoe = row.get("years_of_experience", 0) or 0

        user_prompt = (
            f'Recruiter query: "{query}"\n\n'
            f"Candidate profile:\n"
            f"  Core Skills: {core}\n"
            f"  Secondary Skills: {secondary}\n"
            f"  Soft Skills: {soft}\n"
            f"  Potential Roles: {roles}\n"
            f"  Years of Experience: {yoe}\n"
            f"  Summary: {summary}\n\n"
            f'Rate relevance 0–3. Respond ONLY with: {{"score": N, "reason": "brief explanation"}}'
        )

        result = _call_ollama(
            self.model, user_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.0,
        )
        if result and "score" in result:
            score = int(result["score"])
            score = max(0, min(3, score))
            return {"score": score, "reason": str(result.get("reason", ""))[:200]}
        # Fallback: score 0
        return {"score": 0, "reason": "LLM call failed or returned invalid JSON"}

    def generate_for_query(
        self,
        qid: str,
        query: str,
        cache: dict,
        regen: bool = False,
        verbose: bool = True,
    ) -> dict[str, dict]:
        """
        Generate ground truth judgments for `query`.
        Returns {candidate_id: {score, reason}} dict.
        Uses cache unless `regen=True`.
        """
        if not regen and qid in cache:
            if verbose:
                console.print(f"  [dim]  {qid} — loaded {len(cache[qid])} judgments from cache[/dim]")
            return cache[qid]

        pool_ids = self.prefilter(query, pool_size=GT_POOL_SIZE)
        if not pool_ids:
            console.print(f"  [yellow]  {qid}: BM25 returned 0 results[/yellow]")
            return {}

        if verbose:
            console.print(f"  [cyan]  {qid}[/cyan] — rating {len(pool_ids)} candidates with {self.model}")

        judgments: dict[str, dict] = {}
        for i, cid in enumerate(pool_ids):
            rows = self.df[self.df["id"].apply(_norm_id) == cid]
            if rows.empty:
                continue
            row = rows.iloc[0]
            judgment = self._rate_candidate(query, row)
            judgments[cid] = judgment

            if verbose and (i + 1) % 10 == 0:
                n_rel = sum(1 for j in judgments.values() if j["score"] >= RELEVANCE_THRESHOLD)
                console.print(f"    [{i+1}/{len(pool_ids)}] relevant so far: {n_rel}")

        return judgments

    def generate_all(
        self,
        queries: list[dict],
        cache: dict,
        regen: bool = False,
    ) -> dict:
        """Generate judgments for all queries, updating cache in-place."""
        for q in queries:
            qid, query = q["qid"], q["query"]
            cache[qid] = self.generate_for_query(qid, query, cache, regen=regen)
            # Save cache after each query in case of interruption
            GT_CACHE_PATH.write_text(json.dumps(cache, indent=2))
        return cache


# ═══════════════════════════════════════════════════════════════════════════
#  RETRIEVAL BASELINES
# ═══════════════════════════════════════════════════════════════════════════

class LocalBM25Retriever:
    """
    BM25 lexical baseline built directly from profiles.csv.
    Uses rank_bm25.BM25Okapi with default k1=1.5, b=0.75.
    Core skills are weighted 3× by repeating tokens.
    """

    def __init__(self, df: pd.DataFrame):
        if not BM25_OK:
            raise RuntimeError("rank_bm25 required: pip install rank-bm25")
        self._ids: list[str] = []
        tokenized: list[list[str]] = []
        for _, row in df.iterrows():
            cid = _norm_id(row["id"])
            # Weight core skills 3× (repeat tokens) to match indexer behaviour
            core = str(row.get("core_skills", "") or "")
            rest = _profile_text(row)  # includes core_skills once already
            text = (core + " ") * 2 + rest   # core appears 3× total
            self._ids.append(cid)
            tokenized.append(_tokenize(text))
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = API_K) -> list[str]:
        """Return top-k candidate IDs sorted by BM25 score."""
        q_tokens = _tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [self._ids[i] for i in top_idx if scores[i] > 0]


class LocalSemanticRetriever:
    """
    TF-IDF + TruncatedSVD (LSA) semantic baseline built from profiles.csv.
    Approximates dense semantic retrieval without requiring SBERT.
    """

    def __init__(self, df: pd.DataFrame, n_components: int = 200):
        if not SKLEARN_OK:
            raise RuntimeError("scikit-learn required: pip install scikit-learn")
        self._ids: list[str] = []
        corpus: list[str] = []
        for _, row in df.iterrows():
            self._ids.append(_norm_id(row["id"]))
            corpus.append(_profile_text(row))

        self._vectorizer = TfidfVectorizer(
            max_features=10_000,
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.90,
        )
        tfidf = self._vectorizer.fit_transform(corpus)
        actual_components = min(n_components, tfidf.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=actual_components, random_state=42)
        self._matrix = sk_norm(self._svd.fit_transform(tfidf)).astype(np.float32)

    def retrieve(self, query: str, k: int = API_K) -> list[str]:
        """Return top-k candidate IDs sorted by cosine similarity to query."""
        q_tfidf = self._vectorizer.transform([query])
        q_vec = sk_norm(self._svd.transform(q_tfidf)).astype(np.float32)
        sims = self._matrix @ q_vec.T
        sims = sims.ravel()
        top_idx = np.argsort(sims)[::-1][:k]
        return [self._ids[i] for i in top_idx if sims[i] > 0]


class SystemAPIRetriever:
    """
    Retrieves candidates from the live /search endpoint on port 8000.
    Returns None if the API is unavailable.
    """

    def __init__(self, base_url: str = API_BASE_URL, k: int = API_K):
        self.base_url = base_url.rstrip("/")
        self.k = k
        self._available: bool | None = None  # None = not checked yet

    def _check_health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def retrieve(self, query: str) -> tuple[list[str], dict | None]:
        """
        Returns (list_of_ids, full_api_response).
        Returns (None, None) if API is unavailable.
        """
        if self._available is None:
            self._available = self._check_health()
            if not self._available:
                console.print(f"[yellow]⚠  /search API is not reachable at {self.base_url}[/yellow]")

        if not self._available:
            return None, None

        try:
            url = f"{self.base_url}/search"
            params = {"q": query, "k": self.k, "mode": "hybrid"}
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            ids = [_norm_id(r["id"]) for r in data.get("results", [])]
            return ids, data
        except Exception as exc:
            console.print(f"[red]  API error for '{query[:40]}': {exc}[/red]")
            return None, None


# ═══════════════════════════════════════════════════════════════════════════
#  IR METRICS
# ═══════════════════════════════════════════════════════════════════════════

def _precision_at_k(ranked_ids: list[str], gt: dict[str, int], k: int) -> float:
    """Fraction of top-K results that are relevant (score >= RELEVANCE_THRESHOLD)."""
    top = ranked_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for cid in top if gt.get(str(cid), 0) >= RELEVANCE_THRESHOLD)
    return hits / len(top)


def _recall_at_k(ranked_ids: list[str], gt: dict[str, int], k: int) -> float:
    """Fraction of all relevant candidates found in top-K."""
    n_relevant = sum(1 for s in gt.values() if s >= RELEVANCE_THRESHOLD)
    if n_relevant == 0:
        return 0.0
    top = ranked_ids[:k]
    hits = sum(1 for cid in top if gt.get(str(cid), 0) >= RELEVANCE_THRESHOLD)
    return hits / n_relevant


def _avg_precision(ranked_ids: list[str], gt: dict[str, int]) -> float:
    """Mean average precision (binary relevance at threshold)."""
    relevant = {cid for cid, s in gt.items() if s >= RELEVANCE_THRESHOLD}
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, cid in enumerate(ranked_ids):
        if str(cid) in relevant:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / len(relevant) if relevant else 0.0


def _ndcg_at_k(ranked_ids: list[str], gt: dict[str, int], k: int) -> float:
    """
    nDCG@K with graded relevance (0–3) per TREC convention.
    gain = rel / log2(rank + 1)
    """
    def _dcg(gains):
        return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))

    actual_gains = [gt.get(str(cid), 0) for cid in ranked_ids[:k]]
    ideal_gains = sorted(gt.values(), reverse=True)[:k]

    dcg = _dcg(actual_gains)
    idcg = _dcg(ideal_gains)
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(ranked_ids: list[str], gt_judgments: dict[str, dict]) -> dict:
    """
    Compute all IR metrics for one (query, ranked_list) pair.

    ranked_ids     : ordered list of candidate ID strings
    gt_judgments   : {candidate_id: {score: int, reason: str}}
    """
    if ranked_ids is None:
        return {
            f"precision_at_{k}": None for k in K_VALUES
        } | {f"recall_at_{k}": None for k in K_VALUES
        } | {"map": None} | {f"ndcg_at_{k}": None for k in K_VALUES
        } | {"n_relevant_in_gt": 0, "n_retrieved": 0}

    gt = {cid: j["score"] for cid, j in gt_judgments.items()}
    n_relevant = sum(1 for s in gt.values() if s >= RELEVANCE_THRESHOLD)

    metrics: dict = {"n_relevant_in_gt": n_relevant, "n_retrieved": len(ranked_ids)}
    for k in K_VALUES:
        metrics[f"precision_at_{k}"] = round(_precision_at_k(ranked_ids, gt, k), 4)
        metrics[f"recall_at_{k}"] = round(_recall_at_k(ranked_ids, gt, k), 4)
        metrics[f"ndcg_at_{k}"] = round(_ndcg_at_k(ranked_ids, gt, k), 4)
    metrics["map"] = round(_avg_precision(ranked_ids, gt), 4)
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  RAGAS EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════

class RAGASEvaluator:
    """
    RAGAS-style metrics computed via Ollama (no external RAGAS library needed).

    Four metrics:
      context_precision  — are the retrieved contexts relevant to the query?
      context_recall     — do the contexts cover the key aspects of the query?
      answer_relevance   — does the explanation address the query well?
      faithfulness       — are explanation claims supported by the profile text?
    """

    _CTX_PRECISION_SYS = (
        "You are evaluating retrieval quality for an HR search system. "
        "Rate how relevant the given candidate profile context is for the recruiter query. "
        "Score 0.0 (completely irrelevant) to 1.0 (highly relevant). "
        "Respond ONLY with pure JSON."
    )

    _CTX_RECALL_SYS = (
        "You are evaluating retrieval completeness for an HR search system. "
        "Given multiple candidate profile summaries, assess what fraction of the "
        "recruiter query's key requirements are covered by at least one profile. "
        "Score 0.0 (nothing covered) to 1.0 (all requirements covered). "
        "Respond ONLY with pure JSON."
    )

    _ANSWER_REL_SYS = (
        "You are evaluating AI explanation quality for an HR search system. "
        "Rate how well the explanation addresses the original recruiter query. "
        "Score 0.0 (completely off-topic) to 1.0 (directly addresses the query). "
        "Respond ONLY with pure JSON."
    )

    _FAITHFULNESS_SYS = (
        "You are evaluating AI explanation faithfulness for an HR search system. "
        "Rate whether the claims in the explanation are supported by the candidate profile text. "
        "Score 0.0 (completely fabricated / unsupported) to 1.0 (fully supported by the profile). "
        "Respond ONLY with pure JSON."
    )

    def __init__(self, df: pd.DataFrame, model: str):
        self.df = df
        self.model = model
        self._id_to_row: dict[str, pd.Series] = {}
        for _, row in df.iterrows():
            self._id_to_row[_norm_id(row["id"])] = row

    def _rate(self, prompt: str, system: str) -> float:
        """Call Ollama and extract a 0–1 score. Returns 0.0 on failure."""
        result = _call_ollama(self.model, prompt, system_prompt=system, temperature=0.0)
        if result and "score" in result:
            try:
                return max(0.0, min(1.0, float(result["score"])))
            except (TypeError, ValueError):
                pass
        return 0.0

    def _context_precision(self, query: str, context: str) -> float:
        prompt = (
            f'Recruiter query: "{query}"\n\n'
            f"Candidate profile summary:\n{context[:800]}\n\n"
            f'Rate relevance 0.0–1.0. Respond: {{"score": 0.0–1.0, "reason": "brief"}}'
        )
        return self._rate(prompt, self._CTX_PRECISION_SYS)

    def _context_recall(self, query: str, contexts: list[str]) -> float:
        combined = "\n---\n".join(ctx[:400] for ctx in contexts)
        prompt = (
            f'Recruiter query: "{query}"\n\n'
            f"Top-{len(contexts)} retrieved candidate profiles:\n{combined}\n\n"
            f"What fraction of the query's key requirements are covered by at least one profile?\n"
            f'Respond: {{"score": 0.0–1.0, "covered": ["req1", ...], "missing": ["req2", ...]}}'
        )
        return self._rate(prompt, self._CTX_RECALL_SYS)

    def _answer_relevance(self, query: str, explanation: str) -> float:
        prompt = (
            f'Recruiter query: "{query}"\n\n'
            f"AI explanation for top candidate:\n{explanation[:600]}\n\n"
            f"How well does this explanation address the recruiter's query needs?\n"
            f'Respond: {{"score": 0.0–1.0, "reason": "brief"}}'
        )
        return self._rate(prompt, self._ANSWER_REL_SYS)

    def _faithfulness(self, profile_text: str, explanation: str) -> float:
        prompt = (
            f"Candidate profile:\n{profile_text[:800]}\n\n"
            f"AI-generated explanation:\n{explanation[:600]}\n\n"
            f"Are the claims in the explanation supported by the profile text?\n"
            f'Respond: {{"score": 0.0–1.0, "unsupported_claims": ["..."]}}'
        )
        return self._rate(prompt, self._FAITHFULNESS_SYS)

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        api_response: dict | None = None,
    ) -> dict:
        """
        Compute all RAGAS metrics for one query.

        api_response: full JSON from /search (contains explanation fields)
        """
        if not retrieved_ids:
            return {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_relevance": 0.0,
                "faithfulness": 0.0,
                "_note": "no results to evaluate",
            }

        top_ids = retrieved_ids[:RAGAS_TOP_N]

        # Build context strings from skill_summary
        contexts: list[str] = []
        for cid in top_ids:
            row = self._id_to_row.get(cid)
            if row is not None:
                summary = str(row.get("skill_summary", "") or "")
                if summary:
                    contexts.append(summary)

        # Context Precision — rate each context individually
        cp_scores = [self._context_precision(query, ctx) for ctx in contexts]
        cp = float(np.mean(cp_scores)) if cp_scores else 0.0

        # Context Recall — all contexts together
        cr = self._context_recall(query, contexts) if contexts else 0.0

        # Answer Relevance — from top result's explanation.summary
        ar = 0.0
        if api_response:
            results = api_response.get("results", [])
            if results:
                top_explanation = results[0].get("explanation", {})
                answer_text = top_explanation.get("summary", "")
                if answer_text:
                    ar = self._answer_relevance(query, answer_text)

        # Faithfulness — top result profile vs explanation
        f_score = 0.0
        if api_response:
            results = api_response.get("results", [])
            if results:
                top_cid = _norm_id(results[0]["id"])
                top_row = self._id_to_row.get(top_cid)
                if top_row is not None:
                    profile_text = _profile_text(top_row)
                    expl_text = results[0].get("explanation", {}).get("summary", "")
                    if profile_text and expl_text:
                        f_score = self._faithfulness(profile_text, expl_text)

        return {
            "context_precision": round(cp, 4),
            "context_recall": round(cr, 4),
            "answer_relevance": round(ar, 4),
            "faithfulness": round(f_score, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════

class EvaluationRunner:
    """Orchestrates the full evaluation across all queries and retrieval modes."""

    def __init__(
        self,
        df: pd.DataFrame,
        gt_cache: dict,
        model: str,
        skip_ragas: bool = False,
    ):
        self.df = df
        self.gt_cache = gt_cache
        self.model = model
        self.skip_ragas = skip_ragas

        console.print("[bold cyan]Building local retrievers…[/bold cyan]")
        self.bm25_retriever = LocalBM25Retriever(df)
        self.semantic_retriever = LocalSemanticRetriever(df)
        self.api_retriever = SystemAPIRetriever()
        self.ragas = RAGASEvaluator(df, model) if not skip_ragas else None

    def _eval_one_mode(
        self,
        qid: str,
        query: str,
        retrieved_ids: list[str] | None,
        api_response: dict | None = None,
    ) -> dict:
        """Compute all metrics for one (query, mode) combination."""
        gt = self.gt_cache.get(qid, {})
        metrics = compute_metrics(retrieved_ids, gt)

        ragas_scores: dict = {}
        if not self.skip_ragas and self.ragas and retrieved_ids and api_response:
            ragas_scores = self.ragas.evaluate(query, retrieved_ids, api_response)

        return {
            "retrieved_ids": retrieved_ids or [],
            "metrics": metrics,
            "ragas": ragas_scores,
        }

    def run(self, queries: list[dict]) -> dict:
        """Run evaluation for all `queries`. Returns full results dict."""
        results: dict = {
            "run_info": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "ollama_model": self.model,
                "n_queries": len(queries),
                "profiles_count": len(self.df),
                "k_values": K_VALUES,
                "relevance_threshold": RELEVANCE_THRESHOLD,
                "gt_pool_size": GT_POOL_SIZE,
                "skip_ragas": self.skip_ragas,
            },
            "per_query": {},
        }

        for qi, q in enumerate(queries, 1):
            qid, query, intent = q["qid"], q["query"], q["intent"]
            console.rule(f"[bold]{qid} ({qi}/{len(queries)}): {query[:55]}[/bold]")

            gt_judgments = self.gt_cache.get(qid, {})
            n_rel = sum(1 for j in gt_judgments.values() if j.get("score", 0) >= RELEVANCE_THRESHOLD)
            n_high = sum(1 for j in gt_judgments.values() if j.get("score", 0) == 3)
            console.print(
                f"  GT pool: {len(gt_judgments)} judged  |  "
                f"relevant(≥{RELEVANCE_THRESHOLD}): [green]{n_rel}[/green]  |  "
                f"highly relevant(3): [bold green]{n_high}[/bold green]"
            )

            # ── System API ───────────────────────────────────────────────
            console.print("  [cyan]→ System API[/cyan]", end="  ")
            time.sleep(API_SLEEP_S)
            api_ids, api_resp = self.api_retriever.retrieve(query)
            api_result = self._eval_one_mode(qid, query, api_ids, api_resp)
            if api_ids is not None:
                console.print(f"retrieved {len(api_ids)}  P@10={api_result['metrics'].get('precision_at_10', '—')}")
            else:
                console.print("[yellow]unavailable[/yellow]")

            # ── BM25 Baseline ────────────────────────────────────────────
            console.print("  [cyan]→ BM25 baseline[/cyan]", end="  ")
            bm25_ids = self.bm25_retriever.retrieve(query, k=API_K)
            bm25_result = self._eval_one_mode(qid, query, bm25_ids)
            console.print(f"retrieved {len(bm25_ids)}  P@10={bm25_result['metrics'].get('precision_at_10', '—')}")

            # ── Semantic Baseline ────────────────────────────────────────
            console.print("  [cyan]→ Semantic baseline[/cyan]", end="  ")
            sem_ids = self.semantic_retriever.retrieve(query, k=API_K)
            sem_result = self._eval_one_mode(qid, query, sem_ids)
            console.print(f"retrieved {len(sem_ids)}  P@10={sem_result['metrics'].get('precision_at_10', '—')}")

            results["per_query"][qid] = {
                "qid": qid,
                "query": query,
                "intent": intent,
                "gt_summary": {
                    "pool_size": len(gt_judgments),
                    "n_relevant": n_rel,
                    "n_highly_relevant": n_high,
                },
                "system_api": api_result,
                "bm25_baseline": bm25_result,
                "semantic_baseline": sem_result,
            }

        # ── Aggregate across queries ─────────────────────────────────────
        results["aggregate"] = self._aggregate(results["per_query"])
        return results

    @staticmethod
    def _aggregate(per_query: dict) -> dict:
        """Compute mean metrics across all evaluated queries per mode."""
        modes = ["system_api", "bm25_baseline", "semantic_baseline"]
        metric_keys = (
            [f"precision_at_{k}" for k in K_VALUES]
            + [f"recall_at_{k}" for k in K_VALUES]
            + ["map"]
            + [f"ndcg_at_{k}" for k in K_VALUES]
        )
        ragas_keys = ["context_precision", "context_recall", "answer_relevance", "faithfulness"]

        agg: dict = {}
        for mode in modes:
            mode_metrics: dict[str, list] = defaultdict(list)
            mode_ragas: dict[str, list] = defaultdict(list)
            for qdata in per_query.values():
                mdata = qdata.get(mode, {})
                m = mdata.get("metrics", {})
                r = mdata.get("ragas", {})
                for key in metric_keys:
                    val = m.get(key)
                    if val is not None:
                        mode_metrics[key].append(val)
                for key in ragas_keys:
                    val = r.get(key)
                    if val is not None:
                        mode_ragas[key].append(val)

            agg[mode] = {
                key: round(float(np.mean(vals)), 4) if vals else None
                for key, vals in mode_metrics.items()
            }
            if mode_ragas:
                agg[mode]["ragas"] = {
                    key: round(float(np.mean(vals)), 4) if vals else None
                    for key, vals in mode_ragas.items()
                }
        return agg


# ═══════════════════════════════════════════════════════════════════════════
#  REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(v) -> str:
    """Format a metric value for display."""
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def print_rich_report(results: dict):
    """Print a formatted report to the console using Rich tables."""
    if not RICH_OK:
        return

    agg = results.get("aggregate", {})
    modes = ["system_api", "bm25_baseline", "semantic_baseline"]
    mode_labels = ["System API", "BM25", "Semantic"]

    console.rule("[bold yellow]AGGREGATE IR METRICS[/bold yellow]")

    # IR metrics table
    table = Table(
        Column("Metric", style="bold cyan"),
        Column("System API", style="green"),
        Column("BM25 Baseline", style="yellow"),
        Column("Semantic Baseline", style="magenta"),
        title="[bold]IR Metrics (mean across all queries)[/bold]",
        show_header=True,
        header_style="bold white on dark_blue",
    )

    metric_rows = []
    for k in K_VALUES:
        metric_rows.append((f"Precision@{k}", f"precision_at_{k}"))
    for k in K_VALUES:
        metric_rows.append((f"Recall@{k}", f"recall_at_{k}"))
    metric_rows.append(("MAP", "map"))
    for k in K_VALUES:
        metric_rows.append((f"nDCG@{k}", f"ndcg_at_{k}"))

    for label, key in metric_rows:
        row_vals = [agg.get(mode, {}).get(key) for mode in modes]
        # Highlight best value
        valid_vals = [v for v in row_vals if v is not None]
        best = max(valid_vals) if valid_vals else None
        cells = []
        for v in row_vals:
            s = _fmt(v)
            if v is not None and v == best and best > 0:
                s = f"[bold]{s}[/bold]"
            cells.append(s)
        table.add_row(label, *cells)

    console.print(table)

    # RAGAS table (only if data exists)
    ragas_data = {mode: agg.get(mode, {}).get("ragas", {}) for mode in modes}
    if any(ragas_data[m] for m in modes):
        console.rule("[bold yellow]RAGAS METRICS (System API only)[/bold yellow]")
        rtable = Table(
            Column("RAGAS Metric", style="bold cyan"),
            Column("System API", style="green"),
            title="[bold]RAGAS Scores[/bold]",
            show_header=True,
        )
        for key, label in [
            ("context_precision", "Context Precision"),
            ("context_recall", "Context Recall"),
            ("answer_relevance", "Answer Relevance"),
            ("faithfulness", "Faithfulness"),
        ]:
            val = ragas_data.get("system_api", {}).get(key)
            rtable.add_row(label, _fmt(val))
        console.print(rtable)

    # Per-query nDCG@10
    console.rule("[bold yellow]PER-QUERY nDCG@10[/bold yellow]")
    qtable = Table(
        Column("QID", style="bold"),
        Column("Query", style="dim", max_width=40),
        Column("Intent", style="cyan"),
        Column("GT-Rel", style="white"),
        Column("API nDCG@10", style="green"),
        Column("BM25 nDCG@10", style="yellow"),
        Column("Sem nDCG@10", style="magenta"),
        title="[bold]Per-Query Results[/bold]",
        show_header=True,
    )
    for qid, qdata in sorted(results["per_query"].items()):
        n_rel = qdata["gt_summary"]["n_relevant"]
        intent = qdata["intent"]
        vals = []
        for mode in modes:
            v = qdata.get(mode, {}).get("metrics", {}).get("ndcg_at_10")
            vals.append(_fmt(v))
        qtable.add_row(qid, qdata["query"][:38], intent, str(n_rel), *vals)
    console.print(qtable)


def make_markdown_report(results: dict) -> str:
    """Generate a markdown report suitable for inserting into README."""
    info = results.get("run_info", {})
    agg = results.get("aggregate", {})
    modes = ["system_api", "bm25_baseline", "semantic_baseline"]
    mode_labels = ["System API", "BM25 Baseline", "Semantic Baseline"]
    ts = info.get("timestamp", "")
    model = info.get("ollama_model", "unknown")
    n_queries = info.get("n_queries", 25)

    lines = [
        "## EmployeeKHOJO.ai — Evaluation Report",
        "",
        f"**Generated:** `{ts}`  ",
        f"**Ground truth model:** `{model}`  ",
        f"**Queries evaluated:** {n_queries}  ",
        f"**Profile corpus:** {info.get('profiles_count', 1782)} candidates  ",
        f"**Relevance threshold:** score ≥ {info.get('relevance_threshold', 2)} / 3  ",
        "",
        "---",
        "",
        "### Aggregate IR Metrics",
        "",
        "| Metric | " + " | ".join(mode_labels) + " |",
        "|--------|" + "|".join([":------:"] * len(mode_labels)) + "|",
    ]

    metric_rows = (
        [(f"Precision@{k}", f"precision_at_{k}") for k in K_VALUES]
        + [(f"Recall@{k}", f"recall_at_{k}") for k in K_VALUES]
        + [("MAP", "map")]
        + [(f"nDCG@{k}", f"ndcg_at_{k}") for k in K_VALUES]
    )
    for label, key in metric_rows:
        row_vals = [agg.get(mode, {}).get(key) for mode in modes]
        cells = " | ".join(_fmt(v) for v in row_vals)
        lines.append(f"| {label} | {cells} |")

    # RAGAS section
    ragas_data = {mode: agg.get(mode, {}).get("ragas", {}) for mode in modes}
    if ragas_data.get("system_api"):
        lines += [
            "",
            "### RAGAS Metrics (System API)",
            "",
            "| Metric | System API |",
            "|--------|:----------:|",
        ]
        for key, label in [
            ("context_precision", "Context Precision"),
            ("context_recall", "Context Recall"),
            ("answer_relevance", "Answer Relevance"),
            ("faithfulness", "Faithfulness"),
        ]:
            v = ragas_data["system_api"].get(key)
            lines.append(f"| {label} | {_fmt(v)} |")

    # Per-query breakdown
    lines += [
        "",
        "### Per-Query nDCG@10",
        "",
        "| QID | Query | Intent | GT-Relevant | API | BM25 | Semantic |",
        "|-----|-------|--------|:-----------:|:---:|:----:|:--------:|",
    ]
    for qid, qdata in sorted(results["per_query"].items()):
        query = qdata["query"][:45].replace("|", "\\|")
        n_rel = qdata["gt_summary"]["n_relevant"]
        intent = qdata["intent"]
        vals = []
        for mode in modes:
            v = qdata.get(mode, {}).get("metrics", {}).get("ndcg_at_10")
            vals.append(_fmt(v))
        lines.append(f"| {qid} | {query} | {intent} | {n_rel} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "---",
        "",
        "> *Generated by `evaluate_real.py` — EmployeeKHOJO.ai evaluation framework*",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def _check_deps():
    """Abort early if critical deps are missing."""
    missing = []
    if not OLLAMA_OK:
        missing.append("ollama (pip install ollama)")
    if not BM25_OK:
        missing.append("rank-bm25 (pip install rank-bm25)")
    if not SKLEARN_OK:
        missing.append("scikit-learn (pip install scikit-learn)")
    if missing:
        console.print(f"[red]Missing dependencies:[/red]")
        for m in missing:
            console.print(f"  • {m}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="EmployeeKHOJO.ai — Production Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--regen-gt", action="store_true",
        help="Regenerate ground truth (ignore existing cache)",
    )
    parser.add_argument(
        "--skip-ragas", action="store_true",
        help="Skip RAGAS metrics (faster run; no Ollama calls during eval)",
    )
    parser.add_argument(
        "--queries", nargs="+", metavar="QID",
        help="Evaluate only specific query IDs, e.g. --queries Q01 Q05",
    )
    parser.add_argument(
        "--gt-model", metavar="MODEL",
        help="Override Ollama model for ground truth and RAGAS (e.g. qwen2.5:7b)",
    )
    parser.add_argument(
        "--csv", default=str(CSV_PATH), metavar="PATH",
        help=f"Path to profiles.csv (default: {CSV_PATH})",
    )
    parser.add_argument(
        "--api-url", default=API_BASE_URL, metavar="URL",
        help=f"System API base URL (default: {API_BASE_URL})",
    )
    args = parser.parse_args()

    console.rule("[bold yellow]EmployeeKHOJO.ai — Evaluation Script[/bold yellow]")
    _check_deps()

    # ── Load CSV ─────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if not csv_path.exists():
        console.print(f"[red]profiles.csv not found at {csv_path}[/red]")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    df["id"] = df["id"].apply(_norm_id)
    console.print(f"[green]✓[/green] Loaded {len(df)} profiles from [cyan]{csv_path}[/cyan]")

    # ── Select Ollama model ───────────────────────────────────────────────
    model = _select_model(args.gt_model)
    if model is None:
        console.print("[red]❌ No Ollama model available. Start Ollama and pull a model.[/red]")
        console.print("   sudo ollama serve  &&  ollama pull qwen2.5:7b")
        sys.exit(1)
    console.print(f"[green]✓[/green] Ollama model selected: [bold]{model}[/bold]")

    # ── Filter queries ────────────────────────────────────────────────────
    queries = TEST_QUERIES
    if args.queries:
        qids_wanted = set(args.queries)
        queries = [q for q in TEST_QUERIES if q["qid"] in qids_wanted]
        if not queries:
            console.print(f"[red]No queries matched: {args.queries}[/red]")
            sys.exit(1)
    console.print(f"[green]✓[/green] Evaluating {len(queries)} queries")

    # ── Load or generate ground truth ────────────────────────────────────
    console.rule("[bold]Ground Truth Generation[/bold]")
    cache: dict = {}
    if GT_CACHE_PATH.exists() and not args.regen_gt:
        cache = json.loads(GT_CACHE_PATH.read_text())
        n_cached = sum(1 for q in queries if q["qid"] in cache)
        console.print(
            f"[green]✓[/green] Loaded ground truth cache: {len(cache)} queries, "
            f"{n_cached}/{len(queries)} target queries already cached."
        )
    else:
        if args.regen_gt:
            console.print("[yellow]--regen-gt flag set: regenerating all ground truth.[/yellow]")
        else:
            console.print("[yellow]No cache found — generating ground truth from scratch.[/yellow]")

    # Check which queries still need ground truth
    queries_needing_gt = [q for q in queries if q["qid"] not in cache or args.regen_gt]
    if queries_needing_gt:
        console.print(
            f"[cyan]Generating ground truth for {len(queries_needing_gt)} queries "
            f"({GT_POOL_SIZE} candidates each → {len(queries_needing_gt) * GT_POOL_SIZE} LLM calls).[/cyan]"
        )
        console.print(
            f"[dim]Estimated time with {model}: varies. Ground truth is cached after each query.[/dim]"
        )
        gen = GroundTruthGenerator(df, model)
        for q in queries_needing_gt:
            console.print(f"\n[bold cyan]Ground truth: {q['qid']} — {q['query']}[/bold cyan]")
            cache[q["qid"]] = gen.generate_for_query(
                q["qid"], q["query"], cache, regen=args.regen_gt, verbose=True
            )
            GT_CACHE_PATH.write_text(json.dumps(cache, indent=2))
            n_rel = sum(1 for j in cache[q["qid"]].values() if j.get("score", 0) >= RELEVANCE_THRESHOLD)
            console.print(f"  → {len(cache[q['qid']])} judged, {n_rel} relevant (≥{RELEVANCE_THRESHOLD})")
    else:
        console.print("[green]✓[/green] All target queries have cached ground truth.")

    # ── Run evaluation ────────────────────────────────────────────────────
    console.rule("[bold]Retrieval Evaluation[/bold]")
    runner = EvaluationRunner(
        df=df,
        gt_cache=cache,
        model=model,
        skip_ragas=args.skip_ragas,
    )
    results = runner.run(queries)

    # ── Save JSON results ─────────────────────────────────────────────────
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    console.print(f"\n[green]✓[/green] Results saved to [bold]{RESULTS_PATH}[/bold]")

    # ── Print rich report ─────────────────────────────────────────────────
    console.rule("[bold yellow]EVALUATION RESULTS[/bold yellow]")
    print_rich_report(results)

    # ── Save markdown report ──────────────────────────────────────────────
    report_md = make_markdown_report(results)
    REPORT_PATH.write_text(report_md)
    console.print(f"\n[green]✓[/green] Markdown report saved to [bold]{REPORT_PATH}[/bold]")
    console.print(f"[green]✓[/green] Ground truth cache: [bold]{GT_CACHE_PATH}[/bold]")

    # ── Summary ───────────────────────────────────────────────────────────
    console.rule("[bold]Summary[/bold]")
    agg = results.get("aggregate", {})
    for mode, label in [("system_api", "System API"), ("bm25_baseline", "BM25"), ("semantic_baseline", "Semantic")]:
        m = agg.get(mode, {})
        ndcg10 = m.get("ndcg_at_10")
        map_v = m.get("map")
        p10 = m.get("precision_at_10")
        console.print(
            f"  [bold]{label:18s}[/bold] "
            f"nDCG@10={_fmt(ndcg10)}  MAP={_fmt(map_v)}  P@10={_fmt(p10)}"
        )


if __name__ == "__main__":
    main()
