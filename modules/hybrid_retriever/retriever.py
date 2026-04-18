"""
Hybrid Retrieval Engine — Component B
Intent-Aware Retrieval combining BM25 (lexical) + FAISS (semantic) with RRF fusion.

Input  : Parsed intent JSON from Component A (query understanding)
         + index artifacts: dense_matrix.npy, dense.pkl, bm25.pkl, metadata.pkl, skills.pkl
Output : Ranked JSON with per-result explanations ready for the Explainability Bot.

Design priorities
-----------------
- Core skills are BOOSTED (3x weight) vs secondary/soft skills
- RRF (Reciprocal Rank Fusion) merges lexical + semantic rank lists
- Per-result explanation dict tracks WHICH signals fired and WHY
- Zero heavy I/O at query time — all indexes loaded once at startup
"""

from __future__ import annotations

import json
import math
import re
import time
import pickle
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

# ============================================================================
# FLEX UNPICKLER (To handle missing indexer module classes)
# ============================================================================

class FlexUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type(name, (), {})


# ---------------------------------------------------------------------------
# Optional heavy deps — fail gracefully so unit tests can import this file
# ---------------------------------------------------------------------------
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


# ============================================================================
# CONSTANTS
# ============================================================================

# RRF rank-smoothing constant (standard value = 60)
RRF_K = 60

# Weight split between lexical and semantic in final score
ALPHA_BM25   = 0.40   # lexical contribution
ALPHA_DENSE  = 0.60   # semantic contribution

# Core-skill boost — candidates whose core_skills match query get multiplied
CORE_SKILL_BOOST    = 3.0   # multiplier on BM25 TF for core skill tokens
SECONDARY_BOOST     = 1.2   # multiplier for secondary skill tokens

# Experience-band → minimum years mapping (used for hard filtering)
EXPERIENCE_BAND_MIN = {
    "junior":   0,
    "mid":      2,
    "senior":   5,
    "lead":     8,
    "principal":10,
}

# Maximum candidates to return
DEFAULT_TOP_K = 20

# SBERT model — swappable
SBERT_MODEL = "all-MiniLM-L6-v2"


# ============================================================================
# BM25 ENGINE  (pure-python, loaded from bm25.pkl)
# ============================================================================

class BM25Engine:
    """
    Wraps the serialised BM25 index produced by Component A (indexer.py).

    Expected pickle keys
    --------------------
    corpus_tokens : list[list[str]]   — tokenised docs
    idf           : dict[str, float]  — precomputed IDF per term
    avgdl         : float             — average document length
    k1, b         : float             — Robertson BM25 params
    doc_freqs     : list[dict]        — TF per doc (optional; recomputed if absent)

    The engine adds core-skill boosting: tokens that appear in a candidate's
    core_skills string receive CORE_SKILL_BOOST multiplied into their TF.
    """

    def __init__(self, bm25_path: str | Path, metadata: list[dict]) -> None:
        with open(bm25_path, "rb") as f:
            data = FlexUnpickler(f).load()

        # Extract from BM25Index object
        self._doc_freqs: list[dict[str, int]] = getattr(data, "_doc_freqs", [])
        self._df: dict[str, int]              = getattr(data, "_df", {})
        self.avgdl: float                   = getattr(data, "_avgdl", 50.0)
        self.k1: float                      = getattr(data, "k1", 1.5)
        self.b: float                       = getattr(data, "b", 0.75)
        self.n_docs: int                    = len(self._doc_freqs)
        self._doc_lengths                   = getattr(data, "_doc_lengths", [])
        self.normalizer                     = getattr(data, "_normalizer", None)

        # Convert _df to IDF
        self.idf: dict[str, float] = {}
        for term, freq in self._df.items():
            self.idf[term] = math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1.0)

        # Build per-doc TF dicts (with core-skill boosting baked in)
        self._doc_tfs: list[dict[str, float]] = []
        for idx, tf_dict in enumerate(self._doc_freqs):
            core_text = (metadata[idx].get("core_skills") or "").lower()
            sec_text = (metadata[idx].get("secondary_skills") or "").lower()
            
            tf: dict[str, float] = {}
            for tok, base_tf in tf_dict.items():
                boosted_tf = float(base_tf)
                # Apply boost if the token appears in core_skills field
                if tok in core_text:
                    boosted_tf *= CORE_SKILL_BOOST
                elif tok in sec_text:
                    boosted_tf *= SECONDARY_BOOST
                tf[tok] = boosted_tf
            self._doc_tfs.append(tf)

    # ------------------------------------------------------------------
    def score_query(
        self,
        query_tokens: list[str],
        core_query_tokens: set[str],
    ) -> np.ndarray:
        """
        Return BM25 scores for every document.

        core_query_tokens — tokens extracted from detected skills in the intent;
                            these receive an additional IDF boost.
        """
        scores = np.zeros(self.n_docs, dtype=np.float32)
        term_scores: dict[str, np.ndarray] = {}
        
        dl_arr = np.array(self._doc_lengths, dtype=np.float32)
        norm   = 1.0 - self.b + self.b * (dl_arr / self.avgdl)

        for term in query_tokens:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            # Extra IDF amplification for skill tokens from parsed intent
            if term in core_query_tokens:
                idf_val *= CORE_SKILL_BOOST

            t_scores = np.zeros(self.n_docs, dtype=np.float32)
            for doc_id, tf_dict in enumerate(self._doc_tfs):
                tf = tf_dict.get(term, 0.0)
                if tf == 0:
                    continue
                tf_norm = (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm[doc_id])
                t_scores[doc_id] = idf_val * tf_norm

            scores += t_scores
            term_scores[term] = t_scores

        return scores, term_scores

    def tokenize(self, text: str) -> list[str]:
        # Simple fallback
        if not self.normalizer:
            text = text.lower()
            text = re.sub(r"[^a-z0-9\s]", " ", text)
            return [t for t in text.split() if len(t) > 1]
            
        # Use canonical normalizer logic
        text = text.lower()
        # 1. Expand abbreviations
        abbrev = getattr(self.normalizer, '_abbrev_map', {})
        for word, expansion in abbrev.items():
            text = re.sub(rf'\b{re.escape(word)}\b', expansion, text)
            
        # 2. Entity aliases
        aliases = getattr(self.normalizer, '_entity_aliases', {})
        for phrase, canonical in aliases.items():
            text = text.replace(phrase, canonical)
            
        # 3. Strip suffixes
        suffixes = getattr(self.normalizer, '_strip_suffixes', [])
        for suffix in suffixes:
            text = text.replace(suffix, '')
            
        tokens = []
        syn_map = getattr(self.normalizer, '_synonym_to_canonical', {})
        # Simple greedy matching for phrases
        words = re.sub(r"[^a-z0-9\s]", " ", text).split()
        
        # Extremely basic tokenizer simulating what the canonicalizer does
        # Real one is likely more complex, but this will get us "aws" -> "amazon_web_services"
        # We'll just map individual words and bigrams if possible, or fallback to the exact dict
        
        for w in words:
            if w in syn_map:
                tokens.append(syn_map[w].replace(' ', '_'))
            else:
                tokens.append(w.replace(' ', '_'))
                
        # Handle full phrases mapping (like "amazon web services")
        full_text = " ".join(words)
        for phrase, canonical in syn_map.items():
            if phrase in full_text:
                tokens.append(canonical.replace(' ', '_'))
                
        return list(set(tokens))


# ============================================================================
# SEMANTIC ENGINE  (SBERT + FAISS)
# ============================================================================

class SemanticEngine:
    """
    Wraps the FAISS index + SBERT encoder.

    dense.pkl is expected to contain:
        model_name    : str  (optional, falls back to SBERT_MODEL constant)
        index_type    : str  ("flat_ip" | "flat_l2")
        n_candidates  : int
        embedding_dim : int

    dense_matrix.npy contains the raw 384-d float32 embeddings (n_docs × 384).
    """

    def __init__(
        self,
        dense_pkl_path: str | Path,
        dense_matrix_path: str | Path,
    ) -> None:
        with open(dense_pkl_path, "rb") as f:
            cfg = FlexUnpickler(f).load()

        # Load raw matrix directly from object
        if hasattr(cfg, "_matrix"):
            matrix: np.ndarray = cfg._matrix.astype(np.float32)
        elif Path(dense_matrix_path).exists():
            matrix = np.load(dense_matrix_path).astype(np.float32)
        else:
            raise FileNotFoundError(f"Matrix not found in object or {dense_matrix_path}")

        strategy = getattr(cfg, "strategy", None)
        self.model_name: str = SBERT_MODEL if strategy == "sbert" or not strategy else strategy
        self.embedding_dim: int = matrix.shape[1]
        self.n_docs = matrix.shape[0]

        # Normalise for cosine similarity via inner product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        self._matrix = matrix / norms          # shape (n_docs, dim)

        if FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(self.embedding_dim)
            self._index.add(self._matrix)
        else:
            self._index = None

        # Always lazy-load encoder to avoid using broken unpickled dummy objects
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None and SBERT_AVAILABLE:
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def encode_query(self, text: str) -> np.ndarray:
        enc = self._get_encoder()
        if enc is None:
            # Fallback: random unit vector (for testing without GPU/model)
            v = np.random.randn(self.embedding_dim).astype(np.float32)
            return v / np.linalg.norm(v)
        vec = enc.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return vec[0].astype(np.float32)

    def score_query(self, query_vec: np.ndarray) -> np.ndarray:
        """Return cosine similarity scores for all docs."""
        if self._index is not None:
            # FAISS batch inner product — O(n·d) but extremely fast in C++
            scores, _ = self._index.search(query_vec.reshape(1, -1), self.n_docs)
            return scores[0]
        else:
            # Pure numpy fallback
            return self._matrix @ query_vec


# ============================================================================
# RRF FUSION
# ============================================================================

def reciprocal_rank_fusion(
    rank_lists: list[np.ndarray],
    weights: list[float] | None = None,
    k: int = RRF_K,
) -> np.ndarray:
    """
    Fuse multiple rank lists via Reciprocal Rank Fusion.

    rank_lists : each array is a *score* vector (higher = better) over all docs.
                 Converted internally to rank positions.
    weights    : optional per-list weights (default uniform)
    Returns    : fused score array (higher = better)
    """
    if weights is None:
        weights = [1.0] * len(rank_lists)

    n_docs = rank_lists[0].shape[0]
    fused  = np.zeros(n_docs, dtype=np.float64)

    for score_arr, w in zip(rank_lists, weights):
        # argsort descending → rank positions (0-based)
        order = np.argsort(score_arr)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(n_docs)
        fused += w * (1.0 / (k + ranks + 1))

    return fused.astype(np.float32)


# ============================================================================
# EXPERIENCE FILTER
# ============================================================================

def passes_experience_filter(
    candidate: dict,
    experience_band: str | None,
    experience_years: str | None,
) -> bool:
    """
    Hard filter. Returns False if the candidate clearly doesn't meet requirements.
    Lenient: if years_of_experience is missing/zero AND band is junior, allow.
    """
    cand_years = float(candidate.get("years_of_experience") or 0)

    if experience_years:
        try:
            req = float(experience_years)
            # Allow ±1 year tolerance (don't hard-reject borderline candidates)
            if cand_years < req - 1.5:
                return False
        except ValueError:
            pass

    if experience_band:
        min_yr = EXPERIENCE_BAND_MIN.get(experience_band.lower(), 0)
        if cand_years < max(0, min_yr - 1):
            return False

    return True


# ============================================================================
# EXPLANATION BUILDER
# ============================================================================

def build_explanation(
    candidate: dict,
    matched_skills: list[str],
    bm25_score: float,
    dense_score: float,
    rrf_score: float,
    rank: int,
    parsed_intent: dict,
    term_contributions: dict[str, float],
) -> dict:
    """
    Human-readable explanation dict for each result.
    This feeds directly into the Explainability Bot (Component F).
    """
    core_skills_raw   = candidate.get("core_skills", "")
    secondary_raw     = candidate.get("secondary_skills", "")
    core_skill_hits   = [s for s in matched_skills if s.lower() in core_skills_raw.lower()]
    sec_skill_hits    = [s for s in matched_skills if s.lower() in secondary_raw.lower()
                         and s not in core_skill_hits]

    signals = []
    if core_skill_hits:
        signals.append(f"Core skill match: {', '.join(core_skill_hits)}")
    if sec_skill_hits:
        signals.append(f"Secondary skill match: {', '.join(sec_skill_hits)}")
    if bm25_score > 0:
        signals.append(f"Keyword relevance (BM25={bm25_score:.3f})")
    if dense_score > 0.5:
        signals.append(f"Semantic similarity (cos={dense_score:.3f})")

    exp_band = parsed_intent.get("experience_band")
    exp_yrs  = candidate.get("years_of_experience", 0)
    if exp_band:
        signals.append(f"Experience band '{exp_band}' — candidate has {exp_yrs} yrs")

    matched_keywords = sorted(
        [{"term": t, "contribution": round(float(v), 4)} for t, v in term_contributions.items() if v > 0],
        key=lambda x: x["contribution"],
        reverse=True,
    )

    return {
        "rank":              rank,
        "rrf_score":         round(float(rrf_score), 6),
        "bm25_score":        round(float(bm25_score), 4),
        "semantic_score":    round(float(dense_score), 4),
        "core_skill_hits":   core_skill_hits,
        "secondary_hits":    sec_skill_hits,
        "matched_keywords":  matched_keywords,
        "retrieval_signals": signals,
        "why_retrieved":     " | ".join(signals) if signals else "General profile relevance",
    }


# ============================================================================
# MAIN HYBRID RETRIEVER
# ============================================================================

class HybridRetriever:
    """
    Loads all index artifacts once and answers queries in milliseconds.

    Parameters
    ----------
    index_dir : path to folder containing bm25.pkl, dense.pkl,
                dense_matrix.npy, metadata.pkl, skills.pkl (optional)
    top_k     : number of candidates to return
    """

    def __init__(self, index_dir: str | Path, top_k: int = DEFAULT_TOP_K) -> None:
        index_dir  = Path(index_dir)
        self.top_k = top_k

        # ---------- Load Dense Index first to get texts & doc_ids ----------
        dense_path = index_dir / "dense.pkl"
        with open(dense_path, "rb") as f:
            dense_obj = FlexUnpickler(f).load()
            
        doc_ids = getattr(dense_obj, "_doc_ids", [])
        texts = getattr(dense_obj, "_texts", [])

        # ---------- Build metadata from _texts ----------
        self.metadata = []
        for i, text in enumerate(texts):
            doc_id = doc_ids[i] if i < len(doc_ids) else str(i)
            
            # Parse skills and roles from text
            # Format usually: "... Skills: X (Competent), Y (Beginner). Suitable for roles: Z"
            skills_part = ""
            roles_part = ""
            if "Skills:" in text:
                parts = text.split("Skills:")
                skills_raw = parts[1]
                if "Suitable for roles:" in skills_raw:
                    skills_part, roles_part = skills_raw.split("Suitable for roles:")
                else:
                    skills_part = skills_raw
            
            core_skills = []
            secondary_skills = []
            max_years = 0.0
            
            # Simple heuristic: Advanced/Competent -> Core (more years), Beginner -> Secondary (less years)
            for skill_match in re.finditer(r"([^,]+?)\s*\(([^)]+)\)", skills_part):
                skill = skill_match.group(1).strip()
                prof = skill_match.group(2).strip().lower()
                if "competent" in prof or "advanced" in prof or "expert" in prof:
                    core_skills.append(f"{skill} ({prof})")
                    max_years = max(max_years, 3.0 if "competent" in prof else 5.0)
                else:
                    secondary_skills.append(f"{skill} ({prof})")
                    max_years = max(max_years, 1.0)
                    
            if max_years == 0 and ("fresher" in text.lower() or "intern" in text.lower()):
                max_years = 0.5
            elif max_years == 0:
                max_years = 2.0  # Fallback

            # Look for explicit mentions of years
            years_match = re.search(r'(\d+)\+?\s*years?(?:\s*of)?\s*experience', text.lower())
            if years_match:
                max_years = max(max_years, float(years_match.group(1)))
            
            # Boost based on role mentions
            text_lower = text.lower()
            if "manager" in text_lower or "architect" in text_lower:
                max_years = max(max_years, 8.0)
            elif "lead" in text_lower or "senior" in text_lower:
                max_years = max(max_years, 6.0)

            self.metadata.append({
                "id": doc_id,
                "name": f"Candidate {doc_id}",
                "core_skills": ", ".join(core_skills),
                "secondary_skills": ", ".join(secondary_skills),
                "soft_skills": "",
                "years_of_experience": max_years,
                "potential_roles": roles_part.strip(),
                "skill_summary": text,
            })

        self.n_docs = len(self.metadata)

        # ---------- BM25 ----------
        self.bm25 = BM25Engine(index_dir / "bm25.pkl", self.metadata)

        # ---------- FAISS / SBERT ----------
        self.semantic = SemanticEngine(
            index_dir / "dense.pkl",
            index_dir / "dense_matrix.npy",
        )

        # ---------- Skills inverted index (optional) ----------
        skills_path = index_dir / "skills.pkl"
        self._skills_index: dict[str, list[int]] = {}
        if skills_path.exists():
            with open(skills_path, "rb") as f:
                self._skills_index = FlexUnpickler(f).load()

        print(f"[HybridRetriever] Loaded {self.n_docs} candidates from {index_dir}")

    # ------------------------------------------------------------------
    def retrieve(self, intent_json: dict) -> dict:
        """
        Main entry point.

        Parameters
        ----------
        intent_json : the full parsed-intent object from Component A

        Returns
        -------
        dict — full retrieval output JSON (see OUTPUT FORMAT below)
        """
        intent_block   = intent_json.get("intent") or {}
        t0             = time.perf_counter()
        parsed         = intent_json.get("entities") or intent_json.get("parsed") or {}
        modifiers      = intent_block.get("modifiers") or []
        confidence     = float(intent_block.get("confidence") or 0.70)
        primary_intent = intent_block.get("primary") or intent_block.get("primary_intent") or "general"
        skills         = parsed.get("skills") or []
        domain         = parsed.get("domain") or ""
        role           = parsed.get("role") or ""
        negated        = parsed.get("negated_skills") or []
        kg_ready       = bool(intent_json.get("kg_ready", False))
        top3           = intent_block.get("top3_scores") or {}

        queries   = intent_json.get("queries", [])
        strategy_map = intent_json.get("strategy_map", {})

        skill_tokens: list[str]   = [s.lower() for s in (skills)]
        core_skill_set: set[str]  = set(skill_tokens)   # from intent → core query terms
        exp_band: str | None      = parsed.get("experience_band")
        exp_years: str | None     = parsed.get("experience_years")
        negated_list: list[str]   = [s.lower() for s in (negated)]

        # ---- 1. Build BM25 score vector (aggregate over all expanded queries) ----
        bm25_agg = np.zeros(self.n_docs, dtype=np.float32)
        term_contrib_agg: dict[str, np.ndarray] = {}
        bm25_per_query: list[tuple[str, str, np.ndarray]] = []

        # Canonicalize the core skill tokens for matching
        canonical_core_skills = set()
        for tok in skill_tokens:
            canonical_core_skills.update(self.bm25.tokenize(tok))

        for q in queries:
            tokens = self.bm25.tokenize(q)
            scores, term_scores = self.bm25.score_query(tokens, canonical_core_skills)
            strategy = strategy_map.get(q, "unknown")
            bm25_per_query.append((q, strategy, scores))

            # Weight: original/synonym queries count more than template queries
            w = 1.0 if strategy in ("original", "synonym") else 0.6
            bm25_agg += w * scores

            for term, ts in term_scores.items():
                if term not in term_contrib_agg:
                    term_contrib_agg[term] = np.zeros(self.n_docs, dtype=np.float32)
                term_contrib_agg[term] += w * ts

        # Normalise BM25 aggregate
        mx = bm25_agg.max()
        if mx > 0:
            bm25_agg /= mx
            for t in term_contrib_agg:
                term_contrib_agg[t] /= mx

        # ---- 2. Build Semantic score vector (encode corrected query once) ----
        primary_query = intent_json.get("corrected") or queries[0]
        query_vec = self.semantic.encode_query(primary_query)
        dense_scores = self.semantic.score_query(query_vec)

        # Normalise dense scores (already cosine similarity, clamp to [0,1])
        dense_scores = np.clip(dense_scores, 0, 1)

        # ---- 3. Hard filter: remove negated skills & experience mismatch ----
        valid_mask = np.ones(self.n_docs, dtype=bool)
        for idx, cand in enumerate(self.metadata):
            cand_skills_text = (
                (cand.get("core_skills") or "") + " " +
                (cand.get("secondary_skills") or "")
            ).lower()
            # Negation filter
            if any(neg in cand_skills_text for neg in negated):
                valid_mask[idx] = False
                continue
            # Experience filter
            if not passes_experience_filter(cand, exp_band, exp_years):
                valid_mask[idx] = False

        # ---- 4. RRF Fusion ----
        rrf_scores = reciprocal_rank_fusion(
            [bm25_agg, dense_scores],
            weights=[ALPHA_BM25, ALPHA_DENSE],
        )
        # Zero out filtered candidates
        rrf_scores[~valid_mask] = 0.0

        # ---- 5. Core-skill hard-boost: candidates with core skill hits jump up ----
        for idx, cand in enumerate(self.metadata):
            core_text = (cand.get("core_skills") or "").lower()
            hit_count = sum(1 for s in skill_tokens if s in core_text)
            if hit_count > 0:
                rrf_scores[idx] *= (1.0 + 0.3 * hit_count)  # additive boost per hit

        # Pure lexical scores
        lexical_scores = bm25_agg.copy()
        lexical_scores[~valid_mask] = 0.0

        # Pure semantic scores
        semantic_scores = dense_scores.copy()
        semantic_scores[~valid_mask] = 0.0
        semantic_scores[semantic_scores < 0.5] = 0.0

        # ---- 6. Build output records ----
        def _build_records(score_arr: np.ndarray, mode: str) -> list[dict]:
            top_idx = np.argsort(score_arr)[::-1][: self.top_k]
            records = []
            for rank, idx in enumerate(top_idx, start=1):
                if score_arr[idx] <= 0:
                    break
                cand = self.metadata[idx]

                cand_all_skills = (
                    (cand.get("core_skills") or "") + " " +
                    (cand.get("secondary_skills") or "")
                ).lower()
                matched = [s for s in skill_tokens if s in cand_all_skills]

                # Per-doc term contributions for this candidate
                doc_term_contrib = {
                    t: float(arr[idx])
                    for t, arr in term_contrib_agg.items()
                    if arr[idx] > 0
                }

                expl = build_explanation(
                    candidate   = cand,
                    matched_skills = matched,
                    bm25_score  = float(bm25_agg[idx]),
                    dense_score = float(dense_scores[idx]),
                    rrf_score   = float(score_arr[idx]),
                    rank        = rank,
                    parsed_intent = parsed,
                    term_contributions = doc_term_contrib,
                )

                records.append({
                    "rank":               rank,
                    "candidate_id":       cand.get("id"),
                    "name":               cand.get("name", ""),
                    "years_of_experience":cand.get("years_of_experience", 0),
                    "potential_roles":    cand.get("potential_roles", ""),
                    "core_skills":        cand.get("core_skills", ""),
                    "secondary_skills":   cand.get("secondary_skills", ""),
                    "soft_skills":        cand.get("soft_skills", ""),
                    "skill_summary":      cand.get("skill_summary", ""),
                    "scores": {
                        "primary":  expl["rrf_score"] if mode == "hybrid" else (expl["bm25_score"] if mode == "lexical" else expl["semantic_score"]),
                        "rrf":     expl["rrf_score"],
                        "bm25":    expl["bm25_score"],
                        "semantic":expl["semantic_score"],
                    },
                    "explanation":        expl,
                })
            return records

        hybrid_results = _build_records(rrf_scores, "hybrid")
        lexical_results = _build_records(lexical_scores, "lexical")
        semantic_results = _build_records(semantic_scores, "semantic")

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        # ---- 7. Per-query breakdown ----
        query_breakdown = []
        for q, strategy, scores in bm25_per_query:
            top5_ids = np.argsort(scores)[::-1][:5]
            query_breakdown.append({
                "query":    q,
                "strategy": strategy,
                "top5_bm25_hits": [
                    {"candidate_id": self.metadata[i].get("id"), "bm25": round(float(scores[i]), 4)}
                    for i in top5_ids if scores[i] > 0
                ],
            })

        meta = {
            "original_query":    intent_json.get("original"),
            "corrected_query":   intent_json.get("corrected_query") or intent_json.get("corrected"),
            "primary_intent":    intent_json.get("intent", {}).get("primary") or intent_json.get("intent", {}).get("primary_intent"),
            "detected_skills":   skill_tokens,
            "experience_band":   exp_band,
            "experience_years":  exp_years,
            "total_candidates":  self.n_docs,
            "filtered_out":      int((~valid_mask).sum()),
            "retrieval_time_ms": elapsed_ms,
            "alpha_bm25":        ALPHA_BM25,
            "alpha_dense":       ALPHA_DENSE,
            "rrf_k":             RRF_K,
        }

        return {
            "hybrid": {"meta": {**meta, "mode": "hybrid", "returned": len(hybrid_results)}, "results": hybrid_results, "query_breakdown": query_breakdown},
            "lexical": {"meta": {**meta, "mode": "lexical", "returned": len(lexical_results)}, "results": lexical_results},
            "semantic": {"meta": {**meta, "mode": "semantic", "returned": len(semantic_results)}, "results": semantic_results},
        }
