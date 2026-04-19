"""
Hybrid Retrieval Engine — Component B  (Fixed v4)
Intent-Aware Retrieval combining BM25 (lexical) + FAISS (semantic) with RRF fusion.

Key fix in v4
-------------
- SemanticEngine ALWAYS rebuilds its FAISS index from dense_matrix.npy at load
  time instead of reusing the pickled DenseIndex's internal _faiss_index.
  This eliminates the `assert d == self.d` AssertionError that arose because
  the pickled DenseIndex stored a FAISS index whose dimension came from the
  original training machine, while the live BGE encoder on this machine may
  have a subtly different internal state.

- The `retrieve()` method now calls `self.semantic.score_queries_batch()`
  (the correct plural method) instead of the non-existent `score_query()`.

- `SemanticEngine.encode_queries_batch()` is the single encode entry point;
  a convenience `encode_query()` alias is added for single-query use.
"""

from __future__ import annotations

import math
import re
import time
import pickle
from pathlib import Path
from typing import Any

import numpy as np

# ============================================================================
# FLEX UNPICKLER
# ============================================================================

class FlexUnpickler(pickle.Unpickler):
    """Gracefully handles missing Component A classes when unpickling."""
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type(name, (), {})


# ============================================================================
# OPTIONAL HEAVY DEPS
# ============================================================================

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


# ============================================================================
# CONSTANTS
# ============================================================================

RRF_K            = 60
ALPHA_BM25       = 0.40
ALPHA_DENSE      = 0.60
CORE_SKILL_BOOST = 3.0      # TF multiplier in _doc_tfs (index-time alignment)
SECONDARY_BOOST  = 1.2      # TF multiplier for secondary skills
CORE_IDF_BOOST   = 1.5      # IDF amplifier for core query tokens
DEFAULT_TOP_K    = 50
SBERT_MODEL      = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CORE_WEIGHT_THRESHOLD = 1.2

EXPERIENCE_BAND_MIN = {
    "junior": 0, "mid": 2, "senior": 5, "lead": 8, "principal": 10,
}

STOPWORDS: frozenset[str] = frozenset({
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","as","is","are","was","were","be","been","being","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","shall","can","not","no","nor","so","yet","this","that","it",
    "its","my","your","he","she","we","they","them","their","our","than",
    "then","there","here","when","where","who","which","what","how","if",
    "about","up","out","into","over","after","also",
    "senior","junior","mid","level","entry","lead","principal",
    "years","year","yrs","yr","experience","experienced",
    "candidate","candidates","someone","person","professional",
    "role","roles","position","job","looking","need","required",
    "require","seeking","find","want","skills","skill",
    "proficient","competent","expert","beginner","advanced",
    "suitable","background","strong","good",
    "one","two","three","four","five","six","seven","eight","nine","ten",
})


# ============================================================================
# HELPER — build rich metadata from Component A's MetadataStore
# ============================================================================

def _build_metadata_list(
    meta_store: Any,
    dense_doc_ids: list[str],
    dense_texts: list[str],
) -> list[dict]:
    store: dict[str, dict] = getattr(meta_store, "_store", {})
    metadata = []

    for i, doc_id in enumerate(dense_doc_ids):
        raw  = store.get(doc_id)
        text = dense_texts[i] if i < len(dense_texts) else ""

        if raw is None:
            metadata.append({
                "id":                    doc_id,
                "name":                  f"Candidate {doc_id}",
                "top_skills":            [],
                "core_skill_names":      set(),
                "secondary_skill_names": set(),
                "core_skills_str":       "",
                "secondary_skills_str":  "",
                "soft_skills":           "",
                "years_of_experience":   0.0,
                "potential_roles":       "",
                "skill_summary":         text,
            })
            continue

        top_skills: list[tuple[str, float]] = raw.get("top_skills") or []

        core_skills      = [(s, w) for s, w in top_skills if w >= CORE_WEIGHT_THRESHOLD]
        secondary_skills = [(s, w) for s, w in top_skills if w <  CORE_WEIGHT_THRESHOLD]

        core_skill_names      = {s for s, _ in core_skills}
        secondary_skill_names = {s for s, _ in secondary_skills}

        def _fmt(skill_list: list[tuple[str, float]]) -> str:
            return ", ".join(
                f"{s.replace('_', ' ').title()} ({_weight_to_prof(w)})"
                for s, w in skill_list
            )

        metadata.append({
            "id":                    doc_id,
            "name":                  raw.get("name") or f"Candidate {doc_id}",
            "top_skills":            top_skills,
            "core_skill_names":      core_skill_names,
            "secondary_skill_names": secondary_skill_names,
            "core_skills_str":       _fmt(core_skills),
            "secondary_skills_str":  _fmt(secondary_skills),
            "soft_skills":           "",
            "years_of_experience":   float(raw.get("years_of_experience") or 0),
            "potential_roles":       raw.get("potential_roles") or "",
            "skill_summary":         text,
        })

    return metadata


def _weight_to_prof(w: float) -> str:
    if w >= 2.0: return "Expert"
    if w >= 1.7: return "Advanced"
    if w >= 1.5: return "Proficient"
    if w >= 1.2: return "Competent"
    if w >= 1.0: return "Intermediate"
    if w >= 0.7: return "Advanced Beginner"
    if w >= 0.5: return "Beginner"
    return "Novice"


# ============================================================================
# BM25 ENGINE
# ============================================================================

class BM25Engine:
    def __init__(
        self,
        bm25_path: str | Path,
        metadata: list[dict],
        dense_doc_ids: list[str],
    ) -> None:
        with open(bm25_path, "rb") as f:
            data = FlexUnpickler(f).load()

        self._bm25_doc_ids: list[str]         = getattr(data, "_doc_ids", [])
        self._doc_freqs: list[dict[str, int]]  = getattr(data, "_doc_freqs", [])
        self._doc_lengths: list[int]           = getattr(data, "_doc_lengths", [])
        self._df: dict[str, int]               = getattr(data, "_df", {})
        self.avgdl: float                      = getattr(data, "_avgdl", 50.0)
        self.k1: float                         = getattr(data, "k1", 1.5)
        self.b: float                          = getattr(data, "b", 0.75)
        self.n_bm25: int                       = len(self._bm25_doc_ids)
        self.normalizer                        = getattr(data, "_normalizer", None)

        self.idf: dict[str, float] = {
            term: math.log((self.n_bm25 - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in self._df.items()
        }

        bm25_id_to_pos: dict[str, int] = {
            did: i for i, did in enumerate(self._bm25_doc_ids)
        }
        self.n_docs = len(dense_doc_ids)

        self._doc_tfs: list[dict[str, float]] = []
        for dense_pos, doc_id in enumerate(dense_doc_ids):
            bm25_pos = bm25_id_to_pos.get(doc_id)
            if bm25_pos is None:
                self._doc_tfs.append({})
                continue

            raw_tf = self._doc_freqs[bm25_pos]
            meta   = metadata[dense_pos]

            core_names      = meta.get("core_skill_names", set())
            secondary_names = meta.get("secondary_skill_names", set())

            tf: dict[str, float] = {}
            for tok, base_tf in raw_tf.items():
                tok_space = tok.replace("_", " ")
                if tok_space in core_names or tok in core_names:
                    tf[tok] = float(base_tf) * CORE_SKILL_BOOST
                elif tok_space in secondary_names or tok in secondary_names:
                    tf[tok] = float(base_tf) * SECONDARY_BOOST
                else:
                    tf[tok] = float(base_tf)
            self._doc_tfs.append(tf)

        self._aligned_lengths: list[int] = []
        for doc_id in dense_doc_ids:
            bm25_pos = bm25_id_to_pos.get(doc_id)
            if bm25_pos is not None and bm25_pos < len(self._doc_lengths):
                self._aligned_lengths.append(self._doc_lengths[bm25_pos])
            else:
                self._aligned_lengths.append(int(self.avgdl))

    def score_query(
        self,
        query_tokens: list[str],
        core_query_tokens: set[str],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        term_scores: dict[str, np.ndarray] = {}

        dl_arr = np.array(self._aligned_lengths, dtype=np.float32)
        norm   = 1.0 - self.b + self.b * (dl_arr / max(self.avgdl, 1.0))

        for term in query_tokens:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            if term in core_query_tokens:
                idf_val *= CORE_IDF_BOOST

            t_scores = np.zeros(self.n_docs, dtype=np.float32)
            for i, tf_dict in enumerate(self._doc_tfs):
                tf = tf_dict.get(term, 0.0)
                if tf == 0.0:
                    continue
                tf_norm = (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm[i])
                t_scores[i] = idf_val * tf_norm

            scores          += t_scores
            term_scores[term] = t_scores

        return scores, term_scores

    def tokenize(self, text: str) -> list[str]:
        raw_tokens = re.findall(r"[a-z0-9][a-z0-9+#_]*", text.lower())
        normalized: list[str] = []
        normalize_token = (
            self.normalizer.normalize_token
            if self.normalizer
            else lambda t: t
        )
        for t in raw_tokens:
            normalized.extend(normalize_token(t).split())
        return normalized


# ============================================================================
# SEMANTIC ENGINE  (v4 — always rebuilds FAISS from dense_matrix.npy)
# ============================================================================

class SemanticEngine:
    """
    Wraps Component A's DenseIndex pickle + dense_matrix.npy.

    v4 change: We NEVER reuse cfg._faiss_index from the pickle because
    the pickled FAISS index was built on the training machine and its
    internal dimension may not match what we reconstruct here.  Instead
    we always:
        1. Load the raw float32 matrix from dense_matrix.npy.
        2. L2-normalise the rows.
        3. Build a fresh IndexFlatIP over that matrix.
    This guarantees dim consistency between the stored vectors and any
    live query vector we encode.
    """

    def __init__(
        self,
        dense_pkl_path: str | Path,
        dense_matrix_path: str | Path,
    ) -> None:
        # ── Step 1: load metadata from the pickle (model name, prefix, etc.) ──
        with open(dense_pkl_path, "rb") as f:
            cfg = FlexUnpickler(f).load()

        self.bge_prefix: str = getattr(cfg, "BGE_QUERY_PREFIX", BGE_QUERY_PREFIX)
        strategy             = getattr(cfg, "strategy", "sbert")
        self.model_name: str = (
            getattr(cfg, "SBERT_MODEL_NAME", SBERT_MODEL)
            if strategy == "sbert" else SBERT_MODEL
        )

        # ── Step 2: load the matrix from .npy (always — never from pickle) ───
        matrix_path = Path(dense_matrix_path)
        if not matrix_path.exists():
            # Fallback: try to get it from the pickle if .npy is missing
            if hasattr(cfg, "_matrix") and cfg._matrix is not None:
                matrix = np.array(cfg._matrix, dtype=np.float32)
            else:
                raise FileNotFoundError(
                    f"Embedding matrix not found at {dense_matrix_path} "
                    f"and dense.pkl has no _matrix attribute."
                )
        else:
            matrix = np.load(matrix_path).astype(np.float32)

        self.embedding_dim: int = matrix.shape[1]
        self.n_docs: int        = matrix.shape[0]

        # L2-normalise so inner-product == cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        self._matrix = (matrix / norms).astype(np.float32)

        # ── Step 3: build a FRESH FAISS index ────────────────────────────────
        # This is the critical fix: we never reuse cfg._faiss_index.
        if FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(self.embedding_dim)
            self._index.add(self._matrix)
        else:
            self._index = None

        self._encoder = None

    # ── Encoder (lazy) ───────────────────────────────────────────────────────

    def _get_encoder(self) -> Any:
        if self._encoder is None and SBERT_AVAILABLE:
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    # ── Encoding ─────────────────────────────────────────────────────────────

    def encode_queries_batch(self, queries: list[str]) -> np.ndarray:
        """Batch-encode all queries with BGE prefix. Returns (n_queries, dim)."""
        enc      = self._get_encoder()
        prefixed = [self.bge_prefix + q for q in queries]

        if enc is None:
            # Random fallback when SBERT is unavailable
            rng  = np.random.default_rng(42)
            vecs = rng.standard_normal((len(queries), self.embedding_dim)).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / np.where(norms == 0, 1, norms)

        vecs = enc.encode(
            prefixed,
            batch_size=min(len(prefixed), 32),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vecs, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Convenience wrapper for a single query. Returns shape (dim,)."""
        return self.encode_queries_batch([query])[0]

    # ── Scoring ──────────────────────────────────────────────────────────────

    def score_queries_batch(self, query_vecs: np.ndarray) -> np.ndarray:
        """
        Score all queries against all docs.
        query_vecs: (n_queries, dim)
        Returns: (n_queries, n_docs)
        """
        if self._index is not None:
            scores, _ = self._index.search(query_vecs, self.n_docs)
            return scores
        # Brute-force fallback
        return query_vecs @ self._matrix.T

    def score_query(self, query_vec: np.ndarray) -> np.ndarray:
        """
        Score a single query vector against all docs.
        query_vec: shape (dim,)  OR  (1, dim)
        Returns: shape (n_docs,)

        This method exists so that any existing callers using the old
        single-query API continue to work without changes.
        """
        vec = query_vec.reshape(1, -1)
        result = self.score_queries_batch(vec)   # (1, n_docs)
        return result[0]                          # (n_docs,)


# ============================================================================
# RRF FUSION
# ============================================================================

def reciprocal_rank_fusion(
    rank_lists: list[np.ndarray],
    weights: list[float] | None = None,
    k: int = RRF_K,
) -> np.ndarray:
    if weights is None:
        weights = [1.0] * len(rank_lists)

    n_docs = rank_lists[0].shape[0]
    fused  = np.zeros(n_docs, dtype=np.float64)

    for score_arr, w in zip(rank_lists, weights):
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
    cand_years = float(candidate.get("years_of_experience") or 0)

    if experience_years:
        try:
            req = float(experience_years)
            if cand_years < req - 2.5:
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
    core_names      = candidate.get("core_skill_names", set())
    secondary_names = candidate.get("secondary_skill_names", set())

    core_hits = [s for s in matched_skills if s in core_names]
    sec_hits  = [s for s in matched_skills if s in secondary_names and s not in core_hits]

    signals = []
    if core_hits:
        signals.append(f"Core skill match: {', '.join(core_hits)}")
    if sec_hits:
        signals.append(f"Secondary skill match: {', '.join(sec_hits)}")
    if bm25_score > 0:
        signals.append(f"Keyword relevance (BM25={bm25_score:.3f})")
    if dense_score > 0.5:
        signals.append(f"Semantic similarity (cos={dense_score:.3f})")

    exp_band = parsed_intent.get("experience_band")
    if exp_band:
        signals.append(
            f"Experience band '{exp_band}' — candidate has "
            f"{candidate.get('years_of_experience', 0)} yrs"
        )

    matched_keywords = sorted(
        [{"term": t, "contribution": round(float(v), 4)}
         for t, v in term_contributions.items() if v > 0],
        key=lambda x: x["contribution"],
        reverse=True,
    )

    return {
        "rank":              rank,
        "rrf_score":         round(float(rrf_score), 6),
        "bm25_score":        round(float(bm25_score), 4),
        "semantic_score":    round(float(dense_score), 4),
        "core_skill_hits":   core_hits,
        "secondary_hits":    sec_hits,
        "matched_keywords":  matched_keywords,
        "retrieval_signals": signals,
        "why_retrieved":     " | ".join(signals) if signals else "General profile relevance",
    }


# ============================================================================
# MAIN HYBRID RETRIEVER
# ============================================================================

class HybridRetriever:
    """
    Loads all Component A index artifacts once, answers queries in milliseconds.
    """

    def __init__(self, index_dir: str | Path, top_k: int = DEFAULT_TOP_K) -> None:
        index_dir  = Path(index_dir)
        self.top_k = top_k

        # Step 1: Dense — master ordering
        with open(index_dir / "dense.pkl", "rb") as f:
            dense_obj = FlexUnpickler(f).load()
        dense_doc_ids: list[str] = getattr(dense_obj, "_doc_ids", [])
        dense_texts: list[str]   = getattr(dense_obj, "_texts", [])

        # Step 2: MetadataStore
        with open(index_dir / "metadata.pkl", "rb") as f:
            meta_store = FlexUnpickler(f).load()
        self.metadata: list[dict] = _build_metadata_list(
            meta_store, dense_doc_ids, dense_texts
        )
        self.n_docs = len(self.metadata)

        # Step 3: BM25 aligned to Dense order
        self.bm25 = BM25Engine(
            index_dir / "bm25.pkl",
            self.metadata,
            dense_doc_ids,
        )

        # Step 4: Semantic engine (v4 — always builds fresh FAISS from .npy)
        self.semantic = SemanticEngine(
            index_dir / "dense.pkl",
            index_dir / "dense_matrix.npy",
        )

        # Step 5: Skill inverted index (optional)
        skills_path = index_dir / "skills.pkl"
        self._skills_obj = None
        if skills_path.exists():
            with open(skills_path, "rb") as f:
                self._skills_obj = FlexUnpickler(f).load()

        print(f"[HybridRetriever] Loaded {self.n_docs} candidates from {index_dir}")

    def _extract_skill_tokens(self, skill_list: list[str]) -> set[str]:
        tokens: set[str] = set()
        for s in skill_list:
            tokens.update(self.bm25.tokenize(s))
        return tokens

    def retrieve(self, intent_json: dict) -> dict:
        t0 = time.perf_counter()

        intent_block = intent_json.get("intent") or {}
        parsed       = intent_json.get("entities") or intent_json.get("parsed") or {}
        queries      = intent_json.get("queries", [])
        strategy_map = intent_json.get("strategy_map", {})

        skill_tokens_raw: list[str] = [s.lower() for s in (parsed.get("skills") or [])]
        exp_band: str | None        = parsed.get("experience_band")
        exp_years: str | None       = parsed.get("experience_years")
        negated: list[str]          = [s.lower() for s in (parsed.get("negated_skills") or [])]

        canonical_core_skills: set[str] = self._extract_skill_tokens(skill_tokens_raw)

        # ── 1. BM25 aggregate ─────────────────────────────────────────────
        bm25_agg = np.zeros(self.n_docs, dtype=np.float32)
        term_contrib_agg: dict[str, np.ndarray] = {}
        bm25_per_query: list[tuple[str, str, np.ndarray]] = []

        for q in queries:
            tokens   = self.bm25.tokenize(q)
            scores, term_scores = self.bm25.score_query(tokens, canonical_core_skills)
            strategy = strategy_map.get(q, "unknown")
            bm25_per_query.append((q, strategy, scores))

            w = (1.0  if strategy in ("original", "synonym") else
                 1.2  if strategy == "related_tech" else
                 0.6)

            bm25_agg += w * scores
            for term, ts in term_scores.items():
                if term not in term_contrib_agg:
                    term_contrib_agg[term] = np.zeros(self.n_docs, dtype=np.float32)
                term_contrib_agg[term] += w * ts

        mx = bm25_agg.max()
        if mx > 0:
            bm25_agg /= mx
            for t in term_contrib_agg:
                term_contrib_agg[t] /= mx

        # ── 2. Semantic aggregate (batch) ─────────────────────────────────
        if queries:
            # encode_queries_batch returns (n_queries, dim) — all L2-normed
            query_vecs   = self.semantic.encode_queries_batch(queries)   # (n_q, dim)
            # score_queries_batch returns (n_queries, n_docs)
            dense_matrix = self.semantic.score_queries_batch(query_vecs) # (n_q, n_docs)

            q_weights = np.array([
                1.0 if strategy_map.get(q, "unknown") in ("original", "synonym") else
                1.2 if strategy_map.get(q, "unknown") == "related_tech" else 0.6
                for q in queries
            ], dtype=np.float32)

            # Weighted average across queries → (n_docs,)
            dense_scores = np.average(dense_matrix, axis=0, weights=q_weights)
        else:
            dense_scores = np.zeros(self.n_docs, dtype=np.float32)

        dense_scores = np.clip(dense_scores, 0, 1)

        # ── 3. Hard filters ───────────────────────────────────────────────
        valid_mask = np.ones(self.n_docs, dtype=bool)
        for idx, cand in enumerate(self.metadata):
            all_skill_names = (
                cand.get("core_skill_names", set()) |
                cand.get("secondary_skill_names", set())
            )
            if any(
                neg in all_skill_names or neg in cand.get("skill_summary", "").lower()
                for neg in negated
            ):
                valid_mask[idx] = False
                continue
            if not passes_experience_filter(cand, exp_band, exp_years):
                valid_mask[idx] = False

        # ── 4. RRF fusion ─────────────────────────────────────────────────
        rrf_scores = reciprocal_rank_fusion(
            [bm25_agg, dense_scores],
            weights=[ALPHA_BM25, ALPHA_DENSE],
        )
        rrf_scores[~valid_mask] = 0.0

        # ── 5. Core-skill RRF boost ───────────────────────────────────────
        for idx, cand in enumerate(self.metadata):
            core_names = cand.get("core_skill_names", set())
            hit_count  = sum(1 for s in skill_tokens_raw if s in core_names)
            if hit_count > 0:
                rrf_scores[idx] *= (1.0 + 0.3 * hit_count)

        # ── 6. Quality gates ──────────────────────────────────────────────
        bm25_floor = 0.15

        lexical_scores = bm25_agg.copy()
        lexical_scores[~valid_mask] = 0.0

        max_sem       = dense_scores.max() if self.n_docs > 0 else 0.0
        sem_threshold = max_sem * 0.75 if max_sem > 0 else 0.60

        semantic_scores = dense_scores.copy()
        semantic_scores[~valid_mask] = 0.0
        semantic_scores[semantic_scores < sem_threshold] = 0.0

        hybrid_quality_mask = (
            (bm25_agg >= bm25_floor) |
            ((dense_scores >= sem_threshold) & (bm25_agg > 0.0))
        )
        rrf_scores[~hybrid_quality_mask] = 0.0

        # ── 7. Build result records ───────────────────────────────────────
        SCORE_FLOOR_RATIO = 0.35

        def _build_records(score_arr: np.ndarray, mode: str) -> list[dict]:
            top_score   = score_arr.max()
            score_floor = top_score * SCORE_FLOOR_RATIO if top_score > 0 else 0.0
            records     = []

            for rank, idx in enumerate(np.argsort(score_arr)[::-1], start=1):
                if score_arr[idx] < score_floor or rank > self.top_k:
                    break
                cand = self.metadata[idx]

                all_names = (
                    cand.get("core_skill_names", set()) |
                    cand.get("secondary_skill_names", set())
                )
                matched = [s for s in skill_tokens_raw if s in all_names]

                doc_term_contrib = {
                    t: float(arr[idx])
                    for t, arr in term_contrib_agg.items()
                    if arr[idx] > 0
                }

                expl = build_explanation(
                    candidate          = cand,
                    matched_skills     = matched,
                    bm25_score         = float(bm25_agg[idx]),
                    dense_score        = float(dense_scores[idx]),
                    rrf_score          = float(score_arr[idx]),
                    rank               = rank,
                    parsed_intent      = parsed,
                    term_contributions = doc_term_contrib,
                )

                primary_score = (
                    expl["rrf_score"]      if mode == "hybrid"  else
                    expl["bm25_score"]     if mode == "lexical" else
                    expl["semantic_score"]
                )

                records.append({
                    "rank":                rank,
                    "candidate_id":        cand.get("id"),
                    "name":                cand.get("name", ""),
                    "years_of_experience": cand.get("years_of_experience", 0),
                    "potential_roles":     cand.get("potential_roles", ""),
                    "core_skills":         cand.get("core_skills_str", ""),
                    "secondary_skills":    cand.get("secondary_skills_str", ""),
                    "soft_skills":         cand.get("soft_skills", ""),
                    "skill_summary":       cand.get("skill_summary", ""),
                    "scores": {
                        "primary":  primary_score,
                        "rrf":      expl["rrf_score"],
                        "bm25":     expl["bm25_score"],
                        "semantic": expl["semantic_score"],
                    },
                    "explanation": expl,
                })
            return records

        hybrid_results = _build_records(rrf_scores, "hybrid")

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        # ── 8. Per-query BM25 breakdown ───────────────────────────────────
        query_breakdown = []
        for q, strategy, scores in bm25_per_query:
            top5 = np.argsort(scores)[::-1][:5]
            query_breakdown.append({
                "query":    q,
                "strategy": strategy,
                "top5_bm25_hits": [
                    {
                        "candidate_id": self.metadata[i].get("id"),
                        "name":         self.metadata[i].get("name", ""),
                        "bm25":         round(float(scores[i]), 4),
                    }
                    for i in top5 if scores[i] > 0
                ],
            })

        meta_out = {
            "original_query":    intent_json.get("original"),
            "corrected_query":   intent_json.get("corrected_query") or intent_json.get("corrected"),
            "primary_intent":    intent_block.get("primary") or intent_block.get("primary_intent"),
            "detected_skills":   skill_tokens_raw,
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
            "hybrid": {
                "meta":            {**meta_out, "mode": "hybrid",   "returned": len(hybrid_results)},
                "results":         hybrid_results,
                "query_breakdown": query_breakdown,
            }
        }