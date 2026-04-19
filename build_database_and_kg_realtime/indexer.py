"""
Component A — Intent-Aware Hybrid Retrieval System
Indexing & Embedding Design

Author: Team IITMandiHack60
Goal  : Build dual indexes (BM25 + Sentence-BERT/FAISS dense vectors) over profiles.csv
        with a fully SCHEMA-AGNOSTIC pipeline — no column names are hardcoded.
        The system introspects the DataFrame at runtime, discovers text/numeric/
        categorical fields, and constructs the indexes automatically.

Key design decisions:
  1. Schema-agnostic field discovery   — new columns just work
  2. Skill proficiency-aware parsing   — (Python, Expert) → weight 3x vs (Python, Beginner)
  3. Dual index                        — BM25 (lexical) + Sentence-BERT/FAISS (semantic)
  4. Canonical normalization           — "ML" and "machine learning" resolve to same entity
  5. Persistent index                  — saved to disk so downstream components load instantly
  6. Rich metadata store               — every document stores its raw + parsed form
  7. Skill co-occurrence graph         — foundation for Knowledge Graph (Component E)
  8. Explainability hooks              — matched terms, skills, score contribution for Component F
"""

import re
import math
import json
import pickle
import hashlib
import logging
import collections
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from normalizer import CanonicalNormalizer, get_normalizer

# ── Optional imports: SBERT + FAISS (graceful fallback to LSA) ────────────
_HAS_SBERT = False
_HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except ImportError:
    SentenceTransformer = None

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    faiss = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("ComponentA")

# ─────────────────────────────────────────────────────────────────────────────
# 1. PROFICIENCY SCALE
#    Normalises the noisy "(Expert)", "(Beginner)" strings to numeric weights.
#    Unknown values fall back to 1.0 so nothing breaks when the schema evolves.
# ─────────────────────────────────────────────────────────────────────────────
PROFICIENCY_SCALE: dict[str, float] = {
    "novice":            0.3,
    "beginner":          0.5,
    "basic":             0.5,
    "advanced beginner": 0.7,
    "intermediate":      1.0,
    "competent":         1.2,
    "proficient":        1.5,
    "advanced":          1.7,
    "expert":            2.0,
    "certified":         2.0,
}

_SKILL_PATTERN = re.compile(
    r"([^,(]+?)\s*\(\s*([^)]+?)\s*\)",  # "Skill Name (Proficiency)"
    re.IGNORECASE,
)


def _clean_unicode(text: str) -> str:
    """Fix common encoding artifacts that pollute SBERT embeddings.
    Replaces smart quotes, em-dashes, and other non-ASCII noise with ASCII equivalents."""
    replacements = {
        '\u2019': "'",   # right single quote  → '
        '\u2018': "'",   # left single quote   → '
        '\u201c': '"',   # left double quote   → "
        '\u201d': '"',   # right double quote  → "
        '\u2013': '-',   # en-dash             → -
        '\u2014': '-',   # em-dash             → -
        '\u2026': '...', # ellipsis            → ...
        '\u00e2\u0080\u0099': "'",  # mojibake apostrophe
        '\u00e2\u0080\u0093': '-',  # mojibake en-dash
        '\u00e2\u0080\u009c': '"',  # mojibake left quote
        '\u00e2\u0080\u009d': '"',  # mojibake right quote
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def parse_skill_field(raw: str, normalizer: CanonicalNormalizer | None = None) -> list[tuple[str, float]]:
    """
    Parse a comma-separated skill string like:
        "Python (Expert), SQL (Proficient), Docker (Beginner)"
    into:
        [("python", 2.0), ("sql", 1.5), ("docker", 0.5)]

    Falls back to plain token split if the pattern doesn't match.
    If a normalizer is provided, skill names are canonicalized.
    """
    if not isinstance(raw, str) or not raw.strip():
        return []

    norm = normalizer or get_normalizer()

    results: list[tuple[str, float]] = []
    matched_spans: list[tuple[int, int]] = []

    for m in _SKILL_PATTERN.finditer(raw):
        skill = m.group(1).strip().lower()
        prof_raw = m.group(2).strip().lower()
        weight = PROFICIENCY_SCALE.get(prof_raw, 1.0)
        # ── Canonical normalization ──
        skill = norm.normalize_skill(skill)
        if skill:
            results.append((skill, weight))
        matched_spans.append((m.start(), m.end()))

    # Any text not captured by the pattern is still a skill (weight = 1.0)
    if not results:
        for token in re.split(r"[,;|]+", raw):
            token = token.strip().lower()
            if token:
                token = norm.normalize_skill(token)
                if token:
                    results.append((token, 1.0))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCHEMA-AGNOSTIC FIELD DISCOVERER
#    At runtime, inspect the DataFrame and classify every column.
# ─────────────────────────────────────────────────────────────────────────────
class FieldSchema:
    """Holds the discovered schema of a DataFrame at index-build time."""

    SKILL_KEYWORDS = {"skill", "competenc", "technolog", "tool", "knowledge"}
    SUMMARY_KEYWORDS = {"summary", "description", "about", "bio", "profile"}
    ROLE_KEYWORDS = {"role", "title", "position", "job", "career"}
    NUMERIC_KEYWORDS = {"year", "experience", "salary", "age", "duration"}

    def __init__(self, df: pd.DataFrame):
        self.skill_cols: list[str] = []
        self.summary_cols: list[str] = []
        self.role_cols: list[str] = []
        self.numeric_cols: list[str] = []
        self.text_cols: list[str] = []
        self.id_col: str | None = None
        self.name_col: str | None = None
        self._discover(df)

    def _col_matches(self, col: str, keywords: set[str]) -> bool:
        col_lower = col.lower()
        return any(kw in col_lower for kw in keywords)

    def _discover(self, df: pd.DataFrame):
        for col in df.columns:
            col_l = col.lower()

            if col_l in ("id",):
                self.id_col = col
                continue
            if col_l in ("name",):
                self.name_col = col
                continue

            if pd.api.types.is_numeric_dtype(df[col]) or self._col_matches(col, self.NUMERIC_KEYWORDS):
                self.numeric_cols.append(col)
                continue

            if self._col_matches(col, self.SUMMARY_KEYWORDS):
                self.summary_cols.append(col)
            elif self._col_matches(col, self.SKILL_KEYWORDS):
                self.skill_cols.append(col)
            elif self._col_matches(col, self.ROLE_KEYWORDS):
                self.role_cols.append(col)
            else:
                self.text_cols.append(col)

        log.info("Schema discovered — skills:%s  summaries:%s  roles:%s  numeric:%s  text:%s",
                 self.skill_cols, self.summary_cols, self.role_cols,
                 self.numeric_cols, self.text_cols)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 3. DOCUMENT BUILDER
#    Converts a single DataFrame row → rich Document object.
#    Now applies canonical normalization to all text fields.
# ─────────────────────────────────────────────────────────────────────────────
class Document:
    """Schema-aware representation of one profile."""

    __slots__ = ("doc_id", "raw", "skills", "bm25_text", "semantic_text",
                 "numeric_features", "metadata")

    def __init__(self, doc_id: str, raw: dict, schema: FieldSchema,
                 normalizer: CanonicalNormalizer | None = None):
        self.doc_id = doc_id
        self.raw = raw
        norm = normalizer or get_normalizer()

        # ── Skill parsing (with canonical normalization) ───────────────────
        self.skills: dict[str, float] = {}
        skill_tokens_weighted: list[str] = []
        for col in schema.skill_cols:
            for skill, weight in parse_skill_field(raw.get(col, ""), normalizer=norm):
                # Merge: keep max weight if skill appears in multiple columns
                self.skills[skill] = max(self.skills.get(skill, 0.0), weight)
                # Repeat the token proportional to weight for BM25 boosting
                repetitions = max(1, round(weight * 2))
                skill_tokens_weighted.extend([skill.replace(" ", "_")] * repetitions)

        # ── BM25 text: skill-weighted tokens + roles + free text ──────────
        role_tokens = " ".join(
            norm.normalize_text(str(raw.get(col, "")))
            for col in schema.role_cols
        )
        soft_text = " ".join(
            norm.normalize_text(str(raw.get(col, "")))
            for col in schema.text_cols
        )
        self.bm25_text = " ".join(skill_tokens_weighted) + " " + role_tokens + " " + soft_text

        # ── Semantic text: rich combined input for SBERT ───────────────────
        # Combine ALL relevant columns into a structured natural-language text
        # so SBERT gets the complete picture: summary + skills + roles + experience.
        parts = []

        # 1. Natural language summary (primary context)
        summary_text = " ".join(
            _clean_unicode(str(raw.get(col, "")))
            for col in schema.summary_cols
            if isinstance(raw.get(col), str)
        ).strip()
        if summary_text:
            parts.append(summary_text)

        # 2. Core + secondary skills (specific technical details)
        skill_parts = []
        for col in schema.skill_cols:
            val = str(raw.get(col, ""))
            if val and val.lower() != "nan":
                skill_parts.append(_clean_unicode(val))
        if skill_parts:
            parts.append("Skills: " + ", ".join(skill_parts) + ".")

        # 3. Potential roles
        role_parts = []
        for col in schema.role_cols:
            val = str(raw.get(col, ""))
            if val and val.lower() != "nan":
                role_parts.append(_clean_unicode(val))
        if role_parts:
            parts.append("Suitable for roles: " + ", ".join(role_parts) + ".")

        # 4. Years of experience
        yoe = 0
        for col in schema.numeric_cols:
            if "experience" in col.lower() or "year" in col.lower():
                try:
                    yoe = float(raw.get(col, 0) or 0)
                except (ValueError, TypeError):
                    pass
        if yoe > 0:
            parts.append(f"{int(yoe)} years of experience.")

        self.semantic_text = " ".join(parts) if parts else self.bm25_text

        # ── Numeric features ──────────────────────────────────────────────
        self.numeric_features: dict[str, float] = {}
        for col in schema.numeric_cols:
            try:
                self.numeric_features[col] = float(raw.get(col, 0) or 0)
            except (ValueError, TypeError):
                self.numeric_features[col] = 0.0

        # ── Metadata (preserved for explainability layer) ─────────────────
        self.metadata = {
            "id": doc_id,
            "name": raw.get(schema.name_col, ""),
            "top_skills": sorted(self.skills.items(), key=lambda x: -x[1])[:10],
            "years_of_experience": self.numeric_features.get("years_of_experience", 0),
            "potential_roles": raw.get(
                next((c for c in schema.role_cols if "potential" in c.lower()), ""), ""
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. BM25 INDEX (from scratch — no external library)
#    Pure-Python BM25+ implementation.
#    Parameters k1=1.5, b=0.75 per Robertson et al.
#    Now applies canonical normalization to query tokens.
# ─────────────────────────────────────────────────────────────────────────────
class BM25Index:
    """
    BM25+ lexical index built over tokenised document strings.
    Supports incremental updates via add_document().
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 0.5):
        self.k1 = k1
        self.b = b
        self.delta = delta  # BM25+ lower-bound
        self._doc_ids: list[str] = []
        self._doc_freqs: list[dict[str, int]] = []   # term → count per doc
        self._doc_lengths: list[int] = []
        self._df: dict[str, int] = collections.defaultdict(int)  # term → #docs
        self._avgdl: float = 0.0
        self._built = False
        self._normalizer: CanonicalNormalizer = get_normalizer()

    @staticmethod
    def tokenise(text: str, normalizer: CanonicalNormalizer | None = None) -> list[str]:
        """Tokenise text into normalised terms."""
        norm = normalizer or get_normalizer()
        # Extract alphanumeric words and preserve specific tech symbols (# and +)
        # Allows tokens starting with numbers (e.g. 3d, 2fa, b2b)
        raw_tokens = re.findall(r"[a-z0-9][a-z0-9+#_]*", text.lower())
        normalized = []
        for t in raw_tokens:
            expanded = norm.normalize_token(t)
            # If abbreviation expanded to multi-word, split and add each
            normalized.extend(expanded.split())
        return normalized

    def add_document(self, doc_id: str, text: str):
        tokens = self.tokenise(text, self._normalizer)
        freq: dict[str, int] = collections.Counter(tokens)
        self._doc_ids.append(doc_id)
        self._doc_freqs.append(freq)
        self._doc_lengths.append(len(tokens))
        for term in freq:
            self._df[term] += 1
        self._built = False  # invalidate cached avgdl

    def remove_document(self, doc_id: str):
        """Remove a document from the BM25 index (for incremental updates)."""
        if doc_id not in self._doc_ids:
            return
        idx = self._doc_ids.index(doc_id)
        freq = self._doc_freqs[idx]
        # Decrement document frequencies
        for term, count in freq.items():
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]
        self._doc_ids.pop(idx)
        self._doc_freqs.pop(idx)
        self._doc_lengths.pop(idx)
        self._built = False

    def build(self):
        n = len(self._doc_lengths)
        self._avgdl = sum(self._doc_lengths) / n if n else 1.0
        self._built = True
        log.info("BM25 index built over %d documents | avgdl=%.1f | vocab=%d",
                 n, self._avgdl, len(self._df))

    def score(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return top_k (doc_id, score) pairs."""
        if not self._built:
            self.build()

        terms = self.tokenise(query, self._normalizer)
        n = len(self._doc_ids)
        if n == 0:
            return []
        scores: np.ndarray = np.zeros(n, dtype=np.float32)

        for term in terms:
            if term not in self._df:
                continue
            df_t = self._df[term]
            idf = math.log((n - df_t + 0.5) / (df_t + 0.5) + 1)

            for i, (freq, dl) in enumerate(
                zip(self._doc_freqs, self._doc_lengths)
            ):
                tf = freq.get(term, 0)
                if tf == 0:
                    continue
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                ) + self.delta
                scores[i] += idf * tf_norm

        top_k_actual = min(top_k, n)
        top_idx = np.argpartition(scores, -top_k_actual)[-top_k_actual:]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self._doc_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def get_matching_terms(self, query: str, doc_id: str) -> list[str]:
        """Return which query terms matched a specific doc — for explainability."""
        if doc_id not in self._doc_ids:
            return []
        idx = self._doc_ids.index(doc_id)
        freq = self._doc_freqs[idx]
        return [t for t in self.tokenise(query, self._normalizer) if t in freq]

    def __len__(self):
        return len(self._doc_ids)


# ─────────────────────────────────────────────────────────────────────────────
# 5. DENSE VECTOR INDEX
#    Supports two strategies:
#      - "sbert" : Sentence-BERT embeddings + FAISS index (primary)
#      - "lsa"   : TF-IDF + TruncatedSVD (fallback when SBERT unavailable)
#    Auto-detects available libraries and picks the best strategy.
# ─────────────────────────────────────────────────────────────────────────────
class DenseIndex:
    """
    Dense vector index supporting Sentence-BERT (FAISS) or LSA (TF-IDF+SVD).

    Strategy selection:
      - "sbert" (default): Uses Sentence-BERT for encoding + FAISS for ANN search.
        Produces true semantic embeddings that understand meaning beyond word overlap.
      - "lsa": TF-IDF → TruncatedSVD. Captures term co-occurrence patterns.
        Faster to build, no GPU needed, but weaker semantic understanding.
      - "auto": Try SBERT first, fall back to LSA if libraries unavailable.
    """

    SBERT_MODEL_NAME = "BAAI/bge-base-en-v1.5"  # 768-dim, better semantic quality
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: " 
    def __init__(self, strategy: str = "auto", n_components: int = 256):
        # Resolve strategy
        if strategy == "auto":
            self.strategy = "sbert" if _HAS_SBERT else "lsa"
        else:
            self.strategy = strategy

        if self.strategy == "sbert" and not _HAS_SBERT:
            log.warning("SBERT requested but sentence-transformers not installed. Falling back to LSA.")
            self.strategy = "lsa"

        self.n_components = n_components
        self._doc_ids: list[str] = []
        self._texts: list[str] = []
        self._matrix: np.ndarray | None = None  # shape (N, dim)

        # LSA-specific
        self._vectoriser: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None

        # SBERT-specific
        self._sbert_model: Any = None

        # FAISS-specific
        self._faiss_index: Any = None

        log.info("DenseIndex strategy: %s", self.strategy)

    def fit(self, doc_ids: list[str], texts: list[str]):
        self._doc_ids = doc_ids
        self._texts = texts

        if self.strategy == "sbert":
            self._fit_sbert(texts)
        else:
            self._fit_lsa(texts)

        # Build FAISS index if available
        if _HAS_FAISS and self._matrix is not None:
            self._build_faiss_index()

    # NEW
    def _fit_sbert(self, texts: list[str]):
        log.info("Loading BGE model '%s'…", self.SBERT_MODEL_NAME)
        self._sbert_model = SentenceTransformer(self.SBERT_MODEL_NAME)

        # BGE profiles are encoded WITHOUT any prefix at index time.
        # The prefix is only added to queries at search time.
        # This asymmetry is intentional — it is how BGE was trained.
        log.info("Encoding %d documents with BGE (no prefix)…", len(texts))
        embeddings = self._sbert_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,          # BGE-base is larger — reduce batch size
            normalize_embeddings=True,
        )
        self._matrix = np.array(embeddings, dtype=np.float32)
        log.info("BGE encoding complete | dims=%d", self._matrix.shape[1])

    def _fit_lsa(self, texts: list[str]):
        """Fit TF-IDF + TruncatedSVD (LSA) pipeline."""
        log.info("Fitting TF-IDF vectoriser over %d documents…", len(texts))
        self._vectoriser = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        tfidf_matrix = self._vectoriser.fit_transform(texts)

        actual_components = min(self.n_components, tfidf_matrix.shape[1] - 1)
        log.info("Running TruncatedSVD with %d components…", actual_components)
        self._svd = TruncatedSVD(n_components=actual_components, random_state=42)
        reduced = self._svd.fit_transform(tfidf_matrix)
        self._matrix = normalize(reduced, norm="l2").astype(np.float32)

        explained = self._svd.explained_variance_ratio_.sum()
        log.info(
            "LSA index ready | vocab=%d | dims=%d | variance_explained=%.1f%%",
            len(self._vectoriser.vocabulary_),
            actual_components,
            explained * 100,
        )

    def _build_faiss_index(self):
        """Build a FAISS IndexFlatIP (inner product = cosine on L2-normed vectors)."""
        dim = self._matrix.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(self._matrix)
        log.info("FAISS index built | %d vectors | %d dims", self._faiss_index.ntotal, dim)

    # NEW
# BGE query prefix — required for retrieval tasks
# DO NOT add this prefix to profile text at index time
    

    def encode(self, text: str) -> np.ndarray:
        if self.strategy == "sbert":
            # BGE requires a task-specific prefix on queries ONLY.
            # Profiles were encoded without prefix in _fit_sbert().
            # This asymmetry is how BGE was trained — do not remove prefix.
            prefixed_text = self.BGE_QUERY_PREFIX + text
            vec = self._sbert_model.encode(
                [prefixed_text], normalize_embeddings=True
            )
            return np.array(vec[0], dtype=np.float32)
        else:
            # LSA fallback — no prefix needed
            vec = self._vectoriser.transform([text])
            reduced = self._svd.transform(vec)
            return normalize(reduced, norm="l2")[0].astype(np.float32)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return top_k (doc_id, cosine_similarity) pairs."""
        q_vec = self.encode(query)

        if self._faiss_index is not None:
            # FAISS search (fast ANN)
            scores, indices = self._faiss_index.search(
                q_vec.reshape(1, -1), min(top_k, len(self._doc_ids))
            )
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score > 0:
                    results.append((self._doc_ids[idx], float(score)))
            return results
        else:
            # Brute-force fallback (no FAISS)
            sims = self._matrix @ q_vec
            top_k_actual = min(top_k, len(self._doc_ids))
            top_idx = np.argpartition(sims, -top_k_actual)[-top_k_actual:]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            return [(self._doc_ids[i], float(sims[i])) for i in top_idx if sims[i] > 0]

    def add_documents(self, doc_ids: list[str], texts: list[str]):
        if not texts:
            return

        if self.strategy == "sbert":
            if self._sbert_model is None:
                log.info("Loading BGE model for incremental encoding...")
                self._sbert_model = SentenceTransformer(self.SBERT_MODEL_NAME)

            # Encode new profiles WITHOUT prefix — same as _fit_sbert()
            # Never add BGE_QUERY_PREFIX here — profiles must match
            # the embedding space of existing indexed profiles
            new_embeddings = self._sbert_model.encode(
                texts,
                show_progress_bar=len(texts) > 50,
                batch_size=32,          # reduced for BGE-base memory
                normalize_embeddings=True,
            )
            new_matrix = np.array(new_embeddings, dtype=np.float32)
        else:
            # LSA: transform using existing fitted vectoriser + SVD
            if self._vectoriser is None or self._svd is None:
                log.warning("LSA vectoriser not fitted — cannot add incrementally.")
                return
            tfidf_new = self._vectoriser.transform(texts)
            reduced_new = self._svd.transform(tfidf_new)
            new_matrix = normalize(reduced_new, norm="l2").astype(np.float32)

        # Append to internal tracking
        self._doc_ids.extend(doc_ids)
        self._texts.extend(texts)
        self._matrix = np.vstack([self._matrix, new_matrix])

        # Append to FAISS
        if self._faiss_index is not None:
            self._faiss_index.add(new_matrix)

        log.info("Dense index: added %d documents (total: %d)", len(doc_ids), len(self._doc_ids))

    def remove_documents(self, doc_ids_to_remove: set[str]):
        """
        Remove documents from the dense index by doc_id.
        Rebuilds the FAISS index after removal (FAISS IndexFlatIP doesn't support deletion).
        """
        if not doc_ids_to_remove:
            return
        keep_mask = [did not in doc_ids_to_remove for did in self._doc_ids]
        self._doc_ids = [d for d, k in zip(self._doc_ids, keep_mask) if k]
        self._texts = [t for t, k in zip(self._texts, keep_mask) if k]
        self._matrix = self._matrix[keep_mask]

        # Rebuild FAISS (required because IndexFlatIP doesn't support deletion)
        if _HAS_FAISS and self._matrix is not None and len(self._matrix) > 0:
            self._build_faiss_index()

        log.info("Dense index: removed %d documents (remaining: %d)",
                 len(doc_ids_to_remove), len(self._doc_ids))

    def get_vector(self, doc_id: str) -> np.ndarray | None:
        if doc_id not in self._doc_ids:
            return None
        idx = self._doc_ids.index(doc_id)
        return self._matrix[idx]

    def __len__(self):
        return len(self._doc_ids)


# ─────────────────────────────────────────────────────────────────────────────
# 6. SKILL INVERTED INDEX
#    Maps skill_name → list of (doc_id, weight) pairs.
#    Used by Component E (Knowledge Graph) and Component F (Explainability).
# ─────────────────────────────────────────────────────────────────────────────
class SkillInvertedIndex:
    """Fast lookup: 'python' → all profiles that have Python, sorted by proficiency."""

    def __init__(self):
        self._index: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)

    def add(self, doc_id: str, skills: dict[str, float]):
        for skill, weight in skills.items():
            self._index[skill].append((doc_id, weight))

    def remove_document(self, doc_id: str):
        """Remove all skill entries for a document (for incremental updates)."""
        empty_skills = []
        for skill in self._index:
            self._index[skill] = [
                (did, w) for did, w in self._index[skill] if did != doc_id
            ]
            if not self._index[skill]:
                empty_skills.append(skill)
        for skill in empty_skills:
            del self._index[skill]

    def build(self):
        # Sort each posting list by weight descending
        for skill in self._index:
            self._index[skill].sort(key=lambda x: -x[1])
        log.info("Skill inverted index built | unique_skills=%d", len(self._index))

    def lookup(self, skill: str, top_k: int = 50) -> list[tuple[str, float]]:
        return self._index.get(skill.lower(), [])[:top_k]

    def get_vocabulary(self) -> list[str]:
        return sorted(self._index.keys())

    def get_doc_skills(self, doc_id: str) -> dict[str, float]:
        """Reverse lookup: get all skills for a given doc_id."""
        result = {}
        for skill, postings in self._index.items():
            for did, weight in postings:
                if did == doc_id:
                    result[skill] = weight
                    break
        return result

    def __len__(self):
        return len(self._index)


# ─────────────────────────────────────────────────────────────────────────────
# 7. SKILL CO-OCCURRENCE GRAPH
#    Tracks which skills appear together in profiles.
#    Foundation for Component E (Knowledge Graph).
#    If profile has [python, django, fastapi], records edges:
#      python↔django, python↔fastapi, django↔fastapi
# ─────────────────────────────────────────────────────────────────────────────
class SkillCooccurrenceGraph:
    """
    Undirected weighted graph of skill co-occurrences.
    Edge weight = number of profiles where both skills appear together.
    """

    def __init__(self):
        # Using plain dicts (not defaultdict with lambda) for pickle compatibility
        self._edges: dict[str, dict[str, int]] = {}
        self._skill_freq: dict[str, int] = collections.defaultdict(int)

    def add_profile_skills(self, skills: list[str]):
        """Record co-occurrence edges for all skill pairs in one profile."""
        unique_skills = sorted(set(skills))
        for skill in unique_skills:
            self._skill_freq[skill] += 1
        for i in range(len(unique_skills)):
            for j in range(i + 1, len(unique_skills)):
                s1, s2 = unique_skills[i], unique_skills[j]
                if s1 not in self._edges:
                    self._edges[s1] = {}
                if s2 not in self._edges:
                    self._edges[s2] = {}
                self._edges[s1][s2] = self._edges[s1].get(s2, 0) + 1
                self._edges[s2][s1] = self._edges[s2].get(s1, 0) + 1

    def get_related_skills(self, skill: str, top_k: int = 10) -> list[tuple[str, int]]:
        """
        Return top_k skills most frequently co-occurring with the given skill.
        Returns list of (related_skill, co_occurrence_count).
        """
        skill = skill.lower()
        if skill not in self._edges:
            return []
        neighbors = self._edges[skill]
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: -x[1])
        return sorted_neighbors[:top_k]

    def get_edge_weight(self, skill1: str, skill2: str) -> int:
        """Get co-occurrence count between two skills."""
        return self._edges.get(skill1.lower(), {}).get(skill2.lower(), 0)

    def get_skill_frequency(self, skill: str) -> int:
        """How many profiles contain this skill."""
        return self._skill_freq.get(skill.lower(), 0)

    def get_stats(self) -> dict:
        num_skills = len(self._edges)
        num_edges = sum(len(v) for v in self._edges.values()) // 2  # undirected
        return {"num_skills": num_skills, "num_edges": num_edges}

    def __len__(self):
        return len(self._edges)


# ─────────────────────────────────────────────────────────────────────────────
# 8. METADATA STORE
#    Key-value store: doc_id → Document metadata (JSON-serialisable).
#    Decoupled from indexes so it can be updated without re-indexing.
# ─────────────────────────────────────────────────────────────────────────────
class MetadataStore:
    def __init__(self):
        self._store: dict[str, dict] = {}

    def put(self, doc_id: str, meta: dict):
        self._store[doc_id] = meta

    def get(self, doc_id: str) -> dict | None:
        return self._store.get(doc_id)

    def get_many(self, doc_ids: list[str]) -> list[dict | None]:
        return [self._store.get(d) for d in doc_ids]

    def __len__(self):
        return len(self._store)


# ─────────────────────────────────────────────────────────────────────────────
# 9. SCORE NORMALIZATION UTILITIES
#    Min-max normalization for explainability and cross-signal comparison.
# ─────────────────────────────────────────────────────────────────────────────
def normalize_scores(results: list[dict], score_key: str) -> list[dict]:
    """
    Add a '{score_key}_normalized' field (0–1 range) to each result dict.
    Uses min-max normalization across the result set.
    """
    if not results:
        return results

    scores = [r.get(score_key, 0) for r in results]
    min_s, max_s = min(scores), max(scores)
    spread = max_s - min_s

    norm_key = f"{score_key}_normalized"
    for r in results:
        raw = r.get(score_key, 0)
        r[norm_key] = (raw - min_s) / spread if spread > 0 else 1.0

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 10. MASTER INDEX — orchestrates everything
# ─────────────────────────────────────────────────────────────────────────────
class HybridIndex:
    """
    Top-level index object that holds all sub-indexes and knows how to:
      - Build from a DataFrame (schema-agnostically)
      - Save to / load from disk
      - Answer lexical + semantic queries independently (Components B/D will fuse them)
      - Provide explainability hooks (matched terms, skills, score contribution)
    """

    VERSION = "3.0.0"  # v3: incremental indexing support

    def __init__(self, dense_strategy: str = "auto"):
        self.schema: FieldSchema | None = None
        self.bm25 = BM25Index()
        self.dense = DenseIndex(strategy=dense_strategy)
        self.skills = SkillInvertedIndex()

        self.metadata = MetadataStore()
        self.normalizer = get_normalizer()
        self._fingerprint: str = ""  # SHA256 of source data for cache validation
        self._row_hashes: dict[str, str] = {}  # doc_id → per-row SHA256 for change detection

    # ── Build ─────────────────────────────────────────────────────────────
    def build_from_dataframe(self, df: pd.DataFrame, id_col: str = "id"):
        log.info("=== Component A: Building Hybrid Index ===")
        log.info("Input shape: %s", df.shape)
        log.info("Dense strategy: %s | FAISS available: %s", self.dense.strategy, _HAS_FAISS)

        # Drop entirely empty rows and normalize numeric pandas types
        df = df.dropna(how='all').convert_dtypes()

        # Schema discovery
        self.schema = FieldSchema(df)

        # Data fingerprint (so we know if the CSV changed)
        self._fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:16]
        log.info("Data fingerprint: %s", self._fingerprint)

        # Per-row hashes for incremental change detection
        self._row_hashes = {}

        # Build documents with per-row hashing
        documents: list[Document] = []
        for _, row in df.iterrows():
            if row.isna().all(): continue
            raw = row.to_dict()
            
            # Robust ID normalization (Pandas casts to float64 if there are NaNs)
            raw_id = raw.get(self.schema.id_col or id_col, "")
            doc_id = str(raw_id).strip()
            if doc_id.endswith(".0"): doc_id = doc_id[:-2]
            if not doc_id or doc_id == "nan": continue
            doc = Document(doc_id, raw, self.schema, normalizer=self.normalizer)
            documents.append(doc)
            # Store per-row hash for future incremental updates
            row_hash = hashlib.sha256(
                str(sorted(raw.items())).encode("utf-8")
            ).hexdigest()[:16]
            self._row_hashes[doc_id] = row_hash

        log.info("Parsed %d documents", len(documents))

        # Populate BM25
        log.info("Building BM25 index…")
        for doc in documents:
            self.bm25.add_document(doc.doc_id, doc.bm25_text)
        self.bm25.build()

        # Populate Dense index
        log.info("Building Dense (%s) index…", self.dense.strategy.upper())
        self.dense.fit(
            [d.doc_id for d in documents],
            [d.semantic_text for d in documents],
        )

        # Populate Skill inverted index
        log.info("Building Skill inverted index…")
        for doc in documents:
            self.skills.add(doc.doc_id, doc.skills)
        self.skills.build()



        # Populate Metadata store
        for doc in documents:
            self.metadata.put(doc.doc_id, doc.metadata)

        log.info("=== Index build complete ===")
        self._log_stats()

    def _log_stats(self):
        log.info("──────────────────────────────────────────")
        log.info("  Documents indexed : %d", len(self.bm25))
        log.info("  BM25 vocabulary   : %d terms", len(self.bm25._df))
        log.info("  Dense strategy    : %s", self.dense.strategy)
        log.info("  Dense dimensions  : %d", self.dense._matrix.shape[1] if self.dense._matrix is not None else 0)
        log.info("  FAISS enabled     : %s", self.dense._faiss_index is not None)
        log.info("  Unique skills     : %d", len(self.skills))

        log.info("  Metadata entries  : %d", len(self.metadata))
        log.info("──────────────────────────────────────────")

    # ── Query ─────────────────────────────────────────────────────────────
    def lexical_search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        BM25 lexical search with explainability hooks.
        Returns: doc_id, bm25_score, bm25_score_normalized, matched_terms,
                 matched_skills, skill_overlap_score, + metadata
        """
        results = self.bm25.score(query, top_k=top_k)
        query_skills = self._extract_query_skills(query)
        output = []
        for doc_id, score in results:
            meta = self.metadata.get(doc_id) or {}
            matched_terms = self.bm25.get_matching_terms(query, doc_id)
            doc_skills = self.skills.get_doc_skills(doc_id)
            matched_skills = self._compute_matched_skills(query_skills, doc_skills)
            skill_overlap = len(matched_skills) / max(len(query_skills), 1)
            output.append({
                "doc_id": doc_id,
                "bm25_score": round(score, 4),
                "matched_terms": matched_terms,
                "matched_skills": matched_skills,
                "skill_overlap_score": round(skill_overlap, 3),
                **meta,
            })
        return normalize_scores(output, "bm25_score")

    def semantic_search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Dense vector semantic search with explainability hooks.
        Returns: doc_id, semantic_score, semantic_score_normalized,
                 matched_skills, skill_overlap_score, + metadata
        """
        results = self.dense.search(query, top_k=top_k)
        query_skills = self._extract_query_skills(query)
        output = []
        for doc_id, score in results:
            meta = self.metadata.get(doc_id) or {}
            doc_skills = self.skills.get_doc_skills(doc_id)
            matched_skills = self._compute_matched_skills(query_skills, doc_skills)
            skill_overlap = len(matched_skills) / max(len(query_skills), 1)
            output.append({
                "doc_id": doc_id,
                "semantic_score": round(score, 4),
                "matched_skills": matched_skills,
                "skill_overlap_score": round(skill_overlap, 3),
                **meta,
            })
        return normalize_scores(output, "semantic_score")

    def skill_lookup(self, skill: str, top_k: int = 20) -> list[dict]:
        """Direct skill inverted index lookup."""
        skill_norm = self.normalizer.normalize_skill(skill)
        results = self.skills.lookup(skill_norm, top_k=top_k)
        output = []
        for doc_id, weight in results:
            meta = self.metadata.get(doc_id) or {}
            output.append({
                "doc_id": doc_id,
                "skill_weight": round(weight, 2),
                **meta,
            })
        return output

    # ── Explainability Helpers ────────────────────────────────────────────
    def _extract_query_skills(self, query: str) -> set[str]:
        """
        Extract potential skill terms from query text.
        Uses normalizer to canonicalize, then checks against known skill vocabulary.
        """
        norm_query = self.normalizer.normalize_text(query)
        # Use same robust tokenization regex as BM25 indexer
        tokens = re.findall(r"[a-z0-9][a-z0-9+#_]*", norm_query.lower())
        vocab = set(self.skills.get_vocabulary())
        skills = set()
        for t in tokens:
            t_norm = self.normalizer.normalize_skill(t)
            if t_norm in vocab:
                skills.add(t_norm)
            # Also try multi-word combinations via normalization
            t_token = self.normalizer.normalize_token(t.lower())
            if t_token in vocab:
                skills.add(t_token)
        return skills

    def _compute_matched_skills(self, query_skills: set[str],
                                  doc_skills: dict[str, float]) -> list[dict]:
        """
        Compute which query skills matched document skills.
        Returns list of {skill, weight} dicts for explainability.
        """
        matched = []
        doc_skill_set = set(doc_skills.keys())
        for qs in query_skills:
            if qs in doc_skill_set:
                matched.append({"skill": qs, "weight": doc_skills[qs]})
            else:
                # Check synonyms
                synonyms = self.normalizer.get_synonyms(qs)
                for syn in synonyms:
                    if syn in doc_skill_set:
                        matched.append({
                            "skill": syn,
                            "weight": doc_skills[syn],
                            "matched_via": f"synonym of '{qs}'"
                        })
                        break
        return matched

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": self.VERSION,
            "fingerprint": self._fingerprint,
            "dense_strategy": self.dense.strategy,
            "faiss_enabled": self.dense._faiss_index is not None,
            "schema": self.schema.to_dict() if self.schema else {},
            "sbert_model_name": self.dense.SBERT_MODEL_NAME,
            "num_documents": len(self.bm25),
        }
        with open(path / "manifest.json", "w") as f:
            json.dump(payload, f, indent=2)

        # Save per-row hashes for incremental updates
        with open(path / "row_hashes.pkl", "wb") as f:
            pickle.dump(self._row_hashes, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(path / "schema.pkl", "wb") as f:
            pickle.dump(self.schema, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(path / "dense.pkl", "wb") as f:
            pickle.dump(self.dense, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(path / "skills.pkl", "wb") as f:
            pickle.dump(self.skills, f, protocol=pickle.HIGHEST_PROTOCOL)



        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save dense matrix separately as .npy for easy numpy reloading
        if self.dense._matrix is not None:
            np.save(path / "dense_matrix.npy", self.dense._matrix)

        log.info("Index saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "HybridIndex":
        path = Path(path)
        idx = cls()

        with open(path / "manifest.json") as f:
            manifest = json.load(f)
        log.info("Loading index v%s (fingerprint=%s)", manifest["version"], manifest["fingerprint"])
        idx._fingerprint = manifest["fingerprint"]

        with open(path / "schema.pkl", "rb") as f:
            idx.schema = pickle.load(f)

        with open(path / "bm25.pkl", "rb") as f:
            idx.bm25 = pickle.load(f)
        with open(path / "dense.pkl", "rb") as f:
            idx.dense = pickle.load(f)
        with open(path / "skills.pkl", "rb") as f:
            idx.skills = pickle.load(f)
        with open(path / "metadata.pkl", "rb") as f:
            idx.metadata = pickle.load(f)



        # Load per-row hashes (backward-compatible with v2 indexes)
        hash_path = path / "row_hashes.pkl"
        if hash_path.exists():
            with open(hash_path, "rb") as f:
                idx._row_hashes = pickle.load(f)
        else:
            idx._row_hashes = {}
            log.warning("No row hashes found — incremental updates will treat all rows as new")

        log.info("Index loaded from %s", path)
        return idx

    # ── Incremental update (schema-evolution safe) ────────────────────────
    @staticmethod
    def _row_hash(raw: dict) -> str:
        """Compute a deterministic hash for a single row."""
        return hashlib.sha256(
            str(sorted(raw.items())).encode("utf-8")
        ).hexdigest()[:16]

    def update_from_dataframe(self, df: pd.DataFrame, id_col: str = "id"):
        """
        Incrementally update the index from a (possibly changed) DataFrame.
        Detects new, changed, and deleted rows using per-row SHA256 hashes.
        Only re-processes the delta — no full rebuild required.

        Handles:
          - NEW rows     → add to all indexes
          - CHANGED rows → remove old entry, add updated entry
          - DELETED rows → remove from all indexes
          - SCHEMA CHANGES → new columns are auto-discovered

        Returns a dict summarizing what changed.
        """
        log.info("=== Incremental Update ===")
        log.info("Incoming DataFrame: %d rows", len(df))

        # Drop entirely empty rows and normalize numeric pandas types
        df = df.dropna(how='all').convert_dtypes()

        # Re-discover schema (handles new/renamed columns)
        new_schema = FieldSchema(df)

        # Compute per-row hashes for the incoming data
        incoming_hashes: dict[str, str] = {}
        incoming_rows: dict[str, dict] = {}
        for _, row in df.iterrows():
            if row.isna().all(): continue
            raw = row.to_dict()
            
            # Robust ID normalization against float casting
            raw_id = raw.get(new_schema.id_col or id_col, "")
            doc_id = str(raw_id).strip()
            if doc_id.endswith(".0"): doc_id = doc_id[:-2]
            if not doc_id or doc_id == "nan": continue
            incoming_hashes[doc_id] = self._row_hash(raw)
            incoming_rows[doc_id] = raw

        existing_ids = set(self._row_hashes.keys())
        incoming_ids = set(incoming_hashes.keys())

        # Classify changes
        new_ids = incoming_ids - existing_ids
        deleted_ids = existing_ids - incoming_ids
        common_ids = existing_ids & incoming_ids
        changed_ids = {
            did for did in common_ids
            if incoming_hashes[did] != self._row_hashes.get(did, "")
        }
        unchanged_count = len(common_ids) - len(changed_ids)

        log.info("  New: %d | Changed: %d | Deleted: %d | Unchanged: %d",
                 len(new_ids), len(changed_ids), len(deleted_ids), unchanged_count)

        if not new_ids and not changed_ids and not deleted_ids:
            log.info("  No changes detected — index is up to date.")
            return {"new": 0, "changed": 0, "deleted": 0, "unchanged": unchanged_count}

        # ── Step 1: Remove deleted + changed rows from all indexes ────────
        ids_to_remove = deleted_ids | changed_ids
        if ids_to_remove:
            log.info("  Removing %d documents from indexes...", len(ids_to_remove))
            for doc_id in ids_to_remove:
                self.bm25.remove_document(doc_id)
                self.skills.remove_document(doc_id)
                # Metadata removal
                if doc_id in self.metadata._store:
                    del self.metadata._store[doc_id]
                # Remove from row hash tracker
                if doc_id in self._row_hashes:
                    del self._row_hashes[doc_id]

            # Dense index — batch removal then FAISS rebuild
            self.dense.remove_documents(ids_to_remove)

        # ── Step 2: Add new + changed rows to all indexes ─────────────────
        ids_to_add = new_ids | changed_ids
        if ids_to_add:
            log.info("  Adding %d documents to indexes...", len(ids_to_add))
            new_docs: list[Document] = []
            for doc_id in ids_to_add:
                raw = incoming_rows[doc_id]
                doc = Document(doc_id, raw, new_schema, normalizer=self.normalizer)
                new_docs.append(doc)
                # Update row hash
                self._row_hashes[doc_id] = incoming_hashes[doc_id]

            # BM25: add new documents
            for doc in new_docs:
                self.bm25.add_document(doc.doc_id, doc.bm25_text)

            # Dense: encode only new docs and append to FAISS
            self.dense.add_documents(
                [d.doc_id for d in new_docs],
                [d.semantic_text for d in new_docs],
            )

            # Skills + Co-occurrence + Metadata
            for doc in new_docs:
                self.skills.add(doc.doc_id, doc.skills)

                self.metadata.put(doc.doc_id, doc.metadata)

        # ── Step 3: Recalculate BM25 IDF stats (cheap O(vocab) math) ──────
        self.bm25.build()
        self.skills.build()

        # Update schema reference
        self.schema = new_schema

        # Update fingerprint
        self._fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:16]

        summary = {
            "new": len(new_ids),
            "changed": len(changed_ids),
            "deleted": len(deleted_ids),
            "unchanged": unchanged_count,
            "total_after": len(self.bm25),
        }
        log.info("=== Incremental Update Complete ===")
        self._log_stats()
        log.info("  Summary: %s", summary)
        return summary
