"""
run_retrieval.py — CLI entry point for the Hybrid Retrieval Engine

Usage
-----
# Normal run (with real index_store/)
python run_retrieval.py --index index_store/ --intent intent.json --output results.json

# Demo run (generates mock indexes from CSV so you can test immediately)
python run_retrieval.py --demo --csv profiles.csv --intent intent.json

"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np


# ============================================================================
# MOCK INDEX BUILDER  (used in --demo mode; mirrors real indexer output)
# ============================================================================

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def build_mock_index_from_csv(csv_path: str | Path, index_dir: str | Path) -> None:
    """
    Build minimal index artifacts directly from the candidate CSV.
    Produces: metadata.pkl, bm25.pkl, dense.pkl, dense_matrix.npy

    This lets you run the retriever WITHOUT the full Component A pipeline.
    In production, Component A (indexer.py) writes these files instead.
    """
    import csv
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load CSV ----
    candidates: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append(dict(row))

    n = len(candidates)
    print(f"[MockIndexer] Loaded {n} candidate rows from {csv_path}")

    # ---- Metadata ----
    metadata = []
    for i, row in enumerate(candidates):
        metadata.append({
            "id":                 row.get("id", str(i)),
            "name":               row.get("name", ""),
            "core_skills":        row.get("core_skills", ""),
            "secondary_skills":   row.get("secondary_skills", ""),
            "soft_skills":        row.get("soft_skills", ""),
            "years_of_experience":float(row.get("years_of_experience") or 0),
            "potential_roles":    row.get("potential_roles", ""),
            "skill_summary":      row.get("skill_summary", ""),
        })
    with open(index_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ---- BM25 ----
    # Corpus = core_skills (repeated 3x for weight) + secondary + roles + summary
    corpus_tokens: list[list[str]] = []
    for m in metadata:
        core_text  = " ".join([m["core_skills"]] * 3)   # triple weight for core
        sec_text   = m["secondary_skills"]
        role_text  = m["potential_roles"]
        summ_text  = m["skill_summary"]
        combined   = f"{core_text} {sec_text} {role_text} {summ_text}"
        corpus_tokens.append(_tokenize(combined))

    # Compute IDF
    df: dict[str, int] = {}
    for tokens in corpus_tokens:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    idf: dict[str, float] = {}
    for term, freq in df.items():
        idf[term] = math.log((n - freq + 0.5) / (freq + 0.5) + 1)

    avgdl = sum(len(t) for t in corpus_tokens) / max(n, 1)

    bm25_data = {
        "corpus_tokens": corpus_tokens,
        "idf":           idf,
        "avgdl":         avgdl,
        "k1":            1.5,
        "b":             0.75,
    }
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ---- Dense (mock embeddings — TF-IDF bag-of-words, dim=384) ----
    # In production, SBERT produces these. Here we use a sparse BoW projection
    # so the module works without GPU/transformers for CI/CD testing.
    vocab = list(idf.keys())
    vocab_idx = {t: i for i, t in enumerate(vocab)}
    vocab_size = len(vocab)
    dim = 384

    # Random projection matrix (stable seed for reproducibility)
    rng = np.random.default_rng(42)
    proj = rng.standard_normal((vocab_size, dim)).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=1, keepdims=True)

    matrix = np.zeros((n, dim), dtype=np.float32)
    for i, tokens in enumerate(corpus_tokens):
        bow = np.zeros(vocab_size, dtype=np.float32)
        for t in tokens:
            if t in vocab_idx:
                bow[vocab_idx[t]] += 1.0
        if bow.sum() > 0:
            bow /= bow.sum()
        vec = bow @ proj
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        matrix[i] = vec

    np.save(index_dir / "dense_matrix.npy", matrix)

    dense_cfg = {
        "model_name":    "mock_bow_projection",
        "embedding_dim": dim,
        "index_type":    "flat_ip",
        "n_candidates":  n,
    }
    with open(index_dir / "dense.pkl", "wb") as f:
        pickle.dump(dense_cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[MockIndexer] Index written to {index_dir}/")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Retrieval Engine")
    parser.add_argument("--index",   default=".", help="Path to index_store/ directory")
    parser.add_argument("--intent",  required=True,         help="Path to intent JSON file")
    parser.add_argument("--output",  default="results.json",help="Path to output JSON file")
    parser.add_argument("--top-k",   type=int, default=50,   help="Number of results per mode (use 200 to match KG recall)")
    parser.add_argument("--demo",    action="store_true",   help="Build mock index first (for testing)")
    parser.add_argument("--csv",     default="profiles.csv",help="CSV path (only needed with --demo)")
    args = parser.parse_args()

    # Optionally build mock index
    if args.demo:
        build_mock_index_from_csv(args.csv, args.index)

    # Load intent JSON
    with open(args.intent, "r", encoding="utf-8") as f:
        intent_json = json.load(f)

    # Import and run retriever
    from retriever import HybridRetriever
    retriever = HybridRetriever(index_dir=args.index, top_k=args.top_k)
    output    = retriever.retrieve(intent_json)

    # Write output to a single JSON file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"[Written] {args.output}")

    print(f"\nRetrieval time: {output['hybrid']['meta']['retrieval_time_ms']} ms")

    # Quick preview
    print("\nTop 5 Hybrid results:")
    for r in output["hybrid"]["results"][:5]:
        kw = ", ".join(k["term"] for k in r["explanation"]["matched_keywords"][:5])
        print(f"  #{r['rank']:2d} | {r['name']:<28s} | RRF={r['scores']['rrf']:.4f} | BM25={r['scores']['bm25']:.3f} | SEM={r['scores']['semantic']:.3f}")
        print(f"       keywords: [{kw}]")
        print(f"       {r['explanation']['why_retrieved'][:90]}")

    print("\nTop 5 Lexical results:")
    for r in output["lexical"]["results"][:5]:
        kw = ", ".join(k["term"] for k in r["explanation"]["matched_keywords"][:5])
        print(f"  #{r['rank']:2d} | {r['name']:<28s} | BM25={r['scores']['bm25']:.4f} | keywords: [{kw}]")

    print("\nTop 5 Semantic results:")
    for r in output["semantic"]["results"][:5]:
        print(f"  #{r['rank']:2d} | {r['name']:<28s} | Semantic={r['scores']['semantic']:.4f}")

if __name__ == "__main__":
    main()
