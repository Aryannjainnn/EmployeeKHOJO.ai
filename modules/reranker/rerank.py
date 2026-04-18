from __future__ import annotations
"""
reranker.py
-----------
Intent-aware re-ranking of hybrid + KG retrieval results.

Pipeline:
  1. Read primary_intent + confidence + modifiers from the expanded query.
  2. Look up base (α, β) weights from the intent table.
  3. Dampen weights toward 0.5 proportionally to (1 - confidence).
  4. Take the union of hybrid and KG candidate pools (missing score → 0).
  5. Compute fusion score F = α′·H + β′·K.
  6. Apply modifier bonuses/penalties (experience match, negated skills, location).
  7. Clamp F to [0, 1], rank descending, return top-K with full score breakdown.

Usage (as module):
    from reranker import Reranker
    reranker = Reranker()
    results  = reranker.rerank(expanded_query, hybrid_results, kg_results, top_k=20)

Usage (standalone test):
    python reranker.py
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger("reranker")

# ---------------------------------------------------------------------------
# Tuneable constants — change these without touching logic
# ---------------------------------------------------------------------------

# Intent → (α_hybrid, β_kg).  α + β must always equal 1.0.
INTENT_WEIGHTS: dict[str, tuple[float, float]] = {
    "single_skill":      (0.35, 0.65),   # graph traversal finds related skills better
    "domain_search":     (0.30, 0.70),   # graph domain/subdomain edges are key
    "multi_skill":       (0.40, 0.60),   # KG intersection across multiple skill nodes
    "role_based":        (0.55, 0.45),   # role strings = surface matching advantage
    "experience_filter": (0.60, 0.40),   # numeric/text filter = hybrid advantage
    "location_based":    (0.65, 0.35),   # location is pure text filter
    "hybrid":            (0.50, 0.50),   # explicit hybrid intent → equal weight
}

# Fallback when intent is missing or unrecognised
DEFAULT_WEIGHTS: tuple[float, float] = (0.50, 0.50)

# Below this confidence, weights are blended toward 0.5/0.5
CONFIDENCE_THRESHOLD = 0.0   # always dampen — set to e.g. 0.70 to skip dampening above it

# Modifier adjustments (applied after fusion, before clamp)
MODIFIER_BONUS: dict[str, float] = {
    "experience_filter": +0.05,   # experience band + years matched in KG score
    "location_filter":   +0.03,
    "seniority_match":   +0.03,
}
NEGATED_SKILL_PENALTY = -0.10    # per negated skill found in candidate's core_skills

# Final output cap
TOP_K_DEFAULT = 20


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RankedCandidate:
    candidate_id: str
    final_score:  float
    hybrid_score: float
    kg_score:     float
    alpha:        float            # weight applied to hybrid signal
    beta:         float            # weight applied to KG signal
    modifier_delta: float          # net bonus/penalty from modifiers
    core_skills:  list[str]        = field(default_factory=list)
    matched_terms: list[str]       = field(default_factory=list)
    years_of_experience: float | None = None
    potential_roles: list[str]     = field(default_factory=list)
    source: str                    = "both"   # "hybrid_only" | "kg_only" | "both"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["final_score"] = round(d["final_score"], 4)
        d["hybrid_score"] = round(d["hybrid_score"], 4)
        d["kg_score"] = round(d["kg_score"], 4)
        d["alpha"] = round(d["alpha"], 4)
        d["beta"] = round(d["beta"], 4)
        d["modifier_delta"] = round(d["modifier_delta"], 4)
        return d


# ---------------------------------------------------------------------------
# Core Reranker
# ---------------------------------------------------------------------------

class Reranker:
    """
    Intent-aware score fusion and re-ranking.

    Parameters
    ----------
    intent_weights : dict, optional
        Override the default INTENT_WEIGHTS table.
    modifier_bonus : dict, optional
        Override the default MODIFIER_BONUS table.
    """

    def __init__(
        self,
        intent_weights: dict[str, tuple[float, float]] | None = None,
        modifier_bonus: dict[str, float] | None = None,
    ):
        self._intent_weights = intent_weights or INTENT_WEIGHTS
        self._modifier_bonus = modifier_bonus or MODIFIER_BONUS

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def rerank(
        self,
        expanded_query: dict,
        hybrid_results: list[dict],
        kg_results: list[dict],
        top_k: int = TOP_K_DEFAULT,
    ) -> dict:
        """
        Re-rank the union of hybrid + KG candidate pools (returns ALL candidates).

        Parameters
        ----------
        expanded_query : dict
            Full expanded query object (must contain 'intent' and 'parsed' keys).
        hybrid_results : list[dict]
            Each dict must have 'candidate_id' and 'score' (normalised 0–1).
            May optionally have 'core_skills', 'years_of_experience', 'potential_roles'.
        kg_results : list[dict]
            Each dict must have 'candidate_id' and 'score' (normalised 0–1).
            May optionally have 'core_skills', 'matched_terms', 'years_of_experience'.
        top_k : int
            Deprecated - all candidates are returned.

        Returns
        -------
        dict with keys:
            - 'total_count': int — total number of output candidates
            - 'results': list[dict] — all ranked candidates (no filtering)
        """
        # 1. Extract intent signal
        intent_obj   = expanded_query.get("intent", {})
        primary      = intent_obj.get("primary_intent", "hybrid")
        confidence   = float(intent_obj.get("confidence", 0.5))
        modifiers    = intent_obj.get("modifiers", [])
        parsed       = expanded_query.get("parsed", {})
        negated      = [s.lower() for s in parsed.get("negated_skills", [])]

        # 2. Look up base weights and dampen by confidence
        alpha, beta = self._resolve_weights(primary, confidence)

        logger.info(
            "Rerank — intent=%s conf=%.2f α=%.3f β=%.3f modifiers=%s",
            primary, confidence, alpha, beta, modifiers,
        )

        # 3. Build unified candidate pool
        pool = self._build_pool(hybrid_results, kg_results)

        if not pool:
            logger.warning("Empty candidate pool — returning empty results")
            return {
                "total_count": 0,
                "results": []
            }

        # 4. Score, apply modifiers, clamp
        ranked = []
        for cid, data in pool.items():
            H = data["hybrid_score"]
            K = data["kg_score"]

            fusion = alpha * H + beta * K

            # Modifier bonuses
            delta = self._modifier_delta(modifiers, negated, data, parsed)
            final = max(0.0, min(1.0, fusion + delta))

            ranked.append(RankedCandidate(
                candidate_id       = str(cid),
                final_score        = final,
                hybrid_score       = H,
                kg_score           = K,
                alpha              = alpha,
                beta               = beta,
                modifier_delta     = delta,
                core_skills        = data.get("core_skills", []),
                matched_terms      = data.get("matched_terms", []),
                years_of_experience = data.get("years_of_experience"),
                potential_roles    = data.get("potential_roles", []),
                source             = data["source"],
            ))

        ranked.sort(key=lambda c: c.final_score, reverse=True)

        # Return ALL candidates with total count (no top-k filtering)
        total_count = len(ranked)
        results_list = [c.to_dict() for c in ranked]

        logger.info("Reranked %d candidates → returning all %d (no filtering)", len(ranked), total_count)
        
        return {
            "total_count": total_count,
            "results": results_list
        }

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _resolve_weights(self, primary: str, confidence: float) -> tuple[float, float]:
        """
        Fetch base (α, β) from the intent table, then dampen toward (0.5, 0.5)
        proportionally to (1 - confidence).

        Formula:
            α′ = lerp(0.5, α_base, confidence)
            β′ = 1 - α′           ← guarantees α′ + β′ = 1 always
        """
        alpha_base, _ = self._intent_weights.get(primary, DEFAULT_WEIGHTS)

        # Linear interpolation: at conf=1.0 → α_base, at conf=0.0 → 0.5
        alpha_prime = 0.5 + (alpha_base - 0.5) * confidence
        beta_prime  = 1.0 - alpha_prime

        return round(alpha_prime, 6), round(beta_prime, 6)

    def _build_pool(
        self,
        hybrid_results: list[dict],
        kg_results: list[dict],
    ) -> dict[Any, dict]:
        """
        Build a union pool: {candidate_id → merged_data}.
        Missing score defaults to 0.  Metadata (core_skills, etc.) is taken
        from whichever source has it, preferring KG (richer graph-side data).
        """
        pool: dict[Any, dict] = {}

        # Index hybrid results
        for r in hybrid_results:
            cid = str(r["candidate_id"])
            pool[cid] = {
                "hybrid_score":      float(r.get("score", 0.0)),
                "kg_score":          0.0,
                "core_skills":       r.get("core_skills", []),
                "matched_terms":     r.get("matched_terms", []),
                "years_of_experience": r.get("years_of_experience"),
                "potential_roles":   r.get("potential_roles", []),
                "source":            "hybrid_only",
            }

        # Merge KG results
        for r in kg_results:
            cid = str(r["candidate_id"])
            if cid in pool:
                pool[cid]["kg_score"] = float(r.get("score", 0.0))
                pool[cid]["source"]   = "both"
                # Prefer KG metadata where richer
                if r.get("core_skills"):
                    pool[cid]["core_skills"] = r["core_skills"]
                if r.get("matched_terms"):
                    existing = set(pool[cid]["matched_terms"])
                    pool[cid]["matched_terms"] = list(existing | set(r["matched_terms"]))
                if r.get("years_of_experience") is not None:
                    pool[cid]["years_of_experience"] = r["years_of_experience"]
                if r.get("potential_roles"):
                    pool[cid]["potential_roles"] = r["potential_roles"]
            else:
                pool[cid] = {
                    "hybrid_score":      0.0,
                    "kg_score":          float(r.get("score", 0.0)),
                    "core_skills":       r.get("core_skills", []),
                    "matched_terms":     r.get("matched_terms", []),
                    "years_of_experience": r.get("years_of_experience"),
                    "potential_roles":   r.get("potential_roles", []),
                    "source":            "kg_only",
                }

        return pool

    def _modifier_delta(
        self,
        modifiers: list[str],
        negated_skills: list[str],
        data: dict,
        parsed: dict,
    ) -> float:
        """
        Compute the net bonus/penalty from active modifiers.

        Rules:
          - Each recognised modifier that is satisfied → its bonus.
          - Each negated skill that appears in the candidate's core_skills → penalty.
          - Total is summed and returned un-clamped (caller clamps the final score).
        """
        delta = 0.0

        # Positive modifier bonuses
        for mod in modifiers:
            if mod in self._modifier_bonus:
                delta += self._modifier_bonus[mod]

        # Negated skill penalty — check candidate's core_skills
        candidate_skills_lower = {s.lower() for s in data.get("core_skills", [])}
        for ns in negated_skills:
            if any(ns in cs for cs in candidate_skills_lower):
                delta += NEGATED_SKILL_PENALTY
                logger.debug("Penalty applied for negated skill '%s' on candidate", ns)

        return delta


# ---------------------------------------------------------------------------
# Module-level adapter — required by modules/reranker/__init__.py
# which does: from .rerank import rerank
# ---------------------------------------------------------------------------

def rerank(
    expanded_query: dict,
    hybrid_results: list[dict],
    kg_results: list[dict],
    top_k: int = TOP_K_DEFAULT,
) -> dict:
    """
    Module-level adapter so the pipeline can call rerank() directly without
    instantiating Reranker explicitly.

    Equivalent to: Reranker().rerank(expanded_query, hybrid_results, kg_results, top_k)
    """
    return Reranker().rerank(expanded_query, hybrid_results, kg_results, top_k)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description=(
            "Re-rank hybrid + KG retrieval results using intent-aware score fusion.\n\n"
            "All candidates from both sources are included in the output (union).\n"
            "Candidates that appear in both are deduplicated and scored once."
        )
    )
    p.add_argument("--query",   required=True, help="Path to expanded query JSON file")
    p.add_argument("--hybrid",  required=True, help="Path to hybrid results JSON file")
    p.add_argument("--kg",      required=True, help="Path to KG results JSON file")
    p.add_argument("--out",     default=None,  help="Write output JSON to this file (default: stdout)")
    p.add_argument("--top-k",   type=int, default=TOP_K_DEFAULT, help=f"Max results (default {TOP_K_DEFAULT}, 0 = all)")
    p.add_argument("--verbose", action="store_true", help="Print score breakdown to stderr")
    return p.parse_args()


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s \u2014 %(message)s", stream=sys.stderr)

    args = _parse_args()

    with open(args.query,  "r", encoding="utf-8") as f:
        expanded_query = json.load(f)
    with open(args.hybrid, "r", encoding="utf-8") as f:
        hybrid_results = json.load(f)
    with open(args.kg,     "r", encoding="utf-8") as f:
        kg_results = json.load(f)

    # top_k parameter is now ignored (all candidates returned)
    reranker = Reranker()
    output = reranker.rerank(expanded_query, hybrid_results, kg_results, top_k=args.top_k)

    output_json = json.dumps(output, indent=2, ensure_ascii=False)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output_json)
        logger.info("Results written to %s (total: %d candidates)", args.out, output["total_count"])
    else:
        print(output_json)

    if args.verbose:
        results = output.get("results", [])
        print(f"\n=== Score breakdown (Total: {output.get('total_count', 0)} candidates) ===\n", file=sys.stderr)
        for r in results:
            tag = {"both": "H+KG", "hybrid_only": "H   ", "kg_only": "  KG"}[r["source"]]
            print(
                f"[{tag}] {r['candidate_id']:>8}  "
                f"F={r['final_score']:.4f}  "
                f"H={r['hybrid_score']:.3f}  "
                f"K={r['kg_score']:.3f}  "
                f"\u03b1={r['alpha']:.3f}  \u03b2={r['beta']:.3f}  "
                f"\u0394={r['modifier_delta']:+.3f}  "
                f"src={r['source']}",
                file=sys.stderr,
            )
