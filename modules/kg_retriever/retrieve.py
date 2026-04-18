from __future__ import annotations
"""
kg_retriever.py
---------------
Knowledge Graph Retrieval Engine for HR Candidate Search.

Changes vs previous version:
  - 4-hop traversal (hop1 direct, hop2 related, hop2 domain, hop3 deep, hop4 deep)
  - NO result cap — returns every candidate that matched at any hop depth
  - Output includes match_reasons: [{query_term, matched_node, relationship, hops, score_delta}]
  - Output includes full candidate attributes from CSV:
      name, secondary_skills, soft_skills, skill_summary
  - CLI: --top-k removed, always returns all matches

Usage (standalone):
    python kg_retriever.py --query query.json --csv profiles.csv --out results.json

Usage (as module):
    from kg_retriever import KGRetriever
    retriever = KGRetriever(neo4j_uri, neo4j_user, neo4j_password, csv_path)
    results = retriever.retrieve(expanded_query_dict)
"""

import os
import re
import json
import logging
import argparse
from typing import Any
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

# Load environment variables from .env file manually
env_file = Path.cwd() / '.env'
if not env_file.exists():
    env_file = Path(__file__).parent / '.env'

if env_file.exists():
    try:
        with open(env_file, 'r', encoding='utf-8-sig') as f:  # utf-8-sig strips BOM
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    except Exception:
        pass  # Silent fail - will report in main() if credentials are missing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("kg_retriever")


# ===========================================================================
# SCORING CONSTANTS
# ===========================================================================
W_DIRECT_SKILL  = 3.0
W_RELATED_SKILL = 1.5
W_ROLE_MATCH    = 2.0
W_DOMAIN_MATCH  = 1.0
W_EXP_EXACT     = 2.0
W_EXP_CLOSE     = 1.0
HOP_DECAY       = 0.5   # hop1->x1.0, hop2->x0.5, hop3->x0.25, hop4->x0.125


# ===========================================================================
# CYPHER TEMPLATES
# Each has a unique comment token used by the test mock dispatcher.
# ===========================================================================

CYPHER_HOP1_DIRECT = """
// cypher_hop1_direct
MATCH (c:Candidate)-[:HAS_SKILL]->(s:Skill)
WHERE toLower(s.name) CONTAINS toLower($term)
RETURN DISTINCT c.id AS candidate_id,
       s.name        AS matched_node,
       'HAS_SKILL'   AS relationship,
       1             AS hops
"""

CYPHER_HOP2_RELATED = """
// cypher_hop2_related
MATCH (c:Candidate)-[:HAS_SKILL]->(s:Skill)-[:RELATED_TO|BELONGS_TO]->(n)
WHERE toLower(n.name) CONTAINS toLower($term)
RETURN DISTINCT c.id   AS candidate_id,
       s.name          AS matched_node,
       'RELATED_TO'    AS relationship,
       2               AS hops
"""

CYPHER_HOP2_DOMAIN = """
// cypher_hop2_domain
MATCH (c:Candidate)-[:HAS_SKILL]->(s:Skill)-[:HAS_SUBDOMAIN|BELONGS_TO]->(d)
WHERE toLower(d.name) CONTAINS toLower($term)
RETURN DISTINCT c.id    AS candidate_id,
       d.name           AS matched_node,
       'HAS_SUBDOMAIN'  AS relationship,
       2                AS hops
"""

CYPHER_HOP1_ROLE = """
// cypher_hop1_role
MATCH (c:Candidate)-[:SUITABLE_FOR]->(r:Role)
WHERE toLower(r.name) CONTAINS toLower($term)
RETURN DISTINCT c.id   AS candidate_id,
       r.name          AS matched_node,
       'SUITABLE_FOR'  AS relationship,
       1               AS hops
"""

CYPHER_HOP3_DEEP = """
// cypher_hop3_deep
MATCH (c:Candidate)-[:HAS_SKILL]->(s:Skill)
      -[:RELATED_TO|BELONGS_TO]->(n)
      -[:RELATED_TO|BELONGS_TO]->(m)
WHERE toLower(m.name) CONTAINS toLower($term)
RETURN DISTINCT c.id   AS candidate_id,
       s.name          AS matched_node,
       'HOP3_CHAIN'    AS relationship,
       3               AS hops
"""

CYPHER_HOP4_DEEP = """
// cypher_hop4_deep
MATCH (c:Candidate)-[:HAS_SKILL]->(s:Skill)
      -[:RELATED_TO|BELONGS_TO]->(n)
      -[:RELATED_TO|BELONGS_TO]->(m)
      -[:RELATED_TO|BELONGS_TO]->(p)
WHERE toLower(p.name) CONTAINS toLower($term)
RETURN DISTINCT c.id   AS candidate_id,
       s.name          AS matched_node,
       'HOP4_CHAIN'    AS relationship,
       4               AS hops
"""

CYPHER_CANDIDATE_SKILLS = """
// cypher_candidate_skills
MATCH (c:Candidate)-[:HAS_SKILL]->(s:Skill)
WHERE c.id IN $ids
RETURN c.id AS candidate_id, collect(s.name) AS graph_skills
"""


# ===========================================================================
# HELPERS
# ===========================================================================

def _parse_skill_name(skill_str: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", skill_str).strip()


def _parse_skills_list(raw) -> list[str]:
    if not raw or pd.isna(raw):
        return []
    tokens, depth, current = [], 0, []
    for ch in str(raw):
        if ch == "(":
            depth += 1; current.append(ch)
        elif ch == ")":
            depth -= 1; current.append(ch)
        elif ch == "," and depth == 0:
            tokens.append("".join(current).strip()); current = []
        else:
            current.append(ch)
    if current:
        tokens.append("".join(current).strip())
    return [_parse_skill_name(t) for t in tokens if t]


def _extract_experience_years(query_parsed: dict) -> float | None:
    raw = query_parsed.get("experience_years")
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except (ValueError, TypeError):
        return None


def _experience_score(candidate_years: float, query_years: float | None) -> float:
    if query_years is None or pd.isna(candidate_years):
        return 0.0
    diff = abs(candidate_years - query_years)
    if diff <= 1:
        return W_EXP_EXACT
    if diff <= 3:
        return W_EXP_CLOSE
    return 0.0


def _seniority_keywords(band: str | None) -> list[str]:
    mapping = {
        "senior":  ["senior", "lead", "principal", "staff", "architect"],
        "mid":     ["mid", "intermediate", "engineer", "developer"],
        "junior":  ["junior", "associate", "intern", "entry"],
        "manager": ["manager", "head", "director", "vp"],
    }
    if not band:
        return []
    return mapping.get(band.lower(), [band.lower()])


# ===========================================================================
# NEO4J RUNNER
# ===========================================================================

class _Neo4jRunner:

    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Neo4j driver connected -> %s", uri)

    def close(self):
        self._driver.close()

    def run(self, cypher: str, **params) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(r) for r in result]

    def ping(self) -> bool:
        try:
            with self._driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j ping: OK")
            return True
        except Exception as exc:
            logger.error("Neo4j ping failed: %s", exc)
            return False

    def node_counts(self) -> dict:
        """Return counts of key node labels for connectivity diagnostics."""
        counts = {}
        labels = ["Candidate", "Skill", "Role", "Domain"]
        for label in labels:
            try:
                rows = self.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                counts[label] = rows[0]["cnt"] if rows else 0
            except Exception:
                counts[label] = "error"
        return counts


# ===========================================================================
# CORE RETRIEVER
# ===========================================================================

class KGRetriever:
    """
    Knowledge Graph Retrieval Engine — 4-hop traversal, no result cap.
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        csv_path: str,
    ):
        self._db = _Neo4jRunner(neo4j_uri, neo4j_user, neo4j_password)
        self._profiles = self._load_profiles(csv_path)
        logger.info("Profiles loaded - %d rows", len(self._profiles))

    def ping(self) -> dict:
        """Return DB connectivity status and node counts."""
        ok = self._db.ping()
        counts = self._db.node_counts() if ok else {}
        return {"connected": ok, "node_counts": counts}

    def retrieve(self, expanded_query: dict) -> list[dict]:
        """
        Retrieve ALL matching candidates. No top-k cap.

        Each result contains:
          candidate_id, name, score, graph_score, experience_score,
          core_skills, secondary_skills, soft_skills, skill_summary,
          years_of_experience, potential_roles,
          matched_terms, match_reasons
        """
        parsed  = expanded_query.get("parsed", {})
        skills  = [s.lower() for s in parsed.get("skills", []) if s]
        role    = parsed.get("role")
        band    = parsed.get("experience_band")
        exp_yrs = _extract_experience_years(parsed)

        all_terms = self._collect_terms(expanded_query, skills, role, band)
        logger.info(
            "Retrieval - skills=%s | role=%s | band=%s | exp=%s | terms=%d",
            skills, role, band, exp_yrs, len(all_terms),
        )

        raw_hits = self._graph_search(all_terms, skills)
        if not raw_hits:
            logger.warning("No candidates found.")
            return []

        candidate_ids = list(raw_hits.keys())
        csv_rows = self._profiles[
            self._profiles["id"].astype(str).isin([str(c) for c in candidate_ids])
        ]

        scored = []
        for cid, hit in raw_hits.items():
            csv_row   = csv_rows[csv_rows["id"].astype(str) == str(cid)]
            exp_score = 0.0
            years     = None
            roles_list, secondary, soft, summary, name = [], [], [], "", ""

            if not csv_row.empty:
                row       = csv_row.iloc[0]
                name      = str(row.get("name", "") or "")
                years     = float(row["years_of_experience"]) if not pd.isna(row["years_of_experience"]) else None
                exp_score = _experience_score(years or 0.0, exp_yrs)
                roles_raw = row.get("potential_roles", "")
                roles_list = [r.strip() for r in str(roles_raw).split(",")] if roles_raw else []
                secondary  = _parse_skills_list(row.get("secondary_skills", "") or "")
                soft       = _parse_skills_list(row.get("soft_skills", "") or "")
                summary    = str(row.get("skill_summary", "") or "")

                for kw in _seniority_keywords(band):
                    if any(kw in r.lower() for r in roles_list):
                        exp_score += 0.5
                        break

            total_score = round(hit["graph_score"] + exp_score, 4)

            scored.append({
                "candidate_id":        str(cid),
                "name":                name,
                "score":               total_score,
                "graph_score":         round(hit["graph_score"], 4),
                "experience_score":    round(exp_score, 4),
                "core_skills":         hit["core_skills"],
                "secondary_skills":    secondary,
                "soft_skills":         soft,
                "skill_summary":       summary,
                "years_of_experience": years,
                "potential_roles":     roles_list,
                "matched_terms":       sorted(hit["matched_terms"]),
                "match_reasons":       sorted(
                    hit["match_reasons"],
                    key=lambda x: (-x["score_delta"], x["hops"]),
                ),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        logger.info("Returning %d candidates", len(scored))
        return scored

    def close(self):
        self._db.close()

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    @staticmethod
    def _load_profiles(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df["_core_skills_parsed"] = df["core_skills"].apply(_parse_skills_list)
        return df

    def _collect_terms(
        self,
        expanded_query: dict,
        skills: list[str],
        role: str | None,
        band: str | None,
    ) -> dict[str, str]:
        terms: dict[str, str] = {}
        for s in skills:
            terms[s] = "skill"
        if role:
            terms[role.lower()] = "role"
        for kw in _seniority_keywords(band):
            terms[kw] = "role"

        STOP = {
            "someone", "with", "and", "or", "a", "an", "the", "in", "of",
            "for", "to", "candidate", "skilled", "proficient", "experience",
            "years", "role", "engineer", "developer", "full", "stack",
            "senior", "junior", "mid", "lead", "manager",
        }
        for q_str in expanded_query.get("queries", []):
            strategy = expanded_query.get("strategy_map", {}).get(q_str, "")
            for tok in re.split(r"[\s,]+", q_str.lower()):
                tok = tok.strip(".")
                if len(tok) >= 3 and tok not in STOP and tok not in terms:
                    terms[tok] = "domain" if strategy == "kg_static" else "skill"

        return terms

    def _graph_search(
        self,
        all_terms: dict[str, str],
        primary_skills: list[str],
    ) -> dict[Any, dict]:

        hits: dict[Any, dict] = {}

        def _add_hit(cid, query_term, matched_node, relationship, hops, weight):
            if cid not in hits:
                hits[cid] = {
                    "graph_score":   0.0,
                    "matched_terms": set(),
                    "core_skills":   [],
                    "match_reasons": [],
                }
            decay       = HOP_DECAY ** (hops - 1)
            score_delta = round(weight * decay, 4)
            hits[cid]["graph_score"]   += score_delta
            hits[cid]["matched_terms"].add(query_term)
            hits[cid]["match_reasons"].append({
                "query_term":   query_term,
                "matched_node": matched_node,
                "relationship": relationship,
                "hops":         hops,
                "score_delta":  score_delta,
            })

        # Ordered traversal pipeline: (cypher, weight_fn)
        skill_traversal = [
            (CYPHER_HOP1_DIRECT,  lambda t: W_DIRECT_SKILL if t in primary_skills else W_RELATED_SKILL),
            (CYPHER_HOP2_RELATED, lambda t: W_RELATED_SKILL),
            (CYPHER_HOP2_DOMAIN,  lambda t: W_DOMAIN_MATCH),
            (CYPHER_HOP3_DEEP,    lambda t: W_RELATED_SKILL * HOP_DECAY),
            (CYPHER_HOP4_DEEP,    lambda t: W_RELATED_SKILL * HOP_DECAY * HOP_DECAY),
        ]

        for term, source in all_terms.items():
            for cypher, weight_fn in skill_traversal:
                rows = self._db.run(cypher, term=term)
                w    = weight_fn(term)
                for r in rows:
                    _add_hit(r["candidate_id"], term, r["matched_node"],
                             r["relationship"], r["hops"], w)

            if source in ("role", "skill"):
                rows = self._db.run(CYPHER_HOP1_ROLE, term=term)
                for r in rows:
                    _add_hit(r["candidate_id"], term, r["matched_node"],
                             r["relationship"], r["hops"], W_ROLE_MATCH)

        if not hits:
            return {}

        # Enrich with graph-side skills
        all_ids    = list(hits.keys())
        skill_rows = self._db.run(CYPHER_CANDIDATE_SKILLS, ids=all_ids)
        for row in skill_rows:
            cid = row["candidate_id"]
            if cid in hits:
                hits[cid]["core_skills"] = row["graph_skills"]

        # CSV fallback
        missing = [cid for cid, v in hits.items() if not v["core_skills"]]
        if missing:
            csv_sub = self._profiles[
                self._profiles["id"].astype(str).isin([str(c) for c in missing])
            ]
            for _, row in csv_sub.iterrows():
                cid = row["id"]
                if cid in hits:
                    hits[cid]["core_skills"] = row["_core_skills_parsed"]

        for v in hits.values():
            v["matched_terms"] = list(v["matched_terms"])
            v["graph_score"]   = round(v["graph_score"], 4)

        return hits


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="KG Retriever - 4-hop graph retrieval",
        epilog="Credentials are read from .env file in retrieval/ directory.\nSet NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env",
    )
    # Read environment variables at parse time (after load_dotenv has run)
    default_uri = os.environ.get("NEO4J_URI")
    default_user = os.environ.get("NEO4J_USERNAME")
    default_password = os.environ.get("NEO4J_PASSWORD")
    
    p.add_argument("--query",    required=True,
                   help="Path to expanded query JSON file")
    p.add_argument("--csv",      default="profiles.csv",
                   help="Path to profiles.csv (default: profiles.csv)")
    p.add_argument("--uri",      default=default_uri,
                   help="Neo4j connection URI  (reads from NEO4J_URI in .env if not provided)")
    p.add_argument("--user",     default=default_user,
                   help="Neo4j username        (reads from NEO4J_USERNAME in .env if not provided)")
    p.add_argument("--password", default=default_password,
                   help="Neo4j password        (reads from NEO4J_PASSWORD in .env if not provided)")
    p.add_argument("--out",      default=None,
                   help="Output file path (if not specified, prints to stdout)")
    return p.parse_args()


def main():
    args = _parse_args()
    
    # Validate credentials from .env
    if not args.uri or not args.user or not args.password:
        logger.error("❌ Missing Neo4j credentials!")
        logger.error("   Please ensure .env file has:")
        logger.error("   - NEO4J_URI")
        logger.error("   - NEO4J_USERNAME")
        logger.error("   - NEO4J_PASSWORD")
        return
    
    # Validate query file exists
    query_path = Path(args.query)
    if not query_path.exists():
        logger.error("❌ Query file not found: %s", query_path)
        return
    
    # Validate CSV file exists
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("❌ CSV file not found: %s", csv_path)
        return
    
    with open(query_path, "r", encoding="utf-8") as f:
        expanded_query = json.load(f)

    retriever = KGRetriever(args.uri, args.user, args.password, str(csv_path))
    try:
        results = retriever.retrieve(expanded_query)
    finally:
        retriever.close()

    output = json.dumps(results, indent=2, ensure_ascii=False)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output)
        logger.info("Written -> %s (%d candidates)", out_path, len(results))
    else:
        print(output)


if __name__ == "__main__":
    main()