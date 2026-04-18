"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         LLM-Powered Intent Detection + Query Expansion                     ║
║                    llm_intent_pipeline.py                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Uses a FREE LLM API (Groq by default) for all intelligence:               ║
║    - Spell correction awareness                                             ║
║    - Negation detection                                                     ║
║    - Multi-intent classification                                            ║
║    - Query expansion (all strategies in one prompt)                        ║
║    - Structured JSON output                                                 ║
║                                                                             ║
║  WHY GROQ:                                                                  ║
║    - 100% free, no credit card                                              ║
║    - OpenAI-compatible API (same client, just different base_url)           ║
║    - Llama 3.1 8B Instruct — lightweight, fast, free on Groq               ║
║    - 1,000 requests/day free tier — more than enough for demo              ║
║                                                                             ║
║  SETUP (2 minutes):                                                         ║
║    1. Go to console.groq.com → sign up free                                ║
║    2. Create API key                                                        ║
║    3. pip install openai symspellpy                                         ║
║    4. Set GROQ_API_KEY env var OR pass api_key= directly                   ║
║                                                                             ║
║  ALTERNATIVE FREE PROVIDERS (all OpenAI-compatible, zero code change):     ║
║    - Groq:       base_url="https://api.groq.com/openai/v1"  (fastest)      ║
║    - Cerebras:   base_url="https://api.cerebras.ai/v1"      (high volume)  ║
║    - SambaNova:  base_url="https://api.sambanova.ai/v1"     (backup)       ║
║    - OpenRouter: base_url="https://openrouter.ai/api/v1"    (many models)  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from openai import OpenAI   # pip install openai  (works with Groq — same SDK)

# Spell corrector is still code-based (fast, free, no LLM call needed)
# The LLM handles everything AFTER spell correction
try:
    from symspellpy import SymSpell, Verbosity
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — FREE API PROVIDERS
#  All providers below are OpenAI-compatible — same client, different base_url.
#  Switch between them by changing PROVIDER constant or passing provider= param.
# ══════════════════════════════════════════════════════════════════════════════

PROVIDERS = {
    # WHY GROQ AS DEFAULT:
    # Fastest free inference (~300 tok/s), 1000 req/day, no card, OpenAI-compatible.
    # Llama 3.3 70B follows JSON instructions extremely well.
    "groq": {
        "base_url":    "https://api.groq.com/openai/v1",
        "model":       "llama-3.1-8b-instant",
        "env_var":     "GROQ_API_KEY",
        "description": "Fastest free tier — Llama 3.1 8B at 660+ tok/s on Groq",
    },

    # WHY CEREBRAS AS BACKUP:
    # Even faster raw throughput (~1000 tok/s), 1M tokens/day, no card.
    # Good fallback if Groq rate limits hit during heavy demo use.
    "cerebras": {
        "base_url":    "https://api.cerebras.ai/v1",
        "model":       "llama3.1-8b",
        "env_var":     "CEREBRAS_API_KEY",
        "description": "Highest throughput — Llama 3.1 8B at 2,358 tok/s on Cerebras",
    },

    # OPENROUTER: access many free models through one key.
    # Useful if you want to test different models without code changes.
    "openrouter": {
        "base_url":    "https://openrouter.ai/api/v1",
        "model":       "meta-llama/llama-3.1-8b-instruct:free",
        "env_var":     "OPENROUTER_API_KEY",
        "description": "Multiple free models — Llama 3.1 8B Instruct free tier",
    },

    # SAMBANOVA: another free high-speed option.
    "sambanova": {
        "base_url":    "https://api.sambanova.ai/v1",
        "model":       "Meta-Llama-3.1-8B-Instruct",
        "env_var":     "SAMBANOVA_API_KEY",
        "description": "633 tok/s free — Llama 3.1 8B on SambaNova",
    },
}

DEFAULT_PROVIDER = "groq"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — INTENT TAXONOMY (same as before, used in system prompt)
# ══════════════════════════════════════════════════════════════════════════════

class Intent(str, Enum):
    ROLE_SEARCH       = "role_search"
    SKILL_SEARCH      = "skill_search"
    EXPERIENCE_FILTER = "experience_filter"
    DOMAIN_SEARCH     = "domain_search"
    AVAILABILITY      = "availability"
    COMPARATIVE       = "comparative"
    RANKING           = "ranking"
    MULTI_SKILL       = "multi_skill"
    UNKNOWN           = "unknown"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — DATA STRUCTURES (clean, simple)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMIntentResult:
    """Full structured output from the LLM in one call."""

    # ── Spell correction ──────────────────────────────────────────────────────
    corrected_query: str               # typos fixed by LLM awareness
    corrections_made: list[str]        # ["Pyhton→python", "fintch→fintech"]

    # ── Parsed entities ───────────────────────────────────────────────────────
    skills: list[str]                  # ["python", "react", "aws"]
    negated_skills: list[str]          # ["java"] — from "no java" / "not java"
    role: Optional[str]                # "software engineer"
    experience_band: Optional[str]     # "senior" / "mid" / "entry"
    experience_years: Optional[str]    # "5"
    domain: Optional[str]              # "fintech"
    location: Optional[str]            # "remote"
    negated_location: Optional[str]    # "remote" — from "not remote"

    # ── Intent ────────────────────────────────────────────────────────────────
    primary_intent: Intent
    confidence: float                  # 0.0–1.0
    modifiers: list[Intent]            # secondary intents
    intent_reasoning: str              # LLM explains why (useful for debugging/demo)

    # ── Top-3 scores ──────────────────────────────────────────────────────────
    top3_scores: dict[str, float]      # {"multi_skill":0.92, ...}

    # ── Expanded queries ──────────────────────────────────────────────────────
    expanded_queries: list[str]        # 6–10 diverse query variants
    query_strategies: dict[str, str]   # {query: "strategy_name"} for attribution

    # ── Exclusion filters ─────────────────────────────────────────────────────
    exclusion_filters: dict            # {"must_not_skills":["java"]}

    def to_dict(self) -> dict:
        return {
            "corrected_query":  self.corrected_query,
            "corrections_made": self.corrections_made,
            "entities": {
                "skills":           self.skills,
                "negated_skills":   self.negated_skills,
                "role":             self.role,
                "experience_band":  self.experience_band,
                "experience_years": self.experience_years,
                "domain":           self.domain,
                "location":         self.location,
                "negated_location": self.negated_location,
            },
            "intent": {
                "primary":         self.primary_intent.value,
                "confidence":      round(self.confidence, 3),
                "modifiers":       [m.value for m in self.modifiers],
                "reasoning":       self.intent_reasoning,
                "top3_scores":     self.top3_scores,
            },
            "expanded_queries":  self.expanded_queries,
            "query_strategies":  self.query_strategies,
            "exclusion_filters": self.exclusion_filters,
        }

    def summary(self) -> str:
        mods = " + ".join(m.value for m in self.modifiers)
        mod_str = f"  [{mods}]" if mods else ""
        return (
            f"Intent : {self.primary_intent.value} (conf={self.confidence:.2f}){mod_str}\n"
            f"Corrected: {self.corrected_query}\n"
            f"Skills : {self.skills}  |  Role: {self.role}\n"
            f"Exp    : {self.experience_band}/{self.experience_years}yrs  |  Domain: {self.domain}\n"
            f"Negated: skills={self.negated_skills}  location={self.negated_location}\n"
            f"Top3   : {self.top3_scores}\n"
            f"Queries ({len(self.expanded_queries)}):\n" +
            "\n".join(f"  [{self.query_strategies.get(q,'?'):14}] {q}" for q in self.expanded_queries)
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — SYSTEM PROMPT
#
#  WHY ONE BIG PROMPT (not multiple calls):
#  Each LLM call has network round-trip overhead (~50-200ms on Groq).
#  One well-designed prompt that does spell correction + entity extraction +
#  intent detection + query expansion in a single call is:
#    - Faster overall (1 round trip vs 4-5)
#    - More coherent (the LLM sees all context when generating each part)
#    - Easier to maintain (one prompt to tune, not 5)
#    - Cheaper (uses fewer tokens than 5 separate calls with repeated context)
#
#  WHY STRUCTURED JSON OUTPUT:
#  Free-form text output needs complex parsing. JSON with a fixed schema
#  is parsed with json.loads() — reliable, deterministic, no regex needed.
#  We instruct the model to output ONLY JSON with no preamble.
#
#  WHY THIS SPECIFIC SCHEMA:
#  Every field maps directly to a downstream use:
#    corrected_query     → what gets passed to BM25/FAISS
#    skills/negated      → hard inclusion/exclusion filters
#    intent + modifiers  → which expansion templates to use
#    expanded_queries    → the actual BM25/FAISS query strings
#    exclusion_filters   → must_not clauses for Elasticsearch
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert AI assistant for a job profile retrieval system.

Your job is to analyse a recruiter's search query and return a structured JSON object.
You handle everything in ONE response: spell correction, entity extraction, intent detection,
and query expansion.

DATASET CONTEXT:
Each record in our database is a job profile with: id, candidate_name, skills (list),
years_of_experience, current_role, education, certifications, industry/domain, location.

INTENT CLASSES (pick primary + any modifiers that apply):
- role_search       : searching for a specific job title/role (developer, engineer, analyst)
- skill_search      : looking for candidates with a specific technical skill
- experience_filter : filtering by years of experience OR seniority (senior/junior/lead)
- domain_search     : looking for candidates from a specific industry (fintech, healthtech)
- availability      : asking about work mode or availability (remote, hybrid, open to work)
- comparative       : comparing candidates, roles, or tech stacks
- ranking           : asking for best/top/most qualified candidates
- multi_skill       : candidate must have MULTIPLE specific skills simultaneously
- unknown           : cannot determine intent

QUERY EXPANSION STRATEGIES:
Generate 6-10 diverse query variants using these strategies:
1. synonym       : replace skill names with known aliases (React → ReactJS, k8s → Kubernetes)
2. template      : intent-specific phrasing patterns
3. decomposition : split multi-facet queries into focused single-dimension queries
4. hyde          : write what a PERFECT matching candidate profile would say (3-4 sentences)
5. role_variant  : rephrase using role title variants

NEGATION RULES:
- "not java", "no java", "without java", "excluding java" → java goes to negated_skills
- "not remote", "no remote", "prefer onsite" → goes to negated_location
- Negated terms must NOT appear in expanded_queries

SPELL CORRECTION:
Fix typos in skill names, domain terms, and common English words.
Domain terms to recognise: fintech, healthtech, edtech, ecommerce, saas, blockchain, kubernetes,
fastapi, reactjs, pytorch, tensorflow, typescript, golang, nestjs, graphql, devops, mlops, etc.
Never "correct" valid domain terms (fintech → biotech is WRONG).

OUTPUT FORMAT:
Return ONLY valid JSON, no explanation, no markdown, no code fences.
Exact schema:

{
  "corrected_query": "spell-corrected version of the input",
  "corrections_made": ["original→corrected", ...],
  "skills": ["canonical skill names detected"],
  "negated_skills": ["skills that were negated in the query"],
  "role": "job role if detected, else null",
  "experience_band": "entry|mid|senior|executive or null",
  "experience_years": "number as string or null",
  "domain": "industry domain if detected, else null",
  "location": "location/work-mode if detected positively, else null",
  "negated_location": "location that was negated, else null",
  "primary_intent": "one of the intent class values",
  "confidence": 0.0 to 1.0,
  "modifiers": ["additional intent values that also apply"],
  "intent_reasoning": "1-2 sentences explaining why this intent was chosen",
  "top3_scores": {"intent_name": score, ...},
  "expanded_queries": ["query string 1", "query string 2", ...],
  "query_strategies": {"query string 1": "strategy_name", ...},
  "exclusion_filters": {"must_not_skills": [...], "must_not_location": "..."}
}

RULES:
- expanded_queries must have 6-10 items
- hyde strategy generates a short candidate profile paragraph (not a query keyword string)
- negated_skills and negated_location must NOT appear in expanded_queries
- primary_intent must be exactly one of the intent class strings
- modifiers is a list of additional applicable intent strings (can be empty list)
- top3_scores must have exactly 3 entries
- All field names must match the schema exactly
- Output ONLY the JSON object, nothing before or after it"""


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — DOMAIN-AWARE SPELL CORRECTOR
#  (kept as code — it's fast, free, and handles the fintech→biotech bug
#   before the query even reaches the LLM)
# ══════════════════════════════════════════════════════════════════════════════

# Minimal domain dict — protects key terms from SymSpell corruption
# LLM handles nuanced spell correction for anything not in this list
DOMAIN_PROTECTED: set[str] = {
    "fintech","healthtech","edtech","ecommerce","saas","gaming","blockchain",
    "cybersecurity","adtech","proptech","agritech","hrtech","legaltech",
    "react","reactjs","nodejs","typescript","fastapi","django","kubernetes","k8s",
    "pytorch","tensorflow","sklearn","scikit","pyspark","mlops","devops","devsecops",
    "graphql","grpc","websocket","oauth","jwt","ldap","saml",
    "aws","gcp","azure","eks","gke","aks","ec2","s3","bigquery","cloudfront",
    "elasticsearch","mongodb","cassandra","dynamodb","postgresql","redis","kafka",
    "swe","sde","mlops","devrel","tpm","sre","dba","ux","ui",
    "llm","rag","rlhf","lora","peft","bert","gpt","vllm","langchain","llamaindex",
    "dbt","airflow","prefect","dagster","snowflake","databricks","fivetran",
    "terraform","ansible","helm","kustomize","argocd","jenkins","gitlab",
}

# Alias → canonical normalisation applied after spell correction.
# WHY: "reactjs" typed by a recruiter should reach the LLM as "react" (canonical).
# The LLM also handles aliases, but pre-normalising gives it cleaner input.
ALIAS_TO_CANONICAL: dict[str, str] = {
    # Frontend
    "reactjs":"react", "react.js":"react",
    "vuejs":"vue", "vue.js":"vue", "vue3":"vue",
    "angularjs":"angular", "angular.js":"angular",
    "nextjs":"next.js", "nuxtjs":"nuxt.js",
    "sveltejs":"svelte",
    "vanillajs":"javascript", "ecmascript":"javascript", "es6":"javascript",
    "tsx":"typescript",
    "tailwindcss":"tailwind",
    "scss":"sass",
    # Backend
    "py":"python", "python3":"python",
    "node.js":"nodejs", "node js":"nodejs", "node":"nodejs",
    "express":"nodejs", "expressjs":"nodejs",
    "fast api":"fastapi", "fast-api":"fastapi",
    "drf":"django", "django rest":"django",
    "spring boot":"spring", "springboot":"spring",
    "go":"golang", "go lang":"golang",
    ".net":"dotnet", ".net core":"dotnet", "asp.net":"dotnet",
    # ML / AI
    "ml":"machine learning",
    "dl":"deep learning", "neural networks":"deep learning",
    "torch":"pytorch",
    "tf":"tensorflow", "keras":"tensorflow",
    "sklearn":"scikit-learn", "scikit learn":"scikit-learn",
    "llm":"large language model", "llms":"large language model",
    # Cloud / DevOps
    "k8s":"kubernetes",
    "amazon web services":"aws",
    "google cloud":"gcp", "google cloud platform":"gcp",
    "microsoft azure":"azure",
    "ci/cd":"cicd",
    # Data
    "postgres":"postgresql",
    "mongo":"mongodb",
    "apache spark":"spark",
    "apache kafka":"kafka",
    # Roles
    "swe":"software engineer", "sde":"software engineer",
    "full stack":"fullstack developer", "full-stack":"fullstack developer",
    "pm":"product manager",
}


def _build_domain_dict() -> dict[str, str]:
    """
    Build {token_lower: canonical} from DOMAIN_PROTECTED + alias keys.
    Alias keys are added so SymSpell cannot corrupt them either.
    """
    d = {}
    for term in DOMAIN_PROTECTED:
        d[term.lower()] = term.lower()
    for alias in ALIAS_TO_CANONICAL:
        d[alias.lower()] = alias.lower()
    return d


DOMAIN_DICT = _build_domain_dict()


def _edit_distance(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    if abs(len(a) - len(b)) > 3: return 999
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1] + [0] * len(b)
        for j, cb in enumerate(b):
            curr[j+1] = prev[j] if ca == cb else 1 + min(prev[j], prev[j+1], curr[j])
        prev = curr
    return prev[-1]


def _similarity_ratio(a: str, b: str) -> float:
    d = _edit_distance(a, b)
    m = max(len(a), len(b))
    return 1.0 - d / m if m else 1.0


class SpellCorrector:
    """
    Two-tier corrector — runs BEFORE LLM call.

    WHY KEEP CODE-BASED CORRECTION:
    The LLM also does spell correction, but running both means:
    1. "fintch" gets fixed to "fintech" BEFORE the LLM sees it
       — the LLM gets cleaner input, produces cleaner output
    2. The code-based corrector is near-instant (<1ms)
    3. The LLM's spell correction is a safety net for anything the code misses

    This two-layer approach is more robust than either alone.
    """

    def __init__(self, freq_dict_path: Optional[str] = None):
        self._sym_enabled = False
        if SYMSPELL_AVAILABLE:
            try:
                self._sym = SymSpell(max_dictionary_edit_distance=2)
                if freq_dict_path:
                    self._sym.load_dictionary(freq_dict_path, term_index=0, count_index=1)
                else:
                    import pkg_resources
                    dp = pkg_resources.resource_filename(
                        "symspellpy", "frequency_dictionary_en_82_765.txt")
                    self._sym.load_dictionary(dp, term_index=0, count_index=1)
                self._sym_enabled = True
            except Exception as e:
                logger.warning("SymSpell load failed (%s) — code-only correction", e)

    def correct(self, text: str) -> str:
        """
        Pre-correct obvious domain typos before sending to LLM.
        Also normalises aliases so LLM receives canonical skill names.
        "reactjs develoer fintch" → "react developer fintech"
        """
        tokens = text.split()
        corrected = " ".join(self._correct_token(t) for t in tokens)
        return self._normalise_aliases(corrected)

    def _normalise_aliases(self, text: str) -> str:
        """
        Replace known aliases with canonical forms.
        Runs after spell correction so both typo-fixing and alias-collapsing
        happen before the query reaches the LLM.
        Example: "reactjs" → "react", "k8s" → "kubernetes", "node.js" → "nodejs"
        """
        t = text.lower()
        # Sort by length descending so multi-word aliases match before single words
        for alias, canonical in sorted(ALIAS_TO_CANONICAL.items(), key=lambda x: -len(x[0])):
            t = re.sub(r"\b" + re.escape(alias) + r"\b", canonical, t)
        return t

    def _correct_token(self, token: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9]", "", token).lower()
        if not clean or len(clean) <= 2 or token.isupper() or any(c.isdigit() for c in clean):
            return token

        # Tier 0: exact domain match
        if clean in DOMAIN_DICT:
            return DOMAIN_DICT[clean]

        # Tier 1: fuzzy domain match (catches "fintch" → "fintech")
        best, best_dist = None, 3
        for candidate in DOMAIN_DICT:
            if abs(len(candidate) - len(clean)) > 2: continue
            dist = _edit_distance(clean, candidate)
            if dist <= 2 and _similarity_ratio(clean, candidate) >= 0.70 and dist < best_dist:
                best_dist = dist
                best = DOMAIN_DICT[candidate]
        if best:
            logger.info("Domain spell fix: '%s' → '%s'", clean, best)
            return best

        # Tier 2: SymSpell general English
        if self._sym_enabled:
            sugg = self._sym.lookup(clean, Verbosity.CLOSEST, max_edit_distance=2)
            if sugg and sugg[0].term != clean:
                return self._recase(sugg[0].term, token)

        return token

    @staticmethod
    def _recase(corrected: str, original: str) -> str:
        if original.isupper(): return corrected.upper()
        if original and original[0].isupper(): return corrected.capitalize()
        return corrected


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — RESULT CACHE
#  Same design as before — MD5 key, LRU eviction.
#  WHY CACHE STILL MATTERS: Even at Groq speeds, 50-100ms per call adds up
#  during a demo where the same query is shown multiple times.
# ══════════════════════════════════════════════════════════════════════════════

class ResultCache:
    def __init__(self, max_size: int = 256):
        self._cache: dict[str, dict] = {}
        self._order: list[str] = []
        self.max_size = max_size
        self.hits = self.misses = 0

    def _key(self, q: str) -> str:
        return hashlib.md5(q.lower().strip().encode()).hexdigest()

    def get(self, q: str) -> Optional[dict]:
        k = self._key(q)
        if k in self._cache:
            self.hits += 1
            return self._cache[k]
        self.misses += 1
        return None

    def set(self, q: str, result: dict) -> None:
        k = self._key(q)
        if k not in self._cache:
            if len(self._cache) >= self.max_size:
                del self._cache[self._order.pop(0)]
            self._order.append(k)
        self._cache[k] = result

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{100*self.hits//total}%" if total else "0%",
            "size": len(self._cache),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — JSON PARSER (robust)
#
#  WHY A DEDICATED PARSER:
#  Even with explicit instructions, LLMs occasionally wrap JSON in markdown
#  fences (```json ... ```) or add a one-line preamble. This parser strips
#  those reliably before parsing.
# ══════════════════════════════════════════════════════════════════════════════

def _parse_llm_json(raw: str) -> dict:
    """
    Parse JSON from LLM output robustly.
    Handles: markdown fences, leading/trailing text, partial outputs.
    """
    # Strip markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    raw = raw.strip("`").strip()

    # Find the JSON object boundaries
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM output: {raw[:200]}")

    json_str = raw[start:end]
    return json.loads(json_str)


def _validate_and_fill(data: dict, original_query: str) -> dict:
    """
    Validate parsed JSON and fill missing/malformed fields with safe defaults.
    WHY: Robustness — if any field is missing, the pipeline still works.
    """
    valid_intents = {i.value for i in Intent}

    defaults = {
        "corrected_query":  original_query,
        "corrections_made": [],
        "skills":           [],
        "negated_skills":   [],
        "role":             None,
        "experience_band":  None,
        "experience_years": None,
        "domain":           None,
        "location":         None,
        "negated_location": None,
        "primary_intent":   "unknown",
        "confidence":       0.5,
        "modifiers":        [],
        "intent_reasoning": "No reasoning provided",
        "top3_scores":      {"unknown": 0.5, "role_search": 0.3, "skill_search": 0.2},
        "expanded_queries": [original_query],
        "query_strategies": {original_query: "original"},
        "exclusion_filters": {},
    }

    for key, default in defaults.items():
        if key not in data or data[key] is None and default is not None:
            data[key] = default

    # Sanitise intent values
    if data["primary_intent"] not in valid_intents:
        data["primary_intent"] = "unknown"
    data["modifiers"] = [m for m in data.get("modifiers", []) if m in valid_intents]

    # Clamp confidence
    try:
        data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
    except (TypeError, ValueError):
        data["confidence"] = 0.5

    # Ensure expanded_queries is a non-empty list
    if not isinstance(data.get("expanded_queries"), list) or not data["expanded_queries"]:
        data["expanded_queries"] = [data["corrected_query"]]

    # Ensure query_strategies is a dict
    if not isinstance(data.get("query_strategies"), dict):
        data["query_strategies"] = {q: "llm" for q in data["expanded_queries"]}

    # Ensure exclusion_filters is a dict
    if not isinstance(data.get("exclusion_filters"), dict):
        data["exclusion_filters"] = {}

    # Remove negated terms from expanded_queries (safety check)
    negated = set(data.get("negated_skills", []))
    neg_loc  = data.get("negated_location")
    filtered = []
    for q in data["expanded_queries"]:
        q_lower = q.lower()
        if any(ns.lower() in q_lower for ns in negated):
            logger.debug("Removed negated skill from query: %s", q)
            continue
        if neg_loc and neg_loc.lower() in q_lower:
            logger.debug("Removed negated location from query: %s", q)
            continue
        filtered.append(q)
    data["expanded_queries"] = filtered or [data["corrected_query"]]

    return data


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — MAIN LLM PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class LLMIntentPipeline:
    """
    LLM-powered intent detection and query expansion.

    Everything intelligence-wise (intent, expansion, negation, entities)
    is handled by ONE LLM call per query. The code handles:
    - Domain-safe spell correction (pre-LLM, near-instant)
    - JSON parsing and validation
    - Result caching (repeat queries = <1ms)

    USAGE:
        # Groq (default — free, fastest)
        pipeline = LLMIntentPipeline(api_key="gsk_...")
        result = pipeline.run("senior python dev fintch 5 years no java")
        print(result.summary())

        # Switch to Cerebras (also free):
        pipeline = LLMIntentPipeline(api_key="...", provider="cerebras")

        # Switch to OpenRouter (also free, many models):
        pipeline = LLMIntentPipeline(api_key="...", provider="openrouter")

    ENV VAR (alternative to passing api_key directly):
        export GROQ_API_KEY=gsk_...
        pipeline = LLMIntentPipeline()   # reads from env automatically
    """

    def __init__(
        self,
        api_key:  Optional[str] = None,
        provider: str = DEFAULT_PROVIDER,
        model:    Optional[str] = None,       # override model within provider
        temperature: float = 0.1,             # low = more deterministic JSON
        max_tokens:  int   = 1500,            # 8B needs slightly more room for full JSON schema
        cache_size:  int   = 256,
    ):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS.keys())}")

        self.provider_cfg = PROVIDERS[provider]
        self.model       = model or self.provider_cfg["model"]
        self.temperature = temperature
        self.max_tokens  = max_tokens

        # Resolve API key: argument → env var → error
        resolved_key = api_key or os.environ.get(self.provider_cfg["env_var"], "")
        if not resolved_key:
            raise ValueError(
                f"No API key for {provider}. Either pass api_key='...' or set "
                f"env var {self.provider_cfg['env_var']}.\n"
                f"Get a free key at: {self._signup_url(provider)}"
            )

        # WHY openai.OpenAI CLIENT FOR GROQ:
        # Groq exposes an OpenAI-compatible REST API. The openai Python SDK
        # works identically — just point base_url at Groq's endpoint.
        # This means zero learning curve and one SDK for all providers.
        self.client = OpenAI(
            api_key  = resolved_key,
            base_url = self.provider_cfg["base_url"],
        )

        self.corrector = SpellCorrector()
        self.cache     = ResultCache(max_size=cache_size)

        logger.info(
            "LLMIntentPipeline ready — provider=%s model=%s (%s)",
            provider, self.model, self.provider_cfg["description"]
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, raw_query: str) -> LLMIntentResult:
        """
        Full pipeline: raw query → LLMIntentResult.

        Steps:
          1. Code-based spell correction (domain-safe, instant)
          2. Cache check (if seen before, return immediately)
          3. LLM call — one prompt, all intelligence
          4. Parse + validate JSON
          5. Cache store + return
        """
        logger.info("─── Query: '%s'", raw_query)

        # Step 1: Pre-correct domain typos before LLM sees the query
        pre_corrected = self.corrector.correct(raw_query)
        if pre_corrected != raw_query.lower():
            logger.info("Pre-correction: '%s' → '%s'", raw_query, pre_corrected)

        # Step 2: Cache check
        cached = self.cache.get(pre_corrected)
        if cached:
            logger.info("Cache HIT")
            return self._dict_to_result(cached)

        # Step 3: LLM call
        try:
            raw_json = self._call_llm(pre_corrected)
        except Exception as e:
            logger.error("LLM API call failed: %s", e)
            return self._fallback_result(pre_corrected)

        # Step 4: Parse + validate
        try:
            data = _parse_llm_json(raw_json)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("JSON parse failed: %s\nRaw output: %s", e, raw_json[:500])
            # Return a safe fallback result
            return self._fallback_result(pre_corrected)

        data = _validate_and_fill(data, pre_corrected)

        # Step 5: Cache + return
        self.cache.set(pre_corrected, data)
        return self._dict_to_result(data)

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, query: str) -> str:
        """
        Single LLM call that does everything.

        WHY temperature=0.1:
        Low temperature = more deterministic JSON output.
        High temperature = creative but inconsistent schema adherence.
        0.1 is the sweet spot: follows JSON schema reliably, still generates
        diverse expanded queries.

        WHY max_tokens=1200:
        A full response with 10 expanded queries + hyde paragraph + all
        fields is ~900-1100 tokens on 8B. 1500 gives buffer without wasting quota.
        8B models are slightly less token-efficient than 70B on complex JSON schemas.
        """
        logger.info("LLM call → %s/%s", self.provider_cfg["base_url"].split("/")[2], self.model)

        response = self.client.chat.completions.create(
            model       = self.model,
            temperature = self.temperature,
            max_tokens  = self.max_tokens,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Analyse this recruiter search query: {query}"},
            ],
            # response_format is supported by Groq for JSON — ensures valid JSON output
            # Comment out if your provider doesn't support it
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        logger.info("LLM response: %d chars, %d tokens used",
                    len(raw), response.usage.total_tokens if response.usage else 0)
        return raw

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _dict_to_result(self, data: dict) -> LLMIntentResult:
        return LLMIntentResult(
            corrected_query   = data["corrected_query"],
            corrections_made  = data.get("corrections_made", []),
            skills            = data.get("skills", []),
            negated_skills    = data.get("negated_skills", []),
            role              = data.get("role"),
            experience_band   = data.get("experience_band"),
            experience_years  = data.get("experience_years"),
            domain            = data.get("domain"),
            location          = data.get("location"),
            negated_location  = data.get("negated_location"),
            primary_intent    = Intent(data.get("primary_intent", "unknown")),
            confidence        = float(data.get("confidence", 0.5)),
            modifiers         = [Intent(m) for m in data.get("modifiers", []) if m in {i.value for i in Intent}],
            intent_reasoning  = data.get("intent_reasoning", ""),
            top3_scores       = data.get("top3_scores", {}),
            expanded_queries  = data.get("expanded_queries", [data["corrected_query"]]),
            query_strategies  = data.get("query_strategies", {}),
            exclusion_filters = data.get("exclusion_filters", {}),
        )

    def _fallback_result(self, query: str) -> LLMIntentResult:
        """Fallback to the local Python pipeline when the LLM call or parsing fails completely."""
        try:
            logger.info("Triggering transparent fallback to local pipeline...")
            from intent_pipeline import IntentQueryPipeline
            
            # Lazy initialize the local fallback pipeline
            if not hasattr(self, "_local_fallback"):
                self._local_fallback = IntentQueryPipeline(use_sbert_fallback=True)
                
            local_res = self._local_fallback.run(query).to_dict()
            
            # Map local output format to LLM output format so it's a seamless drop-in
            return LLMIntentResult(
                corrected_query   = local_res.get("corrected", query),
                corrections_made  = [],
                skills            = local_res.get("parsed", {}).get("skills", []),
                negated_skills    = local_res.get("parsed", {}).get("negated_skills", []),
                role              = local_res.get("parsed", {}).get("role"),
                experience_band   = local_res.get("parsed", {}).get("experience_band"),
                experience_years  = local_res.get("parsed", {}).get("experience_years"),
                domain            = local_res.get("parsed", {}).get("domain"),
                location          = local_res.get("parsed", {}).get("location"),
                negated_location  = local_res.get("parsed", {}).get("negated_location"),
                primary_intent    = Intent(local_res.get("intent", {}).get("primary_intent", "unknown")),
                confidence        = local_res.get("intent", {}).get("confidence", 0.0),
                modifiers         = [],  # Avoiding list comprehension to prevent Enum mapping errors if unmapped
                intent_reasoning  = "Fallback triggered: Using local NLP pipeline due to LLM API failure.",
                top3_scores       = local_res.get("intent", {}).get("top3_scores", {}),
                expanded_queries  = local_res.get("queries", [query]),
                query_strategies  = local_res.get("strategy_map", {query: "local_fallback"}),
                exclusion_filters = local_res.get("exclusion_filters", {}),
            )
        except Exception as fallback_err:
            logger.error("Local fallback also failed: %s", fallback_err)
            return LLMIntentResult(
                corrected_query   = query,
                corrections_made  = [],
                skills            = [],
                negated_skills    = [],
                role              = None,
                experience_band   = None,
                experience_years  = None,
                domain            = None,
                location          = None,
                negated_location  = None,
                primary_intent    = Intent.UNKNOWN,
                confidence        = 0.0,
                modifiers         = [],
                intent_reasoning  = "Complete pipeline failure — both LLM and Fallback failed",
                top3_scores       = {"unknown": 1.0, "role_search": 0.0, "skill_search": 0.0},
                expanded_queries  = [query],
                query_strategies  = {query: "fallback"},
                exclusion_filters = {},
            )

    @staticmethod
    def _signup_url(provider: str) -> str:
        urls = {
            "groq":       "https://console.groq.com",
            "cerebras":   "https://cloud.cerebras.ai",
            "openrouter": "https://openrouter.ai",
            "sambanova":  "https://cloud.sambanova.ai",
        }
        return urls.get(provider, "provider website")

    def cache_stats(self) -> dict:
        return self.cache.stats()

    def switch_provider(self, provider: str, api_key: str, model: Optional[str] = None) -> None:
        """
        Switch to a different free provider at runtime.
        Useful for fallback if one provider hits rate limits during demo.

        Example:
            pipeline.switch_provider("cerebras", api_key="csk_...")
        """
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")
        cfg = PROVIDERS[provider]
        self.provider_cfg = cfg
        self.model = model or cfg["model"]
        self.client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
        logger.info("Switched to provider: %s / %s", provider, self.model)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — TEST SUITE
# ══════════════════════════════════════════════════════════════════════════════

TEST_QUERIES = [
    # (query, expected_intent, expect_negation, description)
    ("senior react and node developer",
     "multi_skill", False, "multi-skill + seniority"),

    ("Need AWS and Python engineer with 5 years in fintech",
     "multi_skill", False, "full hackathon judge query"),

    ("Pyhton develoer fintch 5 yeras",
     "experience_filter", False, "typos — spell correction test"),

    ("not remote python developer available for onsite",
     "role_search", True, "negation: location"),

    ("python developer no java experience",
     "role_search", True, "negation: skill"),

    ("best machine learning engineers for our team",
     "ranking", False, "ranking intent"),

    ("senior react and node developer",
     "multi_skill", False, "cache hit (repeat of query 1)"),

    ("who knows kubernetes and docker",
     "multi_skill", False, "multi-skill without seniority"),

    ("fintech senior data scientist 7 years",
     "experience_filter", False, "experience + domain"),
]


def run_test_suite(pipeline: LLMIntentPipeline) -> None:
    print("\n" + "═" * 80)
    print("  LLM INTENT PIPELINE — TEST SUITE")
    print("═" * 80)

    correct = 0
    for i, (query, expected_intent, expect_neg, desc) in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] {desc}")
        print(f"  Input: {query}")

        try:
            result = pipeline.run(query)
            got_intent = result.primary_intent.value
            got_neg    = bool(result.negated_skills or result.negated_location)
            ok_intent  = got_intent == expected_intent
            ok_neg     = got_neg == expect_neg
            ok         = ok_intent and ok_neg
            if ok: correct += 1
            mark = "✓" if ok else "✗"

            print(f"  {mark} Intent : {got_intent} (conf={result.confidence:.2f}) — expected {expected_intent}")
            if result.modifiers:
                print(f"    Modifiers: {[m.value for m in result.modifiers]}")
            print(f"    Top3   : {result.top3_scores}")
            print(f"    Corrected: {result.corrected_query}")
            if result.corrections_made:
                print(f"    Fixed  : {result.corrections_made}")
            print(f"    Skills : {result.skills}  |  Role: {result.role}")
            print(f"    Exp    : {result.experience_band}/{result.experience_years}yrs  |  Domain: {result.domain}")
            if result.negated_skills or result.negated_location:
                print(f"    NEGATED: skills={result.negated_skills} location={result.negated_location}")
                print(f"    Excl   : {result.exclusion_filters}")
            print(f"    Reasoning: {result.intent_reasoning}")
            print(f"    Queries ({len(result.expanded_queries)}):")
            for q in result.expanded_queries:
                strat = result.query_strategies.get(q, "?")
                print(f"      [{strat:14}] {q[:70]}")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")

    print(f"\n{'─'*80}")
    print(f"  Accuracy: {correct}/{len(TEST_QUERIES)} ({100*correct//len(TEST_QUERIES)}%)")
    print(f"  Cache: {pipeline.cache_stats()}")
    print("═" * 80)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Intent Pipeline")
    parser.add_argument("--api-key",  default=None,            help="API key (or set GROQ_API_KEY env var)")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help=f"Provider: {list(PROVIDERS.keys())}")
    parser.add_argument("--model",    default=None,            help="Override model name")
    parser.add_argument("--query",    default=None,            help="Run single query instead of test suite")
    args = parser.parse_args()

    # Resolve key from arg or env
    key = args.api_key or os.environ.get(PROVIDERS[args.provider]["env_var"])
    if not key:
        print(f"\nNo API key found. Either:")
        print(f"  export {PROVIDERS[args.provider]['env_var']}=your_key_here")
        print(f"  python llm_intent_pipeline.py --api-key your_key_here")
        print(f"\nGet a free key at: {LLMIntentPipeline._signup_url(args.provider)}")
        exit(1)

    pipeline = LLMIntentPipeline(api_key=key, provider=args.provider, model=args.model)

    if args.query:
        result = pipeline.run(args.query)
        print("\n" + result.summary())
        print("\nFull JSON:")
        print(json.dumps(result.to_dict(), indent=2))
    else:
        run_test_suite(pipeline)
