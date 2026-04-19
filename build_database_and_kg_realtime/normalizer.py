"""
Canonical Normalizer — Vocabulary Standardization Layer for Component A

Handles:
  1. Abbreviation expansion   — "ml" → "machine learning", "js" → "javascript"
  2. Entity standardization   — "python programming" → "python", "data-science" → "data science"
  3. Synonym group awareness  — "cloud" ↔ ["aws", "azure", "gcp"]

Applied at BOTH index-time and query-time to ensure consistency.
Designed to be extensible — add new mappings as the dataset evolves.
"""

import re
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# ABBREVIATION → CANONICAL FORM
# Bidirectional: we store abbrev→full AND full→full for consistency.
# ─────────────────────────────────────────────────────────────────────────────
ABBREVIATION_MAP: dict[str, str] = {
    # AI / ML
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "rl": "reinforcement learning",
    "genai": "generative ai",
    "llm": "large language model",

    # Programming languages
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "rb": "ruby",
    "cpp": "c++",
    "c#": "csharp",
    "vb": "visual basic",

    # Web / Frameworks
    "rn": "react native",
    "ng": "angular",
    "vue": "vuejs",
    "express": "expressjs",
    "dj": "django",
    "fl": "flask",

    # Cloud / DevOps
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "k8s": "kubernetes",
    "tf": "terraform",
    "ci/cd": "continuous integration continuous deployment",
    "cicd": "continuous integration continuous deployment",

    # Data
    "sql": "sql",
    "nosql": "nosql",
    "etl": "extract transform load",
    "bi": "business intelligence",
    "dw": "data warehouse",
    "ds": "data science",

    # Security
    "vapt": "vulnerability assessment penetration testing",
    "soc": "security operations center",
    "iam": "identity access management",

    # Business / Management
    "pm": "project management",
    "ba": "business analysis",
    "qa": "quality assurance",
    "ui": "user interface",
    "ux": "user experience",
    "erp": "enterprise resource planning",
    "crm": "customer relationship management",
    "hr": "human resources",
    "sap": "sap",
    "rpa": "robotic process automation",

    # Certifications / Compliance
    "pmp": "project management professional",
    "csm": "certified scrum master",
    "fda": "food and drug administration",
    "gdpr": "general data protection regulation",
    "hipaa": "health insurance portability accountability act",
}

# ─────────────────────────────────────────────────────────────────────────────
# ENTITY STANDARDIZATION
# Strips common suffixes/prefixes that create duplicate entities.
# "python programming" and "python" should be the SAME node in a KG.
# ─────────────────────────────────────────────────────────────────────────────
ENTITY_STRIP_SUFFIXES = [
    " programming", " language", " development", " framework",
    " library", " platform", " tool", " tools", " technologies",
    " technology", " skills", " skill", " certification",
    " certified", " developer", " engineer", " engineering",
]

ENTITY_ALIASES: dict[str, str] = {
    "react.js": "react",
    "reactjs": "react",
    "react js": "react",
    "node.js": "nodejs",
    "node js": "nodejs",
    "vue.js": "vuejs",
    "vue js": "vuejs",
    "express.js": "expressjs",
    "express js": "expressjs",
    "next.js": "nextjs",
    "next js": "nextjs",
    "angular.js": "angularjs",
    "angular js": "angularjs",
    "scikit-learn": "scikit learn",
    "sci-kit learn": "scikit learn",
    "power bi": "powerbi",
    "power-bi": "powerbi",
    "machine-learning": "machine learning",
    "deep-learning": "deep learning",
    "data-science": "data science",
    "data science": "data science",
    "devops": "devops",
    "dev ops": "devops",
    "dev-ops": "devops",
    "front end": "frontend",
    "front-end": "frontend",
    "back end": "backend",
    "back-end": "backend",
    "full stack": "fullstack",
    "full-stack": "fullstack",
    "micro services": "microservices",
    "micro-services": "microservices",
    "postgres": "postgresql",
    "mongo db": "mongodb",
    "mongo": "mongodb",
}

# ─────────────────────────────────────────────────────────────────────────────
# SYNONYM GROUPS
# Each group represents concepts that should be linked in the Knowledge Graph.
# The first entry is the "canonical" form; others are synonyms.
# Used for query expansion (Component C) and graph edges (Component E).
# ─────────────────────────────────────────────────────────────────────────────
SYNONYM_GROUPS: list[list[str]] = [
    ["machine learning", "ml", "statistical learning"],
    ["artificial intelligence", "ai"],
    ["deep learning", "dl", "neural networks"],
    ["natural language processing", "nlp", "text mining", "text analytics"],
    ["computer vision", "cv", "image recognition", "image processing"],
    ["data science", "data analytics", "data analysis"],
    ["cloud computing", "cloud", "amazon web services", "azure", "google cloud platform"],
    ["devops", "site reliability engineering", "sre", "infrastructure"],
    ["kubernetes", "k8s", "container orchestration"],
    ["docker", "containerization", "containers"],
    ["react", "reactjs"],
    ["python", "py"],
    ["javascript", "js"],
    ["typescript", "ts"],
    ["sql", "structured query language"],
    ["nosql", "non relational database"],
    ["agile", "scrum", "kanban"],
    ["project management", "pm", "project management professional"],
    ["cybersecurity", "information security", "infosec", "cyber security"],
    ["vulnerability assessment penetration testing", "vapt", "penetration testing", "pen testing"],
    ["robotic process automation", "rpa"],
    ["enterprise resource planning", "erp", "sap"],
    ["business intelligence", "bi", "reporting", "dashboards"],
    ["extract transform load", "etl", "data pipeline", "data engineering"],
    ["frontend", "front end development", "ui development"],
    ["backend", "back end development", "server side"],
    ["fullstack", "full stack development"],
    ["microservices", "micro services architecture"],
    ["rest api", "restful api", "api development"],
    ["postgresql", "postgres"],
    ["mongodb", "mongo"],
]


class CanonicalNormalizer:
    """
    Central normalization engine for the retrieval system.

    Usage:
        normalizer = CanonicalNormalizer()
        normalizer.normalize_skill("Python Programming")  → "python"
        normalizer.normalize_skill("ML")                  → "machine learning"
        normalizer.expand_query_terms(["ml", "cloud"])    → ["machine learning", "cloud", "aws", "azure", "gcp"]
        normalizer.get_synonyms("kubernetes")             → ["k8s", "container orchestration"]
    """

    def __init__(self):
        self._abbrev_map = {k.lower(): v.lower() for k, v in ABBREVIATION_MAP.items()}
        self._entity_aliases = {k.lower(): v.lower() for k, v in ENTITY_ALIASES.items()}
        self._strip_suffixes = [s.lower() for s in ENTITY_STRIP_SUFFIXES]

        # Build synonym lookup: term → canonical form
        self._synonym_to_canonical: dict[str, str] = {}
        # Build canonical → all synonyms
        self._canonical_to_synonyms: dict[str, list[str]] = {}

        for group in SYNONYM_GROUPS:
            canonical = group[0].lower()
            self._canonical_to_synonyms[canonical] = [s.lower() for s in group]
            for term in group:
                self._synonym_to_canonical[term.lower()] = canonical

    def normalize_skill(self, skill: str) -> str:
        """
        Normalize a single skill name to its canonical form.

        Pipeline:
          1. Lowercase + strip whitespace
          2. Apply entity alias mapping (react.js → react)
          3. Expand abbreviation (ml → machine learning)
          4. Strip common suffixes (python programming → python)
          5. Final whitespace cleanup
        """
        if not skill or not isinstance(skill, str):
            return ""

        s = skill.strip().lower()

        # Remove special characters but keep meaningful ones
        s = re.sub(r"['\"]", "", s)
        s = re.sub(r"\s+", " ", s).strip()

        # Entity alias
        if s in self._entity_aliases:
            s = self._entity_aliases[s]

        # Abbreviation expansion
        if s in self._abbrev_map:
            s = self._abbrev_map[s]

        # Strip common suffixes
        for suffix in self._strip_suffixes:
            if s.endswith(suffix) and len(s) > len(suffix):
                s = s[: -len(suffix)].strip()
                break  # only strip one suffix
                
        # Second abbreviation expansion (catches things like "ml" after " engineer" was stripped)
        if s in self._abbrev_map:
            s = self._abbrev_map[s]

        return s.strip()

    def normalize_token(self, token: str) -> str:
        """
        Lighter normalization for BM25 tokens.
        Only applies abbreviation expansion (not suffix stripping).
        """
        t = token.strip().lower()
        return self._abbrev_map.get(t, t)

    def normalize_text(self, text: str) -> str:
        """
        Normalize a full text string by expanding abbreviations in-place.
        Used for BM25 text and semantic text construction.
        """
        if not text:
            return ""
        words = text.lower().split()
        normalized = []
        for w in words:
            clean = re.sub(r"[^a-z0-9+#]", "", w)
            expanded = self._abbrev_map.get(clean, clean)
            # If abbreviation expanded to multi-word, split it
            normalized.extend(expanded.split())
        return " ".join(normalized)

    def get_canonical(self, term: str) -> str:
        """Get the canonical synonym form, or return the term unchanged."""
        return self._synonym_to_canonical.get(term.lower(), term.lower())

    def get_synonyms(self, term: str) -> list[str]:
        """
        Get all synonyms for a term (excluding the term itself).
        Returns empty list if no synonym group exists.
        """
        canonical = self._synonym_to_canonical.get(term.lower())
        if canonical is None:
            return []
        return [s for s in self._canonical_to_synonyms.get(canonical, []) if s != term.lower()]

    def expand_query_terms(self, terms: list[str]) -> list[str]:
        """
        Expand a list of query terms with their synonyms.
        Used by Component C (Query Expansion).

        Returns deduplicated list: original terms + synonym expansions.
        """
        expanded = list(terms)
        seen = set(t.lower() for t in terms)
        for term in terms:
            for syn in self.get_synonyms(term):
                if syn not in seen:
                    expanded.append(syn)
                    seen.add(syn)
        return expanded

    def get_all_synonym_groups(self) -> list[list[str]]:
        """Return all synonym groups — used by Component E for graph edges."""
        return [[self._canonical_to_synonyms[c][0]] + [
            s for s in self._canonical_to_synonyms[c] if s != self._canonical_to_synonyms[c][0]
        ] for c in self._canonical_to_synonyms]


# Module-level singleton for convenience
_default_normalizer: Optional[CanonicalNormalizer] = None


def get_normalizer() -> CanonicalNormalizer:
    """Get or create the default normalizer singleton."""
    global _default_normalizer
    if _default_normalizer is None:
        _default_normalizer = CanonicalNormalizer()
    return _default_normalizer
