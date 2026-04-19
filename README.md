# Nexus — Intent-Aware Hybrid Retrieval System

> **Team BackPropBandits** · Built for the *An Intent‑Aware and Explainable Hybrid Retrieval System* problem statement

---

## What Is This?

Nexus is an end-to-end intelligent search and retrieval system built for HR candidate shortlisting. Given a natural-language recruiter query like *"senior Python developer fintech 5 years no Java"*, the system identifies intent, expands the query, retrieves relevant candidates through multiple complementary signals, re-ranks them intelligently, and explains every result in plain English — all in a single pipeline.

---

## The Problem

Modern search struggles when queries are ambiguous, complex, or context-dependent. Traditional keyword search (BM25, TF-IDF) misses semantic meaning. Pure neural retrieval lacks precision and explainability. Neither alone handles:

- Typos and abbreviations ("fintch", "Pyhton", "k8s")
- Negation ("no Java", "not remote")
- Multi-dimensional intent ("senior React *and* Node developer in fintech, 5 years")
- Relationship traversal ("Python developer" → should surface Django/FastAPI experts too)
- Dynamic, evolving datasets without requiring full re-indexing

The problem statement demanded a system that combines **lexical, semantic, and relational retrieval** while remaining **explainable** and **schema-agnostic**.

---

## Goals

- Retrieve at least 95% of relevant candidates for representative queries
- Understand and infer user intent, automatically reformulating queries when needed
- Combine BM25, dense vector search, and knowledge graph traversal
- Re-rank results using intent-aware signal fusion
- Provide clear, human-readable explanations for every result
- Handle dynamic schemas and incremental dataset updates without full re-indexing

---

## System Architecture

The pipeline has five sequential stages, with hybrid and KG retrieval running in parallel:

```
  Query (HTML / API)
       │
       ▼
  ① Intent Processor ─── spell correction → entity extraction →
                                   intent classification → query expansion
       │
       ├─────────────────────────────────────────────────┐
       ▼                                                 ▼
  ② Hybrid Retriever           ②  KG Retriever
     BM25 + FAISS/SBERT             Neo4j 4-hop traversal
     (lexical + semantic)           (relational + domain)
       │                                                 │
       └─────────────────────────┬───────────────────────┘
                                 ▼
                    ③ Tag & Combine (union pool)
                                 │
                                 ▼
                    ④ Re-ranker (intent-aware fusion)
                                 │
                                 ▼
                    ⑤ Explainability (LLM narration)
                                 │
                                 ▼
                          Nexus Search UI
```

---

## How Each Component Works

### ① Intent Processor (`modules/intent_processor/`)

This is the "brain" of the pipeline. It transforms a raw recruiter query into a rich structured object consumed by all downstream components.

**Spell Correction (two-tier):** A domain-specific dictionary (2500+ technical terms, roles, domains) is checked before general SymSpell correction — preventing "fintech" from becoming "biotech" and "kubernetes" from becoming a mangled English word.

**Entity Extraction:** Structured parsing pulls out skills, role, experience band/years, domain, location, and negated terms using a curated vocabulary of 5000+ skills, synonyms, and role titles (`modules/intent_processor/vocab.py`).

**Intent Detection (three-level):**
- L1: Entity heuristics — if 2+ skills detected → `multi_skill` (< 1ms, ~92% accurate)
- L2: Regex rule patterns — fast matching for common recruiter phrasings
- L3: Zero-shot NLI — `facebook/bart-large-mnli` classifies across 8 intent classes using independent probability scoring per intent (multi-label, so a query can have a primary + modifiers simultaneously)

**Query Expansion (5 strategies):** Original query, synonym variants, static KG skill relations, intent templates (role/skill/experience/domain-specific phrasings), and HyDE (hypothetical document embedding via LLM).

**LLM Path (optional):** If a Groq/Cerebras/OpenRouter API key is configured, the entire spell correction + entity extraction + intent + expansion happens in a single structured JSON LLM call, with a graceful fallback to the local pipeline.

---

### ② Hybrid Retriever (`modules/hybrid_retriever/`)

Handles lexical and semantic retrieval against a pre-built index.

**Index Building (`build_database_and_kg_realtime/indexer.py`):**
- Schema-agnostic: discovers text, skill, numeric, and role columns at runtime — new columns just work
- BM25+ index: pure-Python implementation with k1=1.5, b=0.75, δ=0.5; core skills are boosted 3× in TF during indexing
- Dense index: BGE-base-en-v1.5 (768-dim) via Sentence-BERT + FAISS IndexFlatIP; falls back to TF-IDF + TruncatedSVD (LSA) if transformers unavailable
- Skill inverted index: for direct skill lookup and explainability
- Incremental updates: per-row SHA-256 hashing detects new/changed/deleted rows, only re-processes the delta — no full re-indexing needed
- A watchdog process (`watch_index.py`) monitors `profiles.csv` and triggers incremental updates automatically on file save

**Query-Time Retrieval:**
All expanded query variants are scored in parallel; BM25 scores are aggregated with per-strategy weights (original/synonym > template). Semantic encoding uses a BGE query prefix (asymmetric: profiles indexed without prefix, queries encoded with prefix). Reciprocal Rank Fusion (RRF, k=60) fuses lexical and semantic rank lists with α=0.40 BM25, α=0.60 dense. Hard filters remove negated skills and experience mismatches before final scoring.

---

### ② KG Retriever (`modules/kg_retriever/`)

Traverses a Neo4j knowledge graph built from the same candidate profiles. The graph contains nodes for Candidates, Skills, Roles, Domains, and SubDomains, with relationships like `HAS_SKILL`, `SUITABLE_FOR`, `REQUIRES`, `BELONGS_TO`, `RELATED_TO`, and `HAS_SUBDOMAIN`.

**Graph Construction (`build_database_and_kg_realtime/kg_watcher.py`):** A 3-pass pipeline runs each profile through a local Qwen2.5:7B model (via Ollama), extracting structured KG entities, validating them with Pydantic models, fuzzy-matching against existing canonical entities (TheFuzz, threshold 90%), and pushing to Neo4j. Reference `.md` files for known domains, subdomains, roles, and skills are updated automatically as new entities are discovered. A watchdog process mirrors the CSV watcher for real-time KG sync.

**4-Hop Traversal at Query Time:** For each query term, the retriever runs five parallel Cypher queries: hop-1 direct skill match (`HAS_SKILL`), hop-2 related skill (`RELATED_TO/BELONGS_TO`), hop-2 domain/subdomain, hop-1 role match (`SUITABLE_FOR`), hop-3 and hop-4 deep chains. Scores decay by 0.5× per hop. All results are merged into a union with match reasons tracking which node was matched, via which relationship, at which hop depth.

---

### ③ Tag & Combine

A lightweight merge step that takes the union of hybrid and KG candidate pools, tagging each candidate as `hybrid_only`, `kg_only`, or `both`. Candidates present in both pools get enriched metadata from the richer KG side.

---

### ④ Re-ranker (`modules/reranker/`)

Intent-aware score fusion: `F = α·H + β·K`, where weights are looked up from an intent table (e.g., `domain_search` → α=0.30, β=0.70 since graph domain edges dominate; `experience_filter` → α=0.60, β=0.40 since numeric/text filtering is a hybrid strength). Weights are linearly dampened toward 0.5/0.5 as intent confidence decreases. Modifier bonuses (+0.05 for experience match, +0.03 for seniority) and negated-skill penalties (−0.10 per negated skill found in a candidate's core skills) are applied after fusion.

---

### ⑤ Explainability (`modules/explainability/`)

Each candidate's retrieval signals (BM25 raw score, semantic cosine, RRF rank, KG match reasons) are interpreted into human-readable tiers ("STRONG keyword match", "outstanding conceptual alignment") and passed to a local Llama 3.1 8B model (via Ollama) to produce a 3-paragraph hiring recommendation. The UI surfaces these as "Why Selected", "Signal Analysis", and "Keywords" tabs per candidate.

---

### Frontend (`frontend/`)

A React 18 single-page app (`NexusSearch.jsx`) loaded via Babel standalone — no bundler required. Features a dark-mode search interface with animated candidate cards showing score bars, tabbed explanations, and KG-derived signal breakdowns. Hits `GET /search?q=...` and falls back to demo data if the backend is unavailable.

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js (optional, for frontend development)
- Neo4j instance (local or AuraDB)
- Ollama with `qwen2.5:7b`  pulled (for KG extraction)
- Groq/Cerebras API key for the LLM intent pipeline for query expansion and explainability

### 1. Clone and install dependencies

```bash
git clone https://github.com/Aryannjainnn/hackathon_backprop_bandits.git
cd path/to/hackathon_backprop_bandits
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Neo4j connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Optional: LLM provider for intent pipeline
INTENT_LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...

# Index location
DATA_DIR=./data
```

### 3. Prepare your data

Place your candidate profiles CSV at `data/profiles.csv`. The indexer is schema-agnostic — it will discover skill, role, summary, and numeric columns automatically based on column name keywords. At minimum, include columns for candidate ID, name, skills (core/secondary/soft), years of experience, potential roles, and a skill summary.

### 4. Build the search index

```bash
cd build_database_and_kg_realtime
python build_index.py
```

This produces BM25, FAISS, and skill inverted index artifacts under `data/`. For a dataset of ~1000 candidates this takes 2–5 minutes depending on GPU availability.

### 5. Build the knowledge graph

Ensure Ollama is running with `qwen2.5:7b`:

```bash
ollama pull qwen2.5:7b
cd build_database_and_kg_realtime
python kg_watcher.py
```

On first run this processes every row in `profiles.csv`. Subsequent runs only process changed rows. Leave it running to sync changes automatically.

### 6. Start the API server

```bash
python main.py
# Serves at http://localhost:8000
```

The Nexus search UI is available at `http://localhost:8000`. The API endpoints are:

| Endpoint | Description |
|---|---|
| `GET /search?q=<query>&k=10&mode=hybrid` | Run full pipeline, return ranked results |
| `POST /search` `{"query": "..."}` | Same via JSON body |
| `POST /explain` `{"row_data": {...}}` | Generate LLM explanation for a candidate row |
| `GET /health` | Liveness probe |

### 7. Optional: enable real-time index updates

To automatically update the search index when `profiles.csv` is edited:

```bash
cd build_database_and_kg_realtime
python watch_index.py   # BM25 + FAISS watcher
python kg_watcher.py    # Neo4j KG watcher (separate terminal)
```

### 8. Run diagnostics

```bash
python test_intent_processor.py   # intent pipeline health check
python test_hybrid_module.py      # BM25 + FAISS health check
python test_kg_retriever.py       # Neo4j connectivity + schema check
python test_integration.py        # full end-to-end smoke test
```

---

## Evaluation

The system is evaluated using standard IR metrics (Precision@K, Recall@K, MAP, nDCG) and the RAGAS framework (Context Precision, Context Recall, Answer Relevance, Faithfulness). The hybrid retrieval stage is compared against BM25-only and semantic-only baselines to demonstrate measurable gains from fusion.

---

## Project Structure

```
.
├── main.py                          # FastAPI entry point
├── orchestrator/
│   ├── chain.py                     # LangChain pipeline (RunnableParallel)
│   └── schemas.py                   # Shared Pydantic types
├── modules/
│   ├── intent_processor/
│   │   ├── intent_pipeline.py       # Local NLP pipeline (BART + SBERT)
│   │   ├── llm_intent_pipeline.py   # LLM-powered pipeline (Groq/Cerebras/etc.)
│   │   ├── vocab.py                 # 5000+ skills, synonyms, domains, roles
│   │   └── spell_correct.py        # Domain-aware two-tier spell corrector
│   ├── hybrid_retriever/
│   │   └── retriever.py            # BM25 + FAISS retrieval engine
│   ├── kg_retriever/
│   │   └── retrieve.py             # Neo4j 4-hop graph retrieval
│   ├── reranker/
│   │   └── rerank.py               # Intent-aware score fusion
│   └── explainability/
│       └── explain.py              # LLM hiring recommendation generator
├── build_database_and_kg_realtime/
│   ├── indexer.py                  # Schema-agnostic BM25 + FAISS index builder
│   ├── normalizer.py               # Canonical skill/abbreviation normalizer
│   ├── kg_watcher.py               # Real-time Neo4j sync via Qwen
│   ├── build_index.py              # One-shot index build script
│   ├── update_index.py             # Interactive CLI for incremental updates
│   └── watch_index.py              # File watcher for BM25/FAISS auto-update
├── frontend/
│   ├── index.html                  # Shell HTML (loads React via Babel)
│   └── NexusSearch.jsx             # Full search UI component
├── data/                           # Index artifacts (gitignored)
│   ├── profiles.csv
│   ├── bm25.pkl
│   ├── dense.pkl
│   ├── dense_matrix.npy
│   └── metadata.pkl
├── requirements.txt
└── .env.example
```

---

## Key Design Decisions

**Schema-agnostic indexing** — no column names are hardcoded. The indexer introspects DataFrame column names at runtime using keyword heuristics, so adding new columns to the dataset requires zero code changes.

**Incremental updates without re-indexing** — per-row SHA-256 hashing detects exactly which rows changed. Only those rows are removed and re-added to BM25, FAISS, and the skill index. The KG watcher does the same for Neo4j.

**Multi-label intent detection** — the system uses independent NLI probabilities per intent class rather than softmax, correctly assigning a primary intent *and* modifiers simultaneously (e.g., `multi_skill` + `experience_filter` + `domain_search` for a single compound query).

**Domain-first spell correction** — SymSpell general English correction only runs after domain vocabulary matching fails, preventing technical terms from being corrupted into common English words.

**Asymmetric BGE encoding** — profiles are indexed without a task prefix; queries are encoded with `"Represent this sentence for searching relevant passages: "`. This matches BGE's training setup and significantly improves retrieval precision over symmetric encoding.

**Confidence-dampened weight fusion** — the re-ranker linearly interpolates between intent-specific weights and 50/50 proportional to intent confidence, so uncertain classifications gracefully fall back to equal-weight fusion rather than applying potentially wrong strong weights.

## How parallelism works

`RunnableParallel` in `orchestrator/chain.py` runs the hybrid and KG branches
on a `ThreadPoolExecutor` — they execute concurrently, not sequentially.
Use `pipeline.ainvoke(...)` for async if you hit the GIL on CPU-bound code.
