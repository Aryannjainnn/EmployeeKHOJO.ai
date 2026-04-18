# Intent-Aware Hybrid Retrieval — Integration

Orchestration layer for the HR candidate shortlisting system. Wires 5 teammate-owned
modules behind a FastAPI + HTML frontend.

## Pipeline

```
  query (HTML input)
    │
    ▼
  [intent_processor.process]        ── expands query → JSON with many queries
    │
    ├────────────┬─────────────┐    RunnableParallel
    ▼            ▼             ▼
 [hybrid]     [kg]         (passthrough)
 BM25+FAISS   Neo4j
    │            │
    └────┬───────┘
         ▼
  tag_and_combine                   ── each candidate gets source: hybrid|kg|both
         │
         ▼
  [reranker.rerank]                 ── final ranking (module-owned fusion)
         │
         ▼
  [explainability.explain]          ── LLM narrates per-candidate + process
         │
         ▼
  JSON → frontend
```

## Folders

- `orchestrator/` — LangChain pipeline (`chain.py`) and shared types.
- `modules/` — 5 teammate modules, each with one entry function:
  - `intent_processor.process(query: str) -> dict`
  - `hybrid_retriever.retrieve(queries: list[str]) -> dict`
  - `kg_retriever.retrieve(queries: list[str]) -> dict`
  - `reranker.rerank(combined: dict) -> dict`
  - `explainability.explain(context: dict) -> dict`
- `watchers/` — 3 watchers (`kg_`, `bm25_`, `vector_`). See `watchers/README.md`.
- `frontend/index.html` — search UI, hits `POST /search`.
- `data/` — `profiles.csv`, `bm25_index.pkl`, `faiss_index.pkl`.

## Setup

```bash
cp .env.example .env       # fill NEO4J_* values
pip install -r requirements.txt
python main.py             # serves http://localhost:8000
```

Each module currently raises `NotImplementedError`. Drop the real scripts in,
re-export the main function under the expected name, and the pipeline runs.

## How parallelism works

`RunnableParallel` in `orchestrator/chain.py` runs the hybrid and KG branches
on a `ThreadPoolExecutor` — they execute concurrently, not sequentially.
Use `pipeline.ainvoke(...)` for async if you hit the GIL on CPU-bound code.
