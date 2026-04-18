# Watchers

Each watcher monitors `./data/profiles.csv` (or its corresponding index file)
and keeps its backing store in sync when the CSV changes. They run as
independent long-lived processes — start whichever you need in a separate
terminal.

## Files to drop here (3)

| filename            | what it keeps in sync                               | checkpoint file         |
|---------------------|-----------------------------------------------------|-------------------------|
| `kg_watcher.py`     | Neo4j graph (the one you already have)              | `kg_checkpoint.json`    |
| `bm25_watcher.py`   | BM25 index pkl at `$BM25_INDEX` (lexical)           | `bm25_checkpoint.json`  |
| `vector_watcher.py` | SBERT + FAISS index pkl at `$FAISS_INDEX` (semantic)| `vector_checkpoint.json`|

> Naming convention: `<store>_watcher.py` — the store name is what it writes to,
> not what it reads from. All three read the same `profiles.csv`.

## Shared pattern (same as `kg_watcher.py`)

1. SHA-256 hash each row — skip unchanged, process new/updated, delete removed.
2. Use `watchdog.Observer` on the `data/` directory with a debounce.
3. Maintain a checkpoint file so restarts don't re-process everything.

## Running

```bash
# in three separate terminals
python watchers/kg_watcher.py
python watchers/bm25_watcher.py
python watchers/vector_watcher.py
```

Or wrap them in a single process-manager script later if needed.
