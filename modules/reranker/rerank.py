"""Re-ranker — takes the combined tagged output and produces the final ranking.

ENTRY POINT:  rerank(combined: dict) -> dict

The orchestrator calls this AFTER tagging each candidate with its source
(hybrid | kg | both). The teammate working on this module decides the fusion
strategy (RRF, cross-encoder, weighted, learned-to-rank, etc.).

INPUT shape:
{
    "query": "<raw query>",
    "intent": {...},
    "hybrid": {...raw hybrid retriever output...},
    "kg":     {...raw kg retriever output...},
    "tagged_candidates": [
        {"id", "score", "source": "hybrid|kg|both", "hybrid": {...}|None, "kg": {...}|None},
        ...
    ]
}

Expected output shape (extend as needed):
{
    "ranked": [
        {
            "id": "<candidate_id>",
            "final_score": <float>,
            "rank": 1,
            "source": "hybrid|kg|both",
            "signals": {"hybrid": <float>, "kg": <float>, ...},
            ...any extra fields the explainer wants...
        },
        ...
    ],
    "method": "<name of the fusion technique used>"
}
"""

from __future__ import annotations


def rerank(combined: dict) -> dict:
    raise NotImplementedError(
        "modules/reranker: drop your re-rank script here and re-export its main "
        "function as `rerank(combined: dict) -> dict`."
    )
