"""Explainability — narrates the retrieval+ranking decisions in layman terms.

ENTRY POINT:  explain(context: dict) -> dict

The teammate working on this module: use the matched_terms / matched_paths /
signals already carried by the reranked candidates (hybrid retriever supplies
lexical + semantic signals, kg retriever supplies graph paths) and ask an LLM
to narrate. No need to re-compute anything — the facts are all in `context`.

INPUT shape:
{
    "query": "<raw query>",
    "intent": {...},
    "reranked": {"ranked": [...], "method": "..."},
    "hybrid": {...},
    "kg":     {...},
    "tagged_candidates": [...]
}

Expected output shape:
{
    "per_candidate": {
        "<candidate_id>": {
            "summary": "<short human-readable line>",
            "reasons": ["matched 'aws' and 'flask' via BM25",
                         "KG path: Backend Developer -[REQUIRES]-> Python -[HAS_SKILL]- <id>",
                         "7+ years exp matches 'senior' band"]
        },
        ...
    },
    "process_summary": "<one paragraph: what happened end-to-end, in plain English>",
    "intent_summary":  "<what the system understood the user wanted>"
}
"""

from __future__ import annotations


def explain(context: dict) -> dict:
    raise NotImplementedError(
        "modules/explainability: drop your LLM narration script here and "
        "re-export its main function as `explain(context: dict) -> dict`."
    )
