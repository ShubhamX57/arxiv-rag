"""Pure-function retrieval metrics.

Definitions (1-indexed ranks throughout):
- recall@k: 1 if any gold chunk appears in top-k, else 0. Averaged over queries.
- MRR@k:    1 / rank_of_first_gold if a gold appears in top-k, else 0.
- nDCG@k:   sum over relevant hits of (1 / log2(rank + 1)), normalized by the
            ideal DCG (which assumes all gold chunks were retrieved at the
            very top). All gold chunks are treated as equally relevant (binary
            relevance), which is standard for retrieval where we don't have
            graded relevance judgments.

All metrics return values in [0, 1] for a single query, and we average across
queries to get the dataset-level number. nDCG is a "ranked" metric (cares
where in the top-k the hit appears), while Recall is unranked.

These functions take **lists of chunk_ids** for both retrieved and gold, so
they don't depend on the retriever's score type or any embedding. That makes
them trivial to test and easy to reuse across configs.
"""

from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Fraction of queries where ANY gold chunk is in top-k.

    For a single query: 1.0 if at least one gold appears, else 0.0.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not gold:
        return 0.0
    top_k = set(retrieved[:k])
    return 1.0 if any(g in top_k for g in gold) else 0.0


def reciprocal_rank_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """1 / rank of first gold in top-k, or 0 if none found.

    Rewards retrievers that put the relevant chunk EARLY. A hit at rank 1
    contributes 1.0; rank 2 → 0.5; rank 10 → 0.1.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not gold:
        return 0.0
    gold_set = set(gold)
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


def dcg(retrieved: list[str], gold: list[str], k: int) -> float:
    """Discounted Cumulative Gain at k, binary relevance.

    Each gold chunk found at rank r contributes 1 / log2(r + 1).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    gold_set = set(gold)
    total = 0.0
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in gold_set:
            total += 1.0 / math.log2(rank + 1)
    return total


def ideal_dcg(num_gold: int, k: int) -> float:
    """The best possible DCG@k: all gold chunks at ranks 1..min(num_gold, k)."""
    n = min(num_gold, k)
    return sum(1.0 / math.log2(r + 1) for r in range(1, n + 1))


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """Normalized DCG@k. 1.0 if all gold chunks are at the top, 0 if none in top-k."""
    if not gold:
        return 0.0
    idcg = ideal_dcg(len(gold), k)
    if idcg == 0:
        return 0.0
    return dcg(retrieved, gold, k) / idcg


__all__ = [
    "dcg",
    "ideal_dcg",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
]
