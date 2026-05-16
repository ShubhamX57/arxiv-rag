"""Tests for retrieval metrics. Pure functions — no fixtures, no fakes."""

from __future__ import annotations

import math

import pytest

from arxiv_rag.evals.metrics import (
    dcg,
    ideal_dcg,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

# ----------------------- recall@k -----------------------


def test_recall_hit_at_rank_one() -> None:
    assert recall_at_k(["a", "b", "c"], ["a"], k=3) == 1.0


def test_recall_hit_at_rank_three() -> None:
    assert recall_at_k(["x", "y", "a"], ["a"], k=3) == 1.0


def test_recall_no_hit_in_top_k() -> None:
    assert recall_at_k(["x", "y", "z"], ["a"], k=3) == 0.0


def test_recall_gold_below_k_doesnt_count() -> None:
    # gold at rank 4, k=3 → miss
    assert recall_at_k(["x", "y", "z", "a"], ["a"], k=3) == 0.0


def test_recall_multiple_gold_any_match() -> None:
    # any of multiple gold chunks counts as a hit
    assert recall_at_k(["x", "g2", "y"], ["g1", "g2"], k=3) == 1.0


def test_recall_empty_gold_is_zero() -> None:
    # no gold means there's nothing to recall — defined as 0
    assert recall_at_k(["a", "b", "c"], [], k=3) == 0.0


def test_recall_empty_retrieved_is_zero() -> None:
    assert recall_at_k([], ["a"], k=3) == 0.0


def test_recall_invalid_k_raises() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], ["a"], k=0)
    with pytest.raises(ValueError):
        recall_at_k(["a"], ["a"], k=-1)


# ----------------------- MRR@k -----------------------


def test_mrr_rank_one() -> None:
    assert reciprocal_rank_at_k(["a", "b"], ["a"], k=5) == 1.0


def test_mrr_rank_two() -> None:
    assert reciprocal_rank_at_k(["x", "a"], ["a"], k=5) == 0.5


def test_mrr_rank_four() -> None:
    assert reciprocal_rank_at_k(["x", "y", "z", "a"], ["a"], k=5) == pytest.approx(0.25)


def test_mrr_no_hit() -> None:
    assert reciprocal_rank_at_k(["x", "y", "z"], ["a"], k=5) == 0.0


def test_mrr_uses_first_gold_match() -> None:
    # Two gold chunks; first one appears at rank 2, second at rank 4.
    # MRR should be 1/2, not 1/4 or the sum.
    rr = reciprocal_rank_at_k(["x", "g1", "y", "g2"], ["g1", "g2"], k=5)
    assert rr == 0.5


def test_mrr_gold_beyond_k_is_zero() -> None:
    assert reciprocal_rank_at_k(["x", "y", "z", "a"], ["a"], k=3) == 0.0


# ----------------------- DCG / nDCG -----------------------


def test_dcg_single_hit_at_rank_one() -> None:
    # 1 / log2(2) = 1.0
    assert dcg(["a"], ["a"], k=10) == 1.0


def test_dcg_single_hit_at_rank_three() -> None:
    # 1 / log2(4) = 0.5
    assert dcg(["x", "y", "a"], ["a"], k=10) == pytest.approx(0.5)


def test_dcg_two_hits_sum() -> None:
    # gold at ranks 1 and 3: 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    assert dcg(["g1", "x", "g2"], ["g1", "g2"], k=10) == pytest.approx(1.5)


def test_ideal_dcg_one_gold() -> None:
    # one gold → ideal puts it at rank 1 → DCG = 1.0
    assert ideal_dcg(1, k=10) == 1.0


def test_ideal_dcg_two_gold() -> None:
    # two gold at ranks 1,2 → 1 + 1/log2(3)
    expected = 1.0 + 1.0 / math.log2(3)
    assert ideal_dcg(2, k=10) == pytest.approx(expected)


def test_ndcg_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["a", "b", "c"], ["a"], k=3) == 1.0


def test_ndcg_no_hit_is_zero() -> None:
    assert ndcg_at_k(["x", "y", "z"], ["a"], k=3) == 0.0


def test_ndcg_in_range_zero_to_one() -> None:
    # gold at rank 3 of 10: nDCG should be in (0, 1)
    val = ndcg_at_k(["x", "y", "a", "z", "q"], ["a"], k=5)
    assert 0.0 < val < 1.0


def test_ndcg_empty_gold_is_zero() -> None:
    assert ndcg_at_k(["a", "b"], [], k=3) == 0.0


def test_ndcg_ranks_earlier_hits_higher() -> None:
    """nDCG should reward early hits more than late ones."""
    early = ndcg_at_k(["a", "x", "y", "z"], ["a"], k=4)
    late = ndcg_at_k(["x", "y", "z", "a"], ["a"], k=4)
    assert early > late
