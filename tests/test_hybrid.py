"""Tests for hybrid retrieval and Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from arxiv_rag.retrieval.dense import DenseHit
from arxiv_rag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from arxiv_rag.retrieval.sparse import SparseHit

# ----------------------- Pure RRF -----------------------


def test_rrf_single_list_descending_scores() -> None:
    """Single retriever: first item should have highest fused score."""
    scores = reciprocal_rank_fusion([["a", "b", "c", "d"]], k=60)
    assert scores["a"] > scores["b"] > scores["c"] > scores["d"]


def test_rrf_intersection_boosts_score() -> None:
    """Chunk appearing in both lists scores higher than either alone."""
    list1 = ["a", "b", "c"]
    list2 = ["c", "a", "d"]
    scores = reciprocal_rank_fusion([list1, list2], k=60)
    # "a" is in both → biggest score expected
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["d"]
    assert scores["a"] > scores["c"]


def test_rrf_disjoint_lists() -> None:
    """Items appearing in only one list still rank; first-of-each tie."""
    list1 = ["a", "b"]
    list2 = ["x", "y"]
    scores = reciprocal_rank_fusion([list1, list2], k=60)
    assert pytest.approx(scores["a"]) == scores["x"]
    assert pytest.approx(scores["b"]) == scores["y"]


def test_rrf_empty_lists() -> None:
    assert reciprocal_rank_fusion([], k=60) == {}
    assert reciprocal_rank_fusion([[], []], k=60) == {}


def test_rrf_k_constant_changes_weighting() -> None:
    """Smaller k makes top ranks dominate more."""
    scores_k60 = reciprocal_rank_fusion([["a", "b"]], k=60)
    scores_k10 = reciprocal_rank_fusion([["a", "b"]], k=10)
    ratio_k60 = scores_k60["a"] / scores_k60["b"]
    ratio_k10 = scores_k10["a"] / scores_k10["b"]
    # k=10 should be more top-heavy than k=60
    assert ratio_k10 > ratio_k60


def test_rrf_invalid_k_raises() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-5)


def test_rrf_known_values() -> None:
    """Hand-computed: ['a', 'b'] with k=60 → a: 1/61, b: 1/62."""
    scores = reciprocal_rank_fusion([["a", "b"]], k=60)
    assert scores["a"] == pytest.approx(1.0 / 61)
    assert scores["b"] == pytest.approx(1.0 / 62)


# ----------------------- HybridRetriever with mocks -----------------------


@dataclass
class _FakeEmbedder:
    """Returns same vector for any text."""

    dim: int = 4

    def embed(self, texts, show_progress_bar=False):
        import numpy as np

        return np.ones((len(texts), self.dim), dtype=np.float32)


class _FakeDense:
    def __init__(self, hits: list[DenseHit]) -> None:
        self._hits = hits

    def search(self, qvec, k: int) -> list[DenseHit]:
        return self._hits[:k]


class _FakeSparse:
    def __init__(self, hits: list[SparseHit]) -> None:
        self._hits = hits

    def search(self, query: str, k: int) -> list[SparseHit]:
        return self._hits[:k]


def _dh(cid: str, score: float = 0.9) -> DenseHit:
    return DenseHit(chunk_id=cid, paper_id="p", section_title="t", text=f"text-{cid}", score=score)


def _sh(cid: str, score: float = 12.0) -> SparseHit:
    return SparseHit(chunk_id=cid, paper_id="p", section_title="t", text=f"text-{cid}", score=score)


def test_hybrid_search_intersection_wins() -> None:
    """A chunk in top of both retrievers should rank #1 in fused output."""
    retriever = HybridRetriever(
        embedder=_FakeEmbedder(),
        dense=_FakeDense([_dh("X"), _dh("A"), _dh("B")]),  # type: ignore[arg-type]
        sparse=_FakeSparse([_sh("X"), _sh("C"), _sh("D")]),  # type: ignore[arg-type]
    )
    hits = retriever.search("query", k=5, fan_out_k=3)
    assert hits[0].chunk_id == "X"
    assert hits[0].dense_rank == 1
    assert hits[0].sparse_rank == 1


def test_hybrid_search_records_ranks() -> None:
    retriever = HybridRetriever(
        embedder=_FakeEmbedder(),
        dense=_FakeDense([_dh("A"), _dh("B"), _dh("C")]),  # type: ignore[arg-type]
        sparse=_FakeSparse([_sh("C"), _sh("D")]),  # type: ignore[arg-type]
    )
    hits = retriever.search("query", k=10, fan_out_k=10)
    by_id = {h.chunk_id: h for h in hits}

    # A: dense=1, sparse=None
    assert by_id["A"].dense_rank == 1
    assert by_id["A"].sparse_rank is None
    assert by_id["A"].sparse_score is None

    # C: dense=3, sparse=1
    assert by_id["C"].dense_rank == 3
    assert by_id["C"].sparse_rank == 1

    # D: dense=None, sparse=2
    assert by_id["D"].dense_rank is None
    assert by_id["D"].sparse_rank == 2


def test_hybrid_empty_query() -> None:
    retriever = HybridRetriever(
        embedder=_FakeEmbedder(),
        dense=_FakeDense([_dh("A")]),  # type: ignore[arg-type]
        sparse=_FakeSparse([_sh("B")]),  # type: ignore[arg-type]
    )
    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_hybrid_limits_to_k() -> None:
    retriever = HybridRetriever(
        embedder=_FakeEmbedder(),
        dense=_FakeDense([_dh(c) for c in "ABCDEFGH"]),  # type: ignore[arg-type]
        sparse=_FakeSparse([_sh(c) for c in "ABCDEFGH"]),  # type: ignore[arg-type]
    )
    hits = retriever.search("query", k=3, fan_out_k=10)
    assert len(hits) == 3
