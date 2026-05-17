"""Tests for the cross-encoder reranker.

We don't load the real BGE reranker in tests (568M params, slow to download
and slow to score). Instead, we inject a fake `_model` into a Reranker
instance to bypass `_ensure_model()`. The test verifies the wiring around
the model: scoring, sorting, top_k truncation, RerankedHit construction.
"""

from __future__ import annotations

import pytest

from arxiv_rag.retrieval.hybrid import FusedHit
from arxiv_rag.retrieval.reranker import RerankedHit, Reranker

# ----------------------- helpers -----------------------


class FakeCrossEncoder:
    """A stand-in for sentence_transformers.CrossEncoder.

    `predict(pairs)` returns a pre-set list of scores in the same order as input,
    so a test can engineer the desired reranking outcome.
    """

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.received_pairs: list[tuple[str, str]] | None = None

    def predict(self, pairs: list[tuple[str, str]], show_progress_bar: bool = False) -> list[float]:
        self.received_pairs = pairs
        assert len(pairs) == len(self._scores), (
            f"FakeCrossEncoder configured with {len(self._scores)} scores "
            f"but received {len(pairs)} pairs"
        )
        return list(self._scores)


def _make_reranker_with_scores(scores: list[float]) -> tuple[Reranker, FakeCrossEncoder]:
    """Build a Reranker with a fake model already loaded, so _ensure_model() is bypassed."""
    fake = FakeCrossEncoder(scores)
    r = Reranker()
    r._model = fake  # bypass lazy load
    return r, fake


def _hit(chunk_id: str, text: str, rrf_score: float = 0.5) -> FusedHit:
    return FusedHit(
        chunk_id=chunk_id,
        paper_id="p1",
        section_title="s",
        text=text,
        rrf_score=rrf_score,
        dense_rank=1,
        sparse_rank=2,
        dense_score=0.9,
        sparse_score=10.0,
    )


# ----------------------- score() -----------------------


def test_score_returns_one_float_per_text() -> None:
    r, _ = _make_reranker_with_scores([0.1, 0.5, 0.9])
    out = r.score("query", ["a", "b", "c"])
    assert out == [0.1, 0.5, 0.9]
    assert all(isinstance(s, float) for s in out)


def test_score_empty_texts_is_empty_list() -> None:
    r = Reranker()  # no model needed — early return on empty
    assert r.score("anything", []) == []


def test_score_builds_query_text_pairs() -> None:
    r, fake = _make_reranker_with_scores([0.0, 0.0])
    r.score("my query", ["chunk1", "chunk2"])
    assert fake.received_pairs == [("my query", "chunk1"), ("my query", "chunk2")]


# ----------------------- rerank() ordering -----------------------


def test_rerank_reorders_by_score_descending() -> None:
    """Input is in 'wrong' order; reranker should sort by score desc."""
    cands = [_hit("c1", "junk"), _hit("c2", "gold"), _hit("c3", "noise")]
    # Engineer: rerank should rank c2 first, c1 second, c3 last
    r, _ = _make_reranker_with_scores([0.3, 0.9, 0.1])
    out = r.rerank("q", cands)
    assert [h.chunk_id for h in out] == ["c2", "c1", "c3"]


def test_rerank_preserves_original_rrf_rank() -> None:
    """rrf_rank should reflect position in INPUT list, not output position."""
    cands = [_hit("c1", "a"), _hit("c2", "b"), _hit("c3", "c")]
    r, _ = _make_reranker_with_scores([0.1, 0.9, 0.5])
    out = r.rerank("q", cands)

    # By rerank_score desc: c2 (rrf_rank=2), c3 (rrf_rank=3), c1 (rrf_rank=1)
    assert [h.chunk_id for h in out] == ["c2", "c3", "c1"]
    assert [h.rrf_rank for h in out] == [2, 3, 1]


def test_rerank_top_k_truncates() -> None:
    cands = [_hit(f"c{i}", f"text{i}") for i in range(5)]
    r, _ = _make_reranker_with_scores([0.1, 0.5, 0.9, 0.3, 0.7])
    out = r.rerank("q", cands, top_k=2)
    assert len(out) == 2
    # top-2 by score: c2 (0.9), c4 (0.7)
    assert [h.chunk_id for h in out] == ["c2", "c4"]


def test_rerank_top_k_none_returns_all() -> None:
    cands = [_hit("c1", "a"), _hit("c2", "b")]
    r, _ = _make_reranker_with_scores([0.1, 0.2])
    out = r.rerank("q", cands, top_k=None)
    assert len(out) == 2


def test_rerank_empty_candidates_is_empty() -> None:
    r, _ = _make_reranker_with_scores([])
    assert r.rerank("q", []) == []


def test_rerank_attaches_score() -> None:
    cands = [_hit("c1", "a")]
    r, _ = _make_reranker_with_scores([0.42])
    [out] = r.rerank("q", cands)
    assert isinstance(out, RerankedHit)
    assert out.rerank_score == pytest.approx(0.42)


def test_rerank_preserves_provenance_fields() -> None:
    """RerankedHit should carry all the FusedHit metadata forward."""
    cands = [_hit("c1", "a", rrf_score=0.7)]
    r, _ = _make_reranker_with_scores([0.5])
    [out] = r.rerank("q", cands)
    assert out.chunk_id == "c1"
    assert out.paper_id == "p1"
    assert out.section_title == "s"
    assert out.text == "a"
    assert out.rrf_score == pytest.approx(0.7)
    assert out.dense_rank == 1
    assert out.sparse_rank == 2


def test_rerank_stable_on_ties() -> None:
    """When scores are equal, Python's sort is stable — earlier input wins.

    Important property: ties shouldn't randomize the rest of the order.
    """
    cands = [_hit("c1", "a"), _hit("c2", "b"), _hit("c3", "c")]
    r, _ = _make_reranker_with_scores([0.5, 0.5, 0.5])
    out = r.rerank("q", cands)
    # All tied; should preserve input order
    assert [h.chunk_id for h in out] == ["c1", "c2", "c3"]
