"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

Why RRF?
- Dense and sparse scores live on different scales (cosine in [-1, 1] vs
  BM25 in [0, ~30]). Comparing them directly is meaningless.
- RRF only cares about *ranks*, not raw scores, so calibration is unnecessary.
- Parameter-free in practice (one hyperparameter k, defaults to 60).
- Beats weighted-sum on every public benchmark I'm aware of.

Formula:

    rrf_score(chunk) = sum over retrievers r of: 1 / (k + rank_r(chunk))

where rank_r(chunk) is 1-indexed (best result has rank 1). Chunks missing
from a retriever's list don't contribute from that retriever.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from arxiv_rag.retrieval.dense import DenseStore, chunks_table_name
from arxiv_rag.retrieval.embedder import Embedder
from arxiv_rag.retrieval.sparse import BM25Store, bm25_path_for

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FusedHit:
    """One result from hybrid retrieval, with provenance back to both retrievers."""

    chunk_id: str
    paper_id: str
    section_title: str
    text: str
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None
    dense_score: float | None
    sparse_score: float | None


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """Pure-function RRF: take ranked lists of chunk_ids, return fused scores."""
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank_zero_indexed, chunk_id in enumerate(lst):
            rank = rank_zero_indexed + 1
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    """Dense + BM25 with Reciprocal Rank Fusion."""

    def __init__(
        self,
        strategy: str = "recursive",
        embedder: Embedder | None = None,
        dense: DenseStore | None = None,
        sparse: BM25Store | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.strategy = strategy
        self.rrf_k = rrf_k
        self._embedder = embedder
        self._dense = dense or DenseStore(table_name=chunks_table_name(strategy))
        self._sparse = sparse or BM25Store(index_path=bm25_path_for(strategy))

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def search(
        self,
        query: str,
        k: int = 10,
        fan_out_k: int = 50,
    ) -> list[FusedHit]:
        """Top-k chunks after RRF fusion of dense and sparse retrievers."""
        if not query.strip():
            return []

        qvec = self.embedder.embed([query], show_progress_bar=False)[0]
        dense_hits = self._dense.search(qvec, k=fan_out_k)
        sparse_hits = self._sparse.search(query, k=fan_out_k)

        dense_by_id = {h.chunk_id: h for h in dense_hits}
        sparse_by_id = {h.chunk_id: h for h in sparse_hits}
        dense_rank = {h.chunk_id: i + 1 for i, h in enumerate(dense_hits)}
        sparse_rank = {h.chunk_id: i + 1 for i, h in enumerate(sparse_hits)}

        scores = reciprocal_rank_fusion(
            [
                [h.chunk_id for h in dense_hits],
                [h.chunk_id for h in sparse_hits],
            ],
            k=self.rrf_k,
        )

        ordered_ids = sorted(scores, key=lambda cid: -scores[cid])[:k]
        fused: list[FusedHit] = []
        for cid in ordered_ids:
            meta = dense_by_id.get(cid) or sparse_by_id.get(cid)
            assert meta is not None, "chunk in scores must come from at least one retriever"
            fused.append(
                FusedHit(
                    chunk_id=cid,
                    paper_id=meta.paper_id,
                    section_title=meta.section_title,
                    text=meta.text,
                    rrf_score=scores[cid],
                    dense_rank=dense_rank.get(cid),
                    sparse_rank=sparse_rank.get(cid),
                    dense_score=dense_by_id[cid].score if cid in dense_by_id else None,
                    sparse_score=sparse_by_id[cid].score if cid in sparse_by_id else None,
                )
            )
        return fused


__all__ = [
    "FusedHit",
    "HybridRetriever",
    "reciprocal_rank_fusion",
]
