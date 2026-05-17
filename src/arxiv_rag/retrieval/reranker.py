"""Cross-encoder reranker.

A cross-encoder reads (query, candidate) as a pair through a single transformer
and outputs ONE relevance score. This is fundamentally different from dense
retrieval, where query and chunk are encoded independently and only their
embeddings get compared. Joint attention across both texts makes cross-encoders
much more accurate — but they can't be pre-computed, so we only run them on a
small candidate set (top-K from retrieval).

The canonical RAG pattern:
    1. Hybrid retrieval gets top-50 candidates  (cheap, fast — ~10ms)
    2. Cross-encoder reranks all 50            (expensive — ~80ms x 50 = ~4s on M4 Pro)
    3. Return top-10                            (final result)

We use BAAI/bge-reranker-v2-m3, the standard partner model to BGE-M3 embeddings.

This module exposes a thin wrapper around `sentence-transformers.CrossEncoder`
so we can:
    - Cache the model as a singleton (instantiation is slow, ~3-5s)
    - Run on MPS automatically on Apple Silicon
    - Hide the score-then-sort logic from callers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from arxiv_rag.config import settings

if TYPE_CHECKING:
    from arxiv_rag.retrieval.hybrid import FusedHit

log = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True, slots=True)
class RerankedHit:
    """A FusedHit after reranking. We keep all the original provenance and add
    the rerank score + the original RRF rank so we can see how much the
    reranker moved each result."""

    chunk_id: str
    paper_id: str
    section_title: str
    text: str
    rerank_score: float
    rrf_rank: int  # 1-indexed rank before reranking
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None


class Reranker:
    """Cross-encoder reranker, with the model loaded lazily on first use."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model: object | None = None  # CrossEncoder, but imported lazily

    def _ensure_model(self) -> object:
        """Lazy-load. CrossEncoder import alone pulls in torch and transformers."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            device = settings.embedding_device  # "mps" on M-series, "cpu" elsewhere
            log.info(
                "Loading reranker model %s on %s (this can take a moment)...",
                self.model_name,
                device,
            )
            self._model = CrossEncoder(self.model_name, device=device)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Score each (query, text) pair. Returns list of floats, same order as input.

        Higher score = more relevant. The raw scores are model logits, so they
        aren't bounded — use them for ranking only, not as probabilities.
        """
        if not texts:
            return []
        model = self._ensure_model()
        pairs = [(query, t) for t in texts]
        # CrossEncoder.predict returns a numpy array; cast to plain floats
        scores = model.predict(pairs, show_progress_bar=False)  # type: ignore[attr-defined]
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: list[FusedHit],
        top_k: int | None = None,
    ) -> list[RerankedHit]:
        """Rerank `candidates` by cross-encoder relevance to `query`.

        Returns the top `top_k` (or all of them, if `top_k` is None), sorted
        by rerank_score descending. Each result carries provenance back to
        its original RRF rank so you can see how much the reranker moved it.
        """
        if not candidates:
            return []

        # Score in original RRF order so rrf_rank lines up with the input list
        texts = [c.text for c in candidates]
        scores = self.score(query, texts)

        # Build RerankedHits with original rank attached (1-indexed)
        reranked = [
            RerankedHit(
                chunk_id=c.chunk_id,
                paper_id=c.paper_id,
                section_title=c.section_title,
                text=c.text,
                rerank_score=s,
                rrf_rank=i + 1,
                rrf_score=c.rrf_score,
                dense_rank=c.dense_rank,
                sparse_rank=c.sparse_rank,
            )
            for i, (c, s) in enumerate(zip(candidates, scores, strict=True))
        ]

        # Sort by rerank_score descending; stable sort preserves RRF order on ties
        reranked.sort(key=lambda h: h.rerank_score, reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]
        return reranked


# Module-level singleton: instantiating Reranker is cheap (no model load),
# but creating it once means we share the cached model across calls.
_default_reranker: Reranker | None = None


def get_default_reranker() -> Reranker:
    """Return the module-level Reranker, creating it on first call.

    Use this from CLI / API code so successive eval runs share one loaded model.
    Tests should instantiate their own Reranker to avoid global state.
    """
    global _default_reranker
    if _default_reranker is None:
        _default_reranker = Reranker()
    return _default_reranker


__all__ = [
    "DEFAULT_RERANKER_MODEL",
    "RerankedHit",
    "Reranker",
    "RerankingRetriever",
    "get_default_reranker",
    "replace",  # re-export for convenience in tests that want to tweak a RerankedHit
]


# ---------- High-level pipeline: hybrid retrieve, then rerank ----------

# This sits in reranker.py rather than hybrid.py because it depends on the
# reranker (and its lazy imports); we don't want `hybrid.py` to drag torch in.


class RerankingRetriever:
    """Compose HybridRetriever + Reranker into one search() call.

    Standard pattern: retrieve `fan_out_k` candidates (default 50), rerank them,
    return the top `k` (default 10). The fan_out_k value is the most important
    knob — too small and the reranker can't recover from a bad retrieval; too
    large and you pay for cross-encoder compute on irrelevant chunks.
    """

    def __init__(
        self,
        strategy: str = "recursive",
        rrf_k: int = 60,
        reranker: Reranker | None = None,
    ) -> None:
        # Local import to avoid a circular import (hybrid -> reranker -> hybrid)
        from arxiv_rag.retrieval.hybrid import HybridRetriever

        self.hybrid = HybridRetriever(strategy=strategy, rrf_k=rrf_k)
        self.reranker = reranker or get_default_reranker()

    def search(
        self,
        query: str,
        k: int = 10,
        fan_out_k: int = 50,
    ) -> list[RerankedHit]:
        """Retrieve `fan_out_k` candidates with hybrid, rerank, return top `k`."""
        # Hybrid uses its OWN fan_out internally to populate the candidate pool;
        # we ask it for fan_out_k results back, then rerank those.
        candidates = self.hybrid.search(query, k=fan_out_k, fan_out_k=fan_out_k)
        return self.reranker.rerank(query, candidates, top_k=k)
