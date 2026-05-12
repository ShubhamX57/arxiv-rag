"""Sparse retriever using BM25 (Okapi).

Why BM25 alongside dense?
- Dense embeddings capture semantic similarity (great for paraphrases).
- BM25 captures exact term overlap (great for rare technical terms,
  acronyms like "GQA", "RLHF" — embeddings often miss these).
- Hybrid retrieval combines both via Reciprocal Rank Fusion.

We tokenize with a simple regex+lowercase strategy. Real systems sometimes
use heavier preprocessing (stemming, stopword removal) but for technical
content like ML papers, keeping rare tokens helps more than hurts.
"""

from __future__ import annotations

import logging
import pickle
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from arxiv_rag.config import settings

log = logging.getLogger(__name__)

# Match alphanumeric runs (keeps acronyms like "BERT", numbers like "2024")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase + alphanumeric token split. Stable, deterministic."""
    return _TOKEN_PATTERN.findall(text.lower())


def bm25_path_for(strategy: str, base_dir: Path | None = None) -> Path:
    """Path to the pickled BM25 index for one chunking strategy."""
    base = base_dir or settings.data_dir
    return base / f"bm25_{strategy}.pkl"


@dataclass(frozen=True, slots=True)
class SparseHit:
    """One result from a BM25 search."""

    chunk_id: str
    paper_id: str
    section_title: str
    text: str
    score: float  # raw BM25 score (unnormalized; higher = better)


@dataclass(slots=True)
class _IndexPayload:
    """What we pickle to disk."""

    bm25: BM25Okapi
    chunk_ids: list[str]
    paper_ids: list[str]
    section_titles: list[str]
    texts: list[str]


class BM25Store:
    """Build + query a BM25 index over chunks."""

    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self._payload: _IndexPayload | None = None

    # ---------- build ----------

    def build(self, chunks: Iterable[dict[str, Any]]) -> int:
        """Tokenize + index. Persist as a pickle for fast reload."""
        chunk_list = list(chunks)
        if not chunk_list:
            log.warning("No chunks to index.")
            return 0

        log.info("Tokenizing %d chunks for BM25...", len(chunk_list))
        tokenized: list[list[str]] = [
            tokenize(c["text"]) for c in tqdm(chunk_list, desc="Tokenizing", unit="chunks")
        ]

        log.info("Fitting BM25Okapi...")
        bm25 = BM25Okapi(tokenized)

        payload = _IndexPayload(
            bm25=bm25,
            chunk_ids=[c["chunk_id"] for c in chunk_list],
            paper_ids=[c["paper_id"] for c in chunk_list],
            section_titles=[c["section_title"] for c in chunk_list],
            texts=[c["text"] for c in chunk_list],
        )
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._payload = payload
        log.info("Wrote BM25 index to %s", self.index_path)
        return len(chunk_list)

    # ---------- query ----------

    def _ensure_loaded(self) -> _IndexPayload:
        if self._payload is None:
            if not self.index_path.exists():
                raise FileNotFoundError(
                    f"BM25 index not found at {self.index_path}. Run `arxiv-rag index` first."
                )
            with self.index_path.open("rb") as f:
                self._payload = pickle.load(f)
        return self._payload

    def search(self, query: str, k: int = 10) -> list[SparseHit]:
        """Top-k chunks by BM25 score."""
        p = self._ensure_loaded()
        scores = p.bm25.get_scores(tokenize(query))
        # argsort descending, take top-k
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            SparseHit(
                chunk_id=p.chunk_ids[i],
                paper_id=p.paper_ids[i],
                section_title=p.section_titles[i],
                text=p.texts[i],
                score=float(scores[i]),
            )
            for i in top_idx
            if scores[i] > 0  # skip irrelevant
        ]


__all__ = [
    "BM25Store",
    "SparseHit",
    "bm25_path_for",
    "tokenize",
]
