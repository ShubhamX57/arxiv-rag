"""Chunking strategies for parsed sections.

Why chunk at all?
- Embedding a whole section into one vector loses topical detail; queries
  about specifics fail to match.
- Top-k retrieval works better when each candidate is focused.
- LLM context budgets are finite; smaller relevant chunks beat fewer large
  ones every time.

Three strategies (we benchmark all of them on the eval set in Week 2):
- FIXED: split every N tokens with overlap. Simple, fast, but cuts
  mid-sentence — a fragment of a sentence in chunk i may have its
  meaning completed only in chunk i+1, so the embedding drifts.
- RECURSIVE: split on natural boundaries (paragraph -> line -> sentence ->
  word) keeping each chunk under the size budget. Production workhorse.
- SEMANTIC (deferred to Day 5 once embeddings are wired up): split where
  embedding similarity between adjacent sentences drops below threshold.

Token counting uses tiktoken (cl100k_base) — same encoder the OpenAI / many
HuggingFace models use, so the counts roughly transfer when we swap models.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

import tiktoken
from tqdm import tqdm

from arxiv_rag.config import settings

log = logging.getLogger(__name__)


class ChunkStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrieval unit, with full provenance back to its source paper."""

    chunk_id: str  # f"{paper_id}::{section_index}::{chunk_index}"
    paper_id: str
    section_index: int
    section_title: str
    chunk_index: int  # within the section
    text: str
    char_count: int
    token_count: int
    strategy: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Chunk:
        """Construct a Chunk from a dict loaded from chunks JSONL."""
        return cls(
            chunk_id=str(d["chunk_id"]),
            paper_id=str(d["paper_id"]),
            section_index=int(d["section_index"]),  # type: ignore[call-overload]
            section_title=str(d["section_title"]),
            chunk_index=int(d["chunk_index"]),  # type: ignore[call-overload]
            text=str(d["text"]),
            char_count=int(d["char_count"]),  # type: ignore[call-overload]
            token_count=int(d["token_count"]),  # type: ignore[call-overload]
            strategy=str(d["strategy"]),
        )


# ---------------- Chunker protocol + implementations ----------------


class Chunker(Protocol):
    """Splits a text blob into list of strings under a token budget."""

    chunk_size: int
    chunk_overlap: int

    def chunk(self, text: str) -> list[str]: ...


def _get_encoder(name: str = "cl100k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


class FixedChunker:
    """Splits text into fixed-token windows with sliding overlap.

    Cuts mid-word/mid-sentence by design — fast but loses local coherence.
    Use as a baseline and a fallback for recursive when no separator helps.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        encoding: str = "cl100k_base",
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoder = _get_encoder(encoding)

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        tokens = self._encoder.encode(text)
        if len(tokens) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.chunk_overlap
        out: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            out.append(self._encoder.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start += step
        return out


class RecursiveChunker:
    """Recursively splits on natural boundaries until under chunk_size.

    Tries paragraph breaks first, then line breaks, then sentence ends,
    then spaces. Falls back to token-level splitting if no separator
    helps (e.g. single huge run-on word, equation soup).

    Then merges adjacent fragments back into chunks under chunk_size and
    adds token-level overlap between consecutive chunks.

    This is the production default in most RAG systems for good reason:
    chunks preserve sentence/paragraph integrity, which keeps embeddings
    semantically coherent.
    """

    DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: Iterable[str] | None = None,
        encoding: str = "cl100k_base",
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators: tuple[str, ...] = tuple(separators or self.DEFAULT_SEPARATORS)
        self._encoder = _get_encoder(encoding)

    def _len(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        fragments = self._split(text, self.separators)
        return self._merge_with_overlap(fragments)

    def _split(self, text: str, separators: tuple[str, ...]) -> list[str]:
        """Recursively break `text` into fragments each <= chunk_size tokens."""
        if self._len(text) <= self.chunk_size:
            return [text]

        # Find first separator that actually appears
        for i, sep in enumerate(separators):
            if sep == "":
                # Last resort: token-level fixed cut, no overlap here
                return FixedChunker(self.chunk_size, 0, encoding="cl100k_base").chunk(text)
            if sep in text:
                parts = text.split(sep)
                remaining = separators[i + 1 :]
                out: list[str] = []
                for j, part in enumerate(parts):
                    # Re-attach the separator to all but last for round-trip fidelity
                    piece = part + sep if j < len(parts) - 1 else part
                    if not piece:
                        continue
                    if self._len(piece) <= self.chunk_size:
                        out.append(piece)
                    else:
                        out.extend(self._split(piece, remaining))
                return out
        # Shouldn't reach here (DEFAULT_SEPARATORS ends with "" which always matches)
        return [text]

    def _merge_with_overlap(self, fragments: list[str]) -> list[str]:
        """Greedy merge fragments into <= chunk_size chunks; add token overlap."""
        if not fragments:
            return []

        chunks: list[str] = []
        buf: list[str] = []
        buf_tokens = 0

        for frag in fragments:
            ftok = self._len(frag)
            if buf_tokens + ftok <= self.chunk_size:
                buf.append(frag)
                buf_tokens += ftok
            else:
                if buf:
                    chunks.append("".join(buf))
                buf = [frag]
                buf_tokens = ftok
        if buf:
            chunks.append("".join(buf))

        if self.chunk_overlap == 0 or len(chunks) < 2:
            return chunks

        # Add overlap: prepend last `overlap` tokens of previous chunk to next.
        # Done after merging so overlap is exactly N tokens, not "N tokens of one fragment".
        with_overlap: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tokens = self._encoder.encode(chunks[i - 1])
            overlap_tokens = prev_tokens[-self.chunk_overlap :]
            overlap_text = self._encoder.decode(overlap_tokens)
            with_overlap.append(overlap_text + chunks[i])
        return with_overlap


class SemanticChunker:
    """Splits text where consecutive sentence embeddings diverge.

    Algorithm:
    1. Split text into sentences (regex-based, good enough for prose).
    2. Embed each sentence.
    3. Compute cosine similarity between consecutive embeddings.
    4. Find the Nth percentile *drop* in similarity (default: 95th) — these
       are the strongest topic boundaries.
    5. Build chunks by accumulating sentences and breaking at those points,
       while respecting chunk_size (fallback to recursive splitting if a
       semantic segment is too large).

    The intuition: paragraph breaks within a section often happen for layout
    reasons, but topic shifts happen where the *meaning* changes. Embedding
    similarity captures that.

    Trade-off: slower than recursive (must embed every sentence) and the
    quality vs recursive in practice is mixed — sometimes wins on long
    sections, sometimes worse on short ones with no real topic shift. We
    measure which in Week 2's ablation.
    """

    # Regex captures sentence boundaries: period/question/exclamation followed
    # by whitespace + uppercase. Loses on abbreviations ("e.g.", "Mr.") but
    # that's acceptable noise for chunking purposes.
    _SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        embedder: object,
        breakpoint_percentile: float = 95.0,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        if not 50.0 <= breakpoint_percentile <= 99.5:
            raise ValueError("breakpoint_percentile must be between 50 and 99.5")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        # Use recursive splitter as fallback when a semantic segment is too big
        self._recursive_fallback = RecursiveChunker(chunk_size, chunk_overlap)
        self._encoder = _get_encoder()

    def _split_sentences(self, text: str) -> list[str]:
        # Pre-split on paragraph breaks first to keep sentences cohesive
        sentences: list[str] = []
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            parts = self._SENT_SPLIT.split(para)
            sentences.extend(p.strip() for p in parts if p.strip())
        return sentences

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        # Very short text: don't bother embedding
        if len(sentences) <= 2 or self._token_len(text) <= self.chunk_size:
            return [text]

        # Embed sentences silently — this happens per-section in a loop
        import numpy as np

        vectors = self.embedder.embed(sentences, show_progress_bar=False)  # type: ignore[attr-defined]

        # Cosine similarity between consecutive sentences (vectors are normalized)
        sims = np.einsum("ij,ij->i", vectors[:-1], vectors[1:])
        # Lower similarity = bigger topic shift = better break point
        drops = 1.0 - sims
        threshold = float(np.percentile(drops, self.breakpoint_percentile))

        # Build segments by grouping sentences between break points
        segments: list[str] = []
        current: list[str] = [sentences[0]]
        for i, drop in enumerate(drops):
            if drop >= threshold:
                segments.append(" ".join(current))
                current = []
            current.append(sentences[i + 1])
        if current:
            segments.append(" ".join(current))

        # Enforce chunk_size: fall back to recursive on oversized segments
        out: list[str] = []
        for seg in segments:
            if self._token_len(seg) <= self.chunk_size:
                out.append(seg)
            else:
                out.extend(self._recursive_fallback.chunk(seg))
        return out

    def _token_len(self, text: str) -> int:
        return len(self._encoder.encode(text))


def make_chunker(
    strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
    embedder: object | None = None,
) -> Chunker:
    """Factory: pick a chunker by strategy name.

    Semantic chunking requires an Embedder. The factory accepts it as a generic
    `object` to avoid a circular import with the retrieval package.
    """
    if strategy == ChunkStrategy.FIXED:
        return FixedChunker(chunk_size, chunk_overlap)
    if strategy == ChunkStrategy.RECURSIVE:
        return RecursiveChunker(chunk_size, chunk_overlap)
    if strategy == ChunkStrategy.SEMANTIC:
        if embedder is None:
            raise ValueError(
                "SemanticChunker requires an embedder. "
                "Call chunk_sections(strategy=SEMANTIC) which provides one."
            )
        return SemanticChunker(chunk_size, chunk_overlap, embedder)
    raise ValueError(f"Unknown strategy: {strategy}")  # pragma: no cover


# ---------------- Pipeline driver ----------------


def chunk_sections(
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    sections_path: Path | None = None,
    out_path: Path | None = None,
) -> dict[str, float]:
    """Read sections.jsonl, chunk each, write chunks_<strategy>.jsonl.

    Returns stats: total_chunks, mean_tokens, median_tokens, p90_tokens,
    p99_tokens, papers, sections.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    sections_path = sections_path or settings.sections_path
    if out_path is None:
        out_path = sections_path.parent / f"chunks_{strategy.value}.jsonl"

    # Lazy: construct embedder only for semantic; avoids heavy imports otherwise
    if strategy == ChunkStrategy.SEMANTIC:
        from arxiv_rag.retrieval.embedder import Embedder

        chunker = make_chunker(strategy, chunk_size, chunk_overlap, embedder=Embedder())
    else:
        chunker = make_chunker(strategy, chunk_size, chunk_overlap)
    encoder = _get_encoder()  # for token counts in metadata

    # Read sections (small enough to fit in memory: ~2k for our corpus)
    sections: list[dict[str, object]] = []
    with sections_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sections.append(json.loads(line))

    paper_ids: set[str] = set()
    token_counts: list[int] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_chunks = 0

    log.info(
        "Chunking %d sections with strategy=%s, size=%d, overlap=%d",
        len(sections),
        strategy.value,
        chunk_size,
        chunk_overlap,
    )

    with out_path.open("w") as out_f:
        for section in tqdm(sections, desc=f"Chunking [{strategy.value}]"):
            paper_id = str(section["paper_id"])
            section_index = cast(int, section["section_index"])
            section_title = str(section["section_title"])
            text = str(section["text"])
            paper_ids.add(paper_id)

            for ci, chunk_text in enumerate(chunker.chunk(text)):
                tok = len(encoder.encode(chunk_text))
                chunk = Chunk(
                    chunk_id=f"{paper_id}::{section_index}::{ci}",
                    paper_id=paper_id,
                    section_index=section_index,
                    section_title=section_title,
                    chunk_index=ci,
                    text=chunk_text,
                    char_count=len(chunk_text),
                    token_count=tok,
                    strategy=strategy.value,
                )
                out_f.write(chunk.to_json() + "\n")
                total_chunks += 1
                token_counts.append(tok)

    stats = _compute_stats(token_counts, len(paper_ids), len(sections), total_chunks)
    log.info("Wrote %d chunks to %s", total_chunks, out_path)
    return stats


def _compute_stats(
    token_counts: list[int], papers: int, sections: int, total_chunks: int
) -> dict[str, float]:
    if not token_counts:
        return {
            "papers": float(papers),
            "sections": float(sections),
            "total_chunks": 0.0,
            "mean_tokens": 0.0,
            "median_tokens": 0.0,
            "p90_tokens": 0.0,
            "p99_tokens": 0.0,
        }
    sorted_tc = sorted(token_counts)
    return {
        "papers": float(papers),
        "sections": float(sections),
        "total_chunks": float(total_chunks),
        "mean_tokens": round(statistics.mean(token_counts), 1),
        "median_tokens": float(statistics.median(token_counts)),
        "p90_tokens": float(sorted_tc[int(0.90 * (len(sorted_tc) - 1))]),
        "p99_tokens": float(sorted_tc[int(0.99 * (len(sorted_tc) - 1))]),
    }


__all__ = [
    "Chunk",
    "ChunkStrategy",
    "Chunker",
    "FixedChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "chunk_sections",
    "make_chunker",
]
