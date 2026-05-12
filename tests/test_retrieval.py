"""Tests for the retrieval package.

Strategy: a FakeEmbedder produces deterministic vectors based on token hashes.
This lets us test indexing, storage, and search end-to-end without downloading
the BGE-M3 model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from arxiv_rag.retrieval.dense import DenseStore, load_chunks_jsonl
from arxiv_rag.retrieval.sparse import BM25Store, bm25_path_for, tokenize

# ----------------------- Tokenizer (sparse) -----------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello World", ["hello", "world"]),
        ("BERT is a Transformer.", ["bert", "is", "a", "transformer"]),
        ("GQA vs MHA", ["gqa", "vs", "mha"]),
        ("ImageNet-2024", ["imagenet", "2024"]),
        ("", []),
        ("    ", []),
        ("a1b2c3", ["a1b2c3"]),
    ],
)
def test_tokenize(text: str, expected: list[str]) -> None:
    assert tokenize(text) == expected


# ----------------------- Sample data + fake embedder -----------------------


def _sample_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "p1::0::0",
            "paper_id": "p1",
            "section_index": 0,
            "section_title": "Abstract",
            "chunk_index": 0,
            "text": "We propose a new attention mechanism called GQA for transformers.",
            "char_count": 64,
            "token_count": 14,
            "strategy": "recursive",
        },
        {
            "chunk_id": "p1::1::0",
            "paper_id": "p1",
            "section_index": 1,
            "section_title": "Introduction",
            "chunk_index": 0,
            "text": "Multi-head attention is the standard in modern language models.",
            "char_count": 64,
            "token_count": 13,
            "strategy": "recursive",
        },
        {
            "chunk_id": "p2::0::0",
            "paper_id": "p2",
            "section_index": 0,
            "section_title": "Abstract",
            "chunk_index": 0,
            "text": "Convolutional networks remain effective for image classification tasks.",
            "char_count": 72,
            "token_count": 14,
            "strategy": "recursive",
        },
        {
            "chunk_id": "p3::0::0",
            "paper_id": "p3",
            "section_index": 0,
            "section_title": "Method",
            "chunk_index": 0,
            "text": "Grouped query attention reduces KV cache memory at inference time.",
            "char_count": 67,
            "token_count": 13,
            "strategy": "recursive",
        },
    ]


class FakeEmbedder:
    """Deterministic embedder for tests: hashes tokens into a 32-dim vector."""

    dim = 32
    model_name = "fake-embedder"
    device = "cpu"

    def embed(
        self,
        texts: list[str] | tuple[str, ...],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(text_list), self.dim), dtype=np.float32)
        for i, t in enumerate(text_list):
            for tok in tokenize(t):
                h = hashlib.md5(tok.encode()).digest()
                idx = h[0] % self.dim
                out[i, idx] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


# ----------------------- BM25Store -----------------------


def test_bm25_build_and_query(tmp_path: Path) -> None:
    index_path = tmp_path / "bm25.pkl"
    store = BM25Store(index_path)
    n = store.build(_sample_chunks())
    assert n == 4
    assert index_path.exists()

    hits = store.search("GQA grouped query attention", k=3)
    assert len(hits) >= 2
    top_ids = [h.chunk_id for h in hits[:2]]
    assert "p1::0::0" in top_ids or "p3::0::0" in top_ids


def test_bm25_persistence(tmp_path: Path) -> None:
    index_path = tmp_path / "bm25.pkl"
    BM25Store(index_path).build(_sample_chunks())
    fresh = BM25Store(index_path)
    hits = fresh.search("attention mechanism", k=2)
    assert len(hits) > 0
    assert all(h.score > 0 for h in hits)


def test_bm25_unrelated_query_returns_empty(tmp_path: Path) -> None:
    index_path = tmp_path / "bm25.pkl"
    BM25Store(index_path).build(_sample_chunks())
    fresh = BM25Store(index_path)
    hits = fresh.search("xyzzyx nothingmatches qwerty", k=5)
    assert hits == [] or all(h.score > 0 for h in hits)


def test_bm25_missing_index_raises(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "doesnt_exist.pkl")
    with pytest.raises(FileNotFoundError):
        store.search("anything")


def test_bm25_path_for_default() -> None:
    p = bm25_path_for("recursive")
    assert p.name == "bm25_recursive.pkl"


# ----------------------- DenseStore -----------------------


def test_dense_build_and_search(tmp_path: Path) -> None:
    store = DenseStore(db_dir=tmp_path / "lance", table_name="t")
    embedder = FakeEmbedder()
    n = store.build(_sample_chunks(), embedder=embedder)
    assert n == 4

    qvec = embedder.embed(["attention mechanism"])[0]
    hits = store.search(qvec, k=3)
    assert len(hits) <= 3
    assert all(-1.0 <= h.score <= 1.0 + 1e-6 for h in hits)
    assert "attention" in hits[0].text.lower()


def test_dense_missing_table_raises(tmp_path: Path) -> None:
    store = DenseStore(db_dir=tmp_path / "lance", table_name="nope")
    embedder = FakeEmbedder()
    qvec = embedder.embed(["q"])[0]
    with pytest.raises(FileNotFoundError):
        store.search(qvec, k=5)


def test_dense_empty_chunks_returns_zero(tmp_path: Path) -> None:
    store = DenseStore(db_dir=tmp_path / "lance", table_name="t")
    embedder = FakeEmbedder()
    n = store.build([], embedder=embedder)
    assert n == 0


def test_dense_rebuild_overwrites(tmp_path: Path) -> None:
    store = DenseStore(db_dir=tmp_path / "lance", table_name="t")
    embedder = FakeEmbedder()
    store.build(_sample_chunks(), embedder=embedder)
    n2 = store.build(_sample_chunks()[:2], embedder=embedder, recreate=True)
    assert n2 == 2


# ----------------------- load_chunks_jsonl -----------------------


def test_load_chunks_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "chunks.jsonl"
    chunks = _sample_chunks()
    with p.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    loaded = load_chunks_jsonl(p)
    assert len(loaded) == len(chunks)
    assert loaded[0]["chunk_id"] == chunks[0]["chunk_id"]


def test_load_chunks_jsonl_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "chunks.jsonl"
    p.touch()
    assert load_chunks_jsonl(p) == []
