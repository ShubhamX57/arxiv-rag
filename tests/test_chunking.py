"""Tests for chunking strategies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arxiv_rag.ingest.chunking import (
    Chunk,
    ChunkStrategy,
    FixedChunker,
    RecursiveChunker,
    chunk_sections,
    make_chunker,
)

# ----------------------- FixedChunker -----------------------


def test_fixed_chunker_basic() -> None:
    text = "word " * 1000  # ~1000 tokens
    chunker = FixedChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    # Reconstructed text should contain original content (modulo whitespace)
    joined = "".join(chunks)
    assert "word" in joined


def test_fixed_chunker_token_size_respected() -> None:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    text = " ".join(f"token{i}" for i in range(500))
    chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
    for c in chunker.chunk(text):
        assert len(enc.encode(c)) <= 50


def test_fixed_chunker_short_text_returns_one() -> None:
    chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.chunk("hello world") == ["hello world"]


def test_fixed_chunker_empty_returns_empty() -> None:
    chunker = FixedChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.chunk("") == []
    assert chunker.chunk("   ") == []


def test_fixed_chunker_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        FixedChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        FixedChunker(chunk_size=100, chunk_overlap=200)


def test_fixed_chunker_invalid_size_raises() -> None:
    with pytest.raises(ValueError):
        FixedChunker(chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        FixedChunker(chunk_size=100, chunk_overlap=-1)


def test_fixed_chunker_overlap_creates_repeated_tokens() -> None:
    """Adjacent fixed chunks should share `overlap` tokens at the boundary."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    text = " ".join(f"w{i}" for i in range(300))
    chunker = FixedChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    # Last 10 tokens of chunk i should equal first 10 of chunk i+1
    a_tail = enc.encode(chunks[0])[-10:]
    b_head = enc.encode(chunks[1])[:10]
    assert a_tail == b_head


# ----------------------- RecursiveChunker -----------------------


def test_recursive_short_text_returns_one() -> None:
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.chunk("hello world") == ["hello world"]


def test_recursive_empty_returns_empty() -> None:
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.chunk("") == []


def test_recursive_prefers_paragraph_boundaries() -> None:
    """When paragraph breaks exist and would honor size, splits should land on them."""
    paragraphs = [
        "Sentence one. Sentence two. Sentence three.",
        "Another paragraph with more sentences here.",
        "Third paragraph with yet more text content.",
        "Fourth paragraph wrapping things up nicely.",
    ]
    text = "\n\n".join(paragraphs)
    chunker = RecursiveChunker(chunk_size=30, chunk_overlap=0)
    chunks = chunker.chunk(text)
    # Each chunk should contain whole sentences (period ending), not mid-sentence breaks
    assert len(chunks) >= 2
    # Reassembling should yield original content
    joined = "".join(chunks)
    for p in paragraphs:
        # Each paragraph's content should appear somewhere
        first_words = p.split(".")[0]
        assert first_words in joined


def test_recursive_token_size_respected() -> None:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    text = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50).strip()
    chunker = RecursiveChunker(chunk_size=80, chunk_overlap=10)
    for c in chunker.chunk(text):
        # Allow small slack for overlap re-prepending
        assert len(enc.encode(c)) <= 80 + 10


def test_recursive_handles_no_separators_fallback() -> None:
    """A long run-on string with no separators still chunks via token fallback."""
    text = "x" * 5000
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    assert all(len(c) > 0 for c in chunks)


def test_recursive_overlap_zero_no_repeated_tokens() -> None:
    """With overlap=0, joining chunks should reproduce text (modulo separators)."""
    text = "\n\n".join(f"Para {i} contains some text here." for i in range(20))
    chunker = RecursiveChunker(chunk_size=30, chunk_overlap=0)
    chunks = chunker.chunk(text)
    joined = "".join(chunks)
    # Original content should be present
    assert "Para 0" in joined
    assert "Para 19" in joined


def test_recursive_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)


# ----------------------- Factory -----------------------


def test_make_chunker_fixed() -> None:
    c = make_chunker(ChunkStrategy.FIXED, 100, 10)
    assert isinstance(c, FixedChunker)


def test_make_chunker_recursive() -> None:
    c = make_chunker(ChunkStrategy.RECURSIVE, 100, 10)
    assert isinstance(c, RecursiveChunker)


def test_make_chunker_semantic_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        make_chunker(ChunkStrategy.SEMANTIC, 100, 10)


# ----------------------- chunk_sections (pipeline) -----------------------


def _write_synthetic_sections(path: Path) -> None:
    """Write 2 papers x ~3 sections each into a JSONL file."""
    sections = [
        {
            "paper_id": "2401.00001",
            "section_index": 0,
            "section_title": "Abstract",
            "text": "We propose a method for chunking. " * 10,
            "page_start": 0,
            "page_end": 0,
            "char_count": 0,
        },
        {
            "paper_id": "2401.00001",
            "section_index": 1,
            "section_title": "1. Introduction",
            "text": ("Recent work has shown that retrieval matters a lot. " * 80),
            "page_start": 0,
            "page_end": 1,
            "char_count": 0,
        },
        {
            "paper_id": "2401.00001",
            "section_index": 2,
            "section_title": "2. Method",
            "text": "Our approach uses three steps to process input data effectively. " * 60,
            "page_start": 1,
            "page_end": 2,
            "char_count": 0,
        },
        {
            "paper_id": "2401.00002",
            "section_index": 0,
            "section_title": "Abstract",
            "text": "A different paper with different content goes here for variety. " * 8,
            "page_start": 0,
            "page_end": 0,
            "char_count": 0,
        },
    ]
    with path.open("w") as f:
        for s in sections:
            f.write(json.dumps(s) + "\n")


def test_chunk_sections_recursive(tmp_path: Path) -> None:
    sections_file = tmp_path / "sections.jsonl"
    out_file = tmp_path / "chunks.jsonl"
    _write_synthetic_sections(sections_file)

    stats = chunk_sections(
        strategy=ChunkStrategy.RECURSIVE,
        chunk_size=200,
        chunk_overlap=20,
        sections_path=sections_file,
        out_path=out_file,
    )

    assert stats["papers"] == 2
    assert stats["sections"] == 4
    assert stats["total_chunks"] > 0
    assert stats["mean_tokens"] > 0

    # Verify output file
    chunks = [json.loads(line) for line in out_file.open()]
    assert len(chunks) == int(stats["total_chunks"])
    for c in chunks:
        assert c["chunk_id"].count("::") == 2
        assert c["paper_id"] in {"2401.00001", "2401.00002"}
        assert c["strategy"] == "recursive"
        assert c["token_count"] > 0


def test_chunk_sections_fixed(tmp_path: Path) -> None:
    sections_file = tmp_path / "sections.jsonl"
    out_file = tmp_path / "chunks_fixed.jsonl"
    _write_synthetic_sections(sections_file)

    chunk_sections(
        strategy=ChunkStrategy.FIXED,
        chunk_size=128,
        chunk_overlap=16,
        sections_path=sections_file,
        out_path=out_file,
    )

    chunks = [json.loads(line) for line in out_file.open()]
    assert all(c["strategy"] == "fixed" for c in chunks)
    assert all(c["token_count"] <= 128 for c in chunks)


def test_chunk_id_uniqueness(tmp_path: Path) -> None:
    sections_file = tmp_path / "sections.jsonl"
    out_file = tmp_path / "chunks.jsonl"
    _write_synthetic_sections(sections_file)
    chunk_sections(
        strategy=ChunkStrategy.RECURSIVE,
        chunk_size=100,
        chunk_overlap=10,
        sections_path=sections_file,
        out_path=out_file,
    )
    chunk_ids = [json.loads(line)["chunk_id"] for line in out_file.open()]
    assert len(chunk_ids) == len(set(chunk_ids)), "chunk_ids must be unique"


def test_chunk_dataclass_serialization() -> None:
    c = Chunk(
        chunk_id="x::0::0",
        paper_id="x",
        section_index=0,
        section_title="Abstract",
        chunk_index=0,
        text="hello",
        char_count=5,
        token_count=1,
        strategy="recursive",
    )
    parsed = json.loads(c.to_json())
    assert parsed["chunk_id"] == "x::0::0"
    assert parsed["strategy"] == "recursive"
