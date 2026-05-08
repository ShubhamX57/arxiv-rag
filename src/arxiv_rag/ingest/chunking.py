"""Chunking strategies. We benchmark all three in Week 2.

TODO (Week 1, Day 4):
- Implement FIXED (split every N tokens with overlap)
- Implement RECURSIVE (split on \\n\\n then \\n then sentence then word)
- Implement SEMANTIC (split where embedding similarity drops below threshold)
- Each chunk should carry: chunk_id, paper_id, section_title, text, token_count
- Write a function that takes parsed sections and a strategy, returns chunks
"""

from __future__ import annotations

from enum import StrEnum


class ChunkStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


__all__ = ["ChunkStrategy"]
