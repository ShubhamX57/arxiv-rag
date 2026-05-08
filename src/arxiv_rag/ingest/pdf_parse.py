"""PDF → structured sections via PyMuPDF.

TODO (Week 1, Day 3-4):
- Open each PDF in data/pdfs/, extract text per page
- Detect section headers (font size heuristic, regex on numbered headers)
- Preserve metadata: paper_id, section_title, page_range
- Skip references / bibliography sections (low retrieval value, lots of noise)
- Output: list of {paper_id, section_title, text, page_start, page_end}
- Write to data/sections.jsonl
"""

from __future__ import annotations

from pathlib import Path


def parse_pdf(pdf_path: Path) -> list[dict[str, object]]:
    """Parse one PDF into structured sections. Returns list of section dicts.

    Stub for now — implement section detection in Week 1 Day 3.
    """
    import pymupdf  # noqa: F401  # will be used once implemented

    raise NotImplementedError("Implement in Week 1 Day 3")


__all__ = ["parse_pdf"]
