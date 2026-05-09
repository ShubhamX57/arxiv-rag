"""PDF → structured sections via PyMuPDF.

Strategy (heuristic, no font analysis — keeps it portable and predictable):

1. Extract raw text per page using PyMuPDF's natural reading order.
2. Walk lines top-to-bottom; detect section headers via regex patterns:
   - Numbered: "1. Introduction", "3.2 Setup", "II. Methods"
   - Named:    "Abstract", "Introduction", "References", ...
3. Group text between headers into Section objects.
4. STOP at "References" / "Bibliography" / "Acknowledgments" — they're
   low-signal noise for retrieval (author names, titles, bibtex-like rows).
5. Drop sections under 50 chars (likely artifacts like "Figure 3:").

Output schema per section:
    {paper_id, section_index, section_title, text, page_start, page_end, char_count}

Why heuristics instead of font-based detection? ML papers come from many
templates (NeurIPS, ICML, ACL, plain LaTeX) and font sizes don't generalize.
Numbered/named headers are the most reliable signal across templates.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf
from tqdm import tqdm

from arxiv_rag.config import settings

log = logging.getLogger(__name__)


# Stop ingesting at these section names (low-signal for retrieval)
END_SECTION_KEYWORDS = (
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
)

# Named sections common in ML papers (case-insensitive)
NAMED_SECTIONS = (
    "Abstract",
    "Introduction",
    "Background",
    "Related Work",
    "Preliminaries",
    "Methodology",
    "Method",
    "Methods",
    "Approach",
    "Model",
    "Architecture",
    "Experiments",
    "Experimental Setup",
    "Setup",
    "Results",
    "Evaluation",
    "Analysis",
    "Discussion",
    "Limitations",
    "Conclusion",
    "Conclusions",
    "Future Work",
    "References",
    "Bibliography",
    "Acknowledgments",
    "Acknowledgements",
    "Appendix",
)

# Pre-built regexes
_NUMBERED_HEADER = re.compile(r"^\s*(\d+(?:\.\d+)*\.?)\s+([A-Z][\w\s\-:&,/']{1,80})\s*$")
_ROMAN_HEADER = re.compile(r"^\s*([IVX]+\.?)\s+([A-Z][\w\s\-:&,/']{1,80})\s*$")
_NAMED_HEADER = re.compile(
    r"^\s*(" + "|".join(re.escape(s) for s in NAMED_SECTIONS) + r")\s*:?\s*$",
    re.IGNORECASE,
)

# Lines that look like page artifacts — strip them when building section text
_ARTIFACT_PATTERNS = (
    re.compile(r"^\s*\d+\s*$"),  # bare page numbers
    re.compile(r"^\s*Page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*arXiv:\d+\.\d+v?\d*", re.IGNORECASE),
    re.compile(r"^\s*Preprint\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Under review.*$", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class Section:
    """One parsed section from one paper."""

    paper_id: str
    section_index: int
    section_title: str
    text: str
    page_start: int
    page_end: int
    char_count: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _detect_header(line: str) -> str | None:
    """Return canonical section title if `line` is a header, else None."""
    s = line.strip()
    # Headers are short. A 200-char "line" is body text wrapped by the parser.
    if not s or len(s) > 100:
        return None

    if m := _NUMBERED_HEADER.match(s):
        return f"{m.group(1)} {m.group(2).strip()}"
    if m := _ROMAN_HEADER.match(s):
        return f"{m.group(1)} {m.group(2).strip()}"
    if m := _NAMED_HEADER.match(s):
        # normalize capitalization to title case for consistency
        return m.group(1).strip().title()
    return None


def _is_end_section(title: str) -> bool:
    """True if the section title means we should stop ingesting this paper."""
    cleaned = re.sub(r"^[\dIVX.\s]+", "", title).strip().lower()
    return any(cleaned.startswith(kw) for kw in END_SECTION_KEYWORDS)


def _is_artifact(line: str) -> bool:
    return any(p.match(line) for p in _ARTIFACT_PATTERNS)


def _flush_section(
    out: list[Section],
    paper_id: str,
    index: int,
    title: str,
    text_parts: list[str],
    page_start: int,
    page_end: int,
    min_chars: int = 50,
) -> int:
    """Materialize a section if it has enough content. Returns next index."""
    text = "\n".join(text_parts).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse multi-blank lines
    if len(text) >= min_chars:
        out.append(
            Section(
                paper_id=paper_id,
                section_index=index,
                section_title=title,
                text=text,
                page_start=page_start,
                page_end=page_end,
                char_count=len(text),
            )
        )
        return index + 1
    return index


def parse_pdf(pdf_path: Path, paper_id: str) -> list[Section]:
    """Parse one PDF into a list of sections. Returns [] on failure."""
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        log.warning("Failed to open %s: %s", pdf_path, e)
        return []

    try:
        page_texts: list[tuple[int, str]] = [
            (i, page.get_text() or "") for i, page in enumerate(doc)
        ]
    finally:
        doc.close()

    sections: list[Section] = []
    cur_title = "Preamble"
    cur_index = 0
    cur_text: list[str] = []
    cur_pstart = 0
    cur_pend = 0
    stop = False

    for page_num, page_text in page_texts:
        if stop:
            break
        for line in page_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                if cur_text:
                    cur_text.append("")
                continue
            if _is_artifact(stripped):
                continue

            header = _detect_header(stripped)
            if header is not None:
                # Save current section
                cur_index = _flush_section(
                    sections,
                    paper_id,
                    cur_index,
                    cur_title,
                    cur_text,
                    cur_pstart,
                    cur_pend,
                )
                # Stop if references/bibliography
                if _is_end_section(header):
                    stop = True
                    break
                # Start new section
                cur_title = header
                cur_text = []
                cur_pstart = page_num
                cur_pend = page_num
            else:
                cur_text.append(stripped)
                cur_pend = page_num

    # Final flush (only if we didn't stop at end-section)
    if not stop:
        _flush_section(
            sections,
            paper_id,
            cur_index,
            cur_title,
            cur_text,
            cur_pstart,
            cur_pend,
        )

    return sections


def _parse_one_for_pool(args: tuple[str, str]) -> tuple[str, list[dict[str, object]], bool]:
    """Worker function. Returns (paper_id, section_dicts, success)."""
    pdf_path_str, paper_id = args
    sections = parse_pdf(Path(pdf_path_str), paper_id)
    return paper_id, [asdict(s) for s in sections], len(sections) > 0


def parse_all(
    metadata_path: Path | None = None,
    sections_path: Path | None = None,
    max_workers: int | None = None,
) -> dict[str, int]:
    """Parse every PDF listed in metadata.jsonl, write sections.jsonl.

    Returns stats: {papers_parsed, papers_failed, total_sections}.
    """
    metadata_path = metadata_path or settings.metadata_path
    sections_path = sections_path or settings.sections_path

    # Load tasks from metadata
    tasks: list[tuple[str, str]] = []
    missing_pdfs = 0
    with metadata_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                meta = json.loads(line)
                pdf_path = Path(meta["pdf_path"])
                if pdf_path.exists():
                    tasks.append((str(pdf_path), meta["id"]))
                else:
                    missing_pdfs += 1
            except (json.JSONDecodeError, KeyError) as e:
                log.warning("Bad metadata line: %s", e)

    if missing_pdfs:
        log.warning("Skipped %d papers — PDF file not found on disk.", missing_pdfs)

    log.info("Parsing %d PDFs...", len(tasks))
    sections_path.parent.mkdir(parents=True, exist_ok=True)

    parsed_papers = 0
    failed_papers = 0
    total_sections = 0

    with sections_path.open("w") as out_f, ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_parse_one_for_pool, t) for t in tasks]
        with tqdm(total=len(tasks), desc="Parsing") as pbar:
            for fut in as_completed(futures):
                try:
                    _, section_dicts, ok = fut.result()
                    if ok:
                        parsed_papers += 1
                        for sd in section_dicts:
                            out_f.write(json.dumps(sd, ensure_ascii=False) + "\n")
                        total_sections += len(section_dicts)
                    else:
                        failed_papers += 1
                except Exception as e:  # pragma: no cover
                    log.warning("Worker failed: %s", e)
                    failed_papers += 1
                pbar.update(1)

    return {
        "papers_parsed": parsed_papers,
        "papers_failed": failed_papers,
        "total_sections": total_sections,
    }


__all__ = ["Section", "parse_all", "parse_pdf"]
