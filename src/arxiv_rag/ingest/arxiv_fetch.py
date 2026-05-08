"""Fetch papers from the ArXiv API and store metadata + PDFs locally.

Design notes:
- Resumable: we read existing metadata.jsonl and skip already-downloaded papers.
- Polite: ArXiv asks for >=3s between bulk requests; we honor that.
- Idempotent: re-running with the same args won't re-download or duplicate metadata.
- Date filtered: ArXiv search doesn't natively support date ranges in queries,
  so we sort by SubmittedDate desc and filter client-side.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import arxiv
from tqdm import tqdm

from arxiv_rag.config import settings

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PaperMetadata:
    """One ArXiv paper's metadata — what we keep for retrieval."""

    id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    primary_category: str
    published: str  # ISO 8601
    updated: str  # ISO 8601
    pdf_url: str
    pdf_path: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _load_existing_ids(metadata_path: Path) -> set[str]:
    """Read existing metadata.jsonl to find already-downloaded paper IDs."""
    if not metadata_path.exists():
        return set()
    ids: set[str] = set()
    with metadata_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                log.warning("Skipping malformed metadata line: %r", line[:80])
    return ids


def _build_query(categories: list[str]) -> str:
    """Build an ArXiv API query string from category list."""
    return " OR ".join(f"cat:{c}" for c in categories)


def _parse_date(s: str) -> datetime:
    """Parse YYYY-MM-DD into datetime (naive, UTC-ish)."""
    return datetime.strptime(s, "%Y-%m-%d")


def _to_metadata(result: arxiv.Result, pdf_path: Path) -> PaperMetadata:
    """Convert an arxiv.Result to our PaperMetadata."""
    paper_id = result.entry_id.rsplit("/", 1)[-1]
    return PaperMetadata(
        id=paper_id,
        title=result.title.strip(),
        authors=[str(a) for a in result.authors],
        abstract=result.summary.strip(),
        categories=list(result.categories),
        primary_category=result.primary_category,
        published=result.published.isoformat(),
        updated=result.updated.isoformat(),
        pdf_url=result.pdf_url,
        pdf_path=str(pdf_path),
    )


def fetch_papers(
    categories: list[str] | None = None,
    max_results: int | None = None,
    since: str | None = None,
    until: str | None = None,
    pdf_dir: Path | None = None,
    metadata_path: Path | None = None,
) -> list[PaperMetadata]:
    """Fetch papers from ArXiv, download PDFs, append to metadata.jsonl.

    Args:
        categories: ArXiv categories e.g. ["cs.LG", "cs.CL"].
        max_results: cap on papers returned by the API (we may filter further by date).
        since: inclusive lower bound on published date, format "YYYY-MM-DD".
        until: inclusive upper bound on published date, format "YYYY-MM-DD".
        pdf_dir: where PDFs go.
        metadata_path: jsonl file we append to (one paper per line).

    Returns:
        The newly-added PaperMetadata records (skips already-downloaded).
    """
    categories = categories or settings.arxiv_categories
    max_results = max_results or settings.arxiv_max_results
    since = since or settings.arxiv_since
    until = until or settings.arxiv_until
    pdf_dir = pdf_dir or settings.pdf_dir
    metadata_path = metadata_path or settings.metadata_path

    pdf_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    start_dt = _parse_date(since)
    end_dt = _parse_date(until)
    existing_ids = _load_existing_ids(metadata_path)
    log.info("Found %d existing papers in metadata; will skip them.", len(existing_ids))

    client = arxiv.Client(
        page_size=settings.arxiv_page_size,
        delay_seconds=settings.arxiv_delay_seconds,
        num_retries=settings.arxiv_num_retries,
    )

    search = arxiv.Search(
        query=_build_query(categories),
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    new_records: list[PaperMetadata] = []
    results: Iterable[arxiv.Result] = client.results(search)

    pbar = tqdm(results, total=max_results, desc="Fetching")
    with metadata_path.open("a") as out_f:
        for result in pbar:
            paper_id = result.entry_id.rsplit("/", 1)[-1]
            published = result.published.replace(tzinfo=None)

            # Date filter (ArXiv API doesn't filter natively)
            if not (start_dt <= published <= end_dt):
                continue

            if paper_id in existing_ids:
                continue

            pdf_path = pdf_dir / f"{paper_id}.pdf"
            if not pdf_path.exists():
                try:
                    result.download_pdf(dirpath=str(pdf_dir), filename=f"{paper_id}.pdf")
                except Exception as e:
                    log.warning("Failed to download %s: %s", paper_id, e)
                    continue

            meta = _to_metadata(result, pdf_path)
            out_f.write(meta.to_json() + "\n")
            out_f.flush()
            new_records.append(meta)
            existing_ids.add(paper_id)
            pbar.set_postfix(downloaded=len(new_records))

    log.info("Added %d new papers (total in corpus: %d).", len(new_records), len(existing_ids))
    return new_records
