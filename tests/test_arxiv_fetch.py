"""Tests for ArXiv fetcher pure-logic helpers (no network)."""

from __future__ import annotations

import json
from pathlib import Path

from arxiv_rag.ingest.arxiv_fetch import (
    PaperMetadata,
    _build_query,
    _load_existing_ids,
    _parse_date,
)


def test_build_query_single_category() -> None:
    assert _build_query(["cs.LG"]) == "cat:cs.LG"


def test_build_query_multiple_categories() -> None:
    assert _build_query(["cs.LG", "cs.CL"]) == "cat:cs.LG OR cat:cs.CL"


def test_parse_date() -> None:
    d = _parse_date("2024-06-15")
    assert d.year == 2024
    assert d.month == 6
    assert d.day == 15


def test_load_existing_ids_missing_file(tmp_path: Path) -> None:
    assert _load_existing_ids(tmp_path / "no.jsonl") == set()


def test_load_existing_ids_reads_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "metadata.jsonl"
    p.write_text(
        json.dumps({"id": "2401.0001", "title": "A"})
        + "\n"
        + json.dumps({"id": "2401.0002", "title": "B"})
        + "\n"
    )
    assert _load_existing_ids(p) == {"2401.0001", "2401.0002"}


def test_load_existing_ids_skips_malformed(tmp_path: Path) -> None:
    p = tmp_path / "metadata.jsonl"
    p.write_text(
        json.dumps({"id": "2401.0001"}) + "\n"
        "not valid json\n" + json.dumps({"id": "2401.0002"}) + "\n"
    )
    assert _load_existing_ids(p) == {"2401.0001", "2401.0002"}


def test_paper_metadata_to_json_roundtrips() -> None:
    m = PaperMetadata(
        id="2401.0001",
        title="A Test Paper",
        authors=["Alice", "Bob"],
        abstract="An abstract.",
        categories=["cs.LG"],
        primary_category="cs.LG",
        published="2024-01-01T00:00:00",
        updated="2024-01-02T00:00:00",
        pdf_url="https://arxiv.org/pdf/2401.0001",
        pdf_path="/tmp/2401.0001.pdf",
    )
    parsed = json.loads(m.to_json())
    assert parsed["id"] == "2401.0001"
    assert parsed["authors"] == ["Alice", "Bob"]
