"""Tests for PDF parsing.

Strategy: build a synthetic PDF with a known structure (numbered sections,
a references section we should stop at, page-number artifacts) and assert
the parser extracts what we expect. No network needed, runs in <1s.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf  # type: ignore[import-untyped]
import pytest

from arxiv_rag.ingest.pdf_parse import (
    Section,
    _detect_header,
    _is_artifact,
    _is_end_section,
    parse_pdf,
)

# ----------------------- pure-logic tests -----------------------


@pytest.mark.parametrize(
    "line, expected",
    [
        ("1. Introduction", "1. Introduction"),
        ("2 Methodology", "2 Methodology"),
        ("3.1 Experimental Setup", "3.1 Experimental Setup"),
        ("Abstract", "Abstract"),
        ("INTRODUCTION", "Introduction"),
        ("References", "References"),
        ("Related Work", "Related Work"),
        ("II. Methods", "II. Methods"),
    ],
)
def test_detect_header_positive(line: str, expected: str) -> None:
    assert _detect_header(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "",
        "   ",
        "This is just a regular sentence in the body of the paper.",
        "we showed that the model achieves 95% accuracy",
        "1.",  # number alone
        "A" * 200,  # too long to be a header
        "lower case words only",
    ],
)
def test_detect_header_negative(line: str) -> None:
    assert _detect_header(line) is None


@pytest.mark.parametrize(
    "title, expected",
    [
        ("References", True),
        ("6. References", True),
        ("Bibliography", True),
        ("Acknowledgments", True),
        ("Acknowledgements", True),
        ("Introduction", False),
        ("3. Methodology", False),
        ("Conclusion", False),
    ],
)
def test_is_end_section(title: str, expected: bool) -> None:
    assert _is_end_section(title) is expected


@pytest.mark.parametrize(
    "line, expected",
    [
        ("12", True),
        ("Page 5", True),
        ("Page 5 of 20", True),
        ("arXiv:2401.12345v1", True),
        ("Preprint", True),
        ("Under review at ICLR 2026", True),
        ("This paper proposes a new method", False),
        ("Section 3 covers experiments", False),
    ],
)
def test_is_artifact(line: str, expected: bool) -> None:
    assert _is_artifact(line) is expected


# ----------------------- end-to-end with synthetic PDF -----------------------


def _build_synthetic_pdf(path: Path) -> None:
    """Create a 2-page PDF mimicking a real ML paper's structure."""
    doc = pymupdf.open()

    page1 = doc.new_page(width=612, height=792)
    page1_text = (
        "A Tiny Synthetic Paper for Testing\n"
        "\n"
        "Jane Doe, John Smith\n"
        "\n"
        "Abstract\n"
        "We propose a method for testing PDF parsers. Our approach achieves\n"
        "significant improvements over prior baselines. The method is simple\n"
        "and effective on a range of synthetic benchmarks.\n"
        "\n"
        "1. Introduction\n"
        "Recent advances have led to breakthroughs in retrieval. We argue that\n"
        "robust ingestion is the foundation. This paper introduces a synthetic\n"
        "test that exercises the parser end-to-end.\n"
        "\n"
        "1\n"  # page number artifact
    )
    page1.insert_text((50, 50), page1_text, fontsize=11)

    page2 = doc.new_page(width=612, height=792)
    page2_text = (
        "2. Methodology\n"
        "Our method has three steps. First, we extract text. Second, we detect\n"
        "section boundaries with regex. Third, we group lines into sections.\n"
        "We use heuristics tuned for ML paper conventions.\n"
        "\n"
        "3. Results\n"
        "We achieve 99.9% precision on the synthetic benchmark. The method is\n"
        "robust to common artifacts including page numbers and headers.\n"
        "\n"
        "References\n"
        "[1] Smith J. Some prior work. Conf. 2020.\n"
        "[2] Doe J. Other prior work. Conf. 2021.\n"
        "\n"
        "Page 2\n"
    )
    page2.insert_text((50, 50), page2_text, fontsize=11)

    doc.save(str(path))
    doc.close()


@pytest.fixture
def synthetic_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "synthetic.pdf"
    _build_synthetic_pdf(p)
    return p


def test_parse_pdf_returns_sections(synthetic_pdf: Path) -> None:
    sections = parse_pdf(synthetic_pdf, paper_id="test.001")
    assert all(isinstance(s, Section) for s in sections)
    assert len(sections) >= 3  # Abstract, Introduction, Methodology, Results


def test_parse_pdf_section_titles(synthetic_pdf: Path) -> None:
    sections = parse_pdf(synthetic_pdf, paper_id="test.001")
    titles = [s.section_title for s in sections]
    # The synthetic PDF has these in order
    assert "Abstract" in titles
    assert any(t.startswith("1.") and "Introduction" in t for t in titles)
    assert any(t.startswith("2.") and "Methodology" in t for t in titles)
    assert any(t.startswith("3.") and "Results" in t for t in titles)


def test_parse_pdf_drops_references(synthetic_pdf: Path) -> None:
    sections = parse_pdf(synthetic_pdf, paper_id="test.001")
    titles_lower = [s.section_title.lower() for s in sections]
    assert not any("reference" in t for t in titles_lower)
    # And the ref content shouldn't leak into a previous section
    full_text = " ".join(s.text for s in sections).lower()
    assert "smith j. some prior work" not in full_text


def test_parse_pdf_strips_artifacts(synthetic_pdf: Path) -> None:
    sections = parse_pdf(synthetic_pdf, paper_id="test.001")
    full_text = "\n".join(s.text for s in sections)
    # Page number artifacts should have been removed
    assert "\nPage 2\n" not in full_text


def test_parse_pdf_assigns_paper_id(synthetic_pdf: Path) -> None:
    sections = parse_pdf(synthetic_pdf, paper_id="2401.99999")
    assert all(s.paper_id == "2401.99999" for s in sections)


def test_parse_pdf_section_indexes_are_sequential(synthetic_pdf: Path) -> None:
    sections = parse_pdf(synthetic_pdf, paper_id="test.001")
    indexes = [s.section_index for s in sections]
    assert indexes == list(range(len(indexes)))


def test_parse_pdf_handles_missing_file(tmp_path: Path) -> None:
    # Should return [], not raise
    out = parse_pdf(tmp_path / "does_not_exist.pdf", paper_id="x")
    assert out == []


def test_parse_pdf_char_count_matches(synthetic_pdf: Path) -> None:
    sections = parse_pdf(synthetic_pdf, paper_id="test.001")
    for s in sections:
        assert s.char_count == len(s.text)
