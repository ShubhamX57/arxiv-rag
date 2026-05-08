"""Smoke tests for config module."""

from __future__ import annotations

from pathlib import Path

from arxiv_rag.config import Settings, settings


def test_settings_loads() -> None:
    """Settings should load with defaults."""
    assert isinstance(settings, Settings)
    assert settings.arxiv_max_results > 0
    assert settings.embedding_model
    assert settings.llm_model


def test_paths_are_absolute() -> None:
    """All path settings should be absolute paths under project root."""
    assert settings.data_dir.is_absolute()
    assert settings.pdf_dir.is_absolute()
    assert settings.lancedb_dir.is_absolute()


def test_default_categories() -> None:
    """Default categories include the ML core ones."""
    assert "cs.LG" in settings.arxiv_categories
    assert "cs.CL" in settings.arxiv_categories


def test_ensure_dirs(tmp_path: Path, monkeypatch) -> None:
    """ensure_dirs() creates missing directories."""
    s = Settings(
        data_dir=tmp_path / "data",
        pdf_dir=tmp_path / "data" / "pdfs",
        lancedb_dir=tmp_path / "data" / "lancedb",
        evals_dir=tmp_path / "evals",
    )
    s.ensure_dirs()
    assert s.data_dir.exists()
    assert s.pdf_dir.exists()
    assert s.lancedb_dir.exists()
    assert s.evals_dir.exists()
