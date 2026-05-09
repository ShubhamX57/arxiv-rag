"""Centralized configuration using pydantic-settings.

All knobs live here. Override via env vars or .env file.
Never hardcode paths or model names elsewhere — import from here.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Project-wide settings. Loaded from env / .env / defaults."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Paths ---
    project_root: Path = Field(default=PROJECT_ROOT)
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    pdf_dir: Path = Field(default=PROJECT_ROOT / "data" / "pdfs")
    metadata_path: Path = Field(default=PROJECT_ROOT / "data" / "metadata.jsonl")
    sections_path: Path = Field(default=PROJECT_ROOT / "data" / "sections.jsonl")
    chunks_path: Path = Field(default=PROJECT_ROOT / "data" / "chunks.jsonl")
    lancedb_dir: Path = Field(default=PROJECT_ROOT / "data" / "lancedb")
    evals_dir: Path = Field(default=PROJECT_ROOT / "evals")

    # --- ArXiv ingestion ---
    arxiv_categories: list[str] = Field(default=["cs.LG", "cs.CL"])
    arxiv_max_results: int = Field(default=500)
    arxiv_since: str = Field(default="2024-01-01")
    arxiv_until: str = Field(default_factory=lambda: date.today().isoformat())
    arxiv_page_size: int = Field(default=100)
    arxiv_delay_seconds: float = Field(default=3.0)
    arxiv_num_retries: int = Field(default=5)

    # --- Chunking ---
    chunk_size: int = Field(default=512, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=64)

    # --- Embeddings ---
    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_batch_size: int = Field(default=32)
    embedding_device: str = Field(
        default="mps",
        description="'mps' for Apple Silicon, 'cuda', or 'cpu'",
    )

    # --- Reranker ---
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    rerank_top_k: int = Field(default=50, description="Candidates passed to reranker")
    final_top_k: int = Field(default=10, description="Chunks passed to LLM")

    # --- Generation (via LiteLLM) ---
    llm_model: str = Field(default="anthropic/claude-sonnet-4-5")
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=1024)

    # --- API keys (loaded from env) ---
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # --- Observability (optional) ---
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = Field(default="http://localhost:3000")

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        for path in (self.data_dir, self.pdf_dir, self.lancedb_dir, self.evals_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
