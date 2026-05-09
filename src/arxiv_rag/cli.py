"""Typer-based CLI. Single entry point for the whole pipeline.

Usage:
    uv run arxiv-rag fetch
    uv run arxiv-rag fetch --max-results 200 --since 2024-06-01
    uv run arxiv-rag parse
    uv run arxiv-rag chunk
    uv run arxiv-rag index
    uv run arxiv-rag query "What is GQA?"
    uv run arxiv-rag eval --config full
"""

from __future__ import annotations

import logging

import typer

from arxiv_rag.config import settings

app = typer.Typer(
    add_completion=False,
    help="Production RAG over ArXiv ML papers.",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def fetch(
    max_results: int = typer.Option(
        settings.arxiv_max_results, "--max-results", "-n", help="Cap on papers fetched."
    ),
    categories: list[str] = typer.Option(
        settings.arxiv_categories, "--categories", "-c", help="ArXiv categories."
    ),
    since: str = typer.Option(settings.arxiv_since, "--since", help="YYYY-MM-DD lower bound."),
    until: str = typer.Option(settings.arxiv_until, "--until", help="YYYY-MM-DD upper bound."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fetch papers from ArXiv → data/pdfs + data/metadata.jsonl."""
    _setup_logging(verbose)
    settings.ensure_dirs()

    from arxiv_rag.ingest.arxiv_fetch import fetch_papers

    new = fetch_papers(
        categories=categories,
        max_results=max_results,
        since=since,
        until=until,
    )
    typer.echo(f"\n✔ Added {len(new)} new papers to {settings.metadata_path}")


@app.command()
def parse(
    max_workers: int = typer.Option(
        None, "--workers", "-w", help="Parallel workers (default: CPU count)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Parse PDFs in data/pdfs into structured sections."""
    _setup_logging(verbose)
    settings.ensure_dirs()

    from arxiv_rag.ingest.pdf_parse import parse_all

    stats = parse_all(max_workers=max_workers)
    typer.echo(
        f"\n✔ Parsed {stats['papers_parsed']} papers "
        f"({stats['papers_failed']} failed) → "
        f"{stats['total_sections']} sections at {settings.sections_path}"
    )


@app.command()
def chunk(
    strategy: str = typer.Option(
        "recursive",
        "--strategy",
        "-s",
        help="Chunking strategy: fixed | recursive (semantic on Day 5).",
    ),
    chunk_size: int = typer.Option(
        settings.chunk_size, "--chunk-size", help="Target chunk size in tokens."
    ),
    chunk_overlap: int = typer.Option(
        settings.chunk_overlap, "--overlap", help="Token overlap between chunks."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Chunk parsed sections into retrieval units."""
    _setup_logging(verbose)
    settings.ensure_dirs()

    from arxiv_rag.ingest.chunking import ChunkStrategy, chunk_sections

    try:
        strat = ChunkStrategy(strategy)
    except ValueError:
        typer.echo(
            f"Unknown strategy: {strategy!r}. "
            f"Choose from: {', '.join(s.value for s in ChunkStrategy)}"
        )
        raise typer.Exit(code=1) from None

    stats = chunk_sections(
        strategy=strat,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    typer.echo(
        f"\n✔ Strategy={strat.value}  papers={int(stats['papers'])}  "
        f"sections={int(stats['sections'])}  chunks={int(stats['total_chunks'])}"
    )
    typer.echo(
        f"  Token length — mean={stats['mean_tokens']}  "
        f"median={stats['median_tokens']}  p90={stats['p90_tokens']}  "
        f"p99={stats['p99_tokens']}"
    )


@app.command()
def index(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Build dense (LanceDB) and sparse (BM25) indexes."""
    _setup_logging(verbose)
    typer.echo("Not implemented yet — Week 1, Day 5.")
    raise typer.Exit(code=1)


@app.command()
def query(
    question: str = typer.Argument(..., help="The question to answer."),
    config: str = typer.Option("full", "--config", help="naive | hybrid | rerank | full"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a single query end-to-end."""
    _setup_logging(verbose)
    typer.echo(f"Not implemented yet (config={config}, q={question!r}) — Week 2.")
    raise typer.Exit(code=1)


@app.command()
def eval(
    config: str = typer.Option("full", "--config", help="Which retrieval config to evaluate."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the eval suite (RAGAS + custom)."""
    _setup_logging(verbose)
    typer.echo(f"Not implemented yet (config={config}) — Week 2.")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
