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
from pathlib import Path

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
def index(
    strategy: str = typer.Option(
        "recursive",
        "--strategy",
        "-s",
        help="Which chunking strategy's chunks to index.",
    ),
    skip_dense: bool = typer.Option(False, "--skip-dense", help="Don't build LanceDB."),
    skip_sparse: bool = typer.Option(False, "--skip-sparse", help="Don't build BM25."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build dense (LanceDB) and sparse (BM25) indexes from chunks."""
    _setup_logging(verbose)
    settings.ensure_dirs()

    from arxiv_rag.ingest.chunking import ChunkStrategy
    from arxiv_rag.retrieval import (
        BM25Store,
        DenseStore,
        Embedder,
        bm25_path_for,
        chunks_table_name,
        load_chunks_jsonl,
    )

    try:
        strat = ChunkStrategy(strategy)
    except ValueError:
        typer.echo(f"Unknown strategy: {strategy!r}")
        raise typer.Exit(code=1) from None

    chunks_path = settings.data_dir / f"chunks_{strat.value}.jsonl"
    if not chunks_path.exists():
        typer.echo(
            f"No chunks at {chunks_path}. Run `arxiv-rag chunk --strategy {strat.value}` first."
        )
        raise typer.Exit(code=1)

    chunks = load_chunks_jsonl(chunks_path)
    typer.echo(f"Loaded {len(chunks)} chunks for strategy={strat.value}")

    if not skip_dense:
        embedder = Embedder()
        dense = DenseStore(table_name=chunks_table_name(strat.value))
        typer.echo(f"Building dense index with {embedder.model_name} on {embedder.device}...")
        n = dense.build(chunks, embedder=embedder, recreate=True)
        typer.echo(f"✔ LanceDB: {n} rows in table {dense.table_name}")

    if not skip_sparse:
        bm25 = BM25Store(index_path=bm25_path_for(strat.value))
        n = bm25.build(chunks)
        typer.echo(f"✔ BM25: {n} docs → {bm25.index_path}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Query text."),
    strategy: str = typer.Option(
        "recursive", "--strategy", "-s", help="Which strategy's index to query."
    ),
    mode: str = typer.Option("dense", "--mode", "-m", help="dense | sparse"),
    k: int = typer.Option(5, "--k", help="How many results."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Quick sanity check: search the index without LLM generation."""
    _setup_logging(verbose)

    from arxiv_rag.retrieval import (
        BM25Store,
        DenseStore,
        Embedder,
        bm25_path_for,
        chunks_table_name,
    )

    if mode == "dense":
        embedder = Embedder()
        qvec = embedder.embed([query], show_progress_bar=False)[0]
        store = DenseStore(table_name=chunks_table_name(strategy))
        hits = store.search(qvec, k=k)
        for i, dh in enumerate(hits, 1):
            text = dh.text[:200].replace("\n", " ")
            typer.echo(f"\n[{i}] {dh.paper_id} / {dh.section_title} (sim={dh.score:.3f})")
            typer.echo(f"    {text}...")
    elif mode == "sparse":
        sparse_store = BM25Store(index_path=bm25_path_for(strategy))
        sparse_hits = sparse_store.search(query, k=k)
        for i, sh in enumerate(sparse_hits, 1):
            text = sh.text[:200].replace("\n", " ")
            typer.echo(f"\n[{i}] {sh.paper_id} / {sh.section_title} (bm25={sh.score:.3f})")
            typer.echo(f"    {text}...")
    else:
        typer.echo(f"Unknown mode: {mode}. Use 'dense' or 'sparse'.")
        raise typer.Exit(code=1)


@app.command(name="search-hybrid")
def search_hybrid(
    query: str = typer.Argument(..., help="Query text."),
    strategy: str = typer.Option(
        "recursive", "--strategy", "-s", help="Which strategy's index to query."
    ),
    k: int = typer.Option(10, "--k", help="Final top-k after fusion."),
    fan_out_k: int = typer.Option(
        50, "--fan-out", help="Candidates fetched from each retriever before fusion."
    ),
    rrf_k: int = typer.Option(60, "--rrf-k", help="RRF constant (paper default = 60)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Hybrid (dense + BM25 via RRF) retrieval. Shows ranks from both retrievers."""
    _setup_logging(verbose)

    from arxiv_rag.retrieval import HybridRetriever

    retriever = HybridRetriever(strategy=strategy, rrf_k=rrf_k)
    hits = retriever.search(query, k=k, fan_out_k=fan_out_k)

    for i, h in enumerate(hits, 1):
        d = f"d{h.dense_rank}" if h.dense_rank else "d—"
        s = f"s{h.sparse_rank}" if h.sparse_rank else "s—"
        text = h.text[:200].replace("\n", " ")
        typer.echo(f"\n[{i}] {h.paper_id} / {h.section_title}  rrf={h.rrf_score:.4f}  ({d} {s})")
        typer.echo(f"    {text}...")
        typer.echo(f"    {text}...")


@app.command(name="generate-evals")
def generate_evals(
    strategy: str = typer.Option(
        "recursive", "--strategy", "-s", help="Which chunking strategy to draw from."
    ),
    n_single: int = typer.Option(200, "--n-single", help="Single-hop questions to generate."),
    n_no_answer: int = typer.Option(30, "--n-no-answer", help="No-answer questions."),
    seed: int = typer.Option(42, "--seed", help="Random seed for chunk sampling."),
    min_confidence: float = typer.Option(
        0.6, "--min-confidence", help="Filter out items below this confidence."
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="Output path (default: evals/eval_set_raw.jsonl)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate a raw synthetic Q&A eval set. Manual review required after."""
    _setup_logging(verbose)
    settings.ensure_dirs()

    from arxiv_rag.evals import generate_eval_set, write_jsonl
    from arxiv_rag.retrieval import load_chunks_jsonl

    chunks_path = settings.data_dir / f"chunks_{strategy}.jsonl"
    if not chunks_path.exists():
        typer.echo(
            f"No chunks at {chunks_path}. Run `arxiv-rag chunk --strategy {strategy}` first."
        )
        raise typer.Exit(code=1)

    out_path = output or (settings.evals_dir / "eval_set_raw.jsonl")
    chunks = load_chunks_jsonl(chunks_path)
    typer.echo(f"Loaded {len(chunks)} chunks. Generating eval items...")

    items, stats = generate_eval_set(
        chunks,
        n_single_hop=n_single,
        n_no_answer=n_no_answer,
        seed=seed,
        min_confidence=min_confidence,
    )
    write_jsonl(items, out_path)
    typer.echo(f"\n✔ Wrote {len(items)} eval items to {out_path}")
    typer.echo(f"  Stats: {stats}")
    typer.echo(f"\n⚠ This is the RAW set. Now manually review {out_path} → evals/eval_set.jsonl")


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


@app.command(name="eval-retrieval")
def eval_retrieval(
    config: str = typer.Option(
        "hybrid", "--config", "-c", help="Which retriever to evaluate: dense | sparse | hybrid"
    ),
    strategy: str = typer.Option(
        "recursive", "--strategy", "-s", help="Index strategy to evaluate against."
    ),
    eval_set: Path = typer.Option(
        None, "--eval-set", help="Eval set JSONL. Default: evals/eval_set.jsonl"
    ),
    top_k: int = typer.Option(10, "--top-k", help="Top-k for retrieval metrics."),
    rrf_k: int = typer.Option(60, "--rrf-k", help="RRF constant for hybrid."),
    fan_out_k: int = typer.Option(
        50, "--fan-out", help="Per-retriever candidates before fusion (hybrid only)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Compute Recall@k, MRR@k, nDCG@k against the eval set, save to disk, append ablation row."""
    _setup_logging(verbose)
    settings.ensure_dirs()

    from arxiv_rag.evals import (
        append_to_ablation_md,
        evaluate_retriever,
        format_ablation_row,
        load_eval_set,
        save_run,
    )
    from arxiv_rag.retrieval import (
        BM25Store,
        DenseStore,
        Embedder,
        HybridRetriever,
        bm25_path_for,
        chunks_table_name,
    )

    # Resolve eval set path: prefer the reviewed gold set; fall back to raw.
    eval_path = eval_set or (settings.evals_dir / "eval_set.jsonl")
    if not eval_path.exists():
        raw = settings.evals_dir / "eval_set_raw.jsonl"
        if raw.exists():
            typer.echo(
                f"No reviewed set at {eval_path}. Falling back to RAW set: {raw}\n"
                f"(Reviewed sets give cleaner numbers — do the manual filter pass.)"
            )
            eval_path = raw
        else:
            typer.echo("No eval set found. Run `arxiv-rag generate-evals` first.")
            raise typer.Exit(code=1)

    items = load_eval_set(eval_path)
    typer.echo(f"Loaded {len(items)} eval items from {eval_path}")

    # Build a retrieve_fn for the chosen config. All return list[chunk_id] for the harness.
    if config == "dense":
        embedder = Embedder()
        dense_store = DenseStore(table_name=chunks_table_name(strategy))

        def retrieve_fn(q: str, k: int) -> list[str]:
            qv = embedder.embed([q], show_progress_bar=False)[0]
            return [h.chunk_id for h in dense_store.search(qv, k=k)]

    elif config == "sparse":
        sparse_store = BM25Store(index_path=bm25_path_for(strategy))

        def retrieve_fn(q: str, k: int) -> list[str]:
            return [h.chunk_id for h in sparse_store.search(q, k=k)]

    elif config == "hybrid":
        retriever = HybridRetriever(strategy=strategy, rrf_k=rrf_k)

        def retrieve_fn(q: str, k: int) -> list[str]:
            return [h.chunk_id for h in retriever.search(q, k=k, fan_out_k=fan_out_k)]

    else:
        typer.echo(f"Unknown config: {config!r}. Use dense | sparse | hybrid.")
        raise typer.Exit(code=1)

    aggregate, per_query = evaluate_retriever(
        retrieve_fn, items, config_name=config, strategy=strategy, top_k=top_k
    )

    runs_dir = settings.evals_dir / "runs"
    save_run(aggregate, per_query, runs_dir)
    append_to_ablation_md(aggregate, settings.evals_dir / "ablation.md")

    typer.echo(f"\n=== Results: config={config}, strategy={strategy}, k={top_k} ===")
    typer.echo(
        f"Queries evaluated: {aggregate.n_evaluated} (no-answer skipped: {aggregate.n_no_answer})"
    )
    typer.echo(f"Recall@{top_k}: {aggregate.recall_at_k:.3f}")
    typer.echo(f"MRR@{top_k}:    {aggregate.mrr_at_k:.3f}")
    typer.echo(f"nDCG@{top_k}:   {aggregate.ndcg_at_k:.3f}")
    typer.echo(f"\nAblation row appended to {settings.evals_dir / 'ablation.md'}:")
    typer.echo(format_ablation_row(aggregate))


if __name__ == "__main__":
    app()
