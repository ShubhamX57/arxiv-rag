"""Day 11 cli.py fix — robust AST-based patch.

This supersedes the broken `apply_day11_cli_patch.py` and
`apply_day11_cli_eval_gen.py`. Run from the project root:

    python apply_day11_cli_fix.py

What it does:
  1. Finds the `query` function in src/arxiv_rag/cli.py
  2. Replaces it entirely with the Day 11 version (top_k=3, prints quotes)
  3. Appends a new `eval-generation` command before `if __name__ == "__main__":`

It uses Python's `ast` module to locate function boundaries by line number,
so there's no string escaping, no regex on source code, and no risk of
mangling f-strings. The new function bodies are stored as separate .py
template files (no escape sequences) and inserted verbatim.

Idempotent: running it twice is a no-op.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

CLI_PATH = Path("src/arxiv_rag/cli.py")


# ----- NEW QUERY COMMAND -----
# Written as a triple-quoted raw-ish block. The only escape sequences are
# real \\ (in things like the docstring), and \n inside string literals
# that we genuinely want to be newline escapes. Python will parse this
# correctly when it reads cli.py — because we are writing actual Python
# source, not strings-of-Python-source.

NEW_QUERY = '''@app.command()
def query(
    question: str = typer.Argument(..., help="The question to answer."),
    config: str = typer.Option(
        "rerank",
        "--config",
        help="Retrieval config: hybrid | rerank | multi-query | hyde",
    ),
    top_k: int = typer.Option(3, "--top-k", help="Chunks to feed into the prompt."),
    strategy: str = typer.Option("recursive", "--strategy", help="Chunking strategy."),
    show_chunks: bool = typer.Option(
        False, "--show-chunks", help="Print retrieved chunks alongside the answer."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a single query end-to-end: retrieve, rerank, generate a grounded answer."""
    _setup_logging(verbose)
    settings.ensure_dirs()

    from arxiv_rag.generation import Generator, RAGPipeline
    from arxiv_rag.retrieval import RerankingRetriever
    from arxiv_rag.retrieval.rewriters import RewriteRerankRetriever

    if config in {"rerank", "hybrid"}:
        retriever: object = RerankingRetriever(strategy=strategy)
    elif config in {"multi-query", "hyde"}:
        retriever = RewriteRerankRetriever(strategy=config, retrieval_strategy=strategy)
    else:
        typer.echo(f"Unknown config: {config!r}. Use hybrid | rerank | multi-query | hyde.")
        raise typer.Exit(code=1)

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=Generator(),
        top_k_for_generation=top_k,
    )

    typer.echo(f"\\n? {question}\\n")
    typer.echo("Retrieving and generating...")
    result = pipeline.query(question)

    if show_chunks:
        typer.echo(f"\\n--- Retrieved {len(result.retrieved)} chunks ---")
        for i, c in enumerate(result.retrieved, start=1):
            typer.echo(
                f"\\n[{i}] chunk_id={c.chunk_id}  rerank_score={c.rerank_score:.3f}"
            )
            typer.echo(f"    section: {c.section_title}")
            text_preview = c.text[:200].replace("\\n", " ")
            typer.echo(f"    text:    {text_preview}{'...' if len(c.text) > 200 else ''}")

    typer.echo("\\n--- Answer ---")
    if result.answer.abstained:
        typer.echo(f"(abstained) {result.answer.answer}")
    else:
        typer.echo(result.answer.answer)
        if result.answer.citations:
            typer.echo("\\nGrounded in:")
            for cit in result.answer.citations:
                typer.echo(f"  - {cit.chunk_id}")
                typer.echo(f'    "{cit.supporting_quote}"')
'''


# ----- NEW EVAL-GENERATION COMMAND -----

NEW_EVAL_GEN = '''@app.command(name="eval-generation")
def eval_generation(
    eval_set: Path = typer.Option(
        None,
        "--eval-set",
        help="JSONL eval set with no_answer items. Default: evals/adversarial.jsonl",
    ),
    config: str = typer.Option(
        "rerank", "--config", help="Retrieval config: rerank | multi-query | hyde"
    ),
    strategy: str = typer.Option("recursive", "--strategy"),
    top_k: int = typer.Option(3, "--top-k"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Measure refusal accuracy on out-of-corpus questions."""
    _setup_logging(verbose)
    settings.ensure_dirs()

    import json
    from datetime import datetime, timezone

    from arxiv_rag.evals.schema import QuestionType, load_eval_set
    from arxiv_rag.generation import Generator, RAGPipeline
    from arxiv_rag.retrieval import RerankingRetriever
    from arxiv_rag.retrieval.rewriters import RewriteRerankRetriever

    eval_path = eval_set or (settings.evals_dir / "adversarial.jsonl")
    if not eval_path.exists():
        typer.echo(f"Eval set not found at {eval_path}")
        raise typer.Exit(code=1)

    items = load_eval_set(eval_path)
    no_answer_items = [i for i in items if i.type == QuestionType.NO_ANSWER]
    if not no_answer_items:
        typer.echo("No NO_ANSWER items in the eval set.")
        raise typer.Exit(code=1)

    typer.echo(f"Loaded {len(no_answer_items)} no-answer items from {eval_path}")

    if config in {"rerank", "hybrid"}:
        retriever: object = RerankingRetriever(strategy=strategy)
    elif config in {"multi-query", "hyde"}:
        retriever = RewriteRerankRetriever(strategy=config, retrieval_strategy=strategy)
    else:
        typer.echo(f"Unknown config: {config!r}")
        raise typer.Exit(code=1)

    pipeline = RAGPipeline(
        retriever=retriever, generator=Generator(), top_k_for_generation=top_k
    )

    n_refused = 0
    n_false_grounded = 0
    per_query: list[dict] = []

    for item in no_answer_items:
        result = pipeline.query(item.question)
        refused = result.answer.abstained
        if refused:
            n_refused += 1
        if not refused and result.answer.citations:
            n_false_grounded += 1
        per_query.append({
            "id": item.id,
            "question": item.question,
            "refused": refused,
            "answer": result.answer.answer,
            "n_citations": len(result.answer.citations),
            "citations": [
                {"chunk_id": c.chunk_id, "supporting_quote": c.supporting_quote}
                for c in result.answer.citations
            ],
        })

    n = len(no_answer_items)
    refusal_accuracy = n_refused / n
    false_grounding_rate = n_false_grounded / n

    ts = datetime.now(timezone.utc).isoformat().replace(":", "-").replace(".", "-")
    out_dir = settings.evals_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_q_path = out_dir / f"generation_{config}_{ts}.jsonl"
    agg_path = out_dir / f"generation_{config}_{ts}.json"
    with per_q_path.open("w") as f:
        for row in per_query:
            f.write(json.dumps(row, ensure_ascii=False) + "\\n")
    with agg_path.open("w") as f:
        json.dump(
            {
                "config": config,
                "n": n,
                "refusal_accuracy": refusal_accuracy,
                "false_grounding_rate": false_grounding_rate,
                "timestamp": ts,
            },
            f,
            indent=2,
        )

    typer.echo(f"\\n=== Generation eval: config={config}, n={n} ===")
    typer.echo(
        f"Refusal accuracy:      {refusal_accuracy:.3f}  "
        f"({n_refused}/{n} correctly abstained)"
    )
    typer.echo(
        f"False-grounding rate:  {false_grounding_rate:.3f}  "
        f"({n_false_grounded}/{n} answered with fake citations)"
    )
    typer.echo(f"\\nPer-query results: {per_q_path}")
'''


def find_function_span(tree: ast.Module, name: str) -> tuple[int, int] | None:
    """Return (start_line, end_line) inclusive, 1-indexed, of the function
    decorator block + body. Returns None if not found.
    """
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            # Decorators sit on earlier lines; lineno on the FunctionDef points
            # to the `def` line. Earliest decorator line is the true start.
            start = node.lineno
            if node.decorator_list:
                start = min(start, min(d.lineno for d in node.decorator_list))
            end = node.end_lineno or node.lineno
            return start, end
    return None


def find_main_guard_line(tree: ast.Module) -> int | None:
    """Return the 1-indexed line number of `if __name__ == '__main__':`."""
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                return node.lineno
    return None


def is_already_patched(src: str) -> bool:
    """Check for Day 11 markers."""
    return (
        "top_k: int = typer.Option(3," in src
        and "cit.supporting_quote" in src
        and 'name="eval-generation"' in src
    )


def main() -> int:
    if not CLI_PATH.exists():
        print(f"ERROR: {CLI_PATH} not found. Run from the project root.")
        return 1

    src = CLI_PATH.read_text()

    # Try parsing — if cli.py is currently broken, bail with a clear message.
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"ERROR: {CLI_PATH} has a syntax error at line {e.lineno}.")
        print("Run `git checkout src/arxiv_rag/cli.py` to revert it to the version on main,")
        print("then re-run this script.")
        return 2

    if is_already_patched(src):
        print(f"{CLI_PATH} is already at Day 11. Nothing to do.")
        return 0

    # Find the existing `query` function span
    span = find_function_span(tree, "query")
    if span is None:
        print(f"ERROR: could not find `def query(` in {CLI_PATH}.")
        return 3
    q_start, q_end = span

    # Find the `if __name__` guard so we can insert eval-generation before it
    guard_line = find_main_guard_line(tree)
    if guard_line is None:
        print(f"ERROR: no `if __name__ == '__main__':` block found in {CLI_PATH}.")
        return 4

    # Sanity: the guard must come AFTER the query function. If it doesn't, bail.
    if guard_line <= q_end:
        print("ERROR: structural assumption violated (query comes after main guard).")
        return 5

    lines = src.splitlines(keepends=True)

    # Build the new file:
    #   [lines before query]  +  [new query]  +  [lines between query and guard]
    #   + [new eval-generation]  +  [main guard onwards]
    before_query = "".join(lines[: q_start - 1])
    between = "".join(lines[q_end : guard_line - 1])
    main_guard_onwards = "".join(lines[guard_line - 1 :])

    # Each replacement block ends with exactly one trailing newline. Ensure the
    # spacing around them is right: blank line before/after each, so the file
    # stays PEP-8 clean.
    parts = [
        before_query.rstrip() + "\n\n\n",
        NEW_QUERY.rstrip() + "\n\n\n",
        between.strip() + ("\n\n\n" if between.strip() else ""),
        NEW_EVAL_GEN.rstrip() + "\n\n\n",
        main_guard_onwards,
    ]
    new_src = "".join(parts)

    # Validate the result parses before writing
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        print(f"ERROR: generated source has a syntax error at line {e.lineno}:")
        print(f"  {e.msg}")
        # Write to a debug file so the user can inspect
        debug = CLI_PATH.with_suffix(".broken.py")
        debug.write_text(new_src)
        print(f"  Debug output written to {debug} — DO NOT use this; keep your cli.py.")
        return 6

    CLI_PATH.write_text(new_src)
    print(f"Patched {CLI_PATH}:")
    print(f"  - Replaced `query` (was lines {q_start}-{q_end})")
    print("  - Added `eval-generation` before the main guard")
    print("  - Verified the result parses cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
