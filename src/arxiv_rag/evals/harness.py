"""Retrieval evaluation harness.

Given:
- A `retrieve_fn`: takes (query, k) and returns a ranked list of chunk_ids.
- A list of `EvalItem`s with gold chunk_ids.
- A top_k value.

Produces:
- Per-query: rank-of-first-gold, Recall@k indicator, MRR@k, nDCG@k.
- Aggregate: means across all queries, plus a count of queries evaluated.
- A markdown row that can be appended to evals/runs/ablation.md.

The harness is retriever-agnostic — pass any function with the right signature
and you can benchmark it. That's how we'll compare dense vs sparse vs hybrid
(and later, hybrid+rerank, hybrid+rerank+HyDE, etc.) on the same eval set.

No-answer questions are EXCLUDED from the main retrieval metrics for now —
they need a separate "did the system know to abstain?" metric that lives at
the generation layer (Day 9), not here. We track them in stats so they're
visible but they don't pollute Recall@k numbers.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from arxiv_rag.evals.metrics import ndcg_at_k, recall_at_k, reciprocal_rank_at_k
from arxiv_rag.evals.schema import EvalItem, QuestionType

log = logging.getLogger(__name__)


# A retriever function: (query, k) -> list of chunk_ids, ranked best-first.
RetrieveFn = Callable[[str, int], list[str]]


@dataclass(frozen=True, slots=True)
class PerQueryResult:
    """One row of the per-query results JSONL."""

    eval_id: str
    question: str
    gold_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    rank_of_first_gold: int | None  # None if no gold in top-k
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float


@dataclass(slots=True)
class AggregateMetrics:
    """Summary stats for one config run."""

    config_name: str
    strategy: str
    top_k: int
    n_evaluated: int  # number of queries used in metrics (excludes no-answer)
    n_no_answer: int  # number of no-answer queries in the set (not scored)
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def evaluate_retriever(
    retrieve_fn: RetrieveFn,
    eval_items: list[EvalItem],
    config_name: str,
    strategy: str = "recursive",
    top_k: int = 10,
) -> tuple[AggregateMetrics, list[PerQueryResult]]:
    """Run `retrieve_fn` over `eval_items` and compute metrics.

    Returns (aggregate_metrics, per_query_results). Caller can then save both.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not eval_items:
        raise ValueError("eval_items is empty")

    per_query: list[PerQueryResult] = []
    n_no_answer = 0
    recall_sum = 0.0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    n_scored = 0

    for item in eval_items:
        if item.type == QuestionType.NO_ANSWER:
            n_no_answer += 1
            continue

        retrieved = retrieve_fn(item.question, top_k)
        gold = item.source_chunk_ids
        gold_set = set(gold)

        # Rank of first gold (1-indexed), or None if not in top-k
        first_rank: int | None = None
        for rank, cid in enumerate(retrieved[:top_k], start=1):
            if cid in gold_set:
                first_rank = rank
                break

        r = recall_at_k(retrieved, gold, top_k)
        m = reciprocal_rank_at_k(retrieved, gold, top_k)
        nd = ndcg_at_k(retrieved, gold, top_k)

        per_query.append(
            PerQueryResult(
                eval_id=item.id,
                question=item.question,
                gold_chunk_ids=list(gold),
                retrieved_chunk_ids=retrieved[:top_k],
                rank_of_first_gold=first_rank,
                recall_at_k=r,
                mrr_at_k=m,
                ndcg_at_k=nd,
            )
        )
        recall_sum += r
        mrr_sum += m
        ndcg_sum += nd
        n_scored += 1

    if n_scored == 0:
        raise ValueError("No scorable items (all were no-answer?). Check your eval set.")

    aggregate = AggregateMetrics(
        config_name=config_name,
        strategy=strategy,
        top_k=top_k,
        n_evaluated=n_scored,
        n_no_answer=n_no_answer,
        recall_at_k=recall_sum / n_scored,
        mrr_at_k=mrr_sum / n_scored,
        ndcg_at_k=ndcg_sum / n_scored,
    )
    return aggregate, per_query


def save_run(
    aggregate: AggregateMetrics,
    per_query: list[PerQueryResult],
    runs_dir: Path,
) -> tuple[Path, Path]:
    """Write per-query results and aggregate JSON to runs_dir.

    Returns (per_query_path, aggregate_path).
    """
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Use config + strategy + timestamp for filename to avoid collisions across runs
    # Replace ':' in ISO timestamp with '-' for filesystem safety
    safe_ts = aggregate.timestamp.replace(":", "-").replace(".", "-")
    base = f"{aggregate.config_name}_{aggregate.strategy}_k{aggregate.top_k}_{safe_ts}"

    per_query_path = runs_dir / f"{base}.jsonl"
    aggregate_path = runs_dir / f"{base}.json"

    with per_query_path.open("w") as f:
        for row in per_query:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    with aggregate_path.open("w") as f:
        json.dump(asdict(aggregate), f, indent=2)

    log.info("Wrote %s and %s", per_query_path, aggregate_path)
    return per_query_path, aggregate_path


# ------------- ablation table rendering -------------

ABLATION_HEADER = (
    "| Config | Strategy | k | Recall@k | MRR@k | nDCG@k | n_eval | timestamp |\n"
    "|--------|----------|---|----------|-------|--------|--------|-----------|"
)


def format_ablation_row(m: AggregateMetrics) -> str:
    """Format one aggregate as a markdown table row."""
    # Truncate ISO timestamp to date+hour for readability
    ts_short = m.timestamp[:16]
    return (
        f"| {m.config_name} | {m.strategy} | {m.top_k} "
        f"| {m.recall_at_k:.3f} | {m.mrr_at_k:.3f} | {m.ndcg_at_k:.3f} "
        f"| {m.n_evaluated} | {ts_short} |"
    )


def append_to_ablation_md(m: AggregateMetrics, path: Path) -> None:
    """Append one row to the ablation table. Create file with header if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Retrieval ablation\n\n"
            "Recall@k = at least one gold chunk in top-k. "
            "MRR@k = 1/rank of first gold. "
            "nDCG@k = position-weighted gold hits, normalized.\n\n"
            f"{ABLATION_HEADER}\n"
        )
    with path.open("a") as f:
        f.write(format_ablation_row(m) + "\n")


__all__ = [
    "ABLATION_HEADER",
    "AggregateMetrics",
    "PerQueryResult",
    "RetrieveFn",
    "append_to_ablation_md",
    "evaluate_retriever",
    "format_ablation_row",
    "save_run",
]
