"""Tests for the eval harness. Uses fake retrievers — no real LanceDB / BM25 needed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arxiv_rag.evals.harness import (
    ABLATION_HEADER,
    AggregateMetrics,
    append_to_ablation_md,
    evaluate_retriever,
    format_ablation_row,
    save_run,
)
from arxiv_rag.evals.schema import EvalItem, QuestionType


def _make_eval_set() -> list[EvalItem]:
    """3 single-hop + 1 no-answer; total 4 items."""
    return [
        EvalItem(
            id="sh-0",
            question="What is X?",
            gold_answer="X is foo.",
            type=QuestionType.SINGLE_HOP,
            source_paper_id="p1",
            source_chunk_ids=["c1"],
        ),
        EvalItem(
            id="sh-1",
            question="What is Y?",
            gold_answer="Y is bar.",
            type=QuestionType.SINGLE_HOP,
            source_paper_id="p2",
            source_chunk_ids=["c2"],
        ),
        EvalItem(
            id="sh-2",
            question="What is Z?",
            gold_answer="Z is baz.",
            type=QuestionType.SINGLE_HOP,
            source_paper_id="p3",
            source_chunk_ids=["c3"],
        ),
        EvalItem(
            id="na-0",
            question="What is the capital of Mars?",
            gold_answer="Not answerable from corpus.",
            type=QuestionType.NO_ANSWER,
            source_paper_id="p1",
            source_chunk_ids=["c1"],  # near-miss chunk
        ),
    ]


def test_perfect_retriever_metrics_are_one() -> None:
    """If gold always appears at rank 1, all metrics should be 1.0."""
    gold_by_q = {"What is X?": "c1", "What is Y?": "c2", "What is Z?": "c3"}

    def perfect(q: str, k: int) -> list[str]:
        return [gold_by_q[q], "noise1", "noise2"][:k]

    agg, per_q = evaluate_retriever(perfect, _make_eval_set(), config_name="perfect", top_k=5)
    assert agg.n_evaluated == 3  # 4 items minus 1 no-answer
    assert agg.n_no_answer == 1
    assert agg.recall_at_k == 1.0
    assert agg.mrr_at_k == 1.0
    assert agg.ndcg_at_k == 1.0
    assert all(r.rank_of_first_gold == 1 for r in per_q)


def test_random_retriever_metrics_are_zero() -> None:
    """If gold never appears, all metrics should be 0."""

    def useless(q: str, k: int) -> list[str]:
        return ["junk1", "junk2", "junk3"][:k]

    agg, per_q = evaluate_retriever(useless, _make_eval_set(), config_name="useless", top_k=5)
    assert agg.n_evaluated == 3
    assert agg.recall_at_k == 0.0
    assert agg.mrr_at_k == 0.0
    assert agg.ndcg_at_k == 0.0
    assert all(r.rank_of_first_gold is None for r in per_q)


def test_mixed_results() -> None:
    """Gold at rank 1 for one query, rank 3 for another, miss for the third."""
    plans = {
        "What is X?": ["c1", "junk", "junk"],  # rank 1
        "What is Y?": ["x", "y", "c2"],  # rank 3
        "What is Z?": ["a", "b", "c"],  # miss
    }

    def retriever(q: str, k: int) -> list[str]:
        return plans[q][:k]

    agg, _ = evaluate_retriever(retriever, _make_eval_set(), config_name="mixed", top_k=5)
    assert agg.n_evaluated == 3
    assert agg.recall_at_k == pytest.approx(2 / 3)
    # MRR: (1.0 + 1/3 + 0) / 3
    assert agg.mrr_at_k == pytest.approx((1.0 + 1.0 / 3 + 0.0) / 3)


def test_no_answer_excluded_from_scoring() -> None:
    """No-answer items must not be passed to the retriever or affect averages."""
    asked: list[str] = []

    def retriever(q: str, k: int) -> list[str]:
        asked.append(q)
        return ["c1", "c2", "c3"][:k]

    agg, _ = evaluate_retriever(retriever, _make_eval_set(), config_name="track", top_k=5)
    assert "What is the capital of Mars?" not in asked
    assert len(asked) == 3
    assert agg.n_no_answer == 1


def test_empty_eval_items_raises() -> None:
    def retriever(q: str, k: int) -> list[str]:
        return []

    with pytest.raises(ValueError):
        evaluate_retriever(retriever, [], config_name="x", top_k=5)


def test_invalid_top_k_raises() -> None:
    def retriever(q: str, k: int) -> list[str]:
        return []

    with pytest.raises(ValueError):
        evaluate_retriever(retriever, _make_eval_set(), config_name="x", top_k=0)


def test_all_no_answer_raises() -> None:
    """If every item is no-answer, there's nothing to score."""
    items = [
        EvalItem(
            id="na-0",
            question="q",
            gold_answer="a",
            type=QuestionType.NO_ANSWER,
            source_paper_id="p",
            source_chunk_ids=["c"],
        )
    ]

    def retriever(q: str, k: int) -> list[str]:
        return []

    with pytest.raises(ValueError, match="No scorable items"):
        evaluate_retriever(retriever, items, config_name="x", top_k=5)


# ----------------------- save_run / ablation -----------------------


def test_save_run_writes_both_files(tmp_path: Path) -> None:
    def perfect(q: str, k: int) -> list[str]:
        return ["c1", "c2", "c3"][:k]

    items = [
        EvalItem(
            id="sh-0",
            question="q",
            gold_answer="a",
            type=QuestionType.SINGLE_HOP,
            source_paper_id="p",
            source_chunk_ids=["c1"],
        )
    ]

    agg, per_q = evaluate_retriever(perfect, items, config_name="t", top_k=3)
    per_path, agg_path = save_run(agg, per_q, tmp_path)

    assert per_path.exists()
    assert agg_path.exists()

    # Aggregate file round-trips
    loaded = json.loads(agg_path.read_text())
    assert loaded["config_name"] == "t"
    assert loaded["recall_at_k"] == 1.0

    # Per-query JSONL has one line per query
    lines = per_path.read_text().strip().split("\n")
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["eval_id"] == "sh-0"
    assert row["rank_of_first_gold"] == 1


def test_format_ablation_row_renders() -> None:
    m = AggregateMetrics(
        config_name="hybrid",
        strategy="recursive",
        top_k=10,
        n_evaluated=100,
        n_no_answer=20,
        recall_at_k=0.85,
        mrr_at_k=0.67,
        ndcg_at_k=0.74,
        timestamp="2026-05-14T10:30:00+00:00",
    )
    row = format_ablation_row(m)
    assert "| hybrid | recursive | 10 |" in row
    assert "0.850" in row
    assert "0.670" in row
    assert "0.740" in row


def test_append_to_ablation_md_creates_with_header(tmp_path: Path) -> None:
    m = AggregateMetrics(
        config_name="dense",
        strategy="recursive",
        top_k=10,
        n_evaluated=50,
        n_no_answer=10,
        recall_at_k=0.5,
        mrr_at_k=0.3,
        ndcg_at_k=0.4,
        timestamp="2026-05-14T10:00:00+00:00",
    )
    path = tmp_path / "ablation.md"
    append_to_ablation_md(m, path)

    content = path.read_text()
    assert "Retrieval ablation" in content
    assert ABLATION_HEADER in content
    assert "| dense |" in content


def test_append_to_ablation_md_appends_to_existing(tmp_path: Path) -> None:
    """Second call should append without re-writing the header."""
    m1 = AggregateMetrics(
        config_name="dense",
        strategy="recursive",
        top_k=10,
        n_evaluated=50,
        n_no_answer=10,
        recall_at_k=0.5,
        mrr_at_k=0.3,
        ndcg_at_k=0.4,
    )
    m2 = AggregateMetrics(
        config_name="hybrid",
        strategy="recursive",
        top_k=10,
        n_evaluated=50,
        n_no_answer=10,
        recall_at_k=0.8,
        mrr_at_k=0.6,
        ndcg_at_k=0.7,
    )
    path = tmp_path / "ablation.md"
    append_to_ablation_md(m1, path)
    append_to_ablation_md(m2, path)

    content = path.read_text()
    # Header appears exactly once
    assert content.count("| Config | Strategy | k |") == 1
    # Both rows present
    assert "| dense |" in content
    assert "| hybrid |" in content
