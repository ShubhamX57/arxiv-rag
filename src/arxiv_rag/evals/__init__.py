"""Evaluation: schema, generator, metrics, harness."""

from arxiv_rag.evals.generate import generate_eval_set, write_jsonl
from arxiv_rag.evals.harness import (
    ABLATION_HEADER,
    AggregateMetrics,
    PerQueryResult,
    RetrieveFn,
    append_to_ablation_md,
    evaluate_retriever,
    format_ablation_row,
    save_run,
)
from arxiv_rag.evals.metrics import (
    dcg,
    ideal_dcg,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from arxiv_rag.evals.schema import EvalItem, QuestionType, load_eval_set, save_eval_set

__all__ = [
    "ABLATION_HEADER",
    "AggregateMetrics",
    "EvalItem",
    "PerQueryResult",
    "QuestionType",
    "RetrieveFn",
    "append_to_ablation_md",
    "dcg",
    "evaluate_retriever",
    "format_ablation_row",
    "generate_eval_set",
    "ideal_dcg",
    "load_eval_set",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
    "save_eval_set",
    "save_run",
    "write_jsonl",
]
