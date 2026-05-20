"""Retrieval: dense (BGE-M3 + LanceDB), sparse (BM25), hybrid (RRF), rerank (BGE cross-encoder), rewriters (multi-query + HyDE)."""

from arxiv_rag.retrieval.dense import (
    DenseHit,
    DenseStore,
    chunks_table_name,
    load_chunks_jsonl,
)
from arxiv_rag.retrieval.embedder import Embedder
from arxiv_rag.retrieval.hybrid import FusedHit, HybridRetriever, reciprocal_rank_fusion
from arxiv_rag.retrieval.reranker import (
    DEFAULT_RERANKER_MODEL,
    RerankedHit,
    Reranker,
    RerankingRetriever,
    get_default_reranker,
)
from arxiv_rag.retrieval.rewriters import (
    DEFAULT_REWRITER_MODEL,
    RewriteRerankRetriever,
    RewriteResult,
    hyde_rewrite,
    multi_query_rewrite,
)
from arxiv_rag.retrieval.sparse import BM25Store, SparseHit, bm25_path_for, tokenize

__all__ = [
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_REWRITER_MODEL",
    "BM25Store",
    "DenseHit",
    "DenseStore",
    "Embedder",
    "FusedHit",
    "HybridRetriever",
    "RerankedHit",
    "Reranker",
    "RerankingRetriever",
    "RewriteRerankRetriever",
    "RewriteResult",
    "SparseHit",
    "bm25_path_for",
    "chunks_table_name",
    "get_default_reranker",
    "hyde_rewrite",
    "load_chunks_jsonl",
    "multi_query_rewrite",
    "reciprocal_rank_fusion",
    "tokenize",
]
