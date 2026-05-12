"""Retrieval: dense (BGE-M3 + LanceDB), sparse (BM25). Hybrid fusion in Week 2."""

from arxiv_rag.retrieval.dense import (
    DenseHit,
    DenseStore,
    chunks_table_name,
    load_chunks_jsonl,
)
from arxiv_rag.retrieval.embedder import Embedder
from arxiv_rag.retrieval.sparse import BM25Store, SparseHit, bm25_path_for, tokenize

__all__ = [
    "BM25Store",
    "DenseHit",
    "DenseStore",
    "Embedder",
    "SparseHit",
    "bm25_path_for",
    "chunks_table_name",
    "load_chunks_jsonl",
    "tokenize",
]
