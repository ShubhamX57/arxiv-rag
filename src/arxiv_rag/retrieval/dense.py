"""Dense vector store using LanceDB.

LanceDB is an embedded columnar vector database — no server, no Docker.
It stores chunks + their embeddings as a single Arrow-backed table that
supports cosine-similarity ANN search via `.search(vector).limit(k)`.

For ~5K chunks an exact (flat) search is fast enough on M-series; we only
build an IVF_PQ index when the table grows past a configurable threshold.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
from tqdm import tqdm

from arxiv_rag.config import settings
from arxiv_rag.retrieval.embedder import Embedder

log = logging.getLogger(__name__)

# Build an ANN index once the table is large enough that flat scans get slow.
ANN_INDEX_THRESHOLD = 50_000


def chunks_table_name(strategy: str) -> str:
    """One LanceDB table per chunking strategy so we can compare side-by-side."""
    return f"chunks_{strategy}"


@dataclass(frozen=True, slots=True)
class DenseHit:
    """One result from a dense search."""

    chunk_id: str
    paper_id: str
    section_title: str
    text: str
    score: float  # cosine similarity in [-1, 1]; 1 = identical


def _make_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("paper_id", pa.string()),
            pa.field("section_index", pa.int32()),
            pa.field("section_title", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("token_count", pa.int32()),
            pa.field("strategy", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
        ]
    )


class DenseStore:
    """Build + query a LanceDB table of chunks with embeddings."""

    def __init__(
        self,
        db_dir: Path | None = None,
        table_name: str = "chunks_recursive",
    ) -> None:
        self.db_dir = db_dir or settings.lancedb_dir
        self.table_name = table_name
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.db_dir))

    # ---------- build ----------

    def _table_exists(self) -> bool:
        try:
            self._db.open_table(self.table_name)
            return True
        except (FileNotFoundError, ValueError):
            return False

    def build(
        self,
        chunks: Iterable[dict[str, Any]],
        embedder: Embedder,
        recreate: bool = True,
        embed_batch: int = 256,
    ) -> int:
        """Embed and insert chunks. Returns number of rows written."""
        chunk_list = list(chunks)
        if not chunk_list:
            log.warning("No chunks to index.")
            return 0

        if recreate and self._table_exists():
            log.info("Dropping existing table %s", self.table_name)
            self._db.drop_table(self.table_name)

        # Embed in larger batches than we insert: amortizes model overhead
        log.info("Embedding %d chunks...", len(chunk_list))
        texts = [c["text"] for c in chunk_list]
        vectors = embedder.embed(texts, show_progress_bar=True)
        assert vectors.shape == (len(chunk_list), embedder.dim), (
            f"Embedding shape mismatch: got {vectors.shape}, "
            f"expected ({len(chunk_list)}, {embedder.dim})"
        )

        schema = _make_schema(embedder.dim)
        rows: list[dict[str, Any]] = []
        for c, vec in zip(chunk_list, vectors, strict=True):
            rows.append(
                {
                    "chunk_id": c["chunk_id"],
                    "paper_id": c["paper_id"],
                    "section_index": int(c["section_index"]),
                    "section_title": c["section_title"],
                    "chunk_index": int(c["chunk_index"]),
                    "text": c["text"],
                    "token_count": int(c["token_count"]),
                    "strategy": c.get("strategy", "unknown"),
                    "vector": vec.tolist(),
                }
            )

        log.info("Writing %d rows to table %s", len(rows), self.table_name)
        # create_table with batched inserts to keep peak RAM manageable
        table = self._db.create_table(self.table_name, schema=schema, mode="overwrite")
        # Insert in batches to give a progress bar
        batch_size = 1024
        for i in tqdm(range(0, len(rows), batch_size), desc="Indexing", unit="batch"):
            table.add(rows[i : i + batch_size])

        # Optional ANN index for large corpora
        if len(rows) >= ANN_INDEX_THRESHOLD:
            log.info("Building IVF_PQ ANN index...")
            table.create_index(
                metric="cosine",
                vector_column_name="vector",
                num_partitions=256,
                num_sub_vectors=16,
            )

        return len(rows)

    # ---------- query ----------

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> list[DenseHit]:
        """Top-k nearest by cosine similarity."""
        if not self._table_exists():
            raise FileNotFoundError(
                f"Table {self.table_name} doesn't exist in {self.db_dir}. "
                f"Run `arxiv-rag index` first."
            )
        table = self._db.open_table(self.table_name)
        results = (
            table.search(query_vector, vector_column_name="vector")
            .metric("cosine")
            .limit(k)
            .to_list()
        )
        # LanceDB returns a `_distance` field; cosine distance = 1 - cosine sim
        return [
            DenseHit(
                chunk_id=r["chunk_id"],
                paper_id=r["paper_id"],
                section_title=r["section_title"],
                text=r["text"],
                score=1.0 - float(r["_distance"]),
            )
            for r in results
        ]


def load_chunks_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a chunks JSONL file into memory."""
    chunks: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


__all__ = [
    "ANN_INDEX_THRESHOLD",
    "DenseHit",
    "DenseStore",
    "chunks_table_name",
    "load_chunks_jsonl",
]
