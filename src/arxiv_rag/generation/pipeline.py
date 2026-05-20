"""End-to-end RAG pipeline.

Composes the best retrieval config we have (hybrid + cross-encoder rerank) with
the generator. The `RAGPipeline` is what the CLI `query` command and the
upcoming FastAPI `/answer` endpoint will both call.

Design notes:

- The pipeline accepts any RerankingRetriever-shaped object, so tests inject a
  fake retriever and we can later swap in a rewriting retriever (multi-query
  or HyDE) by config without changing this code.
- Retrieval and generation are split into two methods (`retrieve` and `answer`)
  in addition to the combined `query` method. This lets callers see the
  retrieved chunks separately (useful for debugging and for the eval harness,
  which wants to score retrieval and generation independently on the same call).
- Pipeline is stateless. All "config" lives in the retriever and generator
  passed in at construction time.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from arxiv_rag.generation.generator import GeneratedAnswer, Generator, RetrievedChunk
from arxiv_rag.retrieval.reranker import RerankedHit, RerankingRetriever

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RAGResult:
    """One end-to-end result: the question, what was retrieved, and the answer.

    Contains everything a UI or eval harness needs to display, debug, or score
    the pipeline output.
    """

    question: str
    retrieved: list[RerankedHit]
    answer: GeneratedAnswer


class RAGPipeline:
    """Retrieve, rerank, and generate a grounded answer."""

    def __init__(
        self,
        retriever: Any = None,
        generator: Generator | None = None,
        top_k_for_generation: int = 5,
        fan_out_for_rerank: int = 50,
        retrieval_strategy: str = "recursive",
    ) -> None:
        """Construct a pipeline.

        - `retriever`: anything with `.search(query, k, fan_out_k=...)` returning
          a list of RerankedHit-shaped objects. Defaults to RerankingRetriever.
        - `generator`: a Generator instance. Defaults to one with model from settings.
        - `top_k_for_generation`: how many chunks to feed into the LLM prompt.
          5 is a defensible default — fewer means missed context, more means
          longer prompts and more noise. Cross-encoder ranks the top of the list
          well enough that 5 captures the answer in nearly all cases for this corpus.
        - `fan_out_for_rerank`: candidate pool for the cross-encoder. Matches the
          eval ablation (50).
        """
        if retriever is None:
            retriever = RerankingRetriever(strategy=retrieval_strategy)
        self.retriever = retriever
        self.generator = generator or Generator()
        self.top_k_for_generation = top_k_for_generation
        self.fan_out_for_rerank = fan_out_for_rerank

    def retrieve(self, question: str) -> list[RerankedHit]:
        """Just the retrieval step. Useful for debugging and eval harnesses."""
        return self.retriever.search(
            question,
            k=self.top_k_for_generation,
            fan_out_k=self.fan_out_for_rerank,
        )

    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> GeneratedAnswer:
        """Just the generation step. Useful for eval harnesses that want to
        score generation independently from a fixed set of retrieved chunks."""
        return self.generator.answer(question, chunks)

    def query(self, question: str) -> RAGResult:
        """End-to-end: retrieve and generate."""
        chunks = self.retrieve(question)
        answer = self.generator.answer(question, chunks)
        return RAGResult(question=question, retrieved=chunks, answer=answer)


__all__ = [
    "RAGPipeline",
    "RAGResult",
]
