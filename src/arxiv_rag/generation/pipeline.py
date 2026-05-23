"""End-to-end RAG pipeline: retrieve, rerank, generate."""

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
    question: str
    retrieved: list[RerankedHit]
    answer: GeneratedAnswer


class RAGPipeline:
    def __init__(
        self,
        retriever: Any = None,
        generator: Generator | None = None,
        top_k_for_generation: int = 3,  # Day 11: lowered from 5; less surface area for false citations
        fan_out_for_rerank: int = 50,
        retrieval_strategy: str = "recursive",
    ) -> None:
        """Pipeline composing retrieval + generation.

        Day 11 default: top_k_for_generation=3. Fewer chunks = less material
        for the LLM to confabulate citations from. The reranker is good enough
        that the top 3 contain the gold ~99% of the time on this corpus.
        """
        if retriever is None:
            retriever = RerankingRetriever(strategy=retrieval_strategy)
        self.retriever = retriever
        self.generator = generator or Generator()
        self.top_k_for_generation = top_k_for_generation
        self.fan_out_for_rerank = fan_out_for_rerank

    def retrieve(self, question: str) -> list[RerankedHit]:
        return self.retriever.search(
            question,
            k=self.top_k_for_generation,
            fan_out_k=self.fan_out_for_rerank,
        )

    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> GeneratedAnswer:
        return self.generator.answer(question, chunks)

    def query(self, question: str) -> RAGResult:
        chunks = self.retrieve(question)
        answer = self.generator.answer(question, chunks)
        return RAGResult(question=question, retrieved=chunks, answer=answer)


__all__ = [
    "RAGPipeline",
    "RAGResult",
]
