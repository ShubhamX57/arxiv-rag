"""Tests for the end-to-end RAG pipeline (Day 11)."""

from __future__ import annotations

import json

from arxiv_rag.generation.generator import GeneratedAnswer, Generator
from arxiv_rag.generation.pipeline import RAGPipeline, RAGResult
from arxiv_rag.retrieval.reranker import RerankedHit


def _make_reranked(
    chunk_id: str, text: str = "default text long enough for matching"
) -> RerankedHit:
    return RerankedHit(
        chunk_id=chunk_id,
        paper_id="p1",
        section_title="Methods",
        text=text,
        rerank_score=0.9,
        rrf_rank=1,
        rrf_score=0.5,
        dense_rank=1,
        sparse_rank=2,
    )


class FakeRetriever:
    def __init__(self, hits: list[RerankedHit]) -> None:
        self._hits = hits
        self.received_query: str | None = None
        self.received_k: int | None = None
        self.received_fan_out: int | None = None

    def search(self, query: str, k: int = 10, fan_out_k: int = 50) -> list[RerankedHit]:
        self.received_query = query
        self.received_k = k
        self.received_fan_out = fan_out_k
        return self._hits[:k]


def test_pipeline_uses_configured_k_and_fan_out() -> None:
    retriever = FakeRetriever(
        [_make_reranked(f"c{i}", f"text c{i} long enough for match here") for i in range(10)]
    )
    gen = Generator(llm_fn=lambda p, m: '{"answer": "x", "abstained": true, "citations": []}')
    p = RAGPipeline(
        retriever=retriever,
        generator=gen,
        top_k_for_generation=3,
        fan_out_for_rerank=40,
    )

    p.query("test question")
    assert retriever.received_query == "test question"
    assert retriever.received_k == 3
    assert retriever.received_fan_out == 40


def test_pipeline_returns_rag_result() -> None:
    chunk = _make_reranked("c1", "Grouped query attention reduces memory cost significantly here.")
    retriever = FakeRetriever([chunk])
    gen = Generator(
        llm_fn=lambda p, m: json.dumps(
            {
                "answer": "GQA reduces memory.",
                "abstained": False,
                "citations": [
                    {
                        "chunk_id": "c1",
                        "supporting_quote": "Grouped query attention reduces memory cost",
                    }
                ],
            }
        )
    )
    p = RAGPipeline(retriever=retriever, generator=gen)

    result = p.query("my question")
    assert isinstance(result, RAGResult)
    assert result.question == "my question"
    assert len(result.retrieved) == 1
    assert isinstance(result.answer, GeneratedAnswer)
    assert result.answer.abstained is False
    assert len(result.answer.citations) == 1


def test_pipeline_empty_retrieval_returns_abstention() -> None:
    retriever = FakeRetriever([])
    gen = Generator(llm_fn=lambda p, m: "")
    p = RAGPipeline(retriever=retriever, generator=gen)

    result = p.query("q?")
    assert result.retrieved == []
    assert result.answer.abstained is True


def test_pipeline_retrieve_and_answer_separable() -> None:
    hits = [_make_reranked("c1", "real chunk text that is long enough for matching")]
    retriever = FakeRetriever(hits)
    gen = Generator(
        llm_fn=lambda p, m: json.dumps(
            {
                "answer": "x",
                "abstained": False,
                "citations": [
                    {
                        "chunk_id": "c1",
                        "supporting_quote": "real chunk text that is long enough",
                    }
                ],
            }
        )
    )
    p = RAGPipeline(retriever=retriever, generator=gen, top_k_for_generation=1)

    retrieved = p.retrieve("q?")
    assert len(retrieved) == 1

    answer = p.answer("q?", retrieved)
    assert answer.abstained is False
    assert answer.citation_chunk_ids == ["c1"]
