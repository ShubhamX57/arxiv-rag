"""Tests for the end-to-end RAG pipeline.

Uses fakes for retriever and generator — no real models, no real LLM calls.
"""

from __future__ import annotations

from arxiv_rag.generation.generator import GeneratedAnswer, Generator
from arxiv_rag.generation.pipeline import RAGPipeline, RAGResult
from arxiv_rag.retrieval.reranker import RerankedHit


def _make_reranked(chunk_id: str) -> RerankedHit:
    return RerankedHit(
        chunk_id=chunk_id,
        paper_id="p1",
        section_title="Methods",
        text=f"text for {chunk_id}",
        rerank_score=0.9,
        rrf_rank=1,
        rrf_score=0.5,
        dense_rank=1,
        sparse_rank=2,
    )


class FakeRetriever:
    """Returns a fixed list of RerankedHits and records what it was called with."""

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


def _fake_llm_factory(response: str):
    """Build a fake llm_fn that returns a fixed response."""

    def fake_llm(prompt: str, model: str) -> str:
        return response

    return fake_llm


def test_pipeline_retrieves_with_configured_k_and_fan_out() -> None:
    retriever = FakeRetriever([_make_reranked(f"c{i}") for i in range(10)])
    gen = Generator(
        llm_fn=_fake_llm_factory(
            '{"answer": "x", "abstained": false, "citation_chunk_ids": ["c0"]}'
        )
    )
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


def test_pipeline_passes_retrieved_chunks_to_generator() -> None:
    hits = [_make_reranked("c1"), _make_reranked("c2")]
    retriever = FakeRetriever(hits)
    received_chunks: list[list] = []

    def capture(prompt: str, model: str) -> str:
        # The chunk ids should appear in the prompt
        received_chunks.append([line for line in prompt.split("\n") if "chunk_id" in line])
        return '{"answer": "x", "abstained": false, "citation_chunk_ids": ["c1"]}'

    gen = Generator(llm_fn=capture)
    p = RAGPipeline(retriever=retriever, generator=gen, top_k_for_generation=2)
    p.query("q?")

    # Both chunk ids should have made it into the prompt
    lines = received_chunks[0]
    assert any("c1" in line for line in lines)
    assert any("c2" in line for line in lines)


def test_pipeline_query_returns_rag_result() -> None:
    retriever = FakeRetriever([_make_reranked("c1")])
    gen = Generator(
        llm_fn=_fake_llm_factory(
            '{"answer": "the answer", "abstained": false, "citation_chunk_ids": ["c1"]}'
        )
    )
    p = RAGPipeline(retriever=retriever, generator=gen)

    result = p.query("my question")

    assert isinstance(result, RAGResult)
    assert result.question == "my question"
    assert len(result.retrieved) == 1
    assert isinstance(result.answer, GeneratedAnswer)
    assert result.answer.answer == "the answer"


def test_pipeline_empty_retrieval_still_returns_result() -> None:
    """Even with no retrieved chunks, the pipeline should not crash."""
    retriever = FakeRetriever([])
    gen = Generator(llm_fn=_fake_llm_factory(""))  # won't be called
    p = RAGPipeline(retriever=retriever, generator=gen)

    result = p.query("q?")

    assert result.retrieved == []
    assert result.answer.abstained is True


def test_pipeline_retrieve_and_answer_can_be_called_independently() -> None:
    """Eval harnesses want to score retrieval and generation separately,
    so the two stages need to be callable on their own."""
    hits = [_make_reranked("c1"), _make_reranked("c2")]
    retriever = FakeRetriever(hits)
    gen = Generator(
        llm_fn=_fake_llm_factory(
            '{"answer": "x", "abstained": false, "citation_chunk_ids": ["c1"]}'
        )
    )
    p = RAGPipeline(retriever=retriever, generator=gen, top_k_for_generation=2)

    retrieved = p.retrieve("q?")
    assert len(retrieved) == 2

    answer = p.answer("q?", retrieved)
    assert answer.abstained is False
    assert answer.citation_chunk_ids == ["c1"]
