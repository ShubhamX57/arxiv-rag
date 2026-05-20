"""Tests for query rewriters and the rewrite-rerank composed retriever.

No real LLM calls — every test injects a fake `llm_fn` or `rewrite_fn`.
No real models — `RewriteRerankRetriever` accepts injected `hybrid` and `reranker`.
"""

from __future__ import annotations

import pytest

from arxiv_rag.retrieval.hybrid import FusedHit
from arxiv_rag.retrieval.reranker import RerankedHit
from arxiv_rag.retrieval.rewriters import (
    RewriteRerankRetriever,
    RewriteResult,
    _extract_json_array,
    hyde_rewrite,
    multi_query_rewrite,
)

# ----------------------- RewriteResult -----------------------


def test_rewrite_result_all_queries_includes_original() -> None:
    r = RewriteResult(
        original="What is X?", rewrites=["Define X.", "Explain X."], strategy="multi-query"
    )
    assert r.all_queries() == ["What is X?", "Define X.", "Explain X."]


def test_rewrite_result_all_queries_dedupes_case_insensitive() -> None:
    r = RewriteResult(
        original="What is X?",
        rewrites=["what is x?", "Define X."],
        strategy="multi-query",
    )
    # "what is x?" matches original "What is X?" case-insensitively → dropped
    assert r.all_queries() == ["What is X?", "Define X."]


def test_rewrite_result_all_queries_drops_empty() -> None:
    r = RewriteResult(original="q", rewrites=["", "  ", "alt"], strategy="multi-query")
    assert r.all_queries() == ["q", "alt"]


# ----------------------- _extract_json_array -----------------------


def test_extract_json_array_clean_input() -> None:
    raw = '["foo", "bar", "baz"]'
    assert _extract_json_array(raw) == ["foo", "bar", "baz"]


def test_extract_json_array_handles_code_fences() -> None:
    raw = '```json\n["foo", "bar"]\n```'
    assert _extract_json_array(raw) == ["foo", "bar"]


def test_extract_json_array_handles_prose_wrapper() -> None:
    raw = 'Here are the rephrasings:\n["foo", "bar"]\nHope this helps!'
    assert _extract_json_array(raw) == ["foo", "bar"]


def test_extract_json_array_returns_empty_on_malformed() -> None:
    assert _extract_json_array("not json") == []
    assert _extract_json_array("[unclosed") == []
    assert _extract_json_array('{"not": "an array"}') == []


def test_extract_json_array_drops_non_strings() -> None:
    raw = '["foo", 42, null, "bar"]'
    assert _extract_json_array(raw) == ["foo", "bar"]


def test_extract_json_array_strips_whitespace() -> None:
    raw = '["  foo  ", "bar  "]'
    assert _extract_json_array(raw) == ["foo", "bar"]


# ----------------------- multi_query_rewrite -----------------------


def test_multi_query_calls_llm_and_returns_rewrites() -> None:
    def fake_llm(p: str, m: str) -> str:
        return '["What is the def of X?", "Explain X technically.", "X meaning?"]'

    result = multi_query_rewrite("What is X?", n=3, llm_fn=fake_llm)
    assert result.original == "What is X?"
    assert len(result.rewrites) == 3
    assert result.strategy == "multi-query"


def test_multi_query_truncates_to_n() -> None:
    def fake_llm(p: str, m: str) -> str:
        return '["a", "b", "c", "d", "e"]'

    result = multi_query_rewrite("q", n=2, llm_fn=fake_llm)
    assert len(result.rewrites) == 2


def test_multi_query_n_zero_returns_no_rewrites() -> None:
    def fake_llm(p: str, m: str) -> str:
        return '["a"]'  # should not be called

    result = multi_query_rewrite("q", n=0, llm_fn=fake_llm)
    assert result.rewrites == []


def test_multi_query_llm_failure_falls_back_to_original() -> None:
    def boom(p: str, m: str) -> str:
        raise RuntimeError("rate limited")

    result = multi_query_rewrite("q", llm_fn=boom)
    assert result.rewrites == []
    assert result.original == "q"


def test_multi_query_unparseable_response_returns_empty() -> None:
    def fake_llm(p: str, m: str) -> str:
        return "I cannot do that, sorry."

    result = multi_query_rewrite("q", llm_fn=fake_llm)
    assert result.rewrites == []


def test_multi_query_includes_query_in_prompt() -> None:
    received: list[str] = []

    def capture(prompt: str, model: str) -> str:
        received.append(prompt)
        return "[]"

    multi_query_rewrite("very specific test query", llm_fn=capture)
    assert "very specific test query" in received[0]


# ----------------------- hyde_rewrite -----------------------


def test_hyde_returns_passage_as_single_rewrite() -> None:
    def fake_llm(p: str, m: str) -> str:
        return "Grouped query attention reduces memory by sharing KV heads."

    result = hyde_rewrite("What is GQA?", llm_fn=fake_llm)
    assert result.strategy == "hyde"
    assert len(result.rewrites) == 1
    assert "Grouped query attention" in result.rewrites[0]


def test_hyde_strips_code_fences() -> None:
    def fake_llm(p: str, m: str) -> str:
        return "```\nHypothetical passage here.\n```"

    result = hyde_rewrite("q", llm_fn=fake_llm)
    assert result.rewrites == ["Hypothetical passage here."]


def test_hyde_strips_whitespace() -> None:
    def fake_llm(p: str, m: str) -> str:
        return "   passage with surrounding whitespace   "

    result = hyde_rewrite("q", llm_fn=fake_llm)
    assert result.rewrites == ["passage with surrounding whitespace"]


def test_hyde_empty_response_returns_no_rewrites() -> None:
    def fake_llm(p: str, m: str) -> str:
        return "   "

    result = hyde_rewrite("q", llm_fn=fake_llm)
    assert result.rewrites == []


def test_hyde_llm_failure_falls_back() -> None:
    def boom(p: str, m: str) -> str:
        raise RuntimeError("network error")

    result = hyde_rewrite("q", llm_fn=boom)
    assert result.rewrites == []
    assert result.original == "q"


# ----------------------- RewriteRerankRetriever -----------------------


class FakeHybrid:
    """Returns a configured candidate list for each query. Records what queries
    it was called with so tests can assert on retrieval behavior."""

    def __init__(self, per_query_hits: dict[str, list[FusedHit]]) -> None:
        self._per_query = per_query_hits
        self.received_queries: list[str] = []

    def search(self, query: str, k: int = 10, fan_out_k: int = 50) -> list[FusedHit]:
        self.received_queries.append(query)
        return self._per_query.get(query, [])


class FakeReranker:
    """Pass-through reranker: returns candidates in input order with synthetic scores.
    Records the query used at rerank time."""

    def __init__(self) -> None:
        self.received_query: str | None = None
        self.received_candidates: list[FusedHit] | None = None

    def rerank(
        self, query: str, candidates: list[FusedHit], top_k: int | None = None
    ) -> list[RerankedHit]:
        self.received_query = query
        self.received_candidates = list(candidates)
        out = [
            RerankedHit(
                chunk_id=c.chunk_id,
                paper_id=c.paper_id,
                section_title=c.section_title,
                text=c.text,
                rerank_score=1.0 - i * 0.1,
                rrf_rank=i + 1,
                rrf_score=c.rrf_score,
                dense_rank=c.dense_rank,
                sparse_rank=c.sparse_rank,
            )
            for i, c in enumerate(candidates)
        ]
        if top_k is not None:
            out = out[:top_k]
        return out


def _hit(chunk_id: str) -> FusedHit:
    return FusedHit(
        chunk_id=chunk_id,
        paper_id="p1",
        section_title="s",
        text=f"text for {chunk_id}",
        rrf_score=0.5,
        dense_rank=1,
        sparse_rank=2,
        dense_score=0.9,
        sparse_score=10.0,
    )


def test_multi_query_retriever_calls_hybrid_for_each_rewrite() -> None:
    """All paraphrases (plus original) should hit hybrid."""

    def fake_rewrite(q: str) -> RewriteResult:
        return RewriteResult(
            original=q, rewrites=["paraphrase 1", "paraphrase 2"], strategy="multi-query"
        )

    hybrid = FakeHybrid(
        {
            "original q": [_hit("c1"), _hit("c2")],
            "paraphrase 1": [_hit("c2"), _hit("c3")],  # c2 dup, c3 new
            "paraphrase 2": [_hit("c4")],
        }
    )
    reranker = FakeReranker()
    r = RewriteRerankRetriever(
        strategy="multi-query",
        rewrite_fn=fake_rewrite,
        hybrid=hybrid,
        reranker=reranker,
    )

    out = r.search("original q", k=10)
    assert hybrid.received_queries == ["original q", "paraphrase 1", "paraphrase 2"]
    # 4 unique chunks across 3 queries (c2 deduped)
    assert reranker.received_candidates is not None
    assert {c.chunk_id for c in reranker.received_candidates} == {"c1", "c2", "c3", "c4"}
    assert len(out) == 4


def test_multi_query_dedupes_candidates_by_chunk_id() -> None:
    """Same chunk found by multiple paraphrases must appear once in reranker input."""

    def fake_rewrite(q: str) -> RewriteResult:
        return RewriteResult(original=q, rewrites=["alt"], strategy="multi-query")

    hybrid = FakeHybrid({"q": [_hit("c1")], "alt": [_hit("c1")]})  # same chunk both queries
    reranker = FakeReranker()
    r = RewriteRerankRetriever(
        strategy="multi-query", rewrite_fn=fake_rewrite, hybrid=hybrid, reranker=reranker
    )

    r.search("q")
    assert reranker.received_candidates is not None
    assert len(reranker.received_candidates) == 1
    assert reranker.received_candidates[0].chunk_id == "c1"


def test_multi_query_reranker_uses_original_query() -> None:
    """Reranking is judged against the user's actual query, not paraphrases."""

    def fake_rewrite(q: str) -> RewriteResult:
        return RewriteResult(original=q, rewrites=["paraphrase"], strategy="multi-query")

    hybrid = FakeHybrid({"original q": [_hit("c1")], "paraphrase": [_hit("c2")]})
    reranker = FakeReranker()
    r = RewriteRerankRetriever(
        strategy="multi-query", rewrite_fn=fake_rewrite, hybrid=hybrid, reranker=reranker
    )

    r.search("original q")
    assert reranker.received_query == "original q"


def test_hyde_retriever_uses_only_hypothetical_passage_for_retrieval() -> None:
    """HyDE retrieves with the rewrite, NOT the original."""

    def fake_rewrite(q: str) -> RewriteResult:
        return RewriteResult(original=q, rewrites=["hypothetical answer passage"], strategy="hyde")

    hybrid = FakeHybrid(
        {
            "original q": [_hit("wrong")],  # should NOT be called
            "hypothetical answer passage": [_hit("c1")],
        }
    )
    reranker = FakeReranker()
    r = RewriteRerankRetriever(
        strategy="hyde", rewrite_fn=fake_rewrite, hybrid=hybrid, reranker=reranker
    )

    r.search("original q")
    assert hybrid.received_queries == ["hypothetical answer passage"]


def test_hyde_falls_back_to_original_if_rewrite_fails() -> None:
    """If HyDE produced no rewrites (LLM failure), retrieve with the original."""

    def fake_rewrite(q: str) -> RewriteResult:
        return RewriteResult(original=q, rewrites=[], strategy="hyde")

    hybrid = FakeHybrid({"original q": [_hit("c1")]})
    reranker = FakeReranker()
    r = RewriteRerankRetriever(
        strategy="hyde", rewrite_fn=fake_rewrite, hybrid=hybrid, reranker=reranker
    )

    out = r.search("original q")
    assert hybrid.received_queries == ["original q"]
    assert len(out) == 1


def test_hyde_reranker_uses_original_query() -> None:
    """Like multi-query: rerank against the user's intent, not the hypothetical."""

    def fake_rewrite(q: str) -> RewriteResult:
        return RewriteResult(original=q, rewrites=["fake passage"], strategy="hyde")

    hybrid = FakeHybrid({"fake passage": [_hit("c1")]})
    reranker = FakeReranker()
    r = RewriteRerankRetriever(
        strategy="hyde", rewrite_fn=fake_rewrite, hybrid=hybrid, reranker=reranker
    )

    r.search("user's real question")
    assert reranker.received_query == "user's real question"


def test_empty_candidates_returns_empty() -> None:
    def fake_rewrite(q: str) -> RewriteResult:
        return RewriteResult(original=q, rewrites=[], strategy="multi-query")

    hybrid = FakeHybrid({})  # returns [] for everything
    reranker = FakeReranker()
    r = RewriteRerankRetriever(
        strategy="multi-query", rewrite_fn=fake_rewrite, hybrid=hybrid, reranker=reranker
    )

    assert r.search("anything") == []


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Unknown rewrite strategy"):
        RewriteRerankRetriever(strategy="bogus")
