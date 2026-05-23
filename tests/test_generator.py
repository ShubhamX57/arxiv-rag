"""Tests for the Day 11 Generator: verbatim quote verification + abstention."""

from __future__ import annotations

import json

from arxiv_rag.generation.generator import (
    Citation,
    GeneratedAnswer,
    Generator,
    _format_context,
    _normalize_for_match,
    _parse_response,
    _quote_appears_in,
    _verify_citations,
)
from arxiv_rag.retrieval.hybrid import FusedHit


def _hit(chunk_id: str, text: str = "some default chunk text") -> FusedHit:
    return FusedHit(
        chunk_id=chunk_id,
        paper_id="p1",
        section_title="Methods",
        text=text,
        rrf_score=0.5,
        dense_rank=1,
        sparse_rank=2,
        dense_score=0.9,
        sparse_score=10.0,
    )


# ----------------------- _normalize_for_match -----------------------


def test_normalize_lowercases_and_collapses_whitespace() -> None:
    assert _normalize_for_match("Hello   World") == "hello world"
    assert _normalize_for_match("Foo\nBar\t Baz") == "foo bar baz"


def test_normalize_strips_edges() -> None:
    assert _normalize_for_match("  text  ") == "text"


# ----------------------- _quote_appears_in -----------------------


def test_quote_appears_in_exact_match() -> None:
    chunk = "Grouped query attention shares KV heads across query groups."
    assert _quote_appears_in("shares KV heads across query groups", chunk) is True


def test_quote_appears_in_whitespace_tolerant() -> None:
    chunk = "Grouped query  attention\nshares KV heads."
    assert _quote_appears_in("Grouped query attention shares KV heads", chunk) is True


def test_quote_appears_in_case_tolerant() -> None:
    chunk = "Grouped Query Attention shares KV heads."
    assert _quote_appears_in("grouped query attention shares kv heads", chunk) is True


def test_quote_not_in_chunk_returns_false() -> None:
    chunk = "Grouped query attention shares KV heads."
    assert _quote_appears_in("MoE uses sparse routing", chunk) is False


def test_short_quote_rejected_even_if_present() -> None:
    """A 5-character quote is meaningless support — drop it."""
    chunk = "The model is GQA-based."
    assert _quote_appears_in("GQA", chunk, min_chars=15) is False


def test_quote_with_extra_words_rejected() -> None:
    """If the LLM 'quotes' a sentence with extra words not in the chunk, reject."""
    chunk = "The model uses GQA."
    # LLM tries to invent additional words
    assert _quote_appears_in("The model uses GQA for memory efficiency", chunk) is False


# ----------------------- _verify_citations -----------------------


def test_verify_keeps_valid_citations() -> None:
    chunks_by_id = {"c1": "The model uses grouped query attention for efficiency."}
    raw = [
        {
            "chunk_id": "c1",
            "supporting_quote": "uses grouped query attention for efficiency",
        }
    ]
    out = _verify_citations(raw, chunks_by_id)
    assert len(out) == 1
    assert out[0].chunk_id == "c1"


def test_verify_drops_citation_with_quote_not_in_chunk() -> None:
    """The Day 10 failure mode: chunk contains 'Paris' and 'France' but the LLM
    claims it states 'Paris is the capital of France', which it does not."""
    chunks_by_id = {"c1": "Authors from Paris, France contributed to this work."}
    raw = [{"chunk_id": "c1", "supporting_quote": "Paris is the capital of France"}]
    out = _verify_citations(raw, chunks_by_id)
    assert out == []


def test_verify_drops_citation_to_chunk_not_in_context() -> None:
    chunks_by_id = {"c1": "real chunk text here"}
    raw = [{"chunk_id": "c99", "supporting_quote": "anything"}]
    out = _verify_citations(raw, chunks_by_id)
    assert out == []


def test_verify_dedupes_by_chunk_id() -> None:
    chunks_by_id = {"c1": "The model uses grouped query attention for efficiency."}
    raw = [
        {
            "chunk_id": "c1",
            "supporting_quote": "uses grouped query attention for efficiency",
        },
        {
            "chunk_id": "c1",
            "supporting_quote": "uses grouped query attention for efficiency",
        },
    ]
    out = _verify_citations(raw, chunks_by_id)
    assert len(out) == 1


def test_verify_ignores_non_dict_entries() -> None:
    chunks_by_id = {"c1": "fine fine fine fine fine fine"}
    raw = [
        "not a dict",
        {"chunk_id": "c1", "supporting_quote": "fine fine fine fine fine"},
    ]
    out = _verify_citations(raw, chunks_by_id)
    assert len(out) == 1


def test_verify_empty_quote_rejected() -> None:
    chunks_by_id = {"c1": "anything"}
    raw = [{"chunk_id": "c1", "supporting_quote": ""}]
    out = _verify_citations(raw, chunks_by_id)
    assert out == []


# ----------------------- _parse_response -----------------------


def test_parse_clean_json() -> None:
    raw = '{"answer": "x", "abstained": false, "citations": []}'
    parsed = _parse_response(raw)
    assert parsed is not None
    assert parsed["answer"] == "x"


def test_parse_strips_code_fences() -> None:
    raw = '```json\n{"answer": "y", "abstained": false, "citations": []}\n```'
    parsed = _parse_response(raw)
    assert parsed is not None
    assert parsed["answer"] == "y"


def test_parse_malformed_returns_none() -> None:
    assert _parse_response("not json") is None
    assert _parse_response("") is None


# ----------------------- _format_context -----------------------


def test_format_context_numbers_chunks() -> None:
    chunks = [_hit("c1", "first"), _hit("c2", "second")]
    ctx = _format_context(chunks)
    assert "[1]" in ctx and "[2]" in ctx
    assert "c1" in ctx and "c2" in ctx


def test_format_context_truncates_long_chunks() -> None:
    chunks = [_hit("c1", "x" * 5000)]
    ctx = _format_context(chunks, max_chars_per_chunk=100)
    assert "..." in ctx


# ----------------------- Generator.answer end-to-end -----------------------


def test_generator_returns_verified_answer() -> None:
    chunk_text = (
        "Grouped query attention shares KV heads across query groups for memory efficiency."
    )
    fake_response = json.dumps(
        {
            "answer": "GQA shares KV heads to reduce memory.",
            "abstained": False,
            "citations": [
                {
                    "chunk_id": "c1",
                    "supporting_quote": "shares KV heads across query groups for memory efficiency",
                }
            ],
        }
    )

    g = Generator(llm_fn=lambda p, m: fake_response)
    result = g.answer("What is GQA?", [_hit("c1", chunk_text)])

    assert isinstance(result, GeneratedAnswer)
    assert result.abstained is False
    assert len(result.citations) == 1
    assert isinstance(result.citations[0], Citation)
    assert result.citation_chunk_ids == ["c1"]  # property still works


def test_generator_converts_unverifiable_answer_to_abstention() -> None:
    """The Day 10 failure: LLM claims an answer but the quote isn't in any chunk."""
    chunk_text = "Authors from Paris, France worked on this paper."
    fake_response = json.dumps(
        {
            "answer": "The capital of France is Paris.",
            "abstained": False,
            "citations": [{"chunk_id": "c1", "supporting_quote": "Paris is the capital of France"}],
        }
    )

    g = Generator(llm_fn=lambda p, m: fake_response)
    result = g.answer("What is the capital of France?", [_hit("c1", chunk_text)])

    # The quote isn't actually in the chunk → citation dropped → abstention
    assert result.abstained is True
    assert result.citations == []


def test_generator_honors_explicit_abstention() -> None:
    fake_response = json.dumps(
        {
            "answer": "The passages do not discuss this.",
            "abstained": True,
            "citations": [],
        }
    )
    g = Generator(llm_fn=lambda p, m: fake_response)
    result = g.answer("Out of corpus?", [_hit("c1", "irrelevant")])
    assert result.abstained is True


def test_generator_no_chunks_short_circuits() -> None:
    called = [False]

    def fake_llm(p: str, m: str) -> str:
        called[0] = True
        return ""

    g = Generator(llm_fn=fake_llm)
    result = g.answer("q?", [])
    assert result.abstained is True
    assert called[0] is False


def test_generator_malformed_llm_response_is_abstention() -> None:
    g = Generator(llm_fn=lambda p, m: "I refuse to follow JSON instructions.")
    result = g.answer("q?", [_hit("c1")])
    assert result.abstained is True


def test_generator_llm_exception_is_abstention() -> None:
    def boom(p: str, m: str) -> str:
        raise RuntimeError("api down")

    g = Generator(llm_fn=boom)
    result = g.answer("q?", [_hit("c1")])
    assert result.abstained is True


def test_generator_partial_verification_keeps_verified_only() -> None:
    """Two citations — one verifiable, one fabricated. Keep only the real one."""
    chunk_text = "The model uses grouped query attention for memory efficiency."
    fake_response = json.dumps(
        {
            "answer": "GQA reduces memory.",
            "abstained": False,
            "citations": [
                {
                    "chunk_id": "c1",
                    "supporting_quote": "uses grouped query attention for memory efficiency",
                },
                {
                    "chunk_id": "c1",
                    "supporting_quote": "totally fabricated quote that does not appear",
                },
            ],
        }
    )
    g = Generator(llm_fn=lambda p, m: fake_response)
    result = g.answer("q?", [_hit("c1", chunk_text)])

    # Dedupe by chunk_id keeps only the first; that one verified
    assert result.abstained is False
    assert len(result.citations) == 1


def test_generator_records_model_name() -> None:
    fake_response = json.dumps(
        {
            "answer": "x",
            "abstained": False,
            "citations": [
                {
                    "chunk_id": "c1",
                    "supporting_quote": "real quote text that is long enough",
                }
            ],
        }
    )
    g = Generator(model="my-model", llm_fn=lambda p, m: fake_response)
    result = g.answer("q?", [_hit("c1", "real quote text that is long enough and more")])
    assert result.model == "my-model"


def test_citation_chunk_ids_property() -> None:
    """The property must return chunk_ids derived from citations."""
    ans = GeneratedAnswer(
        question="q",
        answer="a",
        citations=[
            Citation(chunk_id="c1", supporting_quote="abc"),
            Citation(chunk_id="c2", supporting_quote="def"),
        ],
        abstained=False,
        model="m",
        raw_response="",
    )
    assert ans.citation_chunk_ids == ["c1", "c2"]


def test_generator_passes_question_into_prompt() -> None:
    received: list[str] = []

    def capture(prompt: str, model: str) -> str:
        received.append(prompt)
        return '{"answer": "x", "abstained": true, "citations": []}'

    g = Generator(llm_fn=capture)
    g.answer("Why does my model overfit?", [_hit("c1", "irrelevant text")])
    assert "Why does my model overfit?" in received[0]
