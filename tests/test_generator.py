"""Tests for the Generator. Uses injected llm_fn to avoid real LLM calls."""

from __future__ import annotations

import json

from arxiv_rag.generation.generator import (
    GeneratedAnswer,
    Generator,
    _format_context,
    _parse_response,
)
from arxiv_rag.retrieval.hybrid import FusedHit


def _hit(chunk_id: str, text: str = "some chunk text") -> FusedHit:
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


# ----------------------- _parse_response -----------------------


def test_parse_response_clean_json() -> None:
    raw = '{"answer": "foo", "abstained": false, "citation_chunk_ids": ["c1"]}'
    parsed = _parse_response(raw)
    assert parsed == {"answer": "foo", "abstained": False, "citation_chunk_ids": ["c1"]}


def test_parse_response_strips_code_fence() -> None:
    raw = '```json\n{"answer": "x", "abstained": false, "citation_chunk_ids": []}\n```'
    parsed = _parse_response(raw)
    assert parsed is not None
    assert parsed["answer"] == "x"


def test_parse_response_handles_prose_wrapper() -> None:
    raw = 'Sure, here you go:\n{"answer": "y", "abstained": false, "citation_chunk_ids": []}\nLet me know!'
    parsed = _parse_response(raw)
    assert parsed is not None
    assert parsed["answer"] == "y"


def test_parse_response_returns_none_on_malformed() -> None:
    assert _parse_response("not json") is None
    assert _parse_response("{incomplete") is None
    assert _parse_response("") is None


def test_parse_response_rejects_arrays() -> None:
    # An array isn't a valid response shape
    assert _parse_response('["a", "b"]') is None


# ----------------------- _format_context -----------------------


def test_format_context_numbers_chunks() -> None:
    chunks = [_hit("c1", "first text"), _hit("c2", "second text")]
    ctx = _format_context(chunks)
    assert "[1]" in ctx
    assert "[2]" in ctx
    assert "c1" in ctx
    assert "c2" in ctx
    assert "first text" in ctx
    assert "second text" in ctx


def test_format_context_truncates_long_chunks() -> None:
    long_text = "x" * 5000
    chunks = [_hit("c1", long_text)]
    ctx = _format_context(chunks, max_chars_per_chunk=100)
    # Should contain truncation marker
    assert "..." in ctx
    # Total should be much shorter than the original
    assert len(ctx) < 1000


# ----------------------- Generator.answer -----------------------


def test_generator_returns_grounded_answer() -> None:
    def fake_llm(prompt: str, model: str) -> str:
        return json.dumps(
            {
                "answer": "Grouped query attention shares KV heads across query groups.",
                "abstained": False,
                "citation_chunk_ids": ["c1"],
            }
        )

    g = Generator(llm_fn=fake_llm)
    result = g.answer("What is GQA?", [_hit("c1", "GQA shares KV...")])

    assert isinstance(result, GeneratedAnswer)
    assert result.abstained is False
    assert "Grouped query attention" in result.answer
    assert result.citation_chunk_ids == ["c1"]
    assert result.question == "What is GQA?"


def test_generator_honors_abstention() -> None:
    def fake_llm(prompt: str, model: str) -> str:
        return json.dumps(
            {
                "answer": "The passages do not discuss this.",
                "abstained": True,
                "citation_chunk_ids": [],
            }
        )

    g = Generator(llm_fn=fake_llm)
    result = g.answer("What is the capital of Mars?", [_hit("c1")])

    assert result.abstained is True
    assert result.citation_chunk_ids == []


def test_generator_filters_hallucinated_citations() -> None:
    """LLM cites c99 but only c1 was in the context — must drop c99."""

    def fake_llm(prompt: str, model: str) -> str:
        return json.dumps(
            {
                "answer": "Answer text.",
                "abstained": False,
                "citation_chunk_ids": ["c1", "c99"],
            }
        )

    g = Generator(llm_fn=fake_llm)
    result = g.answer("q?", [_hit("c1")])

    assert result.citation_chunk_ids == ["c1"]
    # c99 should NOT be in the output


def test_generator_dedupes_citations() -> None:
    def fake_llm(prompt: str, model: str) -> str:
        return json.dumps(
            {
                "answer": "Answer.",
                "abstained": False,
                "citation_chunk_ids": ["c1", "c1", "c2"],
            }
        )

    g = Generator(llm_fn=fake_llm)
    result = g.answer("q?", [_hit("c1"), _hit("c2")])

    assert result.citation_chunk_ids == ["c1", "c2"]


def test_generator_converts_no_citation_to_abstention() -> None:
    """If LLM says 'not abstained' but cites nothing, we don't trust it."""

    def fake_llm(prompt: str, model: str) -> str:
        return json.dumps(
            {
                "answer": "Some claim with no citations.",
                "abstained": False,
                "citation_chunk_ids": [],
            }
        )

    g = Generator(llm_fn=fake_llm)
    result = g.answer("q?", [_hit("c1")])

    # Soft-fail: an answer with no citations is treated as abstention.
    assert result.abstained is True


def test_generator_no_chunks_means_abstention() -> None:
    """Empty retrieval should abstain without calling the LLM."""
    called = [False]

    def fake_llm(prompt: str, model: str) -> str:
        called[0] = True
        return ""

    g = Generator(llm_fn=fake_llm)
    result = g.answer("q?", [])

    assert result.abstained is True
    assert called[0] is False  # short-circuited before the LLM call


def test_generator_malformed_response_is_abstention() -> None:
    def fake_llm(prompt: str, model: str) -> str:
        return "I refuse to follow your JSON instructions."

    g = Generator(llm_fn=fake_llm)
    result = g.answer("q?", [_hit("c1")])

    assert result.abstained is True
    assert result.citation_chunk_ids == []


def test_generator_llm_exception_is_abstention() -> None:
    def boom(prompt: str, model: str) -> str:
        raise RuntimeError("api down")

    g = Generator(llm_fn=boom)
    result = g.answer("q?", [_hit("c1")])

    assert result.abstained is True
    assert result.raw_response == ""


def test_generator_preserves_raw_response_for_debugging() -> None:
    raw = '{"answer": "ok", "abstained": false, "citation_chunk_ids": ["c1"]}'

    def fake_llm(prompt: str, model: str) -> str:
        return raw

    g = Generator(llm_fn=fake_llm)
    result = g.answer("q?", [_hit("c1")])

    assert result.raw_response == raw


def test_generator_passes_question_into_prompt() -> None:
    received: list[str] = []

    def capture(prompt: str, model: str) -> str:
        received.append(prompt)
        return '{"answer": "x", "abstained": false, "citation_chunk_ids": ["c1"]}'

    g = Generator(llm_fn=capture)
    g.answer("Why does my model overfit?", [_hit("c1")])
    assert "Why does my model overfit?" in received[0]


def test_generator_passes_chunks_into_prompt() -> None:
    received: list[str] = []

    def capture(prompt: str, model: str) -> str:
        received.append(prompt)
        return '{"answer": "x", "abstained": false, "citation_chunk_ids": ["c1"]}'

    g = Generator(llm_fn=capture)
    g.answer("q?", [_hit("c1", "distinctive_chunk_marker")])
    assert "distinctive_chunk_marker" in received[0]


def test_generator_records_model_name() -> None:
    def fake_llm(prompt: str, model: str) -> str:
        return '{"answer": "x", "abstained": false, "citation_chunk_ids": ["c1"]}'

    g = Generator(model="my-test-model/v1", llm_fn=fake_llm)
    result = g.answer("q?", [_hit("c1")])
    assert result.model == "my-test-model/v1"
