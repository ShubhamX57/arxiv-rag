"""Tests for eval-set generation logic. No network — _call_llm is mocked."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from arxiv_rag.evals import generate as evgen
from arxiv_rag.evals.schema import EvalItem, QuestionType, load_eval_set, save_eval_set

# ----------------------- Schema -----------------------


def test_evalitem_roundtrip(tmp_path: Path) -> None:
    items = [
        EvalItem(
            id="sh-0001",
            question="What is GQA?",
            gold_answer="Grouped Query Attention reduces KV cache memory.",
            type=QuestionType.SINGLE_HOP,
            source_paper_id="2401.0001",
            source_chunk_ids=["2401.0001::0::0"],
            notes="confidence=0.85",
        ),
        EvalItem(
            id="na-0001",
            question="What is the capital of France?",
            gold_answer="Not answerable from corpus.",
            type=QuestionType.NO_ANSWER,
            source_paper_id="2401.0002",
            source_chunk_ids=["2401.0002::1::0"],
        ),
    ]
    path = tmp_path / "evals.jsonl"
    save_eval_set(items, path)
    loaded = load_eval_set(path)
    assert len(loaded) == 2
    assert loaded[0].question == items[0].question
    assert loaded[0].type == QuestionType.SINGLE_HOP
    assert loaded[1].type == QuestionType.NO_ANSWER


# ----------------------- JSON parser -----------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"question": "Q?", "gold_answer": "A.", "confidence": 0.9}',
        '```json\n{"question": "Q?", "gold_answer": "A.", "confidence": 0.9}\n```',
        'Here is the JSON:\n{"question": "Q?", "gold_answer": "A.", "confidence": 0.9}',
        '```\n{"question": "Q?", "gold_answer": "A.", "confidence": 0.9}\n```',
    ],
)
def test_parse_json_robust_to_wrappers(raw: str) -> None:
    parsed = evgen._parse_json_response(raw)
    assert parsed is not None
    assert parsed["question"] == "Q?"
    assert parsed["confidence"] == 0.9


def test_parse_json_returns_none_on_garbage() -> None:
    assert evgen._parse_json_response("no json here") is None
    assert evgen._parse_json_response("") is None
    assert evgen._parse_json_response("{ malformed") is None


# ----------------------- Quality filter -----------------------


def test_quality_filter_rejects_low_confidence() -> None:
    kept, reason = evgen._quality_filter(
        "What is X?",
        "X is a thing that does things in machine learning.",
        confidence=0.3,
        chunk_text="some chunk",
    )
    assert not kept
    assert "confidence" in reason


def test_quality_filter_rejects_short_question() -> None:
    kept, reason = evgen._quality_filter(
        "What?",
        "Some answer that's long enough to pass.",
        confidence=0.9,
        chunk_text="some chunk",
    )
    assert not kept
    assert "question too short" in reason


def test_quality_filter_rejects_short_answer() -> None:
    kept, reason = evgen._quality_filter(
        "Is this a real question?",
        "short",
        confidence=0.9,
        chunk_text="some chunk",
    )
    assert not kept
    assert "answer" in reason


def test_quality_filter_rejects_non_question() -> None:
    kept, reason = evgen._quality_filter(
        "This is a statement not a question",
        "Some answer that is sufficiently long.",
        confidence=0.9,
        chunk_text="some chunk",
    )
    assert not kept
    assert "?" in reason


def test_quality_filter_rejects_quote() -> None:
    chunk = "The quick brown fox jumps over the lazy dog repeatedly?"
    kept, reason = evgen._quality_filter(
        "The quick brown fox jumps over the lazy dog repeatedly?",
        "Some answer that is sufficiently long.",
        confidence=0.9,
        chunk_text=chunk,
    )
    assert not kept
    assert "quote" in reason


def test_quality_filter_keeps_valid() -> None:
    kept, reason = evgen._quality_filter(
        "What does GQA reduce?",
        "GQA reduces KV cache memory usage during transformer inference.",
        confidence=0.85,
        chunk_text="Grouped query attention compresses the KV cache.",
    )
    assert kept
    assert reason == ""


# ----------------------- End-to-end with mocked LLM -----------------------


def _sample_chunks() -> list[dict]:
    return [
        {
            "chunk_id": f"p1::0::{i}",
            "paper_id": "p1",
            "section_index": 0,
            "section_title": "Intro",
            "chunk_index": i,
            "text": "ML paper text that is long enough to anchor a question. " * 10,
            "char_count": 600,
            "token_count": 150,
            "strategy": "recursive",
        }
        for i in range(5)
    ]


def test_generate_with_mock_llm() -> None:
    valid_response = json.dumps(
        {
            "question": "What is the main claim of this paper?",
            "gold_answer": "The paper introduces an ML method that improves on prior baselines.",
            "confidence": 0.85,
        }
    )
    with patch.object(evgen, "_call_llm", return_value=valid_response):
        items, stats = evgen.generate_eval_set(_sample_chunks(), n_single_hop=3, n_no_answer=1)
    assert len(items) == 4  # 3 single-hop + 1 no-answer
    assert stats["kept"] == 4
    assert stats["errors"] == 0


def test_generate_filters_bad_llm_output() -> None:
    bad_response = '{"question": "Q?", "gold_answer": "tiny", "confidence": 0.1}'
    with patch.object(evgen, "_call_llm", return_value=bad_response):
        items, stats = evgen.generate_eval_set(_sample_chunks(), n_single_hop=3, n_no_answer=0)
    assert items == []
    assert stats["filtered_low_confidence"] >= 1 or stats["filtered_other"] >= 1


def test_generate_handles_llm_exception() -> None:
    with patch.object(evgen, "_call_llm", side_effect=RuntimeError("LLM down")):
        items, stats = evgen.generate_eval_set(_sample_chunks(), n_single_hop=3, n_no_answer=1)
    assert items == []
    assert stats["errors"] == 4  # one per attempted call


def test_generate_filters_unparseable() -> None:
    with patch.object(evgen, "_call_llm", return_value="garbage non-json"):
        items, stats = evgen.generate_eval_set(_sample_chunks(), n_single_hop=3, n_no_answer=0)
    assert items == []
    assert stats["filtered_parse_failed"] == 3


def test_generate_requires_long_enough_chunks() -> None:
    short_chunks = [
        {**c, "token_count": 50}
        for c in _sample_chunks()  # below MIN_CHUNK_TOKENS
    ]
    with (
        patch.object(evgen, "_call_llm", return_value="{}"),
        pytest.raises(ValueError, match=r"MIN_CHUNK_TOKENS|tokens"),
    ):
        evgen.generate_eval_set(short_chunks, n_single_hop=1)
