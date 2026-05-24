"""Tests for the hand-rolled RAGAS metrics.

All tests use injected judge_fn fakes. No real LLM or embedder calls.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np

from arxiv_rag.evals.ragas import (
    AnswerRelevanceResult,
    FaithfulnessResult,
    _cosine_similarity,
    _extract_claims,
    _parse_json_array,
    _parse_json_object,
    _verify_claim,
    score_answer_relevance,
    score_faithfulness,
)


def _make_judge(responses: list[str]) -> Callable[[str], str]:
    """Build a fake judge_fn that returns each response in turn."""
    it = iter(responses)
    return lambda _prompt: next(it)


# ----------------------- _parse_json_array -----------------------


def test_parse_array_clean() -> None:
    assert _parse_json_array('["a", "b"]') == ["a", "b"]


def test_parse_array_with_fences() -> None:
    assert _parse_json_array('```json\n["a", "b"]\n```') == ["a", "b"]


def test_parse_array_with_prose_wrapper() -> None:
    raw = 'Sure, here you go:\n["x", "y"]\nLet me know if you need more!'
    assert _parse_json_array(raw) == ["x", "y"]


def test_parse_array_returns_none_on_object() -> None:
    assert _parse_json_array('{"key": "value"}') is None


def test_parse_array_returns_none_on_malformed() -> None:
    assert _parse_json_array("not json") is None
    assert _parse_json_array("") is None


# ----------------------- _parse_json_object -----------------------


def test_parse_object_clean() -> None:
    assert _parse_json_object('{"supported": true}') == {"supported": True}


def test_parse_object_with_fences() -> None:
    assert _parse_json_object('```json\n{"supported": false}\n```') == {"supported": False}


def test_parse_object_returns_none_on_array() -> None:
    assert _parse_json_object('["a"]') is None


# ----------------------- _cosine_similarity -----------------------


def test_cosine_identical_vectors_is_one() -> None:
    v = np.array([1.0, 2.0, 3.0])
    assert _cosine_similarity(v, v) == 1.0


def test_cosine_orthogonal_is_zero() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert _cosine_similarity(a, b) == 0.0


def test_cosine_zero_vector_returns_zero() -> None:
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    assert _cosine_similarity(a, b) == 0.0


# ----------------------- _extract_claims -----------------------


def test_extract_claims_basic() -> None:
    judge = _make_judge(['["Claim one.", "Claim two."]'])
    claims, raw = _extract_claims("Q?", "A.", judge)
    assert claims == ["Claim one.", "Claim two."]


def test_extract_claims_handles_unparseable() -> None:
    judge = _make_judge(["not json at all"])
    claims, raw = _extract_claims("Q?", "A.", judge)
    assert claims == []


def test_extract_claims_skips_empty_strings() -> None:
    judge = _make_judge(['["Real claim.", "", "  ", "Another."]'])
    claims, _raw = _extract_claims("Q?", "A.", judge)
    assert claims == ["Real claim.", "Another."]


# ----------------------- _verify_claim -----------------------


def test_verify_claim_supported() -> None:
    judge = _make_judge(['{"supported": true}'])
    is_supported, _raw = _verify_claim("Claim.", "Context.", judge)
    assert is_supported is True


def test_verify_claim_unsupported() -> None:
    judge = _make_judge(['{"supported": false}'])
    is_supported, _raw = _verify_claim("Claim.", "Context.", judge)
    assert is_supported is False


def test_verify_claim_unparseable_defaults_to_unsupported() -> None:
    """Conservative default: don't credit support for malformed judge responses."""
    judge = _make_judge(["I think yes maybe"])
    is_supported, _raw = _verify_claim("Claim.", "Context.", judge)
    assert is_supported is False


# ----------------------- score_faithfulness -----------------------


def test_score_faithfulness_all_supported() -> None:
    judge = _make_judge(
        [
            '["Claim A.", "Claim B."]',  # extraction
            '{"supported": true}',  # verify A
            '{"supported": true}',  # verify B
        ]
    )
    result = score_faithfulness("Q?", "A.", ["context text"], judge)
    assert isinstance(result, FaithfulnessResult)
    assert result.score == 1.0
    assert result.supported == [True, True]


def test_score_faithfulness_partial() -> None:
    judge = _make_judge(
        [
            '["Claim A.", "Claim B.", "Claim C."]',
            '{"supported": true}',
            '{"supported": false}',
            '{"supported": true}',
        ]
    )
    result = score_faithfulness("Q?", "A.", ["ctx"], judge)
    assert result.score == 2 / 3
    assert result.supported == [True, False, True]


def test_score_faithfulness_no_claims_returns_one() -> None:
    """An answer with no factual claims (e.g. an abstention) is vacuously faithful."""
    judge = _make_judge(["[]"])
    result = score_faithfulness("Q?", "(abstained)", ["ctx"], judge)
    assert result.score == 1.0
    assert result.claims == []


def test_score_faithfulness_zero_when_nothing_supported() -> None:
    judge = _make_judge(
        [
            '["Claim."]',
            '{"supported": false}',
        ]
    )
    result = score_faithfulness("Q?", "A.", ["ctx"], judge)
    assert result.score == 0.0


def test_score_faithfulness_joins_context_chunks() -> None:
    """All context chunks must reach the verification prompt, joined."""
    seen_prompts: list[str] = []

    def capture(prompt: str) -> str:
        seen_prompts.append(prompt)
        if "Extract the atomic" in prompt:
            return '["Claim."]'
        return '{"supported": true}'

    score_faithfulness("Q?", "A.", ["chunk one", "chunk two"], capture)
    # The verification prompt is the second call
    assert "chunk one" in seen_prompts[1]
    assert "chunk two" in seen_prompts[1]


# ----------------------- score_answer_relevance -----------------------


def test_score_answer_relevance_high_similarity() -> None:
    judge = _make_judge(['["What is X?", "Tell me about X.", "Describe X."]'])

    # Mock embedder that returns highly-similar vectors
    embedder = MagicMock()
    embedder.embed.return_value = np.array(
        [
            [1.0, 0.0, 0.0],  # original Q
            [0.99, 0.01, 0.0],  # reverse Q 1
            [0.98, 0.02, 0.0],  # reverse Q 2
            [0.97, 0.03, 0.0],  # reverse Q 3
        ]
    )

    result = score_answer_relevance("What is X?", "X is a thing.", judge, embedder=embedder)
    assert isinstance(result, AnswerRelevanceResult)
    assert result.score > 0.95
    assert len(result.generated_questions) == 3


def test_score_answer_relevance_low_similarity() -> None:
    judge = _make_judge(['["Off-topic question?", "Unrelated?", "Random?"]'])

    embedder = MagicMock()
    embedder.embed.return_value = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    result = score_answer_relevance("Q?", "Off-topic A.", judge, embedder=embedder)
    assert result.score == 0.0


def test_score_answer_relevance_no_questions_returns_zero() -> None:
    judge = _make_judge(["not parseable"])
    embedder = MagicMock()
    result = score_answer_relevance("Q?", "A.", judge, embedder=embedder)
    assert result.score == 0.0
    assert result.generated_questions == []
    # Embedder must not have been called if no questions to embed
    embedder.embed.assert_not_called()


def test_score_answer_relevance_respects_n_reverse() -> None:
    judge = _make_judge(['["q1", "q2"]'])
    embedder = MagicMock()
    embedder.embed.return_value = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    result = score_answer_relevance("Q?", "A.", judge, embedder=embedder, n_reverse=2)
    assert len(result.generated_questions) == 2


# ----------------------- judge integration -----------------------


def test_faithfulness_passes_question_into_extraction_prompt() -> None:
    captured: list[str] = []

    def capture(prompt: str) -> str:
        captured.append(prompt)
        return "[]"

    score_faithfulness("My specific question?", "A.", ["ctx"], capture)
    assert "My specific question?" in captured[0]


def test_faithfulness_records_all_raw_responses() -> None:
    judge = _make_judge(
        [
            '["A.", "B."]',
            '{"supported": true}',
            '{"supported": false}',
        ]
    )
    result = score_faithfulness("Q?", "A.", ["ctx"], judge)
    # 1 extraction + 2 verifications
    assert len(result.raw_responses) == 3
