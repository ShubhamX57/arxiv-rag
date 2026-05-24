"""Hand-rolled RAGAS-style faithfulness + answer-relevance metrics.

Why hand-roll instead of importing ragas?

The ragas library is a thin wrapper around a handful of LLM-judge prompts
and an embedding similarity check. Its public API has churned across
releases and pulls in langchain + datasets as transitive deps. The actual
algorithms are short and well-documented in the RAGAS paper
(https://arxiv.org/abs/2309.15217), so reimplementing keeps the dep tree
clean and gives full control over the prompts.

The two metrics implemented here mirror the paper closely:

1. **Faithfulness** — for each generated answer, ask an LLM judge to break
   it into atomic claims, then for each claim ask whether it can be
   inferred from the retrieved context. Score = fraction supported.
   Catches hallucinated facts even when answers carry citations.

2. **Answer relevance** — for each generated answer, ask an LLM judge to
   write N (default 3) questions that the answer would have been a good
   response to. Compute cosine similarity between an embedding of each
   reverse-generated question and an embedding of the original. Score =
   mean similarity. Catches off-topic answers.

Both metrics take an injected `judge_fn` so tests can supply fakes and
production code can swap judges (Claude, GPT-4, Llama, etc.) without
touching the metric logic.

To break self-validation bias when measuring Llama-generated answers, the
default judge in our CLI is Claude Sonnet via LiteLLM. The generator and
the judge are different model families.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from arxiv_rag.config import settings
from arxiv_rag.retrieval.embedder import Embedder

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- types


# A judge_fn takes a prompt and returns the raw LLM response string.
# Tests pass fakes; production uses _call_judge_llm below.
JudgeFn = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class FaithfulnessResult:
    """Output of faithfulness scoring for one (question, answer, context) triple."""

    question: str
    answer: str
    claims: list[str]
    supported: list[bool]
    score: float  # fraction of claims supported, in [0, 1]
    raw_responses: list[str]  # for debugging


@dataclass(frozen=True, slots=True)
class AnswerRelevanceResult:
    """Output of answer-relevance scoring for one (question, answer) pair."""

    question: str
    answer: str
    generated_questions: list[str]
    similarities: list[float]
    score: float  # mean cosine similarity, in [-1, 1]
    raw_response: str


# ---------------------------------------------------------------- prompts


# Faithfulness prompts mirror the RAGAS paper's two-stage approach:
# (1) extract claims, (2) verify each claim against context.

CLAIM_EXTRACTION_PROMPT = """\
Extract the atomic factual claims from the following answer to a question. \
A claim is a single, verifiable factual statement. Break compound sentences \
into separate claims.

Question: {question}
Answer: {answer}

Output ONLY a JSON array of claim strings, one per claim, with no other text. \
Example: ["Claim 1.", "Claim 2.", "Claim 3."]

If the answer contains no factual claims (e.g. it's an abstention), output [].
"""


CLAIM_VERIFICATION_PROMPT = """\
You are checking whether a factual claim is supported by some retrieved context. \
A claim is supported if and only if the context explicitly states it or directly \
implies it. Outside knowledge does NOT count as support — only what the context \
actually says.

Context:
{context}

Claim: {claim}

Respond with a JSON object of the form:
{{"supported": true}} or {{"supported": false}}

Output ONLY the JSON object, no explanation.
"""


# Answer relevance: reverse-generate questions that the answer would be a good
# response to, then measure similarity to the original question.

REVERSE_QUESTION_PROMPT = """\
Given the following answer, generate {n} different questions that this answer \
would be a good response to. Each question should be specific enough that the \
answer makes sense as a reply.

Answer: {answer}

Output ONLY a JSON array of {n} question strings, with no other text. \
Example: ["Question 1?", "Question 2?", "Question 3?"]
"""


# ---------------------------------------------------------------- judge call


def _call_judge_llm(prompt: str, model: str, temperature: float = 0.0) -> str:
    """Call an LLM judge via LiteLLM. Temperature 0 for repeatable scores."""
    from litellm import completion

    resp = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=600,
        num_retries=5,
    )
    return str(resp.choices[0].message.content)


# ---------------------------------------------------------------- parsing


def _parse_json_array(raw: str) -> list[Any] | None:
    """Extract a JSON array from a possibly-fenced LLM response."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, list):
        return None
    return parsed


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    """Extract a JSON object from a possibly-fenced LLM response."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


# ---------------------------------------------------------------- faithfulness


def _extract_claims(question: str, answer: str, judge_fn: JudgeFn) -> tuple[list[str], str]:
    """Stage 1: break the answer into atomic claims."""
    prompt = CLAIM_EXTRACTION_PROMPT.format(question=question, answer=answer)
    raw = judge_fn(prompt)
    parsed = _parse_json_array(raw)
    if parsed is None:
        log.warning("Claim extraction returned unparseable JSON; treating as no claims")
        return [], raw
    claims = [str(c).strip() for c in parsed if str(c).strip()]
    return claims, raw


def _verify_claim(claim: str, context: str, judge_fn: JudgeFn) -> tuple[bool, str]:
    """Stage 2: ask whether one claim is supported by the context."""
    prompt = CLAIM_VERIFICATION_PROMPT.format(context=context, claim=claim)
    raw = judge_fn(prompt)
    parsed = _parse_json_object(raw)
    if parsed is None:
        # Conservative default: treat unparseable response as unsupported.
        # Better to undercount support than to spuriously inflate the score.
        log.warning("Claim verification returned unparseable JSON; defaulting to unsupported")
        return False, raw
    return bool(parsed.get("supported", False)), raw


def score_faithfulness(
    question: str,
    answer: str,
    context_chunks: Sequence[str],
    judge_fn: JudgeFn,
) -> FaithfulnessResult:
    """Compute the faithfulness score for one (question, answer, context) triple.

    Score is the fraction of atomic claims in `answer` that the judge says
    can be inferred from the joined `context_chunks`. Returns 1.0 if the
    answer contains no factual claims (e.g. an abstention) — there is
    nothing to be unfaithful about.
    """
    claims, claim_raw = _extract_claims(question, answer, judge_fn)
    if not claims:
        return FaithfulnessResult(
            question=question,
            answer=answer,
            claims=[],
            supported=[],
            score=1.0,
            raw_responses=[claim_raw],
        )

    context = "\n\n".join(context_chunks)
    supported: list[bool] = []
    raws: list[str] = [claim_raw]
    for claim in claims:
        is_supported, raw = _verify_claim(claim, context, judge_fn)
        supported.append(is_supported)
        raws.append(raw)

    score = sum(supported) / len(supported)
    return FaithfulnessResult(
        question=question,
        answer=answer,
        claims=claims,
        supported=supported,
        score=score,
        raw_responses=raws,
    )


# ---------------------------------------------------------------- relevance


def _generate_reverse_questions(answer: str, n: int, judge_fn: JudgeFn) -> tuple[list[str], str]:
    """Ask the judge to write N questions that the answer would respond to."""
    prompt = REVERSE_QUESTION_PROMPT.format(answer=answer, n=n)
    raw = judge_fn(prompt)
    parsed = _parse_json_array(raw)
    if parsed is None:
        return [], raw
    questions = [str(q).strip() for q in parsed if str(q).strip()]
    return questions[:n], raw


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_answer_relevance(
    question: str,
    answer: str,
    judge_fn: JudgeFn,
    embedder: Embedder | None = None,
    n_reverse: int = 3,
) -> AnswerRelevanceResult:
    """Compute the answer-relevance score for one (question, answer) pair.

    Asks the judge to reverse-generate N questions that the answer would be
    a good response to, then computes the mean cosine similarity between
    those questions' embeddings and the original question's embedding.
    A high score means the answer addresses the question; a low score
    means the answer is off-topic.

    Returns 0.0 if no reverse questions could be generated (e.g. abstention).
    """
    generated, raw = _generate_reverse_questions(answer, n_reverse, judge_fn)
    if not generated:
        return AnswerRelevanceResult(
            question=question,
            answer=answer,
            generated_questions=[],
            similarities=[],
            score=0.0,
            raw_response=raw,
        )

    if embedder is None:
        embedder = Embedder()

    # Embed the original + reverse questions in one batch — embedder amortizes
    # model load and benefits from batched inference.
    all_texts = [question, *generated]
    embeddings = embedder.embed(all_texts, show_progress_bar=False)
    q_emb = embeddings[0]
    gen_embs = embeddings[1:]

    sims = [_cosine_similarity(q_emb, g) for g in gen_embs]
    return AnswerRelevanceResult(
        question=question,
        answer=answer,
        generated_questions=generated,
        similarities=sims,
        score=float(np.mean(sims)),
        raw_response=raw,
    )


# ---------------------------------------------------------------- defaults


# Claude Sonnet as the default judge for cross-family scoring against
# Llama-generated answers. Override at the call site if you want a different
# judge for a particular run.
DEFAULT_JUDGE_MODEL = getattr(settings, "judge_model", "anthropic/claude-sonnet-4-5")


def default_judge_fn(prompt: str) -> str:
    """Production judge call — Claude via LiteLLM."""
    return _call_judge_llm(prompt, model=DEFAULT_JUDGE_MODEL)


__all__ = [
    "CLAIM_EXTRACTION_PROMPT",
    "CLAIM_VERIFICATION_PROMPT",
    "DEFAULT_JUDGE_MODEL",
    "REVERSE_QUESTION_PROMPT",
    "AnswerRelevanceResult",
    "FaithfulnessResult",
    "JudgeFn",
    "default_judge_fn",
    "score_answer_relevance",
    "score_faithfulness",
]
