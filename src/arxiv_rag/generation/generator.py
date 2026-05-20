"""Answer generation with grounding and abstention.

The generator's job: given a question and a list of retrieved chunks, produce a
grounded answer plus the chunks that support it. Or abstain if the chunks don't
contain the answer.

Three guarantees in the output:

1. **Grounded by design**: the prompt explicitly tells the LLM to use ONLY the
   provided chunks and to cite which ones support each claim. This doesn't
   eliminate hallucination — only evaluation does — but it gives the LLM the
   right priors.
2. **Calibrated abstention**: there's an explicit `abstained` boolean in the
   output schema. The LLM has to choose: answer with citations, or refuse.
   This is what makes the no-answer eval items measurable.
3. **Structured output**: JSON schema means we can parse reliably without
   regex-hunting for citation markers. Tradeoff: prose is slightly stiffer
   than free-form, but evaluation is far more robust.

We use LiteLLM so the same `LLM_MODEL` from .env that drives eval generation
and query rewriting also drives answer generation — one provider switch flips
everything to a different model.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from arxiv_rag.config import settings
from arxiv_rag.retrieval.hybrid import FusedHit
from arxiv_rag.retrieval.reranker import RerankedHit

log = logging.getLogger(__name__)


DEFAULT_GENERATOR_MODEL = settings.llm_model


# A retrieved chunk that's been formatted for the prompt context. We accept either
# a FusedHit (hybrid output) or RerankedHit (rerank output) so the generator
# composes cleanly with any retriever in the ablation table.
RetrievedChunk = FusedHit | RerankedHit


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """Structured output from the generator.

    If `abstained=True`, `answer` will typically be a short refusal message
    ("The provided context does not contain enough information to answer this
    question") and `citation_chunk_ids` will be empty. The downstream eval can
    check `abstained` directly without parsing the prose.
    """

    question: str
    answer: str
    citation_chunk_ids: list[str]
    abstained: bool
    model: str
    # Raw LLM response for debugging — useful when an answer looks wrong and you
    # want to see whether the LLM ignored its instructions vs. the parser dropped
    # something.
    raw_response: str


# ----------------------- prompt -----------------------


SYSTEM_INSTRUCTIONS = """You are a research assistant answering questions about \
machine learning papers. You will be given a question and several passages \
retrieved from an ML paper corpus. Follow these rules strictly:

1. Use ONLY the information in the provided passages. Do not use outside knowledge.
2. If the passages do not contain enough information to answer the question, \
set "abstained" to true and explain briefly what is missing.
3. Cite the passages you used by their chunk_id. Only cite passages that \
directly support a claim in your answer.
4. Keep the answer concise (2-5 sentences). Prefer the exact technical \
vocabulary used in the passages.

Output ONLY a JSON object matching this schema, with no other text:
{
  "answer": "your 2-5 sentence answer, or a brief explanation of what's missing if abstaining",
  "abstained": false,
  "citation_chunk_ids": ["chunk_id_1", "chunk_id_2"]
}"""


USER_TEMPLATE = """Question: {question}

Retrieved passages:
{context}

Answer (JSON only):"""


def _format_context(chunks: Sequence[RetrievedChunk], max_chars_per_chunk: int = 1500) -> str:
    """Format chunks into a numbered context block.

    Each chunk is shown with its chunk_id (so the LLM can cite it) and a short
    section_title for grounding. Long chunks are truncated to keep total prompt
    length manageable; cross-encoder reranking has already put the most-relevant
    parts at the top of the list, so this is safe.
    """
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        text = c.text
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."
        parts.append(
            f"[{i}] chunk_id: {c.chunk_id}\n    section: {c.section_title}\n    text: {text}"
        )
    return "\n\n".join(parts)


# ----------------------- parsing -----------------------


def _parse_response(raw: str) -> dict[str, Any] | None:
    """Pull a JSON object out of the LLM response.

    LLMs sometimes wrap output in ```json fences or trailing prose. This strips
    that and parses what's left. Returns None on any parse failure rather than
    raising — callers handle the fallback.
    """
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


def _coerce_to_answer(
    question: str,
    parsed: dict[str, Any] | None,
    raw: str,
    model: str,
    available_chunk_ids: set[str],
) -> GeneratedAnswer:
    """Convert a parsed dict (or None) into a GeneratedAnswer, defending against
    bad output.

    `available_chunk_ids` is the set of chunk_ids that were actually in the
    context. We filter out citations to chunk_ids the LLM hallucinated — this
    happens occasionally even with strict prompts.
    """
    if parsed is None:
        log.warning("Generator: unparseable response; treating as abstention. raw=%r", raw[:300])
        return GeneratedAnswer(
            question=question,
            answer="(generation failed: response was not valid JSON)",
            citation_chunk_ids=[],
            abstained=True,
            model=model,
            raw_response=raw,
        )

    answer = str(parsed.get("answer", "")).strip()
    abstained = bool(parsed.get("abstained", False))
    raw_citations = parsed.get("citation_chunk_ids", [])
    if not isinstance(raw_citations, list):
        raw_citations = []

    # Keep only citations that actually appeared in our context. Dedupe in order.
    seen: set[str] = set()
    citations: list[str] = []
    for c in raw_citations:
        cid = str(c).strip()
        if cid and cid in available_chunk_ids and cid not in seen:
            seen.add(cid)
            citations.append(cid)

    # Sanity: a non-abstained answer should have at least one citation. If the
    # LLM said abstained=false but cited nothing, treat it as a soft failure
    # rather than trusting the answer — better to abstain than to hallucinate.
    if not abstained and not citations:
        log.info(
            "Generator: claimed non-abstention but cited nothing; converting to abstention. q=%r",
            question[:80],
        )
        abstained = True
        if not answer:
            answer = "(insufficient context: model cited no passages)"

    return GeneratedAnswer(
        question=question,
        answer=answer,
        citation_chunk_ids=citations,
        abstained=abstained,
        model=model,
        raw_response=raw,
    )


# ----------------------- LLM call -----------------------


def _call_llm(prompt: str, model: str, temperature: float = 0.0) -> str:
    """LLM completion via LiteLLM with retry-with-backoff.

    Temperature 0 by default — for generation we want determinism so the same
    (question, chunks) always produces the same answer. That makes evaluation
    repeatable and bugs reproducible.
    """
    from litellm import completion

    resp = completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=600,
        num_retries=5,
    )
    return resp.choices[0].message.content


# ----------------------- public API -----------------------


class Generator:
    """Answer a question from retrieved chunks.

    Stateless except for the model name — instantiate once and call `answer()`
    many times. Tests inject `llm_fn` to avoid real LLM calls.
    """

    def __init__(
        self,
        model: str = DEFAULT_GENERATOR_MODEL,
        llm_fn: Callable[[str, str], str] | None = None,
    ) -> None:
        self.model = model
        self._llm_fn = llm_fn or (lambda p, m: _call_llm(p, m))

    def answer(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
    ) -> GeneratedAnswer:
        """Generate a grounded answer from `chunks`. Abstain if grounding fails."""
        if not chunks:
            return GeneratedAnswer(
                question=question,
                answer="(no context retrieved)",
                citation_chunk_ids=[],
                abstained=True,
                model=self.model,
                raw_response="",
            )

        context = _format_context(chunks)
        prompt = USER_TEMPLATE.format(question=question, context=context)
        available_ids = {c.chunk_id for c in chunks}

        try:
            raw = self._llm_fn(prompt, self.model)
        except Exception:
            log.warning("Generator LLM call failed; abstaining", exc_info=True)
            return GeneratedAnswer(
                question=question,
                answer="(generation failed: LLM call raised an exception)",
                citation_chunk_ids=[],
                abstained=True,
                model=self.model,
                raw_response="",
            )

        parsed = _parse_response(raw)
        return _coerce_to_answer(question, parsed, raw, self.model, available_ids)


__all__ = [
    "DEFAULT_GENERATOR_MODEL",
    "SYSTEM_INSTRUCTIONS",
    "USER_TEMPLATE",
    "GeneratedAnswer",
    "Generator",
    "RetrievedChunk",
]
