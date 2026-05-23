"""Answer generation with grounding, abstention, and verbatim quote verification.

This is the Day 11 version. The Day 10 generator had a known failure mode:
citation-justified hallucination. The LLM would invent answers using outside
knowledge and then attach a plausible-looking citation to a chunk that
contained relevant words but did NOT state the answer. Example:

    Q: "What is the capital of France?"
    A: "Paris, as associations between 'Paris' and 'France' are mentioned."
    Citations: [some chunk that mentioned Paris in another context]

Three changes fix this:

1. **Tighter prompt**. The Day 10 prompt said "use only the passages."
   The new prompt is more specific: do not infer, do not associate, do not
   bridge across chunks unless the bridge is itself stated. If the answer
   requires reasoning beyond what the passages explicitly state, abstain.

2. **Verbatim supporting quote**. The LLM must now produce, for each
   citation, the exact sentence from that chunk that supports the answer.
   We then verify the quote appears verbatim in the chunk text. If it
   doesn't, the citation is dropped. If all citations are dropped, we
   abstain. This catches the "decorative citation" pattern directly.

3. **Stricter post-processing**. An answer with zero verified citations is
   converted to an abstention regardless of what the LLM said about
   `abstained`. The LLM doesn't get to claim grounding if the grounding
   doesn't actually exist.

The verbatim check uses a normalized substring match — whitespace and case
are ignored, but the words and their order must match. This catches actual
fabrications without being defeated by trivial formatting differences.
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

RetrievedChunk = FusedHit | RerankedHit


@dataclass(frozen=True, slots=True)
class Citation:
    """One verified citation: which chunk, and the exact supporting sentence
    from that chunk. Verified means the quote was found verbatim in the chunk
    text (modulo whitespace/case normalization)."""

    chunk_id: str
    supporting_quote: str


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """Structured output from the generator.

    Citations now carry verified quotes, not just chunk_ids. If a citation
    survives in this list, the quote was found in the corresponding chunk.
    """

    question: str
    answer: str
    citations: list[Citation]
    abstained: bool
    model: str
    raw_response: str

    @property
    def citation_chunk_ids(self) -> list[str]:
        """Backwards-compat helper for code that only wants the ids."""
        return [c.chunk_id for c in self.citations]


# ----------------------- prompt -----------------------


SYSTEM_INSTRUCTIONS = """You are a research assistant answering questions about \
machine learning papers using ONLY the passages provided in the user message.

Rules:
1. Use ONLY information explicitly stated in the passages. Do not use any \
outside knowledge. Do not infer facts that are not directly stated. Do not \
combine information across passages unless the combination is itself stated.
2. For EVERY claim in your answer, identify the exact sentence from a passage \
that supports that claim. Quote it verbatim.
3. If the passages do not explicitly state the answer, set "abstained" to \
true and briefly explain what is missing. The fact that a passage mentions \
related words is NOT sufficient to answer — the passage must state the answer.
4. Keep the answer concise (2-4 sentences). Use the technical vocabulary from \
the passages.

Output ONLY a JSON object with this exact schema, with no other text:
{
  "answer": "your concise answer, or a brief explanation if abstaining",
  "abstained": false,
  "citations": [
    {
      "chunk_id": "the chunk_id you used",
      "supporting_quote": "verbatim sentence from that chunk that directly states a claim in your answer"
    }
  ]
}

If you cannot find a verbatim sentence in the passages that supports a claim, \
do not make that claim. If you cannot make any supported claim, abstain."""


USER_TEMPLATE = """Question: {question}

Retrieved passages:
{context}

Answer (JSON only):"""


def _format_context(chunks: Sequence[RetrievedChunk], max_chars_per_chunk: int = 1500) -> str:
    """Format chunks for the prompt with chunk_id and section title visible."""
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
    """Extract a JSON object from raw LLM output. Returns None on failure."""
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


# ----------------------- verbatim quote verification -----------------------


def _normalize_for_match(s: str) -> str:
    """Collapse whitespace and lowercase, for fuzzy substring matching.

    We don't want trivial formatting differences (line breaks, double spaces)
    to make a legitimate quote fail verification. But we DO want word order
    and word identity preserved — that's what proves the LLM didn't fabricate
    the quote.
    """
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _quote_appears_in(quote: str, chunk_text: str, min_chars: int = 15) -> bool:
    """Does `quote` appear (modulo whitespace/case) in `chunk_text`?

    Short quotes (< min_chars) are rejected even if they technically match —
    a 5-character quote is essentially meaningless as "support". This catches
    the failure mode where an LLM quotes one word and calls it support.
    """
    if len(quote.strip()) < min_chars:
        return False
    normalized_quote = _normalize_for_match(quote)
    normalized_chunk = _normalize_for_match(chunk_text)
    if not normalized_quote:
        return False
    return normalized_quote in normalized_chunk


def _verify_citations(
    raw_citations: list[Any],
    chunks_by_id: dict[str, str],
) -> list[Citation]:
    """Keep only citations whose supporting_quote appears verbatim in the
    corresponding chunk. Drops citations that fail any check.
    """
    verified: list[Citation] = []
    seen_chunk_ids: set[str] = set()

    for raw in raw_citations:
        if not isinstance(raw, dict):
            continue
        chunk_id = str(raw.get("chunk_id", "")).strip()
        quote = str(raw.get("supporting_quote", "")).strip()

        if not chunk_id or not quote:
            continue
        if chunk_id not in chunks_by_id:
            log.info("Citation rejected: chunk_id %r not in context", chunk_id)
            continue
        if chunk_id in seen_chunk_ids:
            continue  # dedupe by chunk_id; the first verified quote wins
        if not _quote_appears_in(quote, chunks_by_id[chunk_id]):
            log.info(
                "Citation rejected: supporting_quote not found verbatim in chunk %s. quote=%r",
                chunk_id,
                quote[:120],
            )
            continue

        seen_chunk_ids.add(chunk_id)
        verified.append(Citation(chunk_id=chunk_id, supporting_quote=quote))

    return verified


def _coerce_to_answer(
    question: str,
    parsed: dict[str, Any] | None,
    raw: str,
    model: str,
    chunks_by_id: dict[str, str],
) -> GeneratedAnswer:
    """Convert parsed LLM output into a GeneratedAnswer with verified citations.

    If verification leaves no citations, the answer is converted to abstention
    regardless of what the LLM claimed. Grounding cannot be self-certified.
    """
    if parsed is None:
        log.warning("Generator: unparseable response; abstaining. raw=%r", raw[:300])
        return GeneratedAnswer(
            question=question,
            answer="(generation failed: response was not valid JSON)",
            citations=[],
            abstained=True,
            model=model,
            raw_response=raw,
        )

    answer = str(parsed.get("answer", "")).strip()
    llm_claimed_abstained = bool(parsed.get("abstained", False))

    raw_citations = parsed.get("citations", [])
    if not isinstance(raw_citations, list):
        raw_citations = []

    citations = _verify_citations(raw_citations, chunks_by_id)

    # Hard rule: an answer without verified citations is an abstention. The LLM
    # doesn't get to claim it answered the question if the grounding doesn't hold.
    if not citations:
        if not llm_claimed_abstained:
            log.info(
                "Generator: no verified citations; converting claimed answer to abstention. q=%r",
                question[:80],
            )
        return GeneratedAnswer(
            question=question,
            answer=answer
            or "(insufficient context: no verifiable grounding in retrieved passages)",
            citations=[],
            abstained=True,
            model=model,
            raw_response=raw,
        )

    return GeneratedAnswer(
        question=question,
        answer=answer,
        citations=citations,
        abstained=llm_claimed_abstained,
        model=model,
        raw_response=raw,
    )


# ----------------------- LLM call -----------------------


def _call_llm(prompt: str, model: str, temperature: float = 0.0) -> str:
    """LLM completion via LiteLLM. Temperature 0 for repeatable evals."""
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
    return str(resp.choices[0].message.content)


# ----------------------- public API -----------------------


class Generator:
    """Answer a question from retrieved chunks with verified grounding.

    Day 11: every citation in the output has been verified to contain a
    verbatim supporting quote from the cited chunk. If no citations survive
    verification, the answer is converted to an abstention.
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
        if not chunks:
            return GeneratedAnswer(
                question=question,
                answer="(no context retrieved)",
                citations=[],
                abstained=True,
                model=self.model,
                raw_response="",
            )

        context = _format_context(chunks)
        prompt = USER_TEMPLATE.format(question=question, context=context)
        chunks_by_id = {c.chunk_id: c.text for c in chunks}

        try:
            raw = self._llm_fn(prompt, self.model)
        except Exception:
            log.warning("Generator LLM call failed; abstaining", exc_info=True)
            return GeneratedAnswer(
                question=question,
                answer="(generation failed: LLM call raised an exception)",
                citations=[],
                abstained=True,
                model=self.model,
                raw_response="",
            )

        parsed = _parse_response(raw)
        return _coerce_to_answer(question, parsed, raw, self.model, chunks_by_id)


__all__ = [
    "DEFAULT_GENERATOR_MODEL",
    "SYSTEM_INSTRUCTIONS",
    "USER_TEMPLATE",
    "Citation",
    "GeneratedAnswer",
    "Generator",
    "RetrievedChunk",
]
