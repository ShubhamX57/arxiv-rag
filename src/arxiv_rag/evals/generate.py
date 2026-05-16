"""Generate a synthetic Q&A eval set from chunks via an LLM.

Strategy:
1. Sample N chunks with sufficient content (token_count >= 100).
2. For each, prompt the LLM to produce JSON with a question + gold answer + confidence.
3. Filter on confidence + format checks.
4. Generate no-answer questions on near-miss topics.

Raw output goes to evals/eval_set_raw.jsonl. Manual review trims to gold set.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tqdm import tqdm

from arxiv_rag.config import settings
from arxiv_rag.evals.schema import EvalItem, QuestionType

log = logging.getLogger(__name__)

MIN_CHUNK_TOKENS = 100
MIN_GOLD_ANSWER_CHARS = 20
MIN_QUESTION_CHARS = 15

SINGLE_HOP_PROMPT = """\
You are creating evaluation questions for a retrieval system over ML research papers.

Below is one chunk of text from a paper. Produce ONE specific factual question \
whose answer is contained explicitly in this chunk.

Requirements:
- The question must be answerable ONLY from this chunk.
- The question must be specific enough that a different chunk on a related topic \
would NOT be a sufficient answer.
- The gold answer should be 1-3 sentences, factual, drawn from the chunk's content.
- Avoid trivial questions like "what is X mentioned in section 1?".
- Avoid questions that paraphrase the chunk's first sentence verbatim.

Output ONLY a JSON object with these exact keys (no other text, no markdown fence):
{{"question": "...", "gold_answer": "...", "confidence": 0.0_to_1.0}}

Set confidence based on how clearly the chunk supports a specific question/answer.

CHUNK:
{chunk_text}
"""

NO_ANSWER_PROMPT = """\
You are creating evaluation questions for a retrieval system over ML research papers.

Below is one chunk of text from a paper. Produce ONE question that:
- Is on a RELATED but DIFFERENT topic from what this chunk discusses
- Specifically asks about a fact that is NOT mentioned in this chunk
- A reader could plausibly think this chunk would answer it (it should be a near-miss)
- A correct system should respond "I cannot find this information in the corpus."

Output ONLY a JSON object with these exact keys (no other text, no markdown fence):
{{"question": "...", "gold_answer": "Not answerable from corpus. Reason: ...", \
"confidence": 0.0_to_1.0}}

CHUNK:
{chunk_text}
"""


def _call_llm(prompt: str, model: str, temperature: float = 0.3) -> str:
    """Single LLM call via LiteLLM. Imported lazily.

    `num_retries=5` enables LiteLLM's built-in exponential backoff for rate-limit
    (429) and transient errors. Critical when using free-tier providers like Groq
    that throttle aggressively on TPM.
    """
    from litellm import completion

    resp = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=400,
        num_retries=5,
    )
    return resp.choices[0].message.content


def _parse_json_response(raw: str) -> dict[str, Any] | None:
    """LLMs often wrap JSON in markdown fences. Extract robustly."""
    if not raw:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _quality_filter(
    question: str,
    gold_answer: str,
    confidence: float,
    chunk_text: str,
    min_confidence: float = 0.6,
) -> tuple[bool, str]:
    """Return (kept, reason). Reason is empty if kept."""
    if confidence < min_confidence:
        return False, f"low confidence ({confidence:.2f})"
    if len(question) < MIN_QUESTION_CHARS:
        return False, "question too short"
    if len(gold_answer) < MIN_GOLD_ANSWER_CHARS:
        return False, "answer too short"
    if not question.endswith("?"):
        return False, "question doesn't end with '?'"
    if question.lower() in chunk_text.lower():
        return False, "question is a direct quote from chunk"
    return True, ""


def _sample_chunks(
    chunks: list[dict[str, Any]],
    n: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample chunks long enough to anchor a real question."""
    candidates = [c for c in chunks if c.get("token_count", 0) >= MIN_CHUNK_TOKENS]
    if not candidates:
        raise ValueError(
            f"No chunks with >= {MIN_CHUNK_TOKENS} tokens. Run `arxiv-rag chunk` first."
        )
    rng = random.Random(seed)
    if len(candidates) <= n:
        return candidates
    return rng.sample(candidates, n)


def _generate_one(
    chunk: dict[str, Any],
    prompt_template: str,
    qtype: QuestionType,
    item_id: str,
    model: str,
    min_confidence: float,
    stats: dict[str, int],
) -> EvalItem | None:
    """Generate one eval item from one chunk. None if filtered/failed."""
    try:
        raw = _call_llm(prompt_template.format(chunk_text=chunk["text"]), model=model)
        stats["generated"] += 1
    except Exception as e:
        log.warning("LLM call failed on chunk %s: %s", chunk["chunk_id"], e)
        stats["errors"] += 1
        return None

    parsed = _parse_json_response(raw)
    if parsed is None:
        stats["filtered_parse_failed"] += 1
        return None

    question = str(parsed.get("question", "")).strip()
    gold_answer = str(parsed.get("gold_answer", "")).strip()
    try:
        confidence = float(parsed.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0

    kept, reason = _quality_filter(question, gold_answer, confidence, chunk["text"], min_confidence)
    if not kept:
        if "confidence" in reason:
            stats["filtered_low_confidence"] += 1
        else:
            stats["filtered_other"] += 1
        return None

    return EvalItem(
        id=item_id,
        question=question,
        gold_answer=gold_answer,
        type=qtype,
        source_paper_id=chunk["paper_id"],
        source_chunk_ids=[chunk["chunk_id"]],
        notes=f"confidence={confidence:.2f}",
    )


def generate_eval_set(
    chunks: Iterable[dict[str, Any]],
    n_single_hop: int = 200,
    n_no_answer: int = 30,
    model: str | None = None,
    seed: int = 42,
    min_confidence: float = 0.6,
) -> tuple[list[EvalItem], dict[str, int]]:
    """Generate raw eval items. Returns (items, stats)."""
    model = model or settings.llm_model
    chunk_list = list(chunks)
    log.info(
        "Generating up to %d single-hop + %d no-answer from %d chunks",
        n_single_hop,
        n_no_answer,
        len(chunk_list),
    )

    stats = {
        "generated": 0,
        "kept": 0,
        "filtered_parse_failed": 0,
        "filtered_low_confidence": 0,
        "filtered_other": 0,
        "errors": 0,
    }
    items: list[EvalItem] = []

    # single-hop
    sampled = _sample_chunks(chunk_list, n_single_hop, seed=seed)
    for i, chunk in enumerate(tqdm(sampled, desc="single-hop")):
        item = _generate_one(
            chunk,
            SINGLE_HOP_PROMPT,
            QuestionType.SINGLE_HOP,
            item_id=f"sh-{i:04d}",
            model=model,
            min_confidence=min_confidence,
            stats=stats,
        )
        if item is not None:
            items.append(item)
            stats["kept"] += 1

    # no-answer
    rng = random.Random(seed + 1)
    pool = [c for c in chunk_list if c.get("token_count", 0) >= MIN_CHUNK_TOKENS]
    no_answer_chunks = rng.sample(pool, min(n_no_answer, len(pool)))
    for i, chunk in enumerate(tqdm(no_answer_chunks, desc="no-answer")):
        item = _generate_one(
            chunk,
            NO_ANSWER_PROMPT,
            QuestionType.NO_ANSWER,
            item_id=f"na-{i:04d}",
            model=model,
            min_confidence=min_confidence,
            stats=stats,
        )
        if item is not None:
            items.append(item)
            stats["kept"] += 1

    log.info("Done. Stats: %s", stats)
    return items, stats


def write_jsonl(items: list[EvalItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in items:
            f.write(item.to_json() + "\n")


__all__ = ["generate_eval_set", "write_jsonl"]
