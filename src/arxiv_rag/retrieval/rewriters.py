"""Query rewriters: multi-query and HyDE.

Both strategies aim to bridge the vocabulary/style gap between user queries
(often short, question-shaped) and corpus chunks (often long, statement-shaped).

**Multi-query** generates N paraphrases of the original query. We retrieve for
each paraphrase plus the original, then fuse the rankings via RRF. The intuition:
different wordings exercise different parts of the retriever's representation
space, so the union has higher recall than any single phrasing.

**HyDE** (Hypothetical Document Embeddings) generates a hypothetical *answer*
to the query, then retrieves chunks similar to that answer. This is purely a
representation trick: an answer-shaped string matches chunk-shaped strings
more closely in embedding space than a question-shaped string does. The
hypothetical answer doesn't need to be factually correct — it just needs to
have the right vocabulary and structure. From the paper (Gao et al., 2022):
HyDE can match supervised dense retrievers without any labeled training data.

Both rewriters are pure functions: `(query, model) -> list[str]`. They don't
know about retrievers or chunks — that lets us plug any rewriter into any
retriever, or compose rewriters (multi-query then HyDE on each paraphrase,
if we were being ambitious).

Latency: each rewriter call is one LLM completion (~500ms-2s on Groq). For
an eval run of 76 queries that's an additional 76 calls per config, which
fits comfortably in the free tier with retry-backoff.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from arxiv_rag.config import settings
from arxiv_rag.retrieval.hybrid import FusedHit, HybridRetriever
from arxiv_rag.retrieval.reranker import RerankedHit, get_default_reranker

log = logging.getLogger(__name__)


# Default LLM for rewriting. Pulled from settings so the same .env config that
# drives eval generation also drives rewriting — one less knob to remember.
DEFAULT_REWRITER_MODEL = settings.llm_model


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """The full output of a rewriter: original query + generated queries.

    We keep the original separate from the generated ones so a downstream
    retriever can choose to weight the original differently (or skip the
    rewrites if they look bad — e.g. all identical to the original).
    """

    original: str
    rewrites: list[str]
    strategy: str  # 'multi-query' | 'hyde' | etc.

    def all_queries(self) -> list[str]:
        """Original query first, then rewrites. De-duplicated, order-preserving."""
        seen: set[str] = set()
        out: list[str] = []
        for q in [self.original, *self.rewrites]:
            q_norm = q.strip()
            if q_norm and q_norm.lower() not in seen:
                seen.add(q_norm.lower())
                out.append(q_norm)
        return out


# ----------------------- shared LLM call -----------------------


def _call_llm(prompt: str, model: str, temperature: float = 0.3) -> str:
    """One LLM completion via LiteLLM with built-in retry-with-backoff.

    Matches the pattern in evals/generate.py — same provider, same retry config.
    Lazy import so the module doesn't drag in litellm at top-level.
    """
    from litellm import completion

    resp = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=400,
        num_retries=5,
    )
    return str(resp.choices[0].message.content)


# ----------------------- multi-query -----------------------


MULTI_QUERY_PROMPT = """You are helping improve information retrieval over a corpus of \
machine learning research papers. Given a user query, generate {n} alternative phrasings \
that would retrieve the same information.

Rules:
- Each rephrasing should use different vocabulary or sentence structure from the original.
- Stay faithful to the original intent — do NOT add new requirements, narrow the scope, \
or shift the topic.
- Prefer terms that would appear in a research paper (e.g. "computational complexity" \
over "how hard it is to run").
- Output ONLY a JSON array of strings, no other text.

Query: {query}

JSON array of {n} alternative phrasings:"""


def _extract_json_array(raw: str) -> list[str]:
    """Pull a JSON array of strings out of an LLM response.

    LLMs sometimes wrap their output in ```json fences or prose; this strips
    that and parses what's left. Returns [] on any parse failure rather than
    raising — the caller can fall back to just the original query.
    """
    # Strip code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    # Find the first '[' and matching ']'
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    # Coerce to list[str], drop non-strings
    return [str(x).strip() for x in parsed if isinstance(x, str) and x.strip()]


def multi_query_rewrite(
    query: str,
    n: int = 3,
    model: str = DEFAULT_REWRITER_MODEL,
    llm_fn: Callable[[str, str], str] | None = None,
) -> RewriteResult:
    """Generate `n` paraphrases of `query` via LLM.

    `llm_fn` lets tests inject a fake LLM. Defaults to LiteLLM.
    """
    if n <= 0:
        return RewriteResult(original=query, rewrites=[], strategy="multi-query")

    call = llm_fn or (lambda p, m: _call_llm(p, m))
    try:
        raw = call(MULTI_QUERY_PROMPT.format(query=query, n=n), model)
    except Exception:
        log.warning("multi-query LLM call failed; falling back to original query", exc_info=True)
        return RewriteResult(original=query, rewrites=[], strategy="multi-query")

    rewrites = _extract_json_array(raw)
    if not rewrites:
        log.warning("multi-query produced no parseable rewrites; raw=%r", raw[:200])
    return RewriteResult(original=query, rewrites=rewrites[:n], strategy="multi-query")


# ----------------------- HyDE -----------------------


HYDE_PROMPT = """You are helping improve information retrieval over a corpus of \
machine learning research papers. Given a user query, write a brief hypothetical \
passage (3-5 sentences) that would directly answer the query, as if quoted from a \
research paper.

Rules:
- Write in the declarative style of a paper, not the question style of the query.
- Use technical vocabulary that would appear in ML papers.
- It is OK if the answer is not factually verified — you are generating a SHAPE that \
matches real paper passages so retrieval can find similar real passages.
- Do NOT preface with "Here is a passage..." or include disclaimers. Output ONLY the \
passage text.

Query: {query}

Hypothetical passage:"""


def hyde_rewrite(
    query: str,
    model: str = DEFAULT_REWRITER_MODEL,
    llm_fn: Callable[[str, str], str] | None = None,
) -> RewriteResult:
    """Generate a hypothetical answer passage to be used as a retrieval query.

    The returned RewriteResult has the hypothetical answer as the single rewrite.
    Downstream code typically retrieves using ONLY the rewrite (not the original),
    but `all_queries()` returns both so the caller can fuse.
    """
    call = llm_fn or (lambda p, m: _call_llm(p, m))
    try:
        raw = call(HYDE_PROMPT.format(query=query), model)
    except Exception:
        log.warning("HyDE LLM call failed; falling back to original query", exc_info=True)
        return RewriteResult(original=query, rewrites=[], strategy="hyde")

    passage = raw.strip()
    if not passage:
        return RewriteResult(original=query, rewrites=[], strategy="hyde")

    # Strip code fences if the LLM wrapped its output
    passage = re.sub(r"^```(?:\w+)?\s*", "", passage)
    passage = re.sub(r"\s*```$", "", passage).strip()

    return RewriteResult(original=query, rewrites=[passage], strategy="hyde")


__all__ = [
    "DEFAULT_REWRITER_MODEL",
    "HYDE_PROMPT",
    "MULTI_QUERY_PROMPT",
    "RewriteResult",
    "hyde_rewrite",
    "multi_query_rewrite",
]


# ---------- Composed retriever: rewrite -> hybrid retrieve -> rerank ----------


class RewriteRerankRetriever:
    """End-to-end pipeline: rewrite the query, retrieve candidates from each
    rewritten query, dedupe by chunk_id, and rerank everything with the
    cross-encoder.

    This is what the eval harness calls for the 'multi-query' and 'hyde'
    rows of the ablation table.

    Strategy selection happens at construction time so the harness can
    instantiate one of these per config string.
    """

    def __init__(
        self,
        strategy: str,  # 'multi-query' | 'hyde'
        n_rewrites: int = 3,
        retrieval_strategy: str = "recursive",
        rrf_k: int = 60,
        fan_out_per_query: int = 50,
        rewriter_model: str = DEFAULT_REWRITER_MODEL,
        # Injection points for tests:
        rewrite_fn: Callable[[str], RewriteResult] | None = None,
        hybrid: Any = None,
        reranker: Any = None,
    ) -> None:
        if strategy not in {"multi-query", "hyde"}:
            raise ValueError(f"Unknown rewrite strategy {strategy!r}. Use 'multi-query' or 'hyde'.")
        self.strategy = strategy
        self.fan_out_per_query = fan_out_per_query

        # Default rewriters call the real LLM; tests pass a fake rewrite_fn.
        if rewrite_fn is not None:
            self._rewrite_fn = rewrite_fn
        elif strategy == "multi-query":
            self._rewrite_fn = lambda q: multi_query_rewrite(q, n=n_rewrites, model=rewriter_model)
        else:  # hyde
            self._rewrite_fn = lambda q: hyde_rewrite(q, model=rewriter_model)

        # Default hybrid and reranker are lazily constructed singletons.
        # Tests inject fakes to avoid loading models.
        self._hybrid = hybrid or HybridRetriever(strategy=retrieval_strategy, rrf_k=rrf_k)
        self._reranker = reranker if reranker is not None else get_default_reranker()

    def search(self, query: str, k: int = 10, fan_out_k: int | None = None) -> list[RerankedHit]:
        """Returns RerankedHit list, sorted by rerank_score descending.

        fan_out_k is accepted for interface compatibility with RerankingRetriever
        but is ignored; use fan_out_per_query in the constructor to control fan-out.
        """
        result = self._rewrite_fn(query)

        # Choose which queries to retrieve with based on strategy:
        # - multi-query: original + paraphrases (union)
        # - hyde: ONLY the hypothetical answer (or original if rewrite failed)
        if self.strategy == "hyde":
            queries = result.rewrites if result.rewrites else [result.original]
        else:
            queries = result.all_queries()  # original + dedupe paraphrases

        # Retrieve a candidate pool from each query
        candidates_by_id: dict[str, FusedHit] = {}
        for q in queries:
            hits = self._hybrid.search(
                q, k=self.fan_out_per_query, fan_out_k=self.fan_out_per_query
            )
            for h in hits:
                # First sighting wins — keeps the FusedHit from the query that found it best
                if h.chunk_id not in candidates_by_id:
                    candidates_by_id[h.chunk_id] = h

        candidates = list(candidates_by_id.values())
        if not candidates:
            return []

        # Rerank using the ORIGINAL query — the user's actual intent — not the rewrites.
        # This is important: we used rewrites to *find* candidates, but relevance is
        # judged against what the user actually asked.
        result_hits: list[RerankedHit] = self._reranker.rerank(query, candidates, top_k=k)
        return result_hits


# Re-export the composed retriever
__all__ = [
    "DEFAULT_REWRITER_MODEL",
    "HYDE_PROMPT",
    "MULTI_QUERY_PROMPT",
    "RewriteRerankRetriever",
    "RewriteResult",
    "hyde_rewrite",
    "multi_query_rewrite",
]
