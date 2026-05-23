# ArXiv RAG — Production-Grade Retrieval and Grounded Generation over ML Papers

> Hybrid retrieval (BM25 + dense) with cross-encoder reranking and query rewriting over 204 ArXiv ML papers, plus a generation layer with verbatim-quote citation verification. Evaluated end-to-end on both retrieval quality and grounding faithfulness.

**Status:** retrieval, reranking, query rewriting, and grounded generation shipped; ablated end-to-end across six configurations with both positive and negative findings.

---

## TL;DR

| Configuration                       | Recall@10 | MRR@10 | nDCG@10 | Refusal Accuracy | False-Grounding Rate |
| :---------------------------------- | :-------: | :----: | :-----: | :--------------: | :------------------: |
| Sparse only (BM25)                  |   0.921   | 0.781  |  0.816  |        —         |          —           |
| Dense only (BGE-M3 + LanceDB)       |   0.961   | 0.800  |  0.838  |        —         |          —           |
| Hybrid (RRF fusion, k=60)           |   0.974   | 0.801  |  0.842  |        —         |          —           |
| **+ Cross-encoder rerank (top-50)** | **1.000** | **0.955** | **0.966** |   **1.000**   |      **0.000**       |
| + Multi-query (3 paraphrases)       |   1.000   | 0.955  |  0.966  |      1.000       |        0.000         |
| + HyDE                              |   0.947   | 0.903  |  0.914  |      1.000       |        0.000         |

Retrieval metrics over 76 single-hop questions; refusal metrics over 10 hand-crafted out-of-corpus questions. Raw run artifacts are committed under `evals/runs/`.

The story is in the [Retrieval evaluation](#retrieval-evaluation) and [Grounding evaluation](#grounding-evaluation-and-the-citation-justified-hallucination-fix) sections — including a saturation finding, a regression finding, and the methodology behind the refusal-accuracy measurement.

---

## What this is

Most RAG demos are single-script notebooks that retrieve top-k and stuff into a prompt. This project is the production version: real ingestion pipeline, retrieval that actually works, generation with citation grounding that's *measured* rather than assumed, and an evaluation harness that distinguishes "the system found the right chunk" from "the system gave the right answer" from "the system correctly refused to answer."

### Architecture

```
┌──────────────┐    ┌──────────┐    ┌────────────┐    ┌──────────┐
│ ArXiv API    │───>│ Parse +  │───>│ Embed +    │───>│ LanceDB  │
│ (cs.LG/CL)   │    │ Chunk    │    │ BM25 index │    │          │
└──────────────┘    └──────────┘    └────────────┘    └──────────┘
                                                            │
                ┌──────────────┐    ┌──────────┐            │
       Query ──>│ Query rewrite│───>│ Hybrid   │<───────────┘
                │ (multi-query │    │ retrieval│
                │  or HyDE)    │    └──────────┘
                └──────────────┘          │
                                    ┌──────────┐
                                    │ Reranker │
                                    │ (BGE-CE) │
                                    └──────────┘
                                          │
                                    ┌──────────┐
                                    │ Generator│
                                    │ + quote  │
                                    │ verifier │
                                    └──────────┘
                                          │
                                    Answer + verified citations
                                    (or grounded abstention)
```

---

## Retrieval evaluation

Six retrieval configurations, same corpus and chunking strategy (recursive ~500-token chunks), benchmarked on the same eval set of 82 synthetic Q&A pairs (76 single-hop + 6 no-answer):

| Config | k  | Recall@10 | MRR@10 | nDCG@10 | n  |
|--------|----|-----------|--------|---------|-----|
| sparse (BM25)              | 10 | 0.921 | 0.781 | 0.816 | 76 |
| dense (BGE-M3 + LanceDB)   | 10 | 0.961 | 0.800 | 0.838 | 76 |
| hybrid (RRF fusion, k=60)  | 10 | 0.974 | 0.801 | 0.842 | 76 |
| **+ rerank (BGE-reranker-v2-m3, top-50)** | 10 | **1.000** | **0.955** | **0.966** | 76 |
| + multi-query (3 paraphrases, then rerank) | 10 | 1.000 | 0.955 | 0.966 | 76 |
| + HyDE (hypothetical passage, then rerank) | 10 | 0.947 | 0.903 | 0.914 | 76 |

**Metrics** (1-indexed ranks):

- **Recall@10**: fraction of queries where any gold chunk appears in top-10
- **MRR@10**: mean of 1/rank-of-first-gold (0 if no gold in top-10)
- **nDCG@10**: position-weighted gold hits, normalized to [0, 1]

### Reading the table

**Three positive findings**:

1. **Hybrid > either retriever alone** by a small but consistent margin. Dense embeddings catch semantic paraphrases that BM25 misses (+4pp Recall); BM25 catches specific acronyms and rare terms that dense embeddings blur (+1.3pp on top of dense).
2. **Cross-encoder reranking is the dominant accuracy lever**, jumping MRR@10 from 0.801 → 0.955 — meaning the gold chunk is at rank 1 for ~95% of queries instead of averaging around rank 1.25. Recall@10 hits a ceiling of 1.0.
3. The reranking gain comes from joint attention across (query, chunk) — the cross-encoder reads both texts together and outputs a relevance score, whereas embeddings encode each independently and compare in vector space. Joint attention is more accurate but can't be pre-computed, so we only run it on the top-50 candidates from hybrid retrieval.

**Two negative findings**:

4. **Multi-query rewriting offers zero improvement on this eval set.** Identical numbers to rerank-only. The cross-encoder is already picking the right chunk from the original-query candidates, and the paraphrases don't surface anything new. *Implication*: query rewriting is only valuable when retrieval has headroom. On saturated retrieval, it's wasted LLM calls and latency.
5. **HyDE actively hurts (-5pp on every retrieval metric).** The hypothetical passage shifts retrieval away from the actual gold chunk and toward the LLM's hallucinated vocabulary. *Implication*: HyDE is conditional, not universal. It works when there's a real vocabulary gap between user queries and corpus chunks; it hurts when query and corpus already share vocabulary (as is the case here, since the eval set was generated from the corpus itself).

Both negative findings depend on the *eval set's properties*, not on the techniques being broken. On a hand-curated set with more realistic vocabulary gaps, multi-query and HyDE should help.

### Reproduce

```bash
uv run arxiv-rag eval-retrieval --config sparse      --top-k 10
uv run arxiv-rag eval-retrieval --config dense       --top-k 10
uv run arxiv-rag eval-retrieval --config hybrid      --top-k 10
uv run arxiv-rag eval-retrieval --config rerank      --top-k 10
uv run arxiv-rag eval-retrieval --config multi-query --top-k 10
uv run arxiv-rag eval-retrieval --config hyde        --top-k 10
```

Per-query results land in `evals/runs/`; aggregate rows append to `evals/ablation.md`.

---

## Grounding evaluation and the citation-justified hallucination fix

Retrieval metrics measure *"did we find the right chunk?"* They don't measure *"did the system give a faithful answer?"* The two questions can come apart in ways that matter — and on this project they did.

### The failure mode

After shipping the generation layer, smoke-testing produced this output on an out-of-corpus question:

```
> What is the capital of France?

The capital of France is Paris, as associations between 'Paris' and 'France'
are mentioned, implying Paris is the capital.

Citations: 2605.06510v1::2::1
```

The cited chunk was a real ArXiv paper on factual recall in language models. It contained the strings "Paris" and "France" in a different context (LLMs recalling capital-city associations as part of a benchmark). It did *not* state that Paris is the capital of France. The generator used outside knowledge and then attached a decorative citation to launder it.

This is **citation-justified hallucination** — arguably the worst failure mode in RAG, because the citation hides the failure. A user (or an automated downstream consumer) sees a confident answer with a source attached and trusts both.

### The fix

The grounding defense has three layers, all deterministic (no second LLM-as-judge call):

1. **Stricter prompt.** The system message explicitly forbids associative reasoning and demands a verbatim supporting quote per claim. The earlier prompt said "use only the passages"; the current prompt adds *"do not combine information across passages unless the combination is itself stated."*

2. **Verbatim quote verification.** The generator's output schema requires each citation to include a `supporting_quote` — the exact sentence from that chunk that supports the answer. After parsing, the verifier normalizes whitespace and case, then checks that the quote appears as a literal substring of the cited chunk. If it doesn't, the citation is dropped. Quotes shorter than 15 characters are rejected outright (no "GQA" as "support" for a multi-sentence claim).

3. **No verified citations → forced abstention.** If verification leaves zero citations, the answer is converted to an abstention regardless of what the LLM claimed about `abstained`. Grounding cannot be self-certified.

### Measurement

A 10-item adversarial set (`evals/adversarial.jsonl`) contains questions that have no answer in the corpus but whose vocabulary appears in the corpus in unrelated contexts: *capital of France, who won the 2024 US election, height of Mount Everest, speed of light in vacuum, chemical formula for water,* etc. The `eval-generation` command runs each through the full pipeline and computes:

- **Refusal accuracy**: fraction of out-of-corpus questions for which the system correctly abstained
- **False-grounding rate**: fraction where the system claimed grounding (returned ≥1 verified citation) on an out-of-corpus question — the metric that captures exactly the failure mode above

| Config      | Refusal Accuracy | False-Grounding Rate | n  |
|-------------|:----------------:|:--------------------:|:--:|
| rerank      | 1.000            | 0.000                | 10 |
| multi-query | 1.000            | 0.000                | 10 |
| hyde        | 1.000            | 0.000                | 10 |

All three configs hit perfect refusal with zero false grounding. The HyDE result is the most informative: HyDE's hypothetical passage for *"What is the capital of France?"* contains the exact claim *"Paris is the capital of France,"* and retrieval pulled in the same chunk that triggered the original failure. The verifier still caught the hallucination, because the LLM's `supporting_quote` for the cited chunk could not be found verbatim in the chunk text:

```
INFO arxiv_rag.generation.generator: Citation rejected: supporting_quote not
found verbatim in chunk 2605.06510v1::2::1. quote='This enables LLMs to derive
the associations between "Paris" and "France" (Petroni et al., 2019)...'
INFO arxiv_rag.generation.generator: Generator: no verified citations;
converting claimed answer to abstention.
```

This is the key architectural property: **the grounding defense is independent of retrieval quality.** Even when HyDE actively steers retrieval *into* the trap chunk that previously caused a failure, the verifier holds the line because it checks against the real chunk text, not against the hypothetical passage that led retrieval astray.

### Limitations and what's next

This adversarial set is small (n=10) and hand-crafted, designed to exercise the specific failure mode. Stronger validation is planned in two directions:

- **RAGAS faithfulness + answer relevance** on the 76 single-hop items — measures the dual question: *"when the system **does** answer, are the answers good?"*
- **Larger hand-curated no-answer set** covering more diverse out-of-corpus topics
- **LLM-as-judge correctness with a calibrated rubric** as a sanity check on the deterministic verifier

### Reproduce

```bash
uv run arxiv-rag eval-generation --config rerank
uv run arxiv-rag eval-generation --config multi-query
uv run arxiv-rag eval-generation --config hyde
```

Per-query results land in `evals/runs/generation_*.jsonl`; aggregates in the matching `.json` files. Both are committed so the numerical claims in this README are reproducible from the source.

---

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone & install
git clone https://github.com/ShubhamX57/arxiv-rag && cd arxiv-rag
uv sync --all-extras

# Set API keys (Groq for free generation; Anthropic/OpenAI optional)
cp .env.example .env
# then edit .env

# Verify
uv run arxiv-rag --help
```

---

## Usage

### 1. Fetch papers from ArXiv

```bash
uv run arxiv-rag fetch                                          # defaults: 500 papers, cs.LG + cs.CL
uv run arxiv-rag fetch --max-results 200 --categories cs.LG --since 2024-01-01
```

PDFs land in `data/pdfs/`, metadata in `data/metadata.jsonl`.

### 2. Parse & chunk

```bash
uv run arxiv-rag parse                       # PDF → structured sections
uv run arxiv-rag chunk --strategy recursive  # sections → chunks
```

Chunking strategies: `fixed` (token windows + overlap) or `recursive` (paragraph → sentence → word splits, merge to budget).

### 3. Build indexes

```bash
uv run arxiv-rag index --strategy recursive  # builds dense (LanceDB) + sparse (BM25)
```

### 4. Search interactively

```bash
# Single retriever
uv run arxiv-rag search "grouped query attention" --mode dense  --k 5
uv run arxiv-rag search "grouped query attention" --mode sparse --k 5

# Hybrid (RRF) — shows ranks from each retriever
uv run arxiv-rag search-hybrid "techniques to reduce memory during inference" --k 5
```

### 5. End-to-end query (with grounding)

```bash
# A question the corpus answers
uv run arxiv-rag query "How does Mixture of Experts reduce inference cost?" --config rerank

# A question the corpus does NOT answer — system abstains with grounded reasoning
uv run arxiv-rag query "What is the capital of France?" --config rerank
```

The `query` command runs the full retrieve → rerank → generate pipeline and prints either:
- a grounded answer with verified supporting quotes per citation, or
- an explicit abstention message explaining what the passages do not state

### 6. Generate the eval set

```bash
uv run arxiv-rag generate-evals --n-single 100 --n-no-answer 20
```

### 7. Run the ablations

```bash
uv run arxiv-rag eval-retrieval  --config rerank  --top-k 10   # retrieval metrics
uv run arxiv-rag eval-generation --config rerank               # grounding metrics
```

---

## Project structure

```
arxiv-rag/
├── src/arxiv_rag/
│   ├── config.py           # pydantic-settings, single source of truth
│   ├── cli.py              # typer entry point
│   ├── ingest/             # arxiv fetch, PDF parse, chunking
│   ├── retrieval/          # embedder, dense, sparse, hybrid, reranker, rewriters
│   ├── generation/         # generator, RAG pipeline, citation verifier
│   └── evals/              # schema, synthetic generation, metrics, harness
├── tests/                  # 224 tests; pytest + mypy --strict + ruff
├── evals/
│   ├── eval_set.jsonl      # 82 synthetic Q&A pairs (76 single-hop + 6 no-answer)
│   ├── adversarial.jsonl   # 10 hand-crafted out-of-corpus questions
│   ├── ablation.md         # retrieval ablation rows, appended by eval-retrieval
│   └── runs/               # per-query traces; generation aggregates committed
└── data/                   # gitignored: PDFs, LanceDB, BM25 pickle, metadata
```

---

## Design notes

Decisions worth justifying when an interviewer asks "why":

- **LanceDB** over Chroma/Qdrant: embedded (no server), zero-copy reads via Arrow, fast on Apple Silicon, native cosine similarity.
- **BGE-M3** for embeddings: top of MTEB for its size, produces both dense and sparse representations, runs on MPS at reasonable throughput.
- **Reciprocal Rank Fusion** for hybrid: parameter-free in practice (k=60 from the original paper), avoids the score-calibration problem you get with weighted sums (cosine in [-1, 1] vs BM25 in [0, ~30] — comparing them directly is meaningless).
- **BGE-reranker-v2-m3** as the cross-encoder: trained on the same data distribution as the embeddings. 568M params, runs on MPS.
- **Reranking against the original query, not rewrites.** Multi-query and HyDE use rewrites only to *find* candidates. The cross-encoder judges relevance against what the user actually asked. Otherwise the system optimizes for matching the LLM's paraphrase phrasing instead of the user's intent.
- **Deterministic grounding verification** instead of a second LLM call as judge. Verbatim quote checks are reproducible, free, and have no failure modes of their own; LLM-judge faithfulness checks introduce noise from the judge's own hallucinations and add latency. The tradeoff: a deterministic check is strict; it can over-abstain when the LLM paraphrases a quote slightly. That's the safer error in production.
- **`top_k_for_generation = 3`** (lowered from 5 alongside the grounding fix): fewer chunks in the prompt = less material for the LLM to confabulate citations from. The reranker is good enough that the top 3 contain the gold ~99% of the time on this corpus.
- **LiteLLM** for generation: provider swap via one config string (Anthropic / OpenAI / Gemini / Groq / local Ollama). Built-in retry-with-backoff handles free-tier rate limits.
- **No-answer items excluded from retrieval metrics**: retrieval measures "did we find the gold chunk?", which has no meaning for a question with no gold chunk. Abstention is a generation-layer concern measured separately.

---

## Eval methodology

Two distinct eval sets, measuring two distinct things:

- **`evals/eval_set.jsonl`** (n=82): synthetic Q&A pairs generated from corpus chunks via Llama 3.3 70B (Groq). Each item has `id`, `question`, `gold_answer`, `type` (single_hop | no_answer), `source_paper_id`, `source_chunk_ids`. Used for **retrieval** metrics.
- **`evals/adversarial.jsonl`** (n=10): hand-crafted out-of-corpus questions whose vocabulary appears in the corpus in unrelated contexts. Used for **grounding** metrics (refusal accuracy, false-grounding rate).

Limitations of the current eval setup and what's planned:

- **Synthetic retrieval items aren't manually reviewed.** A v0.2 will trim the synthetic set to ~60 hand-curated items. Synthetic evals tend to inflate reranker performance (the gold chunk really is the most relevant in the corpus by construction); hand-curation typically brings MRR down ~5pp.
- **Multi-hop questions (synthesizing 2+ chunks) aren't generated yet.** Will land alongside query-decomposition as a rewriting strategy.
- **The synthetic generator samples chunks randomly and asks the LLM to write a question grounded in that chunk; questions tend to inherit the chunk's vocabulary.** This biases the retrieval eval toward exact-term matching (visible in the unusually small ~5pp gap between sparse and hybrid) and is the main reason HyDE underperforms on this set — there's no vocabulary gap for it to bridge.
- **The adversarial set is small (n=10).** A larger, more diverse no-answer set is planned.

Planned next:

- **RAGAS faithfulness + answer relevance** on the 76 single-hop items — measures *"when the system **does** answer, are the answers good?"*, complementary to the *"does it correctly refuse?"* metric above.
- **Hand-curated retrieval eval set** + full re-run of all six configs.
- **LLM-as-judge correctness** with a calibrated rubric.
- **p50/p95 latency and cost** per query across configs — the operational axis.

---

## Engineering

- Python 3.12, [uv](https://docs.astral.sh/uv/) for env + dependency management
- 224 tests passing, mypy --strict, ruff lint + format
- Pre-commit hooks: trailing whitespace, large-file check, ruff, mypy, end-of-file fixer
- GitHub Actions CI on every push
- Branch protection on `main`

---

## License

MIT
