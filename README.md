# ArXiv RAG — Production-Grade Retrieval over ML Papers

> Hybrid retrieval (BM25 + dense) with cross-encoder reranking over 204 ArXiv ML papers, evaluated end-to-end. Query rewriting and generation eval coming next.

**Status:** Week 1 complete — retrieval + reranking shipped, ablation measured. Week 2 in progress.

---

## TL;DR

| Configuration                       | Recall@10 | MRR@10 | nDCG@10 | n  |
| :---------------------------------- | :-------: | :----: | :-----: | :-: |
| Sparse only (BM25)                  |   0.921   | 0.781  |  0.816  | 76 |
| Dense only (BGE-M3 + LanceDB)       |   0.961   | 0.800  |  0.838  | 76 |
| Hybrid (RRF fusion, k=60)           |   0.974   | 0.801  |  0.842  | 76 |
| **+ Cross-encoder rerank (top-50)** | **1.000** | **0.955** | **0.966** | 76 |
| + Query rewriting                   |    —      |   —    |    —    | —  |

Eval set: 82 synthetic Q&A pairs (76 single-hop retrieval + 6 no-answer abstention).
Generation-quality metrics (RAGAS faithfulness, answer relevance) and latency benchmarks land in Week 2.

---

## What this is

Most RAG demos are single-script notebooks that retrieve top-k and stuff into a prompt. This project is the production version: real ingestion pipeline, retrieval that actually works, evaluation that measures the right things, and observability so you can debug bad answers.

### Architecture

```
┌──────────────┐    ┌──────────┐    ┌────────────┐    ┌──────────┐
│ ArXiv API    │───>│ Parse +  │───>│ Embed +    │───>│ LanceDB  │
│ (cs.LG/CL)   │    │ Chunk    │    │ BM25 index │    │          │
└──────────────┘    └──────────┘    └────────────┘    └──────────┘
                                                            │
                ┌──────────────┐    ┌──────────┐            │
       Query ──>│ Query rewrite│───>│ Hybrid   │<───────────┘
                │ + HyDE       │    │ retrieval│
                └──────────────┘    └──────────┘
                                          │
                                    ┌──────────┐
                                    │ Reranker │
                                    └──────────┘
                                          │
                                    ┌──────────┐    ┌──────────┐
                                    │ LLM gen  │───>│ Answer + │
                                    │ + ground │    │ citations│
                                    └──────────┘    └──────────┘
                                          │
                              Langfuse traces + RAGAS evals
```

Shipped so far: everything from ArXiv API through hybrid retrieval and cross-encoder reranking, plus the retrieval eval harness.
Coming: query rewriting, generation, citations, observability, API + UI.

---

## Retrieval evaluation

Four retrieval configurations, same corpus and chunking strategy (recursive ~500-token chunks), benchmarked on the same eval set:

| Config | k  | Recall@10 | MRR@10 | nDCG@10 | n  |
|--------|----|-----------|--------|---------|-----|
| sparse (BM25)              | 10 | 0.921 | 0.781 | 0.816 | 76 |
| dense (BGE-M3 + LanceDB)   | 10 | 0.961 | 0.800 | 0.838 | 76 |
| hybrid (RRF fusion, k=60)  | 10 | 0.974 | 0.801 | 0.842 | 76 |
| **+ rerank (BGE-reranker-v2-m3, top-50)** | 10 | **1.000** | **0.955** | **0.966** | 76 |

**Metrics** (1-indexed ranks):

- **Recall@10**: fraction of queries where any gold chunk appears in top-10
- **MRR@10**: mean of 1/rank-of-first-gold (0 if no gold in top-10)
- **nDCG@10**: position-weighted gold hits, normalized to [0, 1]

**Reading the table**:
- **Hybrid via RRF** outperforms either retriever alone by a small but consistent margin. Dense embeddings catch semantic paraphrases that BM25 misses (+4pp Recall); BM25 catches specific acronyms and rare terms that dense embeddings blur (+1.3pp on top of dense).
- **Cross-encoder reranking is the dominant accuracy lever**, jumping MRR@10 from 0.801 → 0.955 — meaning the gold chunk is at rank 1 for ~95% of queries instead of averaging around rank 1.25. Recall@10 hits a ceiling of 1.0 on this eval set, but the more interesting story is the position improvement: when the LLM downstream reads top-k, it sees the most relevant chunk first.

The reranking gain comes from joint attention across (query, chunk) — the cross-encoder reads both texts together and outputs a relevance score, whereas embeddings encode each independently and compare in vector space. Joint attention is more accurate but can't be pre-computed, so we only run it on the top-50 candidates from hybrid retrieval.

**Reproduce**:

```bash
uv run arxiv-rag eval-retrieval --config sparse --top-k 10
uv run arxiv-rag eval-retrieval --config dense  --top-k 10
uv run arxiv-rag eval-retrieval --config hybrid --top-k 10
uv run arxiv-rag eval-retrieval --config rerank --top-k 10
```

Per-query results land in `evals/runs/`; aggregate rows append to `evals/ablation.md`.

**Methodology**:

- Eval questions were generated synthetically (Llama 3.3 70B via Groq) by prompting the LLM on a sampled chunk with a strict JSON schema and confidence threshold. Quality filters discarded ~30% of generations.
- Questions are **not yet manually reviewed** — numbers will move with a curated set, planned for v0.2. Hand-curated evals typically bring rerank MRR down from 0.95 toward ~0.90.
- No-answer items (n=6) measure abstention behavior, which lives at the generation layer; they're excluded from retrieval metrics.
- All configurations use identical chunking, so the table isolates retrieval quality from chunking quality.
- Reranker latency: ~8 seconds per query for 50 candidates on M4 Pro / MPS. Production deployment would need batching or a smaller reranker.

---

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone & install
git clone https://github.com/ShubhamX57/arxiv-rag && cd arxiv-rag
uv sync --all-extras

# Set API keys (Groq for free generation, optional Anthropic/OpenAI)
cp .env.example .env
# then edit .env

# Verify
uv run arxiv-rag --help
```

---

## Usage

### 1. Fetch papers from ArXiv

```bash
# Defaults: 500 papers from cs.LG and cs.CL
uv run arxiv-rag fetch

# Customize
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
uv run arxiv-rag search "grouped query attention"    --mode dense  --k 5
uv run arxiv-rag search "grouped query attention"    --mode sparse --k 5

# Hybrid (RRF) — shows rank from each retriever
uv run arxiv-rag search-hybrid "techniques to reduce memory during inference" --k 5
```

### 5. Generate the eval set

```bash
# Uses LLM_MODEL from .env (default: groq/llama-3.3-70b-versatile)
uv run arxiv-rag generate-evals --n-single 100 --n-no-answer 20
```

### 6. Run the retrieval ablation

```bash
uv run arxiv-rag eval-retrieval --config rerank --top-k 10
```

### 7. End-to-end query *(Week 2)*

```bash
uv run arxiv-rag query "What is the difference between MHA and GQA?"
```

### 8. Serve API + UI *(Week 3)*

```bash
docker-compose up   # api + langfuse + streamlit
```

---

## Project structure

```
arxiv-rag/
├── src/arxiv_rag/
│   ├── config.py           # pydantic-settings, single source of truth
│   ├── cli.py              # typer entry point
│   ├── ingest/             # arxiv fetch, PDF parse, chunking
│   ├── retrieval/          # embedder, dense (LanceDB), sparse (BM25), hybrid (RRF), reranker
│   ├── evals/              # schema, synthetic generation, metrics, harness
│   ├── generation/         # (Week 2) prompts, LLM client, citations
│   └── api/                # (Week 3) FastAPI
├── tests/                  # 163 tests; pytest + mypy + ruff
├── evals/                  # eval_set.jsonl, ablation.md, runs/
├── notebooks/              # experiments only — not core code
└── data/                   # gitignored: pdfs, lancedb, metadata, BM25 pickle
```

---

## Design notes

Decisions worth justifying when an interviewer asks "why":

- **LanceDB** over Chroma/Qdrant: embedded (no server), zero-copy reads via Arrow, fast on Apple Silicon, native cosine similarity. ANN index kicks in above 50k vectors.
- **BGE-M3** for embeddings: top of MTEB for its size, produces both dense and sparse representations, runs on MPS (Apple Silicon) at reasonable throughput.
- **Reciprocal Rank Fusion** for hybrid: parameter-free in practice (k=60 from the original paper), avoids the score-calibration problem you get with weighted-sum (cosine in [-1, 1] vs BM25 in [0, ~30] — comparing them directly is meaningless).
- **BGE-reranker-v2-m3** as the cross-encoder: the canonical partner to BGE-M3 embeddings, trained on the same data distribution. Multilingual, 568M params, runs on MPS.
- **Recursive chunking** as the default: respects paragraph boundaries when possible, falls back to sentence/word splits, prepends overlap. Produces ~5100 chunks averaging 437 tokens for the 204-paper corpus.
- **LiteLLM** for generation: lets the LLM provider swap (Anthropic / OpenAI / Gemini / Groq / local Ollama) via one config string. Built-in retry-with-backoff for handling free-tier rate limits.
- **No-answer items excluded from retrieval metrics**: retrieval measures "did we find the gold chunk?", which has no meaning for a question with no gold chunk. Abstention is a generation-layer concern (Week 2).
- **fan_out_k=50 for reranking**: caps Recall@10 at whatever Recall@50 of pure hybrid was (~1.0 here), trading compute for headroom. Smaller fan-out is faster but limits the reranker's ceiling.

---

## Eval methodology

Current eval set is **synthetic, n=82**. Each item has:

- A question (single-hop or no-answer)
- A gold answer (1–3 sentences)
- Source provenance: paper_id + chunk_ids the answer is grounded in
- Generation metadata: confidence score, generating model

Limitations and what's next:

- Questions aren't manually reviewed yet — v0.2 will trim the synthetic set to ~60 hand-curated items. Synthetic evals tend to inflate reranker performance (the gold chunk really is the most relevant in the corpus, by construction); hand-curation typically brings MRR down ~5pp.
- Multi-hop questions (synthesizing 2+ chunks) aren't generated yet — adding in Week 2.
- The generator samples chunks randomly; questions tend to inherit the chunk's vocabulary, which biases the eval slightly toward BM25 — visible in the numbers (sparse is only ~5pp behind hybrid, smaller gap than typical benchmarks).

Planned in Week 2:

- **Retrieval**: query rewriting, HyDE, multi-hop synthesis
- **Generation**: faithfulness, answer relevance, context precision/recall via RAGAS
- **Custom**: LLM-as-judge correctness with calibrated rubric, refusal accuracy on no-answer items
- **Operational**: p50/p95 latency, $ per query

---

## Engineering

- Python 3.12, [uv](https://docs.astral.sh/uv/) for env + dependency management
- 163 tests passing, mypy --strict, ruff lint + format
- Pre-commit hooks: trailing whitespace, large-file check, ruff, mypy
- GitHub Actions CI on every push
- Branch protection on `main`

---

## License

MIT
