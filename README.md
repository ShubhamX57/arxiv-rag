# ArXiv RAG — Production-Grade Retrieval over ML Papers

> Hybrid retrieval (BM25 + dense) with cross-encoder reranking, query rewriting, and a real evaluation harness over ~500 ArXiv ML papers.

**Status:** 🚧 Week 1 / 3 — ingestion in progress

---

## TL;DR (results to fill in)

| Configuration       | Recall@10 | RAGAS Faithfulness | RAGAS Answer Relevance | p95 Latency |
| :------------------ | :-------: | :----------------: | :--------------------: | :---------: |
| Naive RAG (dense)   |    —      |         —          |           —            |      —      |
| + Hybrid (BM25+RRF) |    —      |         —          |           —            |      —      |
| + Reranker          |    —      |         —          |           —            |      —      |
| + Query Rewriting   |    —      |         —          |           —            |      —      |
| **Full pipeline**   |    —      |         —          |           —            |      —      |

Eval set: 100 manually-reviewed Q&A pairs (single-hop, multi-hop, no-answer-in-corpus).

---

## What this is

Most RAG demos are single-script notebooks that retrieve top-k and stuff into a prompt. This project is the production version: a real ingestion pipeline, retrieval that actually works, evaluation that measures the right things, and observability so you can debug bad answers.

### Architecture

```
┌──────────────┐    ┌──────────┐    ┌────────────┐    ┌──────────┐
│ ArXiv API    │───▶│ Parse +  │───▶│ Embed +    │───▶│ LanceDB  │
│ (cs.LG/CL)   │    │ Chunk    │    │ BM25 index │    │          │
└──────────────┘    └──────────┘    └────────────┘    └──────────┘
                                                            │
                ┌──────────────┐    ┌──────────┐            │
       Query ──▶│ Query rewrite│───▶│ Hybrid   │◀───────────┘
                │ + HyDE       │    │ retrieval│
                └──────────────┘    └──────────┘
                                          │
                                    ┌──────────┐
                                    │ Reranker │
                                    └──────────┘
                                          │
                                    ┌──────────┐    ┌──────────┐
                                    │ LLM gen  │───▶│ Answer + │
                                    │ + ground │    │ citations│
                                    └──────────┘    └──────────┘
                                          │
                                          ▼
                              Langfuse traces + RAGAS evals
```

---

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone & install
git clone <your-fork> arxiv-rag && cd arxiv-rag
uv sync --all-extras

# Set API keys
cp .env.example .env
# then edit .env

# Verify
uv run arxiv-rag --help
```

---

## Usage

### 1. Fetch papers from ArXiv

```bash
# Defaults: 500 papers from cs.LG and cs.CL, 2024–2025
uv run arxiv-rag fetch

# Or customize
uv run arxiv-rag fetch --max-results 200 --categories cs.LG --since 2024-01-01
```

PDFs land in `data/pdfs/`, metadata in `data/metadata.jsonl`.

### 2. Parse & chunk *(week 1, day 3-4)*

```bash
uv run arxiv-rag parse        # PDF → structured sections
uv run arxiv-rag chunk        # sections → chunks
```

### 3. Build indexes *(week 1, day 5)*

```bash
uv run arxiv-rag index
```

### 4. Query *(week 2)*

```bash
uv run arxiv-rag query "What is the difference between MHA and GQA?"
```

### 5. Run evals *(week 2)*

```bash
uv run arxiv-rag eval --config full
```

### 6. Serve API + UI *(week 3)*

```bash
docker-compose up   # api + langfuse + ui
```

---

## Project structure

```
arxiv-rag/
├── src/arxiv_rag/
│   ├── config.py           # pydantic-settings, single source of truth
│   ├── cli.py              # typer entry point
│   ├── ingest/             # arxiv fetch, PDF parse, chunking
│   ├── retrieval/          # BM25, dense, hybrid, reranking
│   ├── generation/         # prompts, LLM client, grounding
│   ├── evals/              # RAGAS + custom LLM-as-judge
│   └── api/                # FastAPI
├── tests/                  # pytest
├── evals/                  # eval set + run results
├── notebooks/              # experiments only — not core code
└── data/                   # gitignored: pdfs, lancedb, metadata
```

---

## Design notes

Decisions worth justifying when an interviewer asks "why":

- **LanceDB** over Chroma/Qdrant: embedded (no server), zero-copy reads via Arrow, fast on Apple Silicon, built-in hybrid search support.
- **BGE-M3** for embeddings: produces both dense and sparse vectors from one model, multilingual, top of MTEB for its size.
- **Reciprocal Rank Fusion** for hybrid: parameter-free, robust, works without score calibration. Beat weighted-sum on the eval set (see blog post).
- **Cross-encoder reranking** on top-50 → top-10: cheaper than reranking everything, better recall than no reranking. The dominant accuracy lever in the ablation.
- **LiteLLM** for generation: lets us swap Claude / GPT-4o / local Qwen without code changes — useful for cost/latency experiments.

---

## Eval methodology

Eval set is 100 Q&A pairs, generated synthetically with Claude then manually reviewed and filtered:

- 60 single-hop (answer in one chunk)
- 30 multi-hop (requires synthesizing 2+ chunks)
- 10 no-answer-in-corpus (model should refuse)

We report:

- **Retrieval:** Recall@k, MRR, nDCG@10
- **Generation (RAGAS):** Faithfulness, Answer Relevance, Context Precision, Context Recall
- **Custom LLM-as-judge:** correctness with calibrated rubric, refusal accuracy
- **Operational:** p50/p95 latency, $ per query

---

## License

MIT
