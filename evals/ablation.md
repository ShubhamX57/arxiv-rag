# Retrieval ablation

Recall@k = at least one gold chunk in top-k. MRR@k = 1/rank of first gold. nDCG@k = position-weighted gold hits, normalized.

| Config | Strategy | k | Recall@k | MRR@k | nDCG@k | n_eval | timestamp |
|--------|----------|---|----------|-------|--------|--------|-----------|
| sparse | recursive | 10 | 0.921 | 0.781 | 0.816 | 76 | 2026-05-16T15:31 |
| dense | recursive | 10 | 0.961 | 0.800 | 0.838 | 76 | 2026-05-16T15:32 |
| hybrid | recursive | 10 | 0.974 | 0.801 | 0.842 | 76 | 2026-05-16T15:32 |
