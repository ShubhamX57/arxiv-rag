"""Schema for the evaluation set.

Each question has:
- A type: single_hop | multi_hop | no_answer
- Provenance: which chunk(s) the gold answer comes from
- A gold answer (free text)

We generate these synthetically with an LLM, then manually review and filter.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path


class QuestionType(StrEnum):
    SINGLE_HOP = "single_hop"  # answerable from one chunk
    MULTI_HOP = "multi_hop"  # requires synthesizing 2+ chunks
    NO_ANSWER = "no_answer"  # answer is NOT in corpus (model should refuse)


@dataclass(frozen=True, slots=True)
class EvalItem:
    """One row in the eval set."""

    id: str
    question: str
    gold_answer: str
    type: QuestionType
    source_paper_id: str
    source_chunk_ids: list[str]
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def load_eval_set(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                EvalItem(
                    id=obj["id"],
                    question=obj["question"],
                    gold_answer=obj["gold_answer"],
                    type=QuestionType(obj["type"]),
                    source_paper_id=obj["source_paper_id"],
                    source_chunk_ids=list(obj["source_chunk_ids"]),
                    notes=obj.get("notes", ""),
                )
            )
    return items


def save_eval_set(items: list[EvalItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in items:
            f.write(item.to_json() + "\n")


__all__ = ["EvalItem", "QuestionType", "load_eval_set", "save_eval_set"]
