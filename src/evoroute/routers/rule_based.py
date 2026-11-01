"""Hand-written if/else routing -- the baseline EvoRoute has to beat.

Written from domain intuition before seeing any results, so it is a fair
comparison target rather than a strawman tuned to lose.
"""
from __future__ import annotations

from typing import Any

from ..features import extract


class RuleBasedRouter:
    name = "rule-based"

    def route(self, tasks: list[dict[str, Any]]) -> dict[int, str]:
        out = {}
        for task in tasks:
            f = extract(task)
            kind = task["task_type"]
            if kind == "reasoning":
                # Multi-step reasoning is where small models fail hardest.
                out[task["id"]] = "large"
            elif kind == "extraction":
                # Longer sentences hold more entities to recall precisely.
                out[task["id"]] = "medium" if f["input_words"] < 25 else "large"
            else:
                # Topic classification is easy; spend the least possible.
                out[task["id"]] = "small" if f["input_words"] < 60 else "medium"
        return out
