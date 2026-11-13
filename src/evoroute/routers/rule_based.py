"""Hand-written if/else routing -- the baseline EvoRoute has to beat.

Written from domain intuition before seeing any results, so it is a fair
comparison target rather than a strawman tuned to lose.
"""
from __future__ import annotations

from typing import Any

from ..config import DEFAULT_WEIGHTS, TIERS
from ..features import extract
from ..objective import local_gains


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


class TunedRuleRouter:
    """The same if/else shape, with its constants fitted on the train split.

    The frozen RuleBasedRouter above was written from intuition before any
    results existed, which makes it an honest record of a human's prior but an
    unfair opponent for routers that get to learn. This one searches the same
    tiny rule space -- per task type, either a constant tier or one length
    threshold between two tiers -- against train utility only. Test data is
    never consulted, so it can be compared with the learned routers directly.
    """

    name = "rule-based-tuned"

    # Percentile-free, human-sized cut points, as a person tuning by hand would try.
    THRESHOLDS = (15, 25, 40, 60, 90)

    def __init__(self) -> None:
        self._rules: dict[str, object] = {}

    def fit(self, tasks, matrix, scales, weights=DEFAULT_WEIGHTS) -> None:
        n = len(tasks)
        gains = {t["id"]: local_gains(t["id"], matrix, scales, weights, n) for t in tasks}
        words = {t["id"]: extract(t)["input_words"] for t in tasks}

        for kind in {t["task_type"] for t in tasks}:
            subset = [t for t in tasks if t["task_type"] == kind]
            candidates: list[tuple[float, object]] = []

            for tier in TIERS:
                candidates.append((sum(gains[t["id"]][tier] for t in subset), tier))

            for cut in self.THRESHOLDS:
                for low in TIERS:
                    for high in TIERS:
                        if low == high:
                            continue
                        total = sum(
                            gains[t["id"]][low if words[t["id"]] < cut else high]
                            for t in subset)
                        candidates.append((total, (cut, low, high)))

            self._rules[kind] = max(candidates, key=lambda kv: kv[0])[1]

    def route(self, tasks) -> dict[int, str]:
        out = {}
        for task in tasks:
            rule = self._rules.get(task["task_type"], TIERS[0])
            if isinstance(rule, str):
                out[task["id"]] = rule
            else:
                cut, low, high = rule
                out[task["id"]] = low if extract(task)["input_words"] < cut else high
        return out

    def describe(self) -> list[str]:
        lines = []
        for kind, rule in sorted(self._rules.items()):
            if isinstance(rule, str):
                lines.append(f"{kind}: always {rule}")
            else:
                cut, low, high = rule
                lines.append(f"{kind}: {low} if words < {cut} else {high}")
        return lines
