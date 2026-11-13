"""Routing strategies as ordered decision lists, and how to execute them fast.

A genome is the proposal's representation: a sequence of rules, each a
conjunction of feature tests mapping to a tier, with a default for anything
unmatched. First match wins.

The search is over this *structure* -- how many rules, which features they
test, in what order -- not over the parameters of a fixed model. That is what
lets a strategy express things a per-task classifier cannot, such as sending
nearly everything to one tier and escalating only on a narrow condition, which
keeps the reload count low.

Evaluation is vectorised over the whole task set: a rule becomes one boolean
mask, so scoring a genome costs a handful of numpy comparisons rather than a
Python loop over tasks. The GA runs thousands of these, and every one of them
reads the cached response matrix -- no model is ever invoked.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ..config import TIERS
from ..features import FEATURE_NAMES

MAX_RULES = 8
MAX_CONDITIONS = 3
OPS = ("<", ">=")


@dataclass(frozen=True)
class Condition:
    feature: int
    op: str
    threshold: float

    def describe(self) -> str:
        return f"{FEATURE_NAMES[self.feature]} {self.op} {self.threshold:g}"


@dataclass(frozen=True)
class Rule:
    conditions: tuple[Condition, ...]
    tier: str

    def describe(self) -> str:
        body = " and ".join(c.describe() for c in self.conditions) or "always"
        return f"if {body} -> {self.tier}"


@dataclass(frozen=True)
class Genome:
    rules: tuple[Rule, ...]
    default: str

    @property
    def size(self) -> int:
        """Total conditions, used as the tie-break against bloat."""
        return sum(len(r.conditions) for r in self.rules)

    def describe(self) -> list[str]:
        return [r.describe() for r in self.rules] + [f"default -> {self.default}"]

    def to_dict(self) -> dict:
        return {
            "rules": [{"tier": r.tier,
                       "conditions": [[c.feature, c.op, c.threshold] for c in r.conditions]}
                      for r in self.rules],
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        return cls(
            rules=tuple(Rule(tuple(Condition(int(f), op, float(t))
                                   for f, op, t in r["conditions"]), r["tier"])
                        for r in data["rules"]),
            default=data["default"],
        )


class FeatureSpace:
    """Candidate thresholds drawn from the data the search may look at.

    Random thresholds would mostly produce conditions that every task passes
    or none does, so the search would spend its budget on rules that cannot
    discriminate. Quantiles of the train features keep every proposed split
    meaningful.
    """

    QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    def __init__(self, X: np.ndarray) -> None:
        self.n_features = X.shape[1]
        self.candidates: list[np.ndarray] = []
        for j in range(self.n_features):
            column = X[:, j]
            unique = np.unique(column)
            if unique.size <= 2:          # binary indicator: one split is enough
                values = np.array([0.5])
            else:
                values = np.unique(np.quantile(column, self.QUANTILES))
            self.candidates.append(values)

    def sample_condition(self, rng: random.Random) -> Condition:
        feature = rng.randrange(self.n_features)
        values = self.candidates[feature]
        return Condition(feature, rng.choice(OPS), float(rng.choice(list(values))))


def random_genome(space: FeatureSpace, rng: random.Random,
                  max_rules: int = MAX_RULES) -> Genome:
    rules = tuple(
        Rule(tuple(space.sample_condition(rng)
                   for _ in range(rng.randint(1, MAX_CONDITIONS))),
             rng.choice(TIERS))
        for _ in range(rng.randint(1, max_rules))
    )
    return Genome(rules=rules, default=rng.choice(TIERS))


def _mask(condition: Condition, X: np.ndarray) -> np.ndarray:
    column = X[:, condition.feature]
    return column < condition.threshold if condition.op == "<" else column >= condition.threshold


def assign(genome: Genome, X: np.ndarray) -> np.ndarray:
    """Tier index per row, first matching rule winning."""
    n = X.shape[0]
    out = np.full(n, TIERS.index(genome.default), dtype=np.int8)
    decided = np.zeros(n, dtype=bool)
    for rule in genome.rules:
        hit = np.ones(n, dtype=bool)
        for condition in rule.conditions:
            hit &= _mask(condition, X)
        fresh = hit & ~decided
        if fresh.any():
            out[fresh] = TIERS.index(rule.tier)
            decided |= fresh
        if decided.all():
            break
    return out


def routing_of(genome: Genome, X: np.ndarray, task_ids: list[int]) -> dict[int, str]:
    return {tid: TIERS[int(i)] for tid, i in zip(task_ids, assign(genome, X))}
