"""Routers backed by the genetic search.

Two of them, and the difference between them is the Week 3 result:

`EvolvedRouter` is the method -- it fits on train and is scored on test, like
every other fittable router in the comparison.

`CeilingRouter` is a diagnostic that cheats on purpose: it runs the same search
directly on the test split. Its score is therefore not a result but a
measurement of the *feature space* -- the best J any decision list over these
features can reach when generalisation is removed from the problem. Comparing
the two separates "the search failed" from "the features cannot express it",
which is the only way a negative result says anything useful.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..config import DEFAULT_WEIGHTS, ObjectiveWeights
from ..features import cascade_vector, vector
from ..evolution.genome import Genome, routing_of
from ..evolution.search import Context, EvolutionConfig, EvolutionResult, evolve
from ..objective import Mode, Scales


class EvolvedRouter:
    """Restarts the GA from several seeds; keeps the best genome *on train*.

    A single GA run is one sample from a stochastic search, so reporting one
    seed reports luck. Restarting and selecting on train fitness is the honest
    version: the test split is scored every generation but never selected on,
    so the generalisation gap stays a measurement.
    """

    def __init__(self, mode: Mode = "online",
                 config: EvolutionConfig | None = None,
                 seeds: Sequence[int] = (0, 1, 2, 3, 4),
                 name: str = "evoroute",
                 base_runs: dict | None = None) -> None:
        self.mode = mode
        self.config = config or EvolutionConfig()
        self.seeds = tuple(seeds)
        self.name = name
        # task id -> the small model's stored run. Present => cascade setting.
        self.base_runs = base_runs
        self.genome: Genome | None = None
        self.results: list[EvolutionResult] = []
        self.result: EvolutionResult | None = None
        self._test_ctx: Context | None = None

    # -- context plumbing ---------------------------------------------------
    def _context(self, tasks, matrix, scales) -> Context:
        runs = (None if self.base_runs is None
                else {t["id"]: self.base_runs[t["id"]] for t in tasks})
        return Context(tasks, matrix, scales, runs)

    def prepare(self, test_tasks, test_matrix, test_scales) -> None:
        """Supply the held-out split so the GA can *record* its curve on it."""
        self._test_ctx = self._context(test_tasks, test_matrix, test_scales)

    # -- fitting ------------------------------------------------------------
    def _fit_contexts(self, train: Context, test: Context,
                      weights: ObjectiveWeights) -> None:
        self.results = []
        for seed in self.seeds:
            config = EvolutionConfig(**{**self.config.__dict__, "seed": seed})
            self.results.append(evolve(train, test, weights, self.mode, config))
        self.result = max(self.results, key=lambda r: r.best_train)
        self.genome = self.result.best

    def fit(self, tasks: list[dict[str, Any]], matrix: dict, scales: Scales,
            weights: ObjectiveWeights = DEFAULT_WEIGHTS) -> None:
        train = self._context(tasks, matrix, scales)
        self._fit_contexts(train, self._test_ctx or train, weights)

    # -- routing ------------------------------------------------------------
    @property
    def cascade(self) -> bool:
        return self.base_runs is not None

    def _vectors(self, tasks: list[dict[str, Any]]) -> np.ndarray:
        if self.cascade:
            rows = [cascade_vector(t, self.base_runs[t["id"]]["raw_output"],
                                   self.base_runs[t["id"]]["eval_count"])
                    for t in tasks]
        else:
            rows = [vector(t) for t in tasks]
        return np.asarray(rows, dtype=float)

    def route(self, tasks: list[dict[str, Any]]) -> dict[int, str]:
        if self.genome is None:
            raise RuntimeError(f"{self.name}.route called before fit")
        return routing_of(self.genome, self._vectors(tasks), [t["id"] for t in tasks])

    def describe(self) -> list[str]:
        return self.genome.describe() if self.genome else ["<unfitted>"]

    def seed_spread(self) -> dict[str, float]:
        """How much of the reported number is the search and how much is luck."""
        if not self.results:
            return {}
        tests = [r.best_test for r in self.results]
        return {"min": min(tests), "median": float(np.median(tests)),
                "max": max(tests), "n": len(tests)}


class CeilingRouter(EvolvedRouter):
    """Fits on the split it will be scored on. Not a method -- a measurement.

    Reported as `evoroute-ceiling*`; the asterisk is the warning. It answers
    "could a decision list over these features have done better at all?", which
    an honest router cannot answer about itself.
    """

    def __init__(self, mode: Mode = "online",
                 config: EvolutionConfig | None = None,
                 seeds: Sequence[int] = (0, 1, 2, 3, 4),
                 name: str = "evoroute-ceiling*",
                 base_runs: dict | None = None) -> None:
        super().__init__(mode, config, seeds, name, base_runs)

    def fit(self, tasks: list[dict[str, Any]], matrix: dict, scales: Scales,
            weights: ObjectiveWeights = DEFAULT_WEIGHTS) -> None:
        if self._test_ctx is None:
            raise RuntimeError("CeilingRouter needs prepare() with the scoring split")
        self._fit_contexts(self._test_ctx, self._test_ctx, weights)
