"""Router backed by a genome found by the genetic search."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..config import DEFAULT_WEIGHTS, ObjectiveWeights
from ..features import vector
from ..evolution.genome import Genome, routing_of
from ..evolution.search import Context, EvolutionConfig, evolve
from ..objective import Mode, Scales


class EvolvedRouter:
    """Fits by running the GA on the train split; routes with the winner."""

    def __init__(self, mode: Mode = "online", config: EvolutionConfig | None = None,
                 name: str = "evoroute") -> None:
        self.mode = mode
        self.config = config or EvolutionConfig()
        self.name = name
        self.genome: Genome | None = None
        self.result = None
        self._test_ctx: Context | None = None

    def prepare(self, test_tasks, test_matrix, test_scales) -> None:
        """Supply the held-out split so the GA can *record* its curve on it.

        Only recorded -- selection reads train fitness alone. Without this the
        overfitting gap could not be plotted per generation.
        """
        self._test_ctx = Context(test_tasks, test_matrix, test_scales)

    def fit(self, tasks: list[dict[str, Any]], matrix: dict, scales: Scales,
            weights: ObjectiveWeights = DEFAULT_WEIGHTS) -> None:
        train = Context(tasks, matrix, scales)
        test = self._test_ctx or train
        self.result = evolve(train, test, weights, self.mode, self.config)
        self.genome = self.result.best

    def route(self, tasks: list[dict[str, Any]]) -> dict[int, str]:
        if self.genome is None:
            raise RuntimeError("EvolvedRouter.route called before fit")
        X = np.asarray([vector(t) for t in tasks], dtype=float)
        return routing_of(self.genome, X, [t["id"] for t in tasks])
