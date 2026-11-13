"""Routers trained on the train split of the cached response matrix.

Two framings of the same supervision, because they behave very differently
here:

LearnedRouter   predicts the best tier as a class label. This is the proposal's
    formulation, and it walks into the imbalance trap: 85% of training tasks
    are best served by `small`, so a classifier maximising label accuracy is
    rewarded for collapsing onto "always small" -- which is exactly
    fixed-small, with none of routing's value.

UtilityRouter   predicts each tier's utility and takes the argmax. Errors are
    then penalised in proportion to the utility actually lost, so the 15%
    minority that routing exists to serve is not averaged away. Nothing about
    it depends on the class balance.

Neither can see what ran before, so neither can trade accuracy for fewer model
reloads. That blind spot is what Week 3 attacks.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..config import DEFAULT_WEIGHTS, TIERS, ObjectiveWeights
from ..features import FEATURE_NAMES, vector
from ..objective import Scales, best_local_tier, local_gains


def _design_matrix(tasks: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([vector(t) for t in tasks], dtype=float)


class LearnedRouter:
    """Tier-as-class-label classifier."""

    def __init__(self, kind: str = "tree", balanced: bool = True,
                 seed: int = 0, max_depth: int = 4) -> None:
        self.kind, self.balanced, self.seed, self.max_depth = kind, balanced, seed, max_depth
        self.name = f"learned-{kind}" + ("" if balanced else "-unweighted")
        self._model = None
        self._fallback: str = TIERS[0]

    def _make(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier

        weight = "balanced" if self.balanced else None
        if self.kind == "tree":
            return DecisionTreeClassifier(max_depth=self.max_depth,
                                          class_weight=weight,
                                          random_state=self.seed)
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight=weight,
                               random_state=self.seed))

    def fit(self, tasks: list[dict[str, Any]], matrix: dict, scales: Scales,
            weights: ObjectiveWeights = DEFAULT_WEIGHTS) -> None:
        n = len(tasks)
        y = [best_local_tier(t["id"], matrix, scales, weights, n) for t in tasks]
        self._fallback = max(set(y), key=y.count)
        if len(set(y)) < 2:          # degenerate split: nothing to learn
            self._model = None
            return
        self._model = self._make()
        self._model.fit(_design_matrix(tasks), y)

    def route(self, tasks: list[dict[str, Any]]) -> dict[int, str]:
        if self._model is None:
            return {t["id"]: self._fallback for t in tasks}
        pred = self._model.predict(_design_matrix(tasks))
        return {t["id"]: str(p) for t, p in zip(tasks, pred)}


class UtilityRouter:
    """One regressor per tier over task utility; argmax at decision time."""

    def __init__(self, kind: str = "tree", seed: int = 0, max_depth: int = 4) -> None:
        self.kind, self.seed, self.max_depth = kind, seed, max_depth
        self.name = f"utility-{kind}"
        self._models: dict[str, Any] = {}

    def _make(self):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeRegressor

        if self.kind == "tree":
            return DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=self.seed)
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=self.seed))

    def fit(self, tasks: list[dict[str, Any]], matrix: dict, scales: Scales,
            weights: ObjectiveWeights = DEFAULT_WEIGHTS) -> None:
        n = len(tasks)
        X = _design_matrix(tasks)
        # Targets are scaled by n so they sit near O(1); argmax is unaffected.
        gains = {t["id"]: local_gains(t["id"], matrix, scales, weights, n) for t in tasks}
        self._models = {}
        for tier in TIERS:
            y = np.asarray([gains[t["id"]][tier] * n for t in tasks], dtype=float)
            model = self._make()
            model.fit(X, y)
            self._models[tier] = model

    def route(self, tasks: list[dict[str, Any]]) -> dict[int, str]:
        X = _design_matrix(tasks)
        preds = np.column_stack([self._models[t].predict(X) for t in TIERS])
        picked = preds.argmax(axis=1)
        return {task["id"]: TIERS[int(i)] for task, i in zip(tasks, picked)}


def feature_importances(router: LearnedRouter) -> list[tuple[str, float]]:
    """Which inputs a fitted tree actually splits on. Reported in Week 4."""
    model = getattr(router, "_model", None)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    pairs = list(zip(FEATURE_NAMES, (float(v) for v in importances)))
    return sorted(pairs, key=lambda kv: -kv[1])
