"""Upper bound. Not a router -- the ceiling real routers are measured against."""
from __future__ import annotations

from typing import Any

from ..config import DEFAULT_WEIGHTS, ObjectiveWeights
from ..objective import Mode, Scales, optimal_route


class OracleRouter:
    """Exact argmax of J over the cached matrix, for one execution mode.

    Reads the gold answers, so it is unachievable by construction. Its value is
    telling us whether routing has headroom at all: if it sits level with the
    best fixed model, no router can win and the setup has to be reframed.
    """

    def __init__(self, matrix: dict, scales: Scales,
                 weights: ObjectiveWeights = DEFAULT_WEIGHTS,
                 mode: Mode = "online") -> None:
        self.matrix, self.scales, self.weights, self.mode = matrix, scales, weights, mode
        self.name = "oracle"

    def route(self, tasks: list[dict[str, Any]]) -> dict[int, str]:
        order = [t["id"] for t in tasks]
        return optimal_route(order, self.matrix, self.scales, self.weights, self.mode)
