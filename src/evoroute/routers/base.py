"""Router interface.

A router maps tasks to model tiers without ever seeing a gold answer. The one
exception is OracleRouter, which is explicitly an upper bound, not a method.
"""
from __future__ import annotations

from typing import Any, Protocol


class Router(Protocol):
    name: str

    def route(self, tasks: list[dict[str, Any]]) -> dict[int, str]:
        """Return {task_id: tier} for every task, in any order."""
        ...


class Fittable(Protocol):
    def fit(self, tasks: list[dict[str, Any]], matrix: dict) -> None:
        """Train on the *train* split only. `matrix` is the cached response
        matrix; using it here is legitimate supervision, not leakage, because
        it is restricted to training tasks."""
        ...
