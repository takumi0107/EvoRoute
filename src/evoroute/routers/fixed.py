"""Fixed-model baselines: every task to one tier."""
from __future__ import annotations

from typing import Any


class FixedRouter:
    def __init__(self, tier: str, name: str | None = None) -> None:
        self.tier = tier
        self.name = name or f"fixed-{tier}"

    def route(self, tasks: list[dict[str, Any]]) -> dict[int, str]:
        return {t["id"]: self.tier for t in tasks}
