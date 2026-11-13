"""The objective J, and the exact optimum of it under each execution mode.

    J = accuracy - w_time * (inference + load)/time_ref - w_mem * memory/mem_ref

Why load and inference share one term
-------------------------------------
An earlier version followed the proposal literally and normalised the switch
penalty by its own reference (n * max_load). That made J blind to the thing it
was supposed to measure: scaling every load time by 6x left the optimum almost
unchanged, because numerator and denominator moved together. Load time and
inference time are the same physical quantity -- wall clock a caller waits on
-- so they are summed first and normalised once. With that fixed, expensive
loads visibly collapse routing onto a single model.

Cascade
-------
Under `cascade=True` the smallest tier runs on every task first and the router
decides whether to escalate. That buys the router evidence a pre-hoc feature
vector cannot carry -- what the small model actually said -- and it is costed
honestly: the small inference is paid on every task, an escalated task pays
both, and the small model is treated as permanently resident (it fits
alongside one larger model in 16GB), so only the escalation tier can thrash.

Execution modes
---------------
online: tasks are answered in arrival order, so every change of tier forces a
    reload. Switch cost depends on the previous choice -> exact optimum by DP.
batch: the whole workload is known up front and may be regrouped by tier, so
    only the *set* of tiers used is paid for, at most once each. Exact optimum
    by enumerating the 2^|TIERS| - 1 non-empty subsets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

from .config import DEFAULT_WEIGHTS, TIERS, ObjectiveWeights

Mode = Literal["online", "batch"]
MODES: tuple[Mode, ...] = ("online", "batch")


@dataclass
class Scales:
    """Reference magnitudes putting each cost term on a [0, 1] scale.

    time_ref is the worst case a caller could face: the entire workload on the
    slowest model, including its one load.
    """

    time_ref: float
    mem_ref: float
    load_ms: dict[str, float] = field(default_factory=dict)
    resident_mb: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_matrix(cls, matrix: dict, load_ms: dict[str, float],
                    resident_mb: dict[str, float]) -> "Scales":
        if not matrix:
            raise ValueError("empty response matrix")
        worst = max(
            sum(row[t]["latency_ms"] for row in matrix.values() if t in row)
            + load_ms.get(t, 0.0)
            for t in TIERS)
        return cls(
            time_ref=worst or 1.0,
            mem_ref=(max(resident_mb.values()) if resident_mb else 1.0) or 1.0,
            load_ms=dict(load_ms),
            resident_mb=dict(resident_mb),
        )


@dataclass
class RouteOutcome:
    accuracy: float
    inference_ms: float
    load_ms: float
    total_ms: float
    mean_latency_ms: float
    switches: int
    mean_resident_mb: float
    peak_resident_mb: float
    objective: float


BASE_TIER = TIERS[0]


def execution_order(routing: dict[int, str], order: list[int], mode: Mode) -> list[int]:
    """The sequence tasks actually run in. Batch mode is free to regroup."""
    if mode == "batch":
        return sorted(order, key=lambda i: TIERS.index(routing[i]))
    return list(order)


def evaluate_route(routing: dict[int, str], order: list[int], matrix: dict,
                   scales: Scales, weights: ObjectiveWeights = DEFAULT_WEIGHTS,
                   mode: Mode = "online", cascade: bool = False) -> RouteOutcome:
    """Score a complete routing. No model is invoked: every number is read from
    the cached matrix plus the per-model load profile."""
    if not order:
        raise ValueError("empty task order")

    score_sum = inference = mem_sum = load_sum = 0.0
    switches = 0
    peak = 0.0
    previous: str | None = None
    base_resident = scales.resident_mb.get(BASE_TIER, 0.0) if cascade else 0.0
    if cascade:
        load_sum += scales.load_ms.get(BASE_TIER, 0.0)   # always resident
        switches += 1

    for task_id in execution_order(routing, order, mode):
        tier = routing[task_id]
        cell = matrix[task_id][tier]
        score_sum += cell["score"]
        resident = scales.resident_mb.get(tier, 0.0)

        if cascade:
            inference += matrix[task_id][BASE_TIER]["latency_ms"]
            if tier != BASE_TIER:
                inference += cell["latency_ms"]
                mem_sum += base_resident + resident
                peak = max(peak, base_resident + resident)
                if tier != previous:      # only the escalation slot reloads
                    switches += 1
                    load_sum += scales.load_ms.get(tier, 0.0)
                    previous = tier
            else:
                mem_sum += base_resident
                peak = max(peak, base_resident)
        else:
            inference += cell["latency_ms"]
            mem_sum += resident
            peak = max(peak, resident)
            if tier != previous:
                switches += 1
                load_sum += scales.load_ms.get(tier, 0.0)
                previous = tier

    n = len(order)
    accuracy = score_sum / n
    total = inference + load_sum
    objective = (accuracy
                 - weights.time * (total / scales.time_ref)
                 - weights.memory * ((mem_sum / n) / scales.mem_ref))
    return RouteOutcome(
        accuracy=accuracy,
        inference_ms=inference,
        load_ms=load_sum,
        total_ms=total,
        mean_latency_ms=inference / n,
        switches=switches,
        mean_resident_mb=mem_sum / n,
        peak_resident_mb=peak,
        objective=objective,
    )


def oracle_accuracy(matrix: dict, order: list[int]) -> float:
    """Ceiling on accuracy alone, ignoring every cost."""
    return sum(max(c["score"] for c in matrix[t].values()) for t in order) / len(order)


def _local_gain(task_id: int, tier: str, matrix: dict, scales: Scales,
                weights: ObjectiveWeights, n: int, cascade: bool = False) -> float:
    """The part of J attributable to one task under one tier."""
    cell = matrix[task_id][tier]
    if cascade:
        latency = matrix[task_id][BASE_TIER]["latency_ms"]
        memory = scales.resident_mb.get(BASE_TIER, 0.0)
        if tier != BASE_TIER:
            latency += cell["latency_ms"]
            memory += scales.resident_mb.get(tier, 0.0)
    else:
        latency = cell["latency_ms"]
        memory = scales.resident_mb.get(tier, 0.0)
    return (cell["score"] / n
            - weights.time / scales.time_ref * latency
            - weights.memory / (scales.mem_ref * n) * memory)


def local_gains(task_id: int, matrix: dict, scales: Scales,
                weights: ObjectiveWeights = DEFAULT_WEIGHTS,
                n: int = 1, cascade: bool = False) -> dict[str, float]:
    """Per-tier utility of one task, with the sequencing term left out.

    Load cost depends on what ran *before*, which a per-task router cannot
    see, so it is excluded here by construction. That exclusion is the gap
    evolutionary search is meant to exploit: a strategy evaluated as a whole
    can trade a little per-task utility for far fewer reloads, and no
    per-task classifier can represent that trade.
    """
    return {t: _local_gain(task_id, t, matrix, scales, weights, n, cascade)
            for t in TIERS}


def best_local_tier(task_id: int, matrix: dict, scales: Scales,
                    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
                    n: int = 1, cascade: bool = False) -> str:
    """The supervision signal for a learned router.

    Not "which model got it right": several may, and when none do the cheapest
    is still correct to pick. Taking the argmax of utility handles both without
    a special case.
    """
    gains = local_gains(task_id, matrix, scales, weights, n, cascade)
    return max(TIERS, key=lambda t: (gains[t], -TIERS.index(t)))


def _optimal_online(order, matrix, scales, weights, cascade=False):
    """Exact argmax of J by DP.

    Plain routing: the state is the loaded tier, so a change costs a reload.
    Cascade: the base tier is always resident, so the state is whichever
    *escalation* tier occupies the second slot -- routing a task to the base
    tier leaves it untouched, and a later task can return to it for free.
    """
    n = len(order)
    w_load = weights.time / scales.time_ref

    def gain(i, tier):
        return _local_gain(order[i], tier, matrix, scales, weights, n, cascade)

    states = ((None,) + tuple(t for t in TIERS if t != BASE_TIER)) if cascade else TIERS

    def step(state, tier):
        """(next state, load cost) for running `tier` while holding `state`."""
        if not cascade:
            return tier, (0.0 if state == tier else scales.load_ms.get(tier, 0.0))
        if tier == BASE_TIER:
            return state, 0.0
        return tier, (0.0 if state == tier else scales.load_ms.get(tier, 0.0))

    best: dict = {}
    choice: list[dict] = [{} for _ in range(n)]
    # Nothing is loaded before the first task, so its tier always pays a load.
    for tier in TIERS:
        landed, cost = step(None, tier)
        value = gain(0, tier) - w_load * cost
        if landed not in best or value > best[landed]:
            best[landed] = value
            choice[0][landed] = tier

    for i in range(1, n):
        nxt_best: dict = {}
        for state in states:
            options = {}
            for prev in states:
                if prev not in best:
                    continue
                for tier in TIERS:
                    landed, cost = step(prev, tier)
                    if landed != state:
                        continue
                    value = best[prev] + gain(i, tier) - w_load * cost
                    if state not in options or value > options[state][0]:
                        options[state] = (value, prev, tier)
            if state in options:
                nxt_best[state] = options[state][0]
                choice[i][state] = options[state][1:]
        best = nxt_best

    state = max(best, key=best.__getitem__)
    routing: dict[int, str] = {}
    for i in range(n - 1, 0, -1):
        prev, tier = choice[i][state]
        routing[order[i]] = tier
        state = prev
    routing[order[0]] = choice[0][state]
    return routing


def _optimal_batch(order, matrix, scales, weights, cascade=False):
    """Exact argmax of J when the workload may be regrouped by tier.

    Regrouping means each tier is loaded at most once, so the sequencing cost
    is a function of the tier *set* alone. Fix the set, and every task
    independently takes its best tier within it.
    """
    n = len(order)
    w_load = weights.time / scales.time_ref
    best_value, best_routing = None, None

    for size in range(1, len(TIERS) + 1):
        for subset in combinations(TIERS, size):
            routing = {
                i: max(subset,
                       key=lambda t: _local_gain(i, t, matrix, scales, weights, n, cascade))
                for i in order
            }
            used = set(routing.values())
            if cascade:
                used = {t for t in used if t != BASE_TIER} | {BASE_TIER}
            value = (sum(_local_gain(i, routing[i], matrix, scales, weights, n, cascade)
                         for i in order)
                     - w_load * sum(scales.load_ms.get(t, 0.0) for t in used))
            if best_value is None or value > best_value:
                best_value, best_routing = value, routing
    return best_routing


def optimal_route(order: list[int], matrix: dict, scales: Scales,
                  weights: ObjectiveWeights = DEFAULT_WEIGHTS,
                  mode: Mode = "online", cascade: bool = False) -> dict[int, str]:
    """Exact J-maximising assignment. Reads the gold answers, so it is a
    ceiling, not a method."""
    if not order:
        raise ValueError("empty task order")
    if mode == "batch":
        return _optimal_batch(order, matrix, scales, weights, cascade)
    return _optimal_online(order, matrix, scales, weights, cascade)
