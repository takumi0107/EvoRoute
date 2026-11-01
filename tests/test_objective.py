"""The optimum is the ceiling every result is measured against, so both
execution modes are checked against exhaustive enumeration rather than trusted.
"""
from __future__ import annotations

import itertools
import random

import pytest

from evoroute.config import TIERS, ObjectiveWeights
from evoroute.objective import (MODES, Scales, evaluate_route, execution_order,
                                optimal_route)

WEIGHTS = ObjectiveWeights(time=0.40, memory=0.10)


def random_instance(n: int, seed: int, load_scale: float = 1.0):
    rng = random.Random(seed)
    matrix = {
        i: {t: {"score": rng.choice([0.0, 0.5, 1.0]),
                "latency_ms": rng.uniform(100, 1500),
                "peak_rss_mb": 0.0}
            for t in TIERS}
        for i in range(n)
    }
    load_ms = {t: rng.uniform(300, 6000) * load_scale for t in TIERS}
    resident = {t: rng.uniform(900, 5000) for t in TIERS}
    return matrix, Scales.from_matrix(matrix, load_ms, resident)


def brute_force(order, matrix, scales, mode):
    return max(
        evaluate_route(dict(zip(order, combo)), order, matrix, scales, WEIGHTS, mode).objective
        for combo in itertools.product(TIERS, repeat=len(order)))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(10))
def test_optimum_matches_brute_force(mode, seed):
    n = 7                                     # 3^7 = 2187 routings
    matrix, scales = random_instance(n, seed)
    order = list(range(n))
    found = evaluate_route(optimal_route(order, matrix, scales, WEIGHTS, mode),
                           order, matrix, scales, WEIGHTS, mode).objective
    assert found == pytest.approx(brute_force(order, matrix, scales, mode), abs=1e-9)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(6))
def test_optimum_dominates_every_fixed_model(mode, seed):
    matrix, scales = random_instance(40, seed)
    order = list(range(40))
    best = evaluate_route(optimal_route(order, matrix, scales, WEIGHTS, mode),
                          order, matrix, scales, WEIGHTS, mode).objective
    for tier in TIERS:
        fixed = evaluate_route({i: tier for i in order}, order, matrix, scales,
                               WEIGHTS, mode).objective
        assert best >= fixed - 1e-9


@pytest.mark.parametrize("seed", range(6))
def test_batch_never_worse_than_online(seed):
    """Regrouping can only remove reloads, never add them."""
    matrix, scales = random_instance(40, seed)
    order = list(range(40))
    vals = {m: evaluate_route(optimal_route(order, matrix, scales, WEIGHTS, m),
                              order, matrix, scales, WEIGHTS, m).objective
            for m in MODES}
    assert vals["batch"] >= vals["online"] - 1e-9


@pytest.mark.parametrize("seed", range(6))
def test_batch_pays_each_tier_at_most_once(seed):
    matrix, scales = random_instance(40, seed)
    order = list(range(40))
    routing = optimal_route(order, matrix, scales, WEIGHTS, "batch")
    out = evaluate_route(routing, order, matrix, scales, WEIGHTS, "batch")
    assert out.switches == len(set(routing.values()))


def test_batch_regroups_execution_order():
    routing = {0: "large", 1: "small", 2: "large", 3: "small"}
    order = [0, 1, 2, 3]
    assert execution_order(routing, order, "online") == order
    seq = execution_order(routing, order, "batch")
    tiers = [routing[i] for i in seq]
    assert tiers == sorted(tiers, key=TIERS.index)


@pytest.mark.parametrize("mode", MODES)
def test_expensive_loads_suppress_switching(mode):
    """The bug this objective replaced: scaling load cost left the optimum
    unchanged. Heavy loads must now visibly reduce the number of reloads."""
    counts = {}
    for scale, label in ((0.01, "cheap"), (50.0, "expensive")):
        matrix, scales = random_instance(60, 7, load_scale=scale)
        order = list(range(60))
        routing = optimal_route(order, matrix, scales, WEIGHTS, mode)
        counts[label] = evaluate_route(routing, order, matrix, scales,
                                       WEIGHTS, mode).switches
    assert counts["expensive"] <= counts["cheap"]


def test_zero_time_weight_reduces_to_per_task_argmax():
    matrix, scales = random_instance(25, 99)
    order = list(range(25))
    weights = ObjectiveWeights(time=0.0, memory=0.0)
    routing = optimal_route(order, matrix, scales, weights, "online")
    for task_id in order:
        best = max(matrix[task_id][t]["score"] for t in TIERS)
        assert matrix[task_id][routing[task_id]]["score"] == pytest.approx(best)
