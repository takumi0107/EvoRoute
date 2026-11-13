"""Statistical checks on the comparison.

Two questions the main table cannot answer on its own:

1. Is a difference in J real, or is it the particular 300 tasks that happened
   to be in the test split? Answered by a paired bootstrap over tasks.
2. Is the online result an artefact of the order tasks arrive in? The split is
   read `ORDER BY id`, which groups every dataset's tasks together and so
   makes a task-type rule look far cheaper than it would be against a mixed
   stream. Answered by re-evaluating under shuffled arrival orders.

Both resample the cached matrix, so neither costs an inference.
"""
from __future__ import annotations

import numpy as np

from .config import DEFAULT_WEIGHTS, ObjectiveWeights
from .objective import Mode, Scales, evaluate_route


def bootstrap_diff(routing_a: dict, routing_b: dict, order: list[int],
                   matrix: dict, scales: Scales,
                   weights: ObjectiveWeights = DEFAULT_WEIGHTS,
                   mode: Mode = "online", resamples: int = 2000,
                   seed: int = 0, block: int = 10) -> dict:
    """Paired moving-block bootstrap on J(a) - J(b) over the task sample.

    Paired because both routers are scored on the *same* resampled workload:
    the question is whether a routes better than b, not whether the two
    happened to be measured on easy draws.

    Blocks rather than single tasks because J is not a mean over tasks. In
    online mode it charges a reload whenever consecutive tasks want different
    tiers, so the statistic depends on the sequence. Drawing tasks
    independently breaks that sequence twice over -- it permutes arrival order
    and it places duplicates of a task next to each other, where they can
    never trigger a switch -- and both push the resampled J above the real one,
    far enough that the interval stops containing the point estimate.
    Resampling contiguous blocks keeps the local structure that the switch
    count is made of. Order sensitivity is measured separately, below.
    """
    rng = np.random.default_rng(seed)
    n = len(order)
    ids = np.asarray(order)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(resamples)
    for i in range(resamples):
        starts = rng.integers(0, n - block + 1, n_blocks)
        picks = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        draw = [int(x) for x in ids[picks]]
        a = evaluate_route(routing_a, draw, matrix, scales, weights, mode).objective
        b = evaluate_route(routing_b, draw, matrix, scales, weights, mode).objective
        diffs[i] = a - b
    point = (evaluate_route(routing_a, order, matrix, scales, weights, mode).objective
             - evaluate_route(routing_b, order, matrix, scales, weights, mode).objective)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "win_rate": float(np.mean(diffs > 0)),
        "tie_rate": float(np.mean(diffs == 0)),
        "resamples": resamples,
        "block": block,
    }


def arrival_sensitivity(routing: dict, order: list[int], matrix: dict,
                        scales: Scales,
                        weights: ObjectiveWeights = DEFAULT_WEIGHTS,
                        shuffles: int = 200, seed: int = 0) -> dict:
    """J under randomised arrival order, online mode only.

    Batch regroups by tier, so its result is order-invariant by construction
    and this check does not apply to it. Online pays a reload on every tier
    change, so the result depends on how the stream is interleaved -- and the
    id-sorted order used elsewhere is the most favourable interleaving there
    is, because it is sorted by dataset.
    """
    rng = np.random.default_rng(seed)
    ids = np.asarray(order)
    values, switches = np.empty(shuffles), np.empty(shuffles)
    for i in range(shuffles):
        draw = [int(x) for x in rng.permutation(ids)]
        out = evaluate_route(routing, draw, matrix, scales, weights, "online")
        values[i] = out.objective
        switches[i] = out.switches
    sorted_out = evaluate_route(routing, order, matrix, scales, weights, "online")
    return {
        "sorted": sorted_out.objective,
        "sorted_switches": sorted_out.switches,
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean_switches": float(switches.mean()),
        "shuffles": shuffles,
    }
