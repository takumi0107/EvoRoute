"""The statistics decide whether the headline result is a tie or a win, so the
resampler is checked for the two ways it could quietly lie: an interval that
does not contain its own point estimate, and a resample whose sequence
structure differs from the data's.
"""
from __future__ import annotations

import pytest

from evoroute.config import TIERS, ObjectiveWeights
from evoroute.objective import MODES, evaluate_route
from evoroute.stats import arrival_sensitivity, bootstrap_diff

from test_objective import random_instance

WEIGHTS = ObjectiveWeights(time=0.40, memory=0.10)


def instance(n: int, seed: int):
    matrix, scales = random_instance(n, seed)
    return matrix, scales, list(range(n))


def routings(order, seed):
    """Two routers to compare: a fixed one and a switch-happy one."""
    import random

    rng = random.Random(seed)
    fixed = {i: TIERS[0] for i in order}
    mixed = {i: rng.choice(TIERS) for i in order}
    return mixed, fixed


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(4))
def test_point_estimate_lies_inside_its_interval(mode, seed):
    """A percentile interval that excludes the statistic it describes means the
    resample is biased, not that the effect is significant. An iid bootstrap
    does exactly that here, because duplicated tasks land adjacent and cannot
    trigger a reload."""
    matrix, scales, order = instance(60, seed)
    a, b = routings(order, seed)
    out = bootstrap_diff(a, b, order, matrix, scales, WEIGHTS, mode,
                         resamples=300, seed=seed)
    assert out["ci_low"] <= out["diff"] <= out["ci_high"]


@pytest.mark.parametrize("seed", range(3))
def test_blocks_preserve_the_switch_rate(seed):
    """What the block length buys: a resampled workload that reloads about as
    often as the real one."""
    matrix, scales, order = instance(120, seed)
    a, _ = routings(order, seed)
    real = evaluate_route(a, order, matrix, scales, WEIGHTS, "online").switches

    import numpy as np

    rng = np.random.default_rng(seed)
    ids = np.asarray(order)
    counts = []
    for _ in range(40):
        starts = rng.integers(0, len(order) - 10 + 1, 12)
        picks = np.concatenate([np.arange(s, s + 10) for s in starts])[:len(order)]
        draw = [int(x) for x in ids[picks]]
        counts.append(evaluate_route(a, draw, matrix, scales, WEIGHTS, "online").switches)
    assert abs(float(np.mean(counts)) - real) / real < 0.25


@pytest.mark.parametrize("seed", range(3))
def test_arrival_order_cannot_touch_a_single_tier_router(seed):
    """A router that never switches is order-invariant by construction; if the
    sensitivity check reports otherwise it is measuring itself."""
    matrix, scales, order = instance(60, seed)
    _, fixed = routings(order, seed)
    out = arrival_sensitivity(fixed, order, matrix, scales, WEIGHTS, shuffles=25)
    assert out["sd"] == pytest.approx(0.0, abs=1e-12)
    assert out["mean"] == pytest.approx(out["sorted"], abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_shuffling_breaks_up_a_grouped_workload(seed):
    """The reason the experiment reports this check at all.

    The test split is read `ORDER BY id`, which lays every dataset's tasks out
    in one contiguous run, so a router keyed on task type switches a handful of
    times. That grouping is an artefact of how the split is stored, not a
    property of a real arrival stream -- interleave the stream and the same
    routing reloads far more often. A routing built in blocks reproduces the
    structure; a routing drawn per-task would not, because it is already
    maximally interleaved and shuffling could only help it.
    """
    matrix, scales, order = instance(90, seed)
    grouped = {i: TIERS[(i // 30) % len(TIERS)] for i in order}
    out = arrival_sensitivity(grouped, order, matrix, scales, WEIGHTS, shuffles=30)
    assert out["sorted_switches"] == len(TIERS)
    assert out["mean_switches"] > 3 * out["sorted_switches"]
    assert out["mean"] < out["sorted"]


def test_identical_routers_compare_as_an_exact_tie():
    matrix, scales, order = instance(40, 0)
    _, fixed = routings(order, 0)
    out = bootstrap_diff(fixed, dict(fixed), order, matrix, scales, WEIGHTS,
                         "online", resamples=100)
    assert out["diff"] == 0.0
    assert out["tie_rate"] == 1.0
    assert (out["ci_low"], out["ci_high"]) == (0.0, 0.0)


def test_snapshot_round_trips(tmp_path):
    """The repo ships the measurement as a file; if the file cannot rebuild the
    database, the analysis is not reproducible from a clone."""
    from evoroute.db import connect, export_snapshot, import_snapshot

    src = connect(tmp_path / "src.db")
    src.execute("INSERT INTO models (tier, ollama_tag, params_b, quant, load_ms) "
                "VALUES ('small', 'x:1b', 1.0, 'Q4', 100.0)")
    src.execute("INSERT INTO tasks (id, dataset, task_type, split, source_idx, "
                "prompt, gold) VALUES (1, 'd', 'classification', 'test', 0, 'p', 'g')")
    src.execute("INSERT INTO runs (task_id, tier, raw_output, score, latency_ms) "
                "VALUES (1, 'small', 'o', 1.0, 50.0)")
    src.commit()
    counts = export_snapshot(src, tmp_path / "snap.json.gz")
    assert counts == {"models": 1, "tasks": 1, "runs": 1}

    dst = connect(tmp_path / "dst.db")
    assert import_snapshot(dst, tmp_path / "snap.json.gz") == counts
    assert dst.execute("SELECT score FROM runs").fetchone()["score"] == 1.0

    with pytest.raises(RuntimeError, match="already holds"):
        import_snapshot(dst, tmp_path / "snap.json.gz")
