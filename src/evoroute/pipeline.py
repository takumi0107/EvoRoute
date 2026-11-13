"""Glue: load the cached matrix, run routers against it, persist results."""
from __future__ import annotations

import json
import sqlite3
import subprocess
from typing import Any

from .config import DEFAULT_WEIGHTS, MODELS, ObjectiveWeights
from .db import response_matrix
from .objective import Mode, RouteOutcome, Scales, evaluate_route, oracle_accuracy


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def load_tasks(conn: sqlite3.Connection, split: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM tasks WHERE split = ? ORDER BY id", (split,)).fetchall()
    return [dict(r) for r in rows]


def model_profile(conn: sqlite3.Connection, load_profile: str = "warm") -> tuple[dict, dict]:
    """Per-model load cost under one regime, plus resident footprint.

    `cold` falls back to `warm` per model when the cold number was never
    measured, and the caller is expected to say so in its output rather than
    silently present a warm number as cold.
    """
    column = "load_ms_cold" if load_profile == "cold" else "load_ms"
    load_ms, resident = {}, {}
    for r in conn.execute("SELECT tier, load_ms, load_ms_cold, resident_mb FROM models"):
        value = r[column]
        load_ms[r["tier"]] = (value if value is not None else r["load_ms"]) or 0.0
        resident[r["tier"]] = r["resident_mb"] or 0.0
    return load_ms, resident


def load_profile_available(conn: sqlite3.Connection, load_profile: str) -> bool:
    if load_profile != "cold":
        return True
    missing = conn.execute(
        "SELECT COUNT(*) FROM models WHERE load_ms_cold IS NULL").fetchone()[0]
    return missing == 0


def build_context(conn: sqlite3.Connection, split: str, load_profile: str = "warm"):
    """Everything a router evaluation needs, all read from cache."""
    tasks = load_tasks(conn, split)
    matrix = response_matrix(conn, split)
    missing = [t["id"] for t in tasks if len(matrix.get(t["id"], {})) < len(MODELS)]
    if missing:
        raise RuntimeError(
            f"response matrix incomplete for split={split}: "
            f"{len(missing)} of {len(tasks)} tasks lack a full row. Run `bench` first.")
    load_ms, resident = model_profile(conn, load_profile)
    scales = Scales.from_matrix(matrix, load_ms, resident)
    return tasks, matrix, scales


def run_router(router, tasks, matrix, scales,
               weights: ObjectiveWeights = DEFAULT_WEIGHTS,
               mode: Mode = "online") -> tuple[RouteOutcome, dict]:
    order = [t["id"] for t in tasks]
    routing = router.route(tasks)
    missing = [i for i in order if i not in routing]
    if missing:
        raise ValueError(f"{router.name} failed to route {len(missing)} tasks")
    return evaluate_route(routing, order, matrix, scales, weights, mode), routing


def persist(conn: sqlite3.Connection, experiment_id: int, router_name: str,
            split: str, seed: int, outcome: RouteOutcome, routing: dict,
            mode: Mode = "online", load_profile: str = "warm") -> None:
    conn.execute(
        "INSERT INTO router_results (experiment_id, router, split, seed, accuracy, "
        "mean_latency_ms, total_latency_ms, inference_ms, total_ms, load_ms, "
        "switches, peak_rss_mb, objective, routing, mode, load_profile) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (experiment_id, router_name, split, seed, outcome.accuracy,
         outcome.mean_latency_ms, outcome.inference_ms, outcome.inference_ms,
         outcome.total_ms, outcome.load_ms, outcome.switches,
         outcome.peak_resident_mb, outcome.objective,
         json.dumps({str(k): v for k, v in routing.items()}), mode, load_profile))


def headroom(matrix: dict, tasks: list[dict], best_fixed_accuracy: float) -> dict:
    """The Week 1 go/no-go number: how much accuracy routing could add."""
    order = [t["id"] for t in tasks]
    ceiling = oracle_accuracy(matrix, order)
    return {
        "oracle_accuracy": ceiling,
        "best_fixed_accuracy": best_fixed_accuracy,
        "headroom": ceiling - best_fixed_accuracy,
    }


def persist_generations(conn: sqlite3.Connection, experiment_id: int,
                        router_name: str, results, mode: Mode,
                        load_profile: str) -> None:
    """Store every seed's curve, not just the winner's.

    The spread across seeds is the difference between "the search found this"
    and "one run got lucky", so discarding the losing seeds would discard the
    only evidence that distinguishes them.
    """
    rows = []
    for result in results:
        seed = result.history[0].get("seed", 0) if result.history else 0
        for record in result.history:
            rows.append((experiment_id, router_name, mode, load_profile, seed,
                         record["generation"], record["best_train"],
                         record["mean_train"], record["best_test"],
                         record["diversity"],
                         json.dumps(record["best_genome"].to_dict()),
                         record["evaluations"], record["elapsed_s"]))
    conn.executemany(
        "INSERT INTO generations (experiment_id, router, mode, load_profile, seed, "
        "generation, best_train, mean_train, best_test, diversity, best_genome, "
        "evaluations, elapsed_s) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
