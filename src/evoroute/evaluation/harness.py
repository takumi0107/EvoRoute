"""Week 1: build the response matrix and profile the models.

Two measurements are taken, and they are deliberately separated:

  profile_models()  -- load time (two regimes) and footprint per model.
                       Measured once; reused by the load cost in every J.
  build_matrix()    -- warm inference cost and score for every (task, model).
                       Measured once; reused by every router.

build_matrix groups work by model so each model is loaded a single time for
its whole pass. That keeps the matrix honest: every latency in `runs` is a
warm-model number, with load cost accounted for separately.
"""
from __future__ import annotations

import json
import sqlite3
import time

from ..config import MODELS, DEFAULT_GEN, ModelSpec
from ..models.ollama_client import OllamaClient
from .scoring import score_output

WARM_LOAD_REPEATS = 5


def _purge_page_cache() -> bool:
    """Drop the OS file cache so the next load reads from disk for real.

    macOS exposes this only through `sudo purge`. If it is unavailable the
    caller is told, rather than a page-cached number being passed off as cold.
    """
    import subprocess
    try:
        subprocess.run(["sudo", "-n", "purge"], check=True, timeout=120,
                       capture_output=True)
        return True
    except Exception:
        return False


def profile_models(conn: sqlite3.Connection, client: OllamaClient,
                   specs: tuple[ModelSpec, ...] = MODELS,
                   repeats: int = WARM_LOAD_REPEATS,
                   purge: bool = False) -> list[dict]:
    """Measure both load regimes.

    warm: evicted from ollama, file still in the OS page cache. This is the
        repeated-swap steady state, and it is what `repeats` samples.
    cold: nothing cached anywhere. Only measurable by purging the page cache
        first, so it is recorded as NULL unless `purge` succeeds -- a
        page-cached number must never be reported as a cold one.
    """
    out = []
    probe = "Reply with the single word: ok"
    for spec in specs:
        cold = None
        if purge:
            client.unload(spec.ollama_tag)
            time.sleep(1.0)
            if _purge_page_cache():
                cold = client.generate(spec.ollama_tag, probe, DEFAULT_GEN).load_ms

        warm_samples, resident = [], None
        for _ in range(repeats):
            client.unload(spec.ollama_tag)
            time.sleep(1.0)  # let the runtime actually release the weights
            c = client.generate(spec.ollama_tag, probe, DEFAULT_GEN)
            warm_samples.append(c.load_ms)
            resident = resident or client.resident_mb(spec.ollama_tag)

        warm = sorted(warm_samples)[len(warm_samples) // 2]
        conn.execute(
            "UPDATE models SET load_ms = ?, load_ms_cold = COALESCE(?, load_ms_cold), "
            "resident_mb = ?, load_samples = ? WHERE tier = ?",
            (warm, cold, resident, json.dumps(warm_samples), spec.tier))
        out.append({"tier": spec.tier, "load_ms": warm, "load_ms_cold": cold,
                    "samples": warm_samples, "resident_mb": resident})
    conn.commit()
    return out


def pending_pairs(conn: sqlite3.Connection, tier: str) -> list[sqlite3.Row]:
    """Tasks this model has not answered yet -- makes the run resumable."""
    return list(conn.execute(
        "SELECT t.* FROM tasks t WHERE NOT EXISTS ("
        "  SELECT 1 FROM runs r WHERE r.task_id = t.id "
        "  AND r.tier = ? AND r.repeat_idx = 0) ORDER BY t.id",
        (tier,)))


def build_matrix(conn: sqlite3.Connection, client: OllamaClient,
                 specs: tuple[ModelSpec, ...] = MODELS,
                 progress=None) -> dict[str, int]:
    """Fill `runs`. Safe to interrupt and re-run: completed pairs are skipped."""
    from ..db import record_run

    done: dict[str, int] = {}
    for spec in specs:
        tasks = pending_pairs(conn, spec.tier)
        done[spec.tier] = 0
        if not tasks:
            continue
        for i, task in enumerate(tasks):
            try:
                c = client.generate(spec.ollama_tag, task["prompt"], DEFAULT_GEN)
                score, parsed, extra = score_output(dict(task), c.text)
                record_run(conn, task_id=task["id"], tier=spec.tier, repeat_idx=0,
                           raw_output=c.text, parsed=parsed, score=score,
                           latency_ms=c.inference_ms, eval_count=c.eval_count,
                           prompt_count=c.prompt_count,
                           peak_rss_mb=client.resident_mb(spec.ollama_tag),
                           error=json.dumps(extra) if extra else None)
            except Exception as exc:  # keep the sweep alive; record the gap
                record_run(conn, task_id=task["id"], tier=spec.tier, repeat_idx=0,
                           raw_output="", parsed=None, score=0.0, latency_ms=0.0,
                           eval_count=0, prompt_count=0, peak_rss_mb=None,
                           error=f"{type(exc).__name__}: {exc}")
            done[spec.tier] += 1
            if i % 10 == 0:
                conn.commit()
            if progress:
                progress(spec.tier, i + 1, len(tasks))
        conn.commit()
    return done
