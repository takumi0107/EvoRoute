"""SQLite persistence.

The schema is built around one idea: `runs` is a dense (task x model) response
matrix, measured exactly once. Every router -- rule-based, learned, evolved --
is then evaluated by *looking up* rows in that matrix instead of calling a
model. That is what makes thousands of evolutionary fitness evaluations cheap.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import DB_PATH

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS models (
    tier        TEXT PRIMARY KEY,
    ollama_tag  TEXT NOT NULL,
    params_b    REAL NOT NULL,
    quant       TEXT NOT NULL,
    -- Two load regimes, both measured in Week 1 and both reported:
    --   load_ms      steady state -- evicted from ollama, file still in the
    --                OS page cache (the repeated-swap case)
    --   load_ms_cold first touch -- nothing cached anywhere
    load_ms       REAL,
    load_ms_cold  REAL,
    load_samples  TEXT,
    disk_bytes    INTEGER,
    resident_mb   REAL
);

CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY,
    dataset     TEXT NOT NULL,
    task_type   TEXT NOT NULL,
    split       TEXT NOT NULL CHECK (split IN ('train', 'test')),
    source_idx  INTEGER NOT NULL,
    prompt      TEXT NOT NULL,
    gold        TEXT NOT NULL,
    meta        TEXT NOT NULL DEFAULT '{}',
    UNIQUE (dataset, split, source_idx)
);

-- The response matrix. One row per (task, model, repeat).
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY,
    task_id       INTEGER NOT NULL REFERENCES tasks(id),
    tier          TEXT NOT NULL REFERENCES models(tier),
    repeat_idx    INTEGER NOT NULL DEFAULT 0,
    raw_output    TEXT NOT NULL,
    parsed        TEXT,
    score         REAL NOT NULL,          -- per-item metric in [0, 1]
    latency_ms    REAL NOT NULL,          -- inference only, model already warm
    eval_count    INTEGER,                -- tokens generated
    prompt_count  INTEGER,
    peak_rss_mb   REAL,
    error         TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (task_id, tier, repeat_idx)
);

CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task_id);
CREATE INDEX IF NOT EXISTS idx_runs_tier ON runs(tier);

-- Cached feature vectors so routers and the GA agree on inputs.
CREATE TABLE IF NOT EXISTS features (
    task_id   INTEGER PRIMARY KEY REFERENCES tasks(id),
    payload   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    config      TEXT NOT NULL,
    git_sha     TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per (router, split, seed): the headline comparison table.
CREATE TABLE IF NOT EXISTS router_results (
    id             INTEGER PRIMARY KEY,
    experiment_id  INTEGER NOT NULL REFERENCES experiments(id),
    router         TEXT NOT NULL,
    split          TEXT NOT NULL,
    seed           INTEGER NOT NULL DEFAULT 0,
    accuracy       REAL NOT NULL,
    mean_latency_ms  REAL NOT NULL,
    total_latency_ms REAL NOT NULL,
    load_ms        REAL NOT NULL,
    switches       INTEGER NOT NULL,
    peak_rss_mb    REAL,
    objective      REAL NOT NULL,
    routing        TEXT NOT NULL,         -- json: task_id -> tier
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Per-generation GA trace. train vs test is recorded side by side so the
-- overfitting gap is visible rather than inferred.
CREATE TABLE IF NOT EXISTS generations (
    id             INTEGER PRIMARY KEY,
    experiment_id  INTEGER NOT NULL REFERENCES experiments(id),
    seed           INTEGER NOT NULL,
    generation     INTEGER NOT NULL,
    best_train     REAL NOT NULL,
    mean_train     REAL NOT NULL,
    best_test      REAL NOT NULL,
    diversity      REAL,
    best_genome    TEXT NOT NULL,
    evaluations    INTEGER NOT NULL,
    elapsed_s      REAL NOT NULL
);
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created.

    The response matrix is expensive to rebuild, so schema changes extend the
    existing file rather than asking for a fresh one.
    """
    have = {r["name"] for r in conn.execute("PRAGMA table_info(models)")}
    for column, decl in (("load_ms_cold", "REAL"), ("load_samples", "TEXT")):
        if column not in have:
            conn.execute(f"ALTER TABLE models ADD COLUMN {column} {decl}")
    have = {r["name"] for r in conn.execute("PRAGMA table_info(router_results)")}
    for column, decl in (("mode", "TEXT"), ("load_profile", "TEXT"),
                         ("inference_ms", "REAL"), ("total_ms", "REAL")):
        if column not in have:
            conn.execute(f"ALTER TABLE router_results ADD COLUMN {column} {decl}")
    # One experiment now holds several GA runs -- two routers across the 2x2
    # grid of execution mode and load regime -- so a curve has to say which.
    have = {r["name"] for r in conn.execute("PRAGMA table_info(generations)")}
    for column, decl in (("router", "TEXT"), ("mode", "TEXT"),
                         ("load_profile", "TEXT")):
        if column not in have:
            conn.execute(f"ALTER TABLE generations ADD COLUMN {column} {decl}")
    conn.commit()


@contextmanager
def session(path: Path | str = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_models(conn: sqlite3.Connection, specs) -> None:
    conn.executemany(
        "INSERT INTO models (tier, ollama_tag, params_b, quant) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(tier) DO UPDATE SET ollama_tag=excluded.ollama_tag, "
        "params_b=excluded.params_b, quant=excluded.quant",
        [(m.tier, m.ollama_tag, m.params_b, m.quant) for m in specs],
    )


def insert_tasks(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    cur = conn.executemany(
        "INSERT OR IGNORE INTO tasks (dataset, task_type, split, source_idx, prompt, gold, meta) "
        "VALUES (:dataset, :task_type, :split, :source_idx, :prompt, :gold, :meta)",
        rows,
    )
    return cur.rowcount


def record_run(conn: sqlite3.Connection, **kw: Any) -> None:
    cols = ("task_id", "tier", "repeat_idx", "raw_output", "parsed", "score",
            "latency_ms", "eval_count", "prompt_count", "peak_rss_mb", "error")
    conn.execute(
        f"INSERT OR REPLACE INTO runs ({', '.join(cols)}) "
        f"VALUES ({', '.join(':' + c for c in cols)})",
        {c: kw.get(c) for c in cols},
    )


def response_matrix(conn: sqlite3.Connection, split: str | None = None) -> dict:
    """Load the cached (task, tier) matrix the routers are scored against."""
    sql = ("SELECT r.task_id, r.tier, r.score, r.latency_ms, r.peak_rss_mb "
           "FROM runs r JOIN tasks t ON t.id = r.task_id WHERE r.repeat_idx = 0")
    params: tuple = ()
    if split:
        sql += " AND t.split = ?"
        params = (split,)
    out: dict[int, dict[str, dict]] = {}
    for row in conn.execute(sql, params):
        out.setdefault(row["task_id"], {})[row["tier"]] = {
            "score": row["score"],
            "latency_ms": row["latency_ms"],
            "peak_rss_mb": row["peak_rss_mb"],
        }
    return out


def base_runs(conn: sqlite3.Connection, split: str, tier: str) -> dict[int, dict]:
    """The stored output of one model on every task in a split.

    The cascade setting needs these: its features are read off the small
    model's answer, so the answer has to be recoverable from cache rather than
    regenerated -- otherwise the whole point of the matrix is lost.
    """
    rows = conn.execute(
        "SELECT r.task_id, r.raw_output, r.eval_count, r.score "
        "FROM runs r JOIN tasks t ON t.id = r.task_id "
        "WHERE t.split = ? AND r.tier = ? AND r.repeat_idx = 0", (split, tier))
    return {r["task_id"]: dict(r) for r in rows}


def start_experiment(conn: sqlite3.Connection, name: str, config: dict,
                     git_sha: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO experiments (name, config, git_sha) VALUES (?, ?, ?)",
        (name, json.dumps(config, ensure_ascii=False, sort_keys=True), git_sha),
    )
    return int(cur.lastrowid)
