"""Command line entry points for the Week 1 pipeline.

    evoroute ingest    load datasets into SQLite
    evoroute profile   measure cold load time + footprint per model
    evoroute bench     fill the (task x model) response matrix   [slow, resumable]
    evoroute baseline  fixed models + oracle -> the go/no-go table
"""
from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn)
from rich.table import Table

from .config import (DATASETS, DB_PATH, LOAD_PROFILES, MODELS, TIERS,
                     DEFAULT_WEIGHTS)
from .db import insert_tasks, session, upsert_models
from .datasets import loaders
from .evaluation.harness import build_matrix, profile_models
from .models.ollama_client import OllamaClient
from .objective import MODES
from .pipeline import (build_context, git_sha, headroom, load_profile_available,
                       persist, run_router)
from .routers.fixed import FixedRouter
from .routers.learned import LearnedRouter, UtilityRouter, feature_importances
from .routers.oracle import OracleRouter
from .routers.rule_based import RuleBasedRouter, TunedRuleRouter

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


@app.command()
def ingest() -> None:
    """Download datasets, build prompts, store train/test splits."""
    with session(DB_PATH) as conn:
        upsert_models(conn, MODELS)
        table = Table("dataset", "type", "train", "test", "new rows")
        for spec in DATASETS:
            rows = loaders.load(spec)
            added = insert_tasks(conn, rows)
            tr = sum(1 for r in rows if r["split"] == "train")
            table.add_row(spec.name, spec.task_type, str(tr),
                          str(len(rows) - tr), str(added))
        console.print(table)
        total = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        console.print(f"[green]tasks in db: {total}[/green]")


@app.command()
def refresh(dataset: str = typer.Option(..., help="dataset name to regenerate"),
            dry_run: bool = False) -> None:
    """Rebuild prompts for one dataset and invalidate the runs they produced.

    Prompts live in the `tasks` table, so editing a template in loaders.py does
    not reach rows already ingested. This rewrites them and deletes exactly the
    matrix cells whose prompt changed, leaving every other dataset's measured
    cells untouched -- `bench` then refills only the hole.
    """
    spec = next((d for d in DATASETS if d.name == dataset), None)
    if spec is None:
        raise typer.BadParameter(
            f"unknown dataset {dataset!r}; known: {[d.name for d in DATASETS]}")

    with session(DB_PATH) as conn:
        fresh = {(r["split"], r["source_idx"]): r["prompt"] for r in loaders.load(spec)}
        existing = conn.execute(
            "SELECT id, split, source_idx, prompt FROM tasks WHERE dataset = ?",
            (dataset,)).fetchall()

        changed = [(row["id"], fresh[(row["split"], row["source_idx"])])
                   for row in existing
                   if (row["split"], row["source_idx"]) in fresh
                   and fresh[(row["split"], row["source_idx"])] != row["prompt"]]

        if not changed:
            console.print(f"[green]{dataset}: prompts already current[/green]")
            return

        ids = [cid for cid, _ in changed]
        marks = ",".join("?" * len(ids))
        stale = conn.execute(
            f"SELECT COUNT(*) FROM runs WHERE task_id IN ({marks})", ids).fetchone()[0]

        console.print(f"[yellow]{dataset}: {len(changed)} prompts changed, "
                      f"{stale} matrix cells to re-measure[/yellow]")
        sample = next(p for _, p in changed)
        console.print(f"[dim]new prompt:[/dim]\n{sample[:400]}")

        if dry_run:
            console.print("[dim]dry run -- nothing written[/dim]")
            return

        conn.executemany("UPDATE tasks SET prompt = ? WHERE id = ?",
                         [(p, i) for i, p in changed])
        conn.execute(f"DELETE FROM runs WHERE task_id IN ({marks})", ids)
        conn.execute(f"DELETE FROM features WHERE task_id IN ({marks})", ids)
        console.print(f"[green]done -- run `evoroute bench` to refill "
                      f"{stale} cells[/green]")


def _routers(matrix, scales, mode):
    """Every method under comparison, oracle last."""
    out = [FixedRouter(t) for t in TIERS]
    out.append(RuleBasedRouter())
    out.append(TunedRuleRouter())
    out.append(LearnedRouter("tree"))
    out.append(LearnedRouter("logreg"))
    out.append(LearnedRouter("tree", balanced=False))
    out.append(UtilityRouter("tree"))
    out.append(UtilityRouter("ridge"))
    out.append(OracleRouter(matrix, scales, DEFAULT_WEIGHTS, mode))
    return out


def _fit_all(routers, train_tasks, train_matrix, train_scales):
    """Fit whatever can be fitted, on the train split only."""
    for router in routers:
        fit = getattr(router, "fit", None)
        if fit is not None:
            fit(train_tasks, train_matrix, train_scales, DEFAULT_WEIGHTS)


@app.command()
def profile(repeats: int = 5,
            purge: bool = typer.Option(
                False, help="run `sudo purge` first to measure a true cold load")) -> None:
    """Measure load time (warm and, with --purge, cold) and resident memory."""
    with session(DB_PATH) as conn, OllamaClient() as client:
        if not client.health():
            raise typer.BadParameter("ollama is not reachable; run `ollama serve`")
        upsert_models(conn, MODELS)
        missing = [m.ollama_tag for m in MODELS if m.ollama_tag not in client.available()]
        if missing:
            raise typer.BadParameter(f"models not pulled: {missing}")
        console.print(f"[cyan]profiling {len(MODELS)} models x {repeats} warm loads"
                      f"{' + 1 purged cold load' if purge else ''}[/cyan]")
        rows = profile_models(conn, client, MODELS, repeats, purge=purge)

    table = Table("tier", "warm load (ms)", "cold load (ms)", "resident (MB)", "warm samples")
    for r in rows:
        table.add_row(r["tier"], f"{r['load_ms']:.0f}",
                      f"{r['load_ms_cold']:.0f}" if r["load_ms_cold"] else "[dim]not measured[/dim]",
                      f"{r['resident_mb']:.0f}" if r["resident_mb"] else "-",
                      ", ".join(f"{s:.0f}" for s in r["samples"]))
    console.print(table)
    if not purge:
        console.print("[yellow]cold load not measured -- re-run with --purge "
                      "(needs sudo) to get a first-touch number[/yellow]")


@app.command()
def bench() -> None:
    """Fill the response matrix. Resumable: finished pairs are skipped."""
    with session(DB_PATH) as conn, OllamaClient() as client:
        if not client.health():
            raise typer.BadParameter("ollama is not reachable; run `ollama serve`")
        todo = conn.execute(
            "SELECT COUNT(*) FROM tasks").fetchone()[0] * len(MODELS)
        have = conn.execute("SELECT COUNT(*) FROM runs WHERE repeat_idx = 0").fetchone()[0]
        console.print(f"[cyan]matrix: {have}/{todo} cells filled[/cyan]")
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                      TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                      console=console) as bar:
            handles: dict[str, int] = {}

            def tick(tier: str, done: int, total: int) -> None:
                if tier not in handles:
                    handles[tier] = bar.add_task(f"[bold]{tier}", total=total)
                bar.update(handles[tier], completed=done)

            build_matrix(conn, client, MODELS, progress=tick)
        _matrix_summary(conn)


def _matrix_summary(conn) -> None:
    table = Table("tier", "cells", "errors", "mean score", "mean latency (ms)")
    for tier in TIERS:
        r = conn.execute(
            "SELECT COUNT(*) n, SUM(error IS NOT NULL AND error NOT LIKE '{%') e, "
            "AVG(score) s, AVG(latency_ms) l FROM runs WHERE tier = ?", (tier,)).fetchone()
        table.add_row(tier, str(r["n"]), str(r["e"] or 0),
                      f"{r['s']:.3f}" if r["s"] is not None else "-",
                      f"{r['l']:.0f}" if r["l"] is not None else "-")
    console.print(table)


@app.command()
def baseline(split: str = "test") -> None:
    """Compare every routing method across execution mode and load regime.

    Fittable routers are trained on the train split and scored on `split`;
    both numbers are shown so the generalisation gap is visible rather than
    inferred.
    """
    with session(DB_PATH) as conn:
        exp = conn.execute(
            "INSERT INTO experiments (name, config, git_sha) VALUES (?,?,?)",
            (f"compare-{split}",
             json.dumps({"split": split, "weights": DEFAULT_WEIGHTS.__dict__,
                         "modes": list(MODES), "load_profiles": list(LOAD_PROFILES)}),
             git_sha())).lastrowid

        grid: dict[tuple[str, str, str], float] = {}
        best_fixed = 0.0
        tasks = matrix = None
        described = False

        for load_profile in LOAD_PROFILES:
            if not load_profile_available(conn, load_profile):
                console.print(f"[yellow]skipping load profile '{load_profile}': "
                              f"not measured (run `evoroute profile --purge`)[/yellow]")
                continue
            train_tasks, train_matrix, train_scales = build_context(conn, "train", load_profile)
            tasks, matrix, scales = build_context(conn, split, load_profile)

            for mode in MODES:
                routers = _routers(matrix, scales, mode)
                _fit_all(routers, train_tasks, train_matrix, train_scales)

                table = Table(title=f"{load_profile} load / {mode} execution",
                              title_style="bold cyan")
                for col in ("router", "acc", "infer (s)", "load (s)", "total (s)",
                            "switch", "J (train)", "J (test)", "gap"):
                    table.add_column(col)

                for router in routers:
                    out, routing = run_router(router, tasks, matrix, scales,
                                              DEFAULT_WEIGHTS, mode)
                    persist(conn, exp, router.name, split, 0, out, routing,
                            mode, load_profile)
                    # The oracle reads a matrix directly, so scoring it on the
                    # train split needs one bound to that split's context.
                    train_router = (OracleRouter(train_matrix, train_scales,
                                                 DEFAULT_WEIGHTS, mode)
                                    if isinstance(router, OracleRouter) else router)
                    tr, _ = run_router(train_router, train_tasks, train_matrix,
                                       train_scales, DEFAULT_WEIGHTS, mode)
                    grid[(router.name, mode, load_profile)] = out.objective
                    if router.name.startswith("fixed-"):
                        best_fixed = max(best_fixed, out.accuracy)
                    gap = tr.objective - out.objective
                    style = "[red]" if gap > 0.02 else ""
                    table.add_row(router.name, f"{out.accuracy:.3f}",
                                  f"{out.inference_ms/1000:.1f}", f"{out.load_ms/1000:.1f}",
                                  f"{out.total_ms/1000:.1f}", str(out.switches),
                                  f"{tr.objective:.4f}", f"{out.objective:.4f}",
                                  f"{style}{gap:+.4f}")
                console.print(table)

                if not described:
                    described = True
                    tuned = next(r for r in routers if r.name == "rule-based-tuned")
                    console.print("[dim]tuned rules (train-fitted): "
                                  + "; ".join(tuned.describe()) + "[/dim]")
                    tree = next(r for r in routers if r.name == "learned-tree")
                    top = feature_importances(tree)[:4]
                    if top:
                        console.print("[dim]learned-tree top features: "
                                      + ", ".join(f"{k}={v:.2f}" for k, v in top) + "[/dim]")

        _print_grid(grid)

        h = headroom(matrix, tasks, best_fixed)
        console.print(
            f"\n[bold]accuracy ceiling[/bold]  oracle={h['oracle_accuracy']:.3f}  "
            f"best fixed={h['best_fixed_accuracy']:.3f}  "
            f"[bold]headroom={h['headroom']:+.3f}[/bold]")


def _print_grid(grid: dict[tuple[str, str, str], float]) -> None:
    """Does routing beat the best single model, per cell? This is the headline."""
    profiles = sorted({lp for _, _, lp in grid})
    if not profiles:
        return
    table = Table(title="best achievable router vs best single model "
                        "(and how much of the ceiling it captures)",
                  title_style="bold magenta")
    table.add_column("load regime")
    for mode in MODES:
        table.add_column(mode)
    for lp in profiles:
        cells = []
        for mode in MODES:
            oracle = grid.get(("oracle", mode, lp))
            fixed = max((v for (r, m, p), v in grid.items()
                         if r.startswith("fixed-") and m == mode and p == lp), default=None)
            real = max(((v, r) for (r, m, p), v in grid.items()
                        if r != "oracle" and not r.startswith("fixed-")
                        and m == mode and p == lp), default=None)
            if oracle is None or fixed is None or real is None:
                cells.append("-")
                continue
            value, name = real
            room = oracle - fixed
            captured = (value - fixed) / room if abs(room) > 1e-9 else 0.0
            verdict = "[green]beats single[/green]" if value > fixed else "[red]loses to single[/red]"
            cells.append(f"{verdict}\n{name} J={value:.4f}\n"
                         f"fixed={fixed:.4f} oracle={oracle:.4f}\n"
                         f"captured {captured*100:.0f}% of headroom")
        table.add_row(lp, *cells)
    console.print(table)


if __name__ == "__main__":
    app()
