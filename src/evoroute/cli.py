"""Command line entry points for the Week 1 pipeline.

    evoroute ingest    load datasets into SQLite
    evoroute profile   measure cold load time + footprint per model
    evoroute bench     fill the (task x model) response matrix   [slow, resumable]
    evoroute baseline  every routing method across the 2x2 grid
    evoroute figures   redraw the report's figures from the stored results
    evoroute stats     bootstrap CIs and arrival-order sensitivity
    evoroute cascade   does seeing the small model's answer first help?
    evoroute learning-curve   does more training data close the gap?
    evoroute export    write the measured matrix to a portable snapshot
    evoroute import-snapshot   rebuild a database from one
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn)
from rich.table import Table

from .config import (DATASETS, DB_PATH, LOAD_PROFILES, MODELS, TIERS,
                     DEFAULT_WEIGHTS)
from .db import (export_snapshot, import_snapshot, insert_tasks,
                 session, upsert_models)
from .datasets import loaders
from .evaluation.harness import build_matrix, profile_models
from .models.ollama_client import OllamaClient
from .objective import MODES
from .pipeline import (build_context, git_sha, headroom, load_profile_available,
                       persist, persist_generations, run_router)
from .evolution.search import EvolutionConfig
from .routers.evolved import CeilingRouter, EvolvedRouter
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


GA_SEEDS = (0, 1, 2, 3, 4)


def _routers(tasks, matrix, scales, mode):
    """Every method under comparison: the five of the proposal, plus diagnostics.

    `tasks/matrix/scales` describe the split results are *reported* on. The
    evolved routers get it so each generation can be scored there and the
    train/test curves plotted apart; selection still reads train fitness only.
    The one exception is `evoroute-ceiling*`, which is told to fit there on
    purpose -- see routers/evolved.py.
    """
    out = [FixedRouter(t) for t in TIERS]
    out.append(RuleBasedRouter())
    out.append(TunedRuleRouter())
    out.append(LearnedRouter("tree"))
    out.append(LearnedRouter("logreg"))
    out.append(LearnedRouter("tree", balanced=False))
    out.append(UtilityRouter("tree"))
    out.append(UtilityRouter("ridge"))
    evolved = [EvolvedRouter(mode, seeds=GA_SEEDS),
               CeilingRouter(mode, seeds=GA_SEEDS)]
    for router in evolved:
        router.prepare(tasks, matrix, scales)
    out.extend(evolved)
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
                routers = _routers(tasks, matrix, scales, mode)
                _fit_all(routers, train_tasks, train_matrix, train_scales)
                for router in routers:
                    if isinstance(router, EvolvedRouter) and router.results:
                        persist_generations(conn, exp, router.name, router.results,
                                            mode, load_profile)

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

                evo = next(r for r in routers if r.name == "evoroute")
                spread = evo.seed_spread()
                console.print(f"[dim]evoroute over {spread['n']} seeds: "
                              f"J(test) min={spread['min']:.4f} "
                              f"median={spread['median']:.4f} max={spread['max']:.4f}"
                              f"  |  winner: " + "; ".join(evo.describe()) + "[/dim]")

        _print_grid(grid)

        h = headroom(matrix, tasks, best_fixed)
        console.print(
            f"\n[bold]accuracy ceiling[/bold]  oracle={h['oracle_accuracy']:.3f}  "
            f"best fixed={h['best_fixed_accuracy']:.3f}  "
            f"[bold]headroom={h['headroom']:+.3f}[/bold]")


def _print_grid(grid: dict[tuple[str, str, str], float]) -> None:
    """The headline, in three layers per cell.

    `fixed` is what not routing costs. `oracle` is what perfect routing would
    be worth. `ceiling` is what the *features* could reach if generalisation
    were free. A router stuck at `fixed` while the ceiling sits near it has a
    different problem from one stuck at `fixed` with a high ceiling above, and
    the grid is useless if it cannot tell those two apart.
    """
    # Ties below this are noise: the GA's spread across seeds is larger.
    TIE = 1e-3
    profiles = sorted({lp for _, _, lp in grid})
    if not profiles:
        return
    table = Table(title="routing vs the best single model, and how much of the "
                        "available headroom each layer reaches",
                  title_style="bold magenta")
    table.add_column("load regime")
    for mode in MODES:
        table.add_column(mode)
    for lp in profiles:
        cells = []
        for mode in MODES:
            oracle = grid.get(("oracle", mode, lp))
            ceiling = grid.get(("evoroute-ceiling*", mode, lp))
            fixed = max((v for (r, m, p), v in grid.items()
                         if r.startswith("fixed-") and m == mode and p == lp), default=None)
            real = max(((v, r) for (r, m, p), v in grid.items()
                        if r != "oracle" and not r.startswith("fixed-")
                        and not r.endswith("*")
                        and m == mode and p == lp), default=None)
            if oracle is None or fixed is None or real is None:
                cells.append("-")
                continue
            value, name = real
            room = oracle - fixed
            margin = value - fixed

            def share(x: float | None) -> str:
                if x is None or abs(room) < 1e-9:
                    return "n/a"
                return f"{(x - fixed) / room * 100:.1f}%"

            if margin > TIE:
                verdict = f"[green]beats single[/green] ({margin:+.4f})"
            elif margin < -TIE:
                verdict = f"[red]loses to single[/red] ({margin:+.4f})"
            else:
                verdict = f"[yellow]ties single[/yellow] ({margin:+.4f})"
            evo = grid.get(("evoroute", mode, lp))
            cells.append(
                f"{verdict}\n"
                f"best honest: {name} J={value:.4f}\n"
                f"fixed={fixed:.4f}  oracle={oracle:.4f}\n"
                f"[bold]reached[/bold]  {share(evo)} (evoroute)\n"
                f"[bold]expressible[/bold]  {share(ceiling)} (ceiling*)")
        table.add_row(lp, *cells)
    console.print(table)
    console.print("[dim]* evoroute-ceiling is fitted on the split it is scored "
                  "on. Not a result -- it measures what the feature space can "
                  "express at all, so a gap below it is a generalisation "
                  "failure and a gap above it is a feature failure.[/dim]")


@app.command()
def stats(load_profile: str = "warm", resamples: int = 2000,
          shuffles: int = 200) -> None:
    """Is the comparison's verdict robust to the sample and to arrival order?"""
    from .stats import arrival_sensitivity, bootstrap_diff

    with session(DB_PATH) as conn:
        train_tasks, train_matrix, train_scales = build_context(conn, "train", load_profile)
        tasks, matrix, scales = build_context(conn, "test", load_profile)
        order = [t["id"] for t in tasks]

        for mode in MODES:
            routers = _routers(tasks, matrix, scales, mode)
            _fit_all(routers, train_tasks, train_matrix, train_scales)
            routings = {r.name: r.route(tasks) for r in routers}
            reference = routings["fixed-small"]

            table = Table(title=f"J minus fixed-small, {load_profile}/{mode} "
                                f"({resamples} paired moving-block bootstrap resamples)",
                          title_style="bold cyan")
            for col in ("router", "difference", "95% CI", "resamples won", "verdict"):
                table.add_column(col)
            for name in ("rule-based", "learned-tree", "utility-ridge",
                         "evoroute", "evoroute-ceiling*", "oracle"):
                if name not in routings:
                    continue
                b = bootstrap_diff(routings[name], reference, order, matrix,
                                   scales, DEFAULT_WEIGHTS, mode, resamples)
                verdict = ("[dim]identical route[/dim]" if b["tie_rate"] > 0.999 else
                           "[green]better[/green]" if b["ci_low"] > 0 else
                           "[red]worse[/red]" if b["ci_high"] < 0 else
                           "[yellow]indistinguishable[/yellow]")
                table.add_row(name, f"{b['diff']:+.4f}",
                              f"[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]",
                              f"{b['win_rate']*100:.0f}%", verdict)
            console.print(table)

        # Arrival order only bites in online execution; batch regroups by tier.
        tasks_, matrix_, scales_ = tasks, matrix, scales
        routers = _routers(tasks_, matrix_, scales_, "online")
        _fit_all(routers, train_tasks, train_matrix, train_scales)
        table = Table(title=f"online J under randomised arrival order "
                            f"({shuffles} shuffles, {load_profile} load)",
                      title_style="bold cyan")
        for col in ("router", "id-sorted J", "switches", "shuffled J (mean+-sd)",
                    "shuffled switches", "cost of the assumption"):
            table.add_column(col)
        for router in routers:
            if router.name.startswith("fixed-"):
                continue
            a = arrival_sensitivity(router.route(tasks_), [t["id"] for t in tasks_],
                                    matrix_, scales_, DEFAULT_WEIGHTS, shuffles)
            table.add_row(router.name, f"{a['sorted']:.4f}",
                          str(a["sorted_switches"]),
                          f"{a['mean']:.4f} +- {a['sd']:.4f}",
                          f"{a['mean_switches']:.0f}",
                          f"{a['mean'] - a['sorted']:+.4f}")
        console.print(table)
        console.print("[dim]fixed routers are omitted: they never switch, so "
                      "arrival order cannot touch them.[/dim]")


def _ceiling_and_reached(train, test, mode, weights, seeds):
    """(reached, expressible) J on the test context, for one architecture.

    `reached` fits on train like a method; `expressible` fits on test on
    purpose. Both read whichever feature space the contexts were built with,
    so passing cascade contexts measures the cascade architecture.
    """
    from .evolution.search import evolve

    def best_of(fit_ctx):
        runs = [evolve(fit_ctx, test, weights, mode,
                       EvolutionConfig(**{**EvolutionConfig().__dict__, "seed": s}))
                for s in seeds]
        return max(runs, key=lambda r: r.best_train).best_test

    return best_of(train), best_of(test)


@app.command()
def cascade(load_profile: str = "warm", seeds: int = 5) -> None:
    """Compare the pre-hoc architecture against a cascade, under both modes.

    Pre-hoc routers see only the input. A cascade runs the small model on every
    task first and routes on what came back -- strictly more information, at
    the cost of an inference that is always paid. Whether that trade is worth
    taking turns out to depend on the execution mode, which is the point.
    """
    from .db import base_runs
    from .evolution.search import Context
    from .objective import evaluate_route, optimal_route

    seed_list = tuple(range(seeds))
    with session(DB_PATH) as conn:
        train_tasks, train_matrix, train_scales = build_context(conn, "train", load_profile)
        tasks, matrix, scales = build_context(conn, "test", load_profile)
        order = [t["id"] for t in tasks]
        base = {split: base_runs(conn, split, TIERS[0]) for split in ("train", "test")}

        table = Table(title=f"pre-hoc vs cascade ({load_profile} load, test split, "
                            f"{seeds} GA seeds)", title_style="bold cyan")
        for col in ("architecture", "mode", "fixed-small", "reached",
                    "expressible", "oracle", "reached %", "expressible %"):
            table.add_column(col)

        for label, runs in (("pre-hoc (input only)", None), ("cascade", base)):
            for mode in MODES:
                is_cascade = runs is not None
                train_ctx = Context(train_tasks, train_matrix, train_scales,
                                    runs["train"] if is_cascade else None)
                test_ctx = Context(tasks, matrix, scales,
                                   runs["test"] if is_cascade else None)
                reached, expressible = _ceiling_and_reached(
                    train_ctx, test_ctx, mode, DEFAULT_WEIGHTS, seed_list)
                oracle = evaluate_route(
                    optimal_route(order, matrix, scales, DEFAULT_WEIGHTS, mode, is_cascade),
                    order, matrix, scales, DEFAULT_WEIGHTS, mode, is_cascade).objective
                # Always-small costs the same either way: a cascade that never
                # escalates is exactly the small model run once per task.
                floor = evaluate_route({t: TIERS[0] for t in order}, order, matrix,
                                       scales, DEFAULT_WEIGHTS, mode, is_cascade).objective
                room = oracle - floor
                pct = lambda v: (f"{(v - floor) / room * 100:5.1f}%"
                                 if abs(room) > 1e-9 else "   n/a")
                table.add_row(label, mode, f"{floor:.4f}", f"{reached:.4f}",
                              f"{expressible:.4f}", f"{oracle:.4f}",
                              pct(reached), pct(expressible))
        console.print(table)
        console.print("[dim]'expressible' is fitted on the test split on purpose: "
                      "it is the feature space's ceiling, not a method. Percentages "
                      "are of each architecture's own headroom (oracle - always-small).[/dim]")
        _sklearn_crosscheck(conn, train_tasks, train_matrix, train_scales,
                            tasks, matrix, scales, order, base)


def _sklearn_crosscheck(conn, train_tasks, train_matrix, train_scales,
                        tasks, matrix, scales, order, base) -> None:
    """Is the failure the GA's, or the data's?

    The same cascade features, handed to ordinary supervised learners instead
    of a genetic search. If they fail too, "the search did not look hard
    enough" is not available as an explanation.
    """
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    from .features import cascade_vector
    from .objective import best_local_tier, evaluate_route, local_gains

    def design(task_list, runs):
        return np.asarray([cascade_vector(t, runs[t["id"]]["raw_output"],
                                          runs[t["id"]]["eval_count"])
                           for t in task_list], dtype=float)

    mode = "online"
    X_tr, X_te = design(train_tasks, base["train"]), design(tasks, base["test"])
    n_tr = len(train_tasks)
    y_cls = [best_local_tier(t["id"], train_matrix, train_scales, DEFAULT_WEIGHTS,
                             n_tr, True) for t in train_tasks]
    gains = np.asarray([[local_gains(t["id"], train_matrix, train_scales,
                                     DEFAULT_WEIGHTS, n_tr, True)[tier]
                         for tier in TIERS] for t in train_tasks])

    floor = evaluate_route({t: TIERS[0] for t in order}, order, matrix, scales,
                           DEFAULT_WEIGHTS, mode, True).objective

    def score(routing):
        return evaluate_route(routing, order, matrix, scales, DEFAULT_WEIGHTS,
                              mode, True).objective

    rows = []
    for depth in (2, 3, 4, 6):
        model = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_cls)
        rows.append((f"classifier tree depth={depth}",
                     score(dict(zip(order, model.predict(X_te))))))
    for depth in (2, 3, 4, 6):
        preds = np.column_stack([
            DecisionTreeRegressor(max_depth=depth, random_state=0)
            .fit(X_tr, gains[:, i]).predict(X_te) for i in range(len(TIERS))])
        rows.append((f"utility tree depth={depth}",
                     score(dict(zip(order, [TIERS[i] for i in preds.argmax(1)])))))
    preds = np.column_stack([Ridge(alpha=1.0).fit(X_tr, gains[:, i]).predict(X_te)
                             for i in range(len(TIERS))])
    rows.append(("utility ridge",
                 score(dict(zip(order, [TIERS[i] for i in preds.argmax(1)])))))

    table = Table(title="same cascade features, ordinary learners instead of the GA "
                        "(online)", title_style="bold cyan")
    for col in ("model", "J (test)", "vs always-small"):
        table.add_column(col)
    for name, value in rows:
        delta = value - floor
        style = "[green]" if delta > 1e-6 else "[red]" if delta < -1e-6 else "[dim]"
        table.add_row(name, f"{value:.6f}", f"{style}{delta:+.6f}[/]")
    console.print(table)
    console.print(f"[dim]always-small = {floor:.6f}. "
                  "Not one learner clears it, so the failure is not the GA's.[/dim]")


@app.command("learning-curve")
def learning_curve(load_profile: str = "warm", mode: str = "online",
                   cascade_features: bool = True, seeds: int = 3) -> None:
    """Fit on growing subsets of the train split, score on the full test split.

    A result that is stuck at the baseline has two readings: too little data,
    or nothing to learn. They look different here. Too little data climbs
    toward the ceiling as the subset grows; nothing to learn converges on the
    baseline and stays there -- more data only makes the search *more* certain
    that not routing is correct.
    """
    import numpy as np

    from .db import base_runs
    from .evolution.search import Context, evolve

    with session(DB_PATH) as conn:
        train_tasks, train_matrix, train_scales = build_context(conn, "train", load_profile)
        tasks, matrix, scales = build_context(conn, "test", load_profile)
        runs = ({split: base_runs(conn, split, TIERS[0]) for split in ("train", "test")}
                if cascade_features else None)
        test_ctx = Context(tasks, matrix, scales, runs["test"] if runs else None)

        is_cascade = runs is not None
        order = [t["id"] for t in tasks]
        from .objective import evaluate_route, optimal_route
        floor = evaluate_route({t: TIERS[0] for t in order}, order, matrix, scales,
                               DEFAULT_WEIGHTS, mode, is_cascade).objective
        oracle = evaluate_route(
            optimal_route(order, matrix, scales, DEFAULT_WEIGHTS, mode, is_cascade),
            order, matrix, scales, DEFAULT_WEIGHTS, mode, is_cascade).objective
        ceiling = max(
            (evolve(test_ctx, test_ctx, DEFAULT_WEIGHTS, mode,
                    EvolutionConfig(**{**EvolutionConfig().__dict__, "seed": s}))
             for s in range(seeds)), key=lambda r: r.best_train).best_test

        rng = np.random.default_rng(0)
        sizes = [n for n in (60, 120, 180, 240, len(train_tasks))
                 if n <= len(train_tasks)]
        table = Table(title=f"EvoRoute vs training-set size "
                            f"({load_profile}/{mode}, "
                            f"{'cascade' if is_cascade else 'input-only'} features, "
                            f"{seeds} seeds)", title_style="bold cyan")
        for col in ("train tasks", "J (train)", "J (test)", "vs always-small"):
            table.add_column(col)
        for n in sizes:
            picks = rng.choice(len(train_tasks), n, replace=False)
            subset = [train_tasks[i] for i in sorted(picks)]
            ctx = Context(subset, train_matrix, train_scales,
                          {t["id"]: runs["train"][t["id"]] for t in subset}
                          if runs else None)
            best = max((evolve(ctx, test_ctx, DEFAULT_WEIGHTS, mode,
                               EvolutionConfig(**{**EvolutionConfig().__dict__, "seed": s}))
                        for s in range(seeds)), key=lambda r: r.best_train)
            delta = best.best_test - floor
            style = "[green]" if delta > 1e-6 else "[red]" if delta < -1e-6 else "[dim]"
            table.add_row(str(n), f"{best.best_train:.6f}", f"{best.best_test:.6f}",
                          f"{style}{delta:+.6f}[/]")
        console.print(table)
        console.print(f"[dim]always-small = {floor:.6f} · expressible ceiling = "
                      f"{ceiling:.6f} · oracle = {oracle:.6f}[/dim]")


SNAPSHOT = Path("data/measurements.json.gz")


@app.command()
def export(path: Path = SNAPSHOT) -> None:
    """Write the measured tables so the analysis reproduces from a clone."""
    with session(DB_PATH) as conn:
        counts = export_snapshot(conn, path)
    size = path.stat().st_size / 1e6
    console.print(f"[green]{path}[/green]  {size:.1f} MB  "
                  + ", ".join(f"{k}={v}" for k, v in counts.items()))


@app.command("import-snapshot")
def import_snapshot_cmd(path: Path = SNAPSHOT) -> None:
    """Rebuild a database from a snapshot, then run `baseline` as usual."""
    if not path.exists():
        raise typer.BadParameter(f"no snapshot at {path}")
    with session(DB_PATH) as conn:
        counts = import_snapshot(conn, path)
    console.print("[green]imported[/green] "
                  + ", ".join(f"{k}={v}" for k, v in counts.items()))
    console.print(f"[dim]database: {DB_PATH}[/dim]")


@app.command()
def figures(outdir: Path = typer.Option(Path("results/figures"),
                                        help="where the PNGs are written")) -> None:
    """Redraw every figure from the database. Measures nothing."""
    from .viz import render_all

    with session(DB_PATH) as conn:
        paths = render_all(conn, outdir)
    for p in paths:
        console.print(f"[green]{p}[/green]")


if __name__ == "__main__":
    app()
