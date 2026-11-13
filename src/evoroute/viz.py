"""Figures for the report.

Every figure reads the database and re-measures nothing: the response matrix
and the generation curves are already on disk, so a full redraw costs a second
and the plots cannot drift away from the numbers in the tables.

Palette: validated categorical slots 1-3 (blue / orange / aqua), light surface.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from .config import DEFAULT_WEIGHTS, LOAD_PROFILES, DB_PATH
from .objective import MODES
from .pipeline import build_context

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
FAINT = "#b9b8b2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# The grid is always read regime-major so the two rows are the two load
# regimes and the two columns the two execution modes.
CELLS = [(lp, mode) for lp in LOAD_PROFILES for mode in MODES]


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": FAINT, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#e8e7e2", "grid.linewidth": 0.8,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
        "lines.linewidth": 2.0, "figure.dpi": 140,
    })


# -- data access ----------------------------------------------------------
def latest_experiment(conn: sqlite3.Connection, name: str = "compare-test") -> int:
    row = conn.execute(
        "SELECT id FROM experiments WHERE name = ? ORDER BY id DESC LIMIT 1",
        (name,)).fetchone()
    if row is None:
        raise RuntimeError(f"no experiment named {name!r}; run `evoroute baseline`")
    return row["id"]


def results(conn: sqlite3.Connection, exp: int) -> dict[tuple[str, str, str], dict]:
    """(router, mode, regime) -> the stored outcome."""
    rows = conn.execute(
        "SELECT * FROM router_results WHERE experiment_id = ?", (exp,))
    return {(r["router"], r["mode"], r["load_profile"]): dict(r) for r in rows}


def curves(conn: sqlite3.Connection, exp: int, router: str, mode: str,
           load_profile: str) -> dict[str, np.ndarray]:
    """Per-generation train/test J, stacked seed-wise.

    Returns arrays of shape (seeds, generations) so the spread can be drawn as
    a band rather than collapsed into a mean that hides it.
    """
    rows = conn.execute(
        "SELECT seed, generation, best_train, best_test FROM generations "
        "WHERE experiment_id = ? AND router = ? AND mode = ? AND load_profile = ? "
        "ORDER BY seed, generation", (exp, router, mode, load_profile)).fetchall()
    if not rows:
        return {}
    seeds = sorted({r["seed"] for r in rows})
    gens = sorted({r["generation"] for r in rows})
    index = {(r["seed"], r["generation"]): r for r in rows}
    train = np.array([[index[(s, g)]["best_train"] for g in gens] for s in seeds])
    test = np.array([[index[(s, g)]["best_test"] for g in gens] for s in seeds])
    return {"generation": np.array(gens), "train": train, "test": test}


# -- figure 1: the search converges on the baseline -----------------------
def fig_generations(conn: sqlite3.Connection, exp: int, out: Path) -> Path:
    import matplotlib.pyplot as plt

    _style()
    res = results(conn, exp)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True)

    for ax, (lp, mode) in zip(axes.flat, CELLS):
        data = curves(conn, exp, "evoroute", mode, lp)
        if not data:
            ax.set_visible(False)
            continue
        g = data["generation"]
        for key, color, label in (("train", BLUE, "J (train) — what is selected on"),
                                  ("test", ORANGE, "J (test) — what is reported")):
            band = data[key]
            ax.fill_between(g, band.min(0), band.max(0), color=color, alpha=0.15,
                            linewidth=0)
            ax.plot(g, np.median(band, 0), color=color, label=label)

        base = res.get(("fixed-small", mode, lp))
        ceil = res.get(("evoroute-ceiling*", mode, lp))
        span = float(np.ptp(np.concatenate([data["train"], data["test"]])))
        # Where the ceiling sits on top of the baseline -- which is the whole
        # finding in the online cells -- two labels would overprint each other.
        merged = (base is not None and ceil is not None
                  and abs(ceil["objective"] - base["objective"]) < 0.04 * span)
        lines = ([(base, (0, (4, 3)), "fixed-small = expressible ceiling", "bottom")]
                 if merged else
                 [(base, (0, (4, 3)), "fixed-small", "top"),
                  (ceil, (0, (1, 2)), "expressible ceiling", "bottom")])
        for row, style, text, va in lines:
            if row is None:
                continue
            ax.axhline(row["objective"], color=MUTED, linestyle=style, linewidth=1.2)
            ax.annotate(text, xy=(g[-1], row["objective"]),
                        xytext=(-2, 3 if va == "bottom" else -4),
                        textcoords="offset points", ha="right", va=va,
                        fontsize=7.5, color=MUTED)

        ax.set_title(f"{lp} load  ·  {mode} execution", color=INK, loc="left")
        ax.grid(axis="y")
        ax.set_axisbelow(True)

    for ax in axes[1]:
        ax.set_xlabel("generation")
    for ax in axes[:, 0]:
        ax.set_ylabel("objective J")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("The search climbs on train and stalls at the baseline on test",
                 x=0.055, ha="left", fontsize=12, color=INK)
    fig.text(0.055, 0.925, "median over 5 seeds; band spans min–max",
             ha="left", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# -- figure 2: three layers of headroom ------------------------------------
def fig_headroom(conn: sqlite3.Connection, exp: int, out: Path) -> Path:
    """How far along the road from "no routing" to "perfect routing" each cell got.

    Two markers per row joined by a bar: what EvoRoute reached, and what a
    decision list over these features could have reached with generalisation
    removed. The distance between them is the generalisation failure; the
    distance from the right marker to 100% is the feature failure.
    """
    import matplotlib.pyplot as plt

    _style()
    res = results(conn, exp)
    rows = []
    for lp, mode in CELLS:
        fixed = max((v["objective"] for (r, m, p), v in res.items()
                     if r.startswith("fixed-") and m == mode and p == lp), default=None)
        oracle = res.get(("oracle", mode, lp))
        evo = res.get(("evoroute", mode, lp))
        ceil = res.get(("evoroute-ceiling*", mode, lp))
        if None in (fixed, oracle, evo, ceil):
            continue
        room = oracle["objective"] - fixed
        rows.append({
            "label": f"{lp} / {mode}",
            "reached": (evo["objective"] - fixed) / room * 100,
            "expressible": (ceil["objective"] - fixed) / room * 100,
            "room": room,
        })

    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    y = np.arange(len(rows))[::-1]
    ax.hlines(y, 0, 100, color="#ecebe6", linewidth=14, zorder=0)
    for yi, r in zip(y, rows):
        lo, hi = sorted((r["reached"], r["expressible"]))
        ax.hlines(yi, lo, hi, color=AQUA, linewidth=14, zorder=1, alpha=0.35)
    ax.scatter([r["expressible"] for r in rows], y, s=100, color=AQUA, zorder=3,
               edgecolor=SURFACE, linewidth=2, label="expressible by the features")
    ax.scatter([r["reached"] for r in rows], y, s=100, color=BLUE, zorder=4,
               edgecolor=SURFACE, linewidth=2, label="reached by EvoRoute")

    # Percentages live in their own column on the right: on the track they
    # would collide every time the two markers land close together, which in
    # this experiment is most of the grid.
    for yi, r in zip(y, rows):
        note = "  (headroom negligible)" if r["room"] < 0.005 else ""
        ax.annotate(f"{r['reached']:.0f}%  →  {r['expressible']:.0f}%{note}",
                    (104, yi), xytext=(0, -3), textcoords="offset points",
                    fontsize=8.5, color=MUTED, annotation_clip=False)

    ax.set_yticks(y, [f"{r['label']}\nheadroom {r['room']:+.3f} J" for r in rows],
                  color=INK)
    ax.set_xlim(-3, 103)
    ax.set_xticks([0, 25, 50, 75, 100],
                  ["0%\nbest single model", "25%", "50%", "75%", "100%\noracle"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), ncols=2)
    ax.text(0, 1.20, "Where routing's headroom goes: expressible, then reached",
            transform=ax.transAxes, fontsize=12, color=INK)
    ax.text(0, 1.09, "share of the gap between the best single model and a "
            "perfect router, test split",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# -- figure 3: where the objective actually goes ---------------------------
def fig_anatomy(conn: sqlite3.Connection, exp: int, out: Path,
                load_profile: str = "warm") -> Path:
    """Accuracy minus what it costs to get it.

    The bar starts at the router's accuracy and is eaten from the right by the
    time and memory penalties; what survives is J. It is the whole argument of
    the project in one picture: the most accurate router keeps the least.
    """
    import matplotlib.pyplot as plt

    _style()
    res = results(conn, exp)
    _, _, scales = build_context(conn, "test", load_profile)

    order = ["fixed-small", "fixed-medium", "fixed-large", "rule-based",
             "learned-tree", "learned-logreg", "utility-ridge", "evoroute", "oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)

    for ax, mode in zip(axes, MODES):
        names, acc, time_pen, mem_pen, obj = [], [], [], [], []
        for name in order:
            row = res.get((name, mode, load_profile))
            if row is None:
                continue
            t = DEFAULT_WEIGHTS.time * (row["total_ms"] / scales.time_ref)
            names.append(name)
            acc.append(row["accuracy"])
            time_pen.append(t)
            mem_pen.append(row["accuracy"] - row["objective"] - t)
            obj.append(row["objective"])

        y = np.arange(len(names))[::-1]
        obj_a, mem_a, time_a = np.array(obj), np.array(mem_pen), np.array(time_pen)
        bars = (("J kept", obj_a, np.zeros_like(obj_a), BLUE),
                ("memory cost", mem_a, obj_a, AQUA),
                ("time cost", time_a, obj_a + mem_a, ORANGE))
        for label, width, left, color in bars:
            ax.barh(y, width, left=left, height=0.62, color=color, label=label,
                    edgecolor=SURFACE, linewidth=1.5)

        for yi, a, j in zip(y, acc, obj):
            ax.annotate(f"acc {a:.3f}", (a, yi), xytext=(5, -3),
                        textcoords="offset points", fontsize=7.5, color=MUTED)
            ax.annotate(f"{j:.3f}", (j / 2, yi), xytext=(0, -3),
                        textcoords="offset points", ha="center", fontsize=7.5,
                        color=SURFACE, fontweight="bold")

        ax.set_yticks(y, names, color=INK)
        ax.set_xlim(0, 1.06)
        ax.set_xlabel("accuracy, decomposed")
        ax.set_title(f"{mode} execution", loc="left", color=INK)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Accuracy is not the product: what each router keeps after costs",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.text(0.02, 0.925, f"{load_profile} load regime, test split", ha="left",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_all(conn: sqlite3.Connection, outdir: Path,
               name: str = "compare-test") -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    exp = latest_experiment(conn, name)
    return [
        fig_anatomy(conn, exp, outdir / "01-objective-anatomy.png"),
        fig_headroom(conn, exp, outdir / "02-headroom-layers.png"),
        fig_generations(conn, exp, outdir / "03-generation-curves.png"),
    ]
