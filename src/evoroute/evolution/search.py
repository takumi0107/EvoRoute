"""The genetic algorithm.

Fitness is J on the *train* split and nothing else. The same genome is also
scored on test every generation, but that number is recorded and never
selected on -- it exists so the generalisation gap is a measurement rather
than an assumption. A 300-task train split is small enough that a structure
search will find and exploit its quirks, and the only honest way to report
that is to watch the two curves separate.

Every evaluation reads the cached response matrix, so a full run costs
seconds rather than the hundred hours a naive implementation would need.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import DEFAULT_WEIGHTS, ObjectiveWeights
from ..features import vector
from ..objective import Mode, Scales, evaluate_route
from .genome import FeatureSpace, Genome, assign, random_genome, routing_of
from .operators import crossover, mutate, tournament


@dataclass
class Context:
    """One split, pre-vectorised so fitness evaluation is pure numpy."""

    tasks: list
    matrix: dict
    scales: Scales
    X: np.ndarray = field(init=False)
    order: list = field(init=False)

    def __post_init__(self) -> None:
        self.X = np.asarray([vector(t) for t in self.tasks], dtype=float)
        self.order = [t["id"] for t in self.tasks]


@dataclass
class EvolutionConfig:
    population: int = 60
    generations: int = 60
    elite: int = 2
    tournament_k: int = 3
    crossover_rate: float = 0.6
    mutation_ops: int = 1
    # Breaks ties toward simpler strategies without distorting real differences.
    parsimony: float = 1e-5
    # Fraction of the train split scored each generation, resampled every time.
    # A genome that only wins by exploiting a handful of specific tasks stops
    # winning once those tasks are often absent, which is the only lever here
    # that actually resists overfitting: unlike a fitted model, a genome has
    # no held-out set of its own -- whatever it is selected on, it overfits.
    subsample: float = 1.0
    seed: int = 0


@dataclass
class EvolutionResult:
    best: Genome
    best_train: float
    best_test: float
    history: list[dict]
    evaluations: int
    elapsed_s: float


def _score(genome: Genome, ctx: Context, weights: ObjectiveWeights,
           mode: Mode, idx: np.ndarray | None = None) -> float:
    if idx is None:
        X, order = ctx.X, ctx.order
    else:
        X, order = ctx.X[idx], [ctx.order[i] for i in idx]
    routing = routing_of(genome, X, order)
    return evaluate_route(routing, order, ctx.matrix, ctx.scales,
                          weights, mode).objective


def _behavioural_diversity(population: list[Genome], X: np.ndarray) -> float:
    """Share of the population with a distinct decision pattern.

    Genotypic diversity overstates variety: many different rule lists route
    identically. Counting distinct assignments measures what actually differs.
    """
    seen = {assign(g, X).tobytes() for g in population}
    return len(seen) / len(population)


def evolve(train: Context, test: Context,
           weights: ObjectiveWeights = DEFAULT_WEIGHTS,
           mode: Mode = "online",
           config: EvolutionConfig = EvolutionConfig(),
           on_generation=None) -> EvolutionResult:
    rng = random.Random(config.seed)
    space = FeatureSpace(train.X)
    started = time.perf_counter()

    population = [random_genome(space, rng) for _ in range(config.population)]
    evaluations = 0
    history: list[dict] = []
    best_genome, best_train, best_test = None, -float("inf"), -float("inf")

    n_train = len(train.order)
    take = max(10, int(round(config.subsample * n_train)))
    sampler = np.random.default_rng(config.seed)

    for generation in range(config.generations):
        # Selection sees a fresh subsample; the reported curves always use the
        # full splits, so the history stays comparable across configurations.
        idx = (None if take >= n_train
               else np.sort(sampler.choice(n_train, size=take, replace=False)))
        raw = [_score(g, train, weights, mode, idx) for g in population]
        fitness = [r - config.parsimony * g.size for r, g in zip(raw, population)]
        evaluations += len(population)

        ranked = sorted(range(len(population)), key=lambda i: -fitness[i])
        champion = population[ranked[0]]
        champion_train = _score(champion, train, weights, mode)
        champion_test = _score(champion, test, weights, mode)

        if champion_train > best_train:
            best_genome, best_train, best_test = champion, champion_train, champion_test

        history.append({
            "generation": generation,
            "best_train": champion_train,
            "mean_train": float(np.mean(raw)),
            "best_test": champion_test,
            "diversity": _behavioural_diversity(population, train.X),
            "best_genome": champion,
            "evaluations": evaluations,
            "elapsed_s": time.perf_counter() - started,
        })
        if on_generation:
            on_generation(history[-1])

        if generation == config.generations - 1:
            break

        # Elitism: carry the top genomes forward untouched so the best result
        # can never regress between generations.
        nxt = [population[i] for i in ranked[:config.elite]]
        while len(nxt) < config.population:
            parent = tournament(population, fitness, rng, config.tournament_k)
            if rng.random() < config.crossover_rate:
                other = tournament(population, fitness, rng, config.tournament_k)
                child = crossover(parent, other, rng)
            else:
                child = parent
            nxt.append(mutate(child, space, rng, config.mutation_ops))
        population = nxt

    return EvolutionResult(best=best_genome, best_train=best_train,
                           best_test=best_test, history=history,
                           evaluations=evaluations,
                           elapsed_s=time.perf_counter() - started)
