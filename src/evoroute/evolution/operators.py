"""Variation and selection over decision lists.

The mutation set deliberately touches structure as well as constants: adding
and removing rules, and reordering them, are what make this a search over
routing logic rather than a threshold tuner. Rule order matters because the
list is first-match, so a swap can change many decisions at once -- a coarse
move that lets the population escape a local optimum a threshold nudge cannot.
"""
from __future__ import annotations

import random

from ..config import TIERS
from .genome import (MAX_CONDITIONS, MAX_RULES, Condition, FeatureSpace,
                     Genome, Rule)


def _mutate_tier(genome: Genome, space: FeatureSpace, rng: random.Random) -> Genome:
    if not genome.rules:
        return genome
    i = rng.randrange(len(genome.rules))
    rules = list(genome.rules)
    rules[i] = Rule(rules[i].conditions, rng.choice(TIERS))
    return Genome(tuple(rules), genome.default)


def _mutate_default(genome: Genome, space: FeatureSpace, rng: random.Random) -> Genome:
    return Genome(genome.rules, rng.choice(TIERS))


def _mutate_threshold(genome: Genome, space: FeatureSpace, rng: random.Random) -> Genome:
    if not genome.rules:
        return genome
    i = rng.randrange(len(genome.rules))
    rule = genome.rules[i]
    if not rule.conditions:
        return genome
    j = rng.randrange(len(rule.conditions))
    old = rule.conditions[j]
    values = list(space.candidates[old.feature])
    if len(values) > 1 and old.threshold in values:
        # Step to a neighbouring quantile: a local move, unlike a resample.
        k = values.index(old.threshold)
        k = max(0, min(len(values) - 1, k + rng.choice((-1, 1))))
        new = Condition(old.feature, old.op, float(values[k]))
    else:
        new = space.sample_condition(rng)
    conditions = list(rule.conditions)
    conditions[j] = new
    rules = list(genome.rules)
    rules[i] = Rule(tuple(conditions), rule.tier)
    return Genome(tuple(rules), genome.default)


def _mutate_condition(genome: Genome, space: FeatureSpace, rng: random.Random) -> Genome:
    if not genome.rules:
        return genome
    i = rng.randrange(len(genome.rules))
    rule = genome.rules[i]
    conditions = list(rule.conditions)
    if not conditions:
        return genome
    conditions[rng.randrange(len(conditions))] = space.sample_condition(rng)
    rules = list(genome.rules)
    rules[i] = Rule(tuple(conditions), rule.tier)
    return Genome(tuple(rules), genome.default)


def _add_condition(genome: Genome, space: FeatureSpace, rng: random.Random) -> Genome:
    if not genome.rules:
        return genome
    i = rng.randrange(len(genome.rules))
    rule = genome.rules[i]
    if len(rule.conditions) >= MAX_CONDITIONS:
        return genome
    rules = list(genome.rules)
    rules[i] = Rule(rule.conditions + (space.sample_condition(rng),), rule.tier)
    return Genome(tuple(rules), genome.default)


def _drop_condition(genome: Genome, space: FeatureSpace, rng: random.Random) -> Genome:
    if not genome.rules:
        return genome
    i = rng.randrange(len(genome.rules))
    rule = genome.rules[i]
    if len(rule.conditions) <= 1:
        return genome
    j = rng.randrange(len(rule.conditions))
    conditions = tuple(c for k, c in enumerate(rule.conditions) if k != j)
    rules = list(genome.rules)
    rules[i] = Rule(conditions, rule.tier)
    return Genome(tuple(rules), genome.default)


def _add_rule(genome: Genome, space: FeatureSpace, rng: random.Random) -> Genome:
    if len(genome.rules) >= MAX_RULES:
        return genome
    rule = Rule(tuple(space.sample_condition(rng)
                      for _ in range(rng.randint(1, MAX_CONDITIONS))),
                rng.choice(TIERS))
    rules = list(genome.rules)
    rules.insert(rng.randint(0, len(rules)), rule)
    return Genome(tuple(rules), genome.default)


def _drop_rule(genome: Genome, space: FeatureSpace, rng: random.Random) -> Genome:
    if len(genome.rules) <= 1:
        return genome
    i = rng.randrange(len(genome.rules))
    return Genome(tuple(r for k, r in enumerate(genome.rules) if k != i), genome.default)


def _swap_rules(genome: Genome, space: FeatureSpace, rng: random.Random) -> Genome:
    if len(genome.rules) < 2:
        return genome
    rules = list(genome.rules)
    i, j = rng.sample(range(len(rules)), 2)
    rules[i], rules[j] = rules[j], rules[i]
    return Genome(tuple(rules), genome.default)


MUTATIONS = (
    (_mutate_tier, 3.0),
    (_mutate_default, 1.5),
    (_mutate_threshold, 3.0),
    (_mutate_condition, 2.0),
    (_add_condition, 1.0),
    (_drop_condition, 1.0),
    (_add_rule, 1.5),
    (_drop_rule, 1.5),
    (_swap_rules, 1.5),
)
_OPS = [op for op, _ in MUTATIONS]
_WEIGHTS = [w for _, w in MUTATIONS]


def mutate(genome: Genome, space: FeatureSpace, rng: random.Random,
           n_ops: int = 1) -> Genome:
    for _ in range(n_ops):
        op = rng.choices(_OPS, weights=_WEIGHTS, k=1)[0]
        genome = op(genome, space, rng)
    return genome


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    """One-point splice of the two rule lists, default inherited from a parent.

    Splicing preserves each parent's rule *prefix*, which is the part that
    fires most often in a first-match list, so a child keeps recognisable
    behaviour from both rather than being a fresh random strategy.
    """
    if not a.rules or not b.rules:
        return Genome(a.rules or b.rules, rng.choice((a.default, b.default)))
    i = rng.randint(0, len(a.rules))
    j = rng.randint(0, len(b.rules))
    rules = (a.rules[:i] + b.rules[j:])[:MAX_RULES]
    if not rules:
        rules = a.rules[:1]
    return Genome(tuple(rules), rng.choice((a.default, b.default)))


def tournament(population: list, fitness: list[float], rng: random.Random,
               k: int = 3):
    """Pick the best of k random contestants. Low k keeps selection pressure
    gentle enough that diversity survives the early generations."""
    best = None
    for _ in range(k):
        i = rng.randrange(len(population))
        if best is None or fitness[i] > fitness[best]:
            best = i
    return population[best]
