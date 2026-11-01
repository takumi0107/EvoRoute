"""Dispatch a raw completion to the right metric for its task type."""
from __future__ import annotations

import json
from typing import Any

from .metrics import (AG_NEWS_LABELS, choice_score, classification_score,
                      entity_em, entity_f1, parse_choice, parse_entities,
                      parse_label)


def score_output(task: dict[str, Any], raw: str) -> tuple[float, str, dict]:
    """Return (score in [0,1], parsed answer, extra metrics)."""
    meta = json.loads(task["meta"]) if isinstance(task["meta"], str) else task["meta"]
    kind = task["task_type"]
    gold = task["gold"]

    if kind == "classification":
        labels = tuple(meta.get("labels", AG_NEWS_LABELS))
        parsed = parse_label(raw, labels)
        return classification_score(raw, gold, labels), parsed or "", {}

    if kind == "reasoning":
        options = meta.get("options", [])
        parsed = parse_choice(raw, options)
        return choice_score(raw, gold, options), parsed or "", {}

    if kind == "extraction":
        parsed = parse_entities(raw)
        return (entity_f1(raw, gold), "|".join(parsed),
                {"exact_match": entity_em(raw, gold)})

    raise ValueError(f"unknown task_type: {kind}")
