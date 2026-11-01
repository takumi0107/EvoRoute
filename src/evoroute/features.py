"""Task features available to a router at decision time.

Hard constraint: every feature is computed from the *input* only. Anything
derived from the gold answer (entity count, correct option, difficulty label)
would leak the answer key into the router and invalidate the comparison
against the oracle, which is the one component allowed to see it.
"""
from __future__ import annotations

import json
import re
from typing import Any

TASK_TYPES = ("classification", "extraction", "reasoning")

# Ordered so feature vectors are stable across runs and comparable in the DB.
NUMERIC_FEATURES = (
    "input_chars", "input_words", "avg_word_len",
    "upper_ratio", "digit_ratio", "punct_ratio",
    "n_options", "long_word_ratio",
)

FEATURE_NAMES = NUMERIC_FEATURES + tuple(f"is_{t}" for t in TASK_TYPES)


def _body(prompt: str) -> str:
    """Strip the instruction scaffolding so length features describe the
    actual item, not the template."""
    for marker in ("Article: ", "Sentence: ", "Question: "):
        if marker in prompt:
            return prompt.split(marker, 1)[1].rsplit("\n\n", 1)[0]
    return prompt


def extract(task: dict[str, Any]) -> dict[str, float]:
    meta = task.get("meta") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)

    text = _body(task["prompt"])
    words = text.split()
    n_words = len(words) or 1
    n_chars = len(text) or 1

    feats = {
        "input_chars": float(n_chars),
        "input_words": float(n_words),
        "avg_word_len": sum(len(w) for w in words) / n_words,
        "upper_ratio": sum(1 for c in text if c.isupper()) / n_chars,
        "digit_ratio": sum(1 for c in text if c.isdigit()) / n_chars,
        "punct_ratio": len(re.findall(r"[^\w\s]", text)) / n_chars,
        "n_options": float(meta.get("n_options", 0)),
        "long_word_ratio": sum(1 for w in words if len(w) >= 9) / n_words,
    }
    for t in TASK_TYPES:
        feats[f"is_{t}"] = 1.0 if task["task_type"] == t else 0.0
    return feats


def vector(task: dict[str, Any]) -> list[float]:
    f = extract(task)
    return [f[name] for name in FEATURE_NAMES]
