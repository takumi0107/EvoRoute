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

# Cascade features: properties of the *small model's own answer*, available
# only once it has run. They are not leakage -- nothing here touches the gold
# label -- but they do change the architecture, since the small inference must
# be paid on every task before the router can consult them.
#
# They exist because the input-only features above turned out to carry almost
# no signal about which tasks need a bigger model: even fitted directly on the
# test split, a rule over them captured just 13% of the oracle's headroom.
# Measured on the train split, over-extraction is the clearest tell -- when the
# small model is wrong it emits 2.5x the entities and 3x the characters.
CASCADE_FEATURES = (
    "out_tokens", "out_chars", "out_lines", "out_items",
    "out_in_ratio", "out_has_prose", "out_repeats", "out_degenerate",
)

CASCADE_FEATURE_NAMES = FEATURE_NAMES + CASCADE_FEATURES

_PROSE = re.compile(r"\b(the|is|are|this|which|because|there|would)\b")


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


def cascade_extract(task: dict[str, Any], raw: str,
                    eval_count: int | None = None) -> dict[str, float]:
    """Features describing what the small model replied, not what it was asked."""
    from .evaluation.metrics import parse_entities

    text = raw.strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    items = parse_entities(raw) if task["task_type"] == "extraction" else ([text] if text else [])
    input_chars = max(len(_body(task["prompt"])), 1)
    lowered = text.lower()

    return {
        "out_tokens": float(eval_count or 0),
        "out_chars": float(len(text)),
        "out_lines": float(len(lines)),
        "out_items": float(len(items)),
        "out_in_ratio": len(text) / input_chars,
        "out_has_prose": 1.0 if _PROSE.search(lowered) else 0.0,
        "out_repeats": 1.0 if len(lines) != len({l.lower() for l in lines}) else 0.0,
        "out_degenerate": 1.0 if (not text or lowered.startswith("none")) else 0.0,
    }


def cascade_vector(task: dict[str, Any], raw: str,
                   eval_count: int | None = None) -> list[float]:
    """Input features followed by cascade features, in CASCADE_FEATURE_NAMES order."""
    f = extract(task)
    f.update(cascade_extract(task, raw, eval_count))
    return [f[name] for name in CASCADE_FEATURE_NAMES]
