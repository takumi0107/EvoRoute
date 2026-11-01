"""Per-item scoring. Every metric returns a float in [0, 1] so the objective
function can mix task types without rescaling.
"""
from __future__ import annotations

import re
import string
from collections import Counter

AG_NEWS_LABELS = ("world", "sports", "business", "sci/tech")


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)


def parse_label(raw: str, labels: tuple[str, ...]) -> str | None:
    """Pull a class label out of a free-form completion.

    Small models ignore 'answer with one word' far more often than large ones,
    so parsing has to be forgiving -- otherwise we would measure instruction
    following and call it accuracy.
    """
    norm = _normalize(raw)
    for label in labels:
        if _normalize(label) in norm:
            return label
    first = norm.split()
    if first and first[0].isdigit():
        idx = int(first[0]) - 1
        if 0 <= idx < len(labels):
            return labels[idx]
    return None


def classification_score(raw: str, gold: str, labels: tuple[str, ...]) -> float:
    return 1.0 if parse_label(raw, labels) == _normalize_label(gold, labels) else 0.0


def _normalize_label(gold: str, labels: tuple[str, ...]) -> str | None:
    for label in labels:
        if _normalize(label) == _normalize(gold):
            return label
    return None


def parse_choice(raw: str, options: list[str]) -> str | None:
    """Multiple choice: accept 'B', 'B)', '(B)', 'Answer: B', or the text itself."""
    letters = [chr(ord("A") + i) for i in range(len(options))]
    m = re.search(r"\b([A-Z])\b", raw.strip().upper()[:40])
    if m and m.group(1) in letters:
        return m.group(1)
    norm = _normalize(raw)
    for letter, text in zip(letters, options):
        if text and _normalize(text) in norm:
            return letter
    return None


def choice_score(raw: str, gold_letter: str, options: list[str]) -> float:
    return 1.0 if parse_choice(raw, options) == gold_letter.strip().upper() else 0.0


def parse_entities(raw: str) -> list[str]:
    """Entities are requested one per line; tolerate bullets and numbering.

    Models that ignore the line format and answer "A, B, C" on one line are
    split on commas instead. The split is restricted to single-line output so
    a model that *did* use one-per-line is never second-guessed -- otherwise an
    entity that legitimately contains a comma ("Washington, D.C.") would be
    torn in half.
    """
    lines = [l for l in raw.splitlines() if l.strip()]
    if len(lines) == 1 and "," in lines[0]:
        lines = [part for part in lines[0].split(",")]

    out = []
    for line in lines:
        line = line.strip().lstrip("-*0123456789.) \t").strip()
        if not line or len(line) > 80:
            continue
        if line.lower().startswith(("none", "no entities", "here", "the ")):
            continue
        out.append(line)
    return out


def entity_f1(raw: str, gold: str) -> float:
    """Set-level F1 over extracted entity strings.

    NOTE: the proposal specified exact match. Exact match over a whole entity
    set floors a 1.5B model at ~0, which would erase the accuracy differences
    the router is supposed to exploit. F1 keeps the gradation; `entity_em`
    is recorded alongside so the strict number is still reportable.
    """
    pred = Counter(_normalize(e) for e in parse_entities(raw) if _normalize(e))
    truth = Counter(_normalize(e) for e in gold.split("|") if _normalize(e))
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    overlap = sum((pred & truth).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(truth.values())
    return 2 * precision * recall / (precision + recall)


def entity_em(raw: str, gold: str) -> float:
    pred = {_normalize(e) for e in parse_entities(raw) if _normalize(e)}
    truth = {_normalize(e) for e in gold.split("|") if _normalize(e)}
    return 1.0 if pred == truth else 0.0


def macro_f1(pairs: list[tuple[str | None, str | None]], labels: tuple[str, ...]) -> float:
    """Corpus-level macro F1, for the dataset summary table (not per item)."""
    scores = []
    for label in labels:
        tp = sum(1 for p, g in pairs if p == label and g == label)
        fp = sum(1 for p, g in pairs if p == label and g != label)
        fn = sum(1 for p, g in pairs if p != label and g == label)
        if tp == 0:
            scores.append(0.0)
            continue
        precision, recall = tp / (tp + fp), tp / (tp + fn)
        scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0
