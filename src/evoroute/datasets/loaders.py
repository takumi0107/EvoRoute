"""Dataset loading and prompt construction.

Each loader emits task rows in a single uniform shape -- prompt, gold, meta --
so downstream code never branches on dataset identity. Sampling is seeded and
train/test are drawn disjointly from a shuffled index so re-running the
pipeline reproduces the exact same item set.
"""
from __future__ import annotations

import json
import random
from typing import Any, Callable

from ..config import DatasetSpec
from ..evaluation.metrics import AG_NEWS_LABELS

LETTERS = "ABCDEFGH"

# Fallback if a mirror drops the ClassLabel metadata.
BIO_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
            "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

CLASSIFY_TMPL = (
    "Classify the news article into exactly one category: "
    "World, Sports, Business, Sci/Tech.\n\n"
    "Article: {text}\n\n"
    "Reply with the category name only."
)

# The trailing "if there are none, reply: none" escape hatch of an earlier
# version made qwen2.5:3b answer "none" on 76% of items -- every one of them
# wrong, since the eval set is filtered to sentences that *do* contain
# entities. It measured prompt fragility, not extraction ability, so it is
# gone. The set guaranteeing at least one entity is stated in the report.
EXTRACT_TMPL = (
    "Extract every named entity (people, organizations, locations) from the "
    "sentence.\n\n"
    "Sentence: {text}\n\n"
    "List the entities, one per line, exactly as written in the sentence. "
    "Output nothing else."
)

REASON_TMPL = (
    "Answer the multiple-choice question.\n\n"
    "Question: {question}\n{options}\n\n"
    "Reply with the letter of the correct answer only."
)


def _split_indices(n: int, n_train: int, n_test: int, seed: int = 20251027):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    need = n_train + n_test
    if n < need:
        raise ValueError(f"dataset has {n} rows, need {need}")
    return idx[:n_train], idx[n_train:need]


def _rows(spec: DatasetSpec, source, build: Callable[[Any, int], dict]) -> list[dict]:
    train_idx, test_idx = _split_indices(len(source), spec.n_train, spec.n_test)
    rows = []
    for split, indices in (("train", train_idx), ("test", test_idx)):
        for i in indices:
            item = build(source[int(i)], int(i))
            if item is None:
                continue
            rows.append({
                "dataset": spec.name,
                "task_type": spec.task_type,
                "split": split,
                "source_idx": int(i),
                "prompt": item["prompt"],
                "gold": item["gold"],
                "meta": json.dumps(item.get("meta", {}), ensure_ascii=False),
            })
    return rows


def load_ag_news(spec: DatasetSpec) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(spec.hf_id, split="test")

    def build(row, _i):
        return {
            "prompt": CLASSIFY_TMPL.format(text=row["text"].strip()),
            "gold": AG_NEWS_LABELS[row["label"]],
            "meta": {"labels": list(AG_NEWS_LABELS),
                     "input_chars": len(row["text"])},
        }

    return _rows(spec, ds, build)


def _entities_from_bio(tokens: list[str], tags: list[int], names: list[str]) -> list[str]:
    spans, current = [], []
    for token, tag in zip(tokens, tags):
        name = names[tag]
        if name.startswith("B-") or (name.startswith("I-") and not current):
            if current:
                spans.append(" ".join(current))
            current = [token]
        elif name.startswith("I-") and current:
            current.append(token)
        else:
            if current:
                spans.append(" ".join(current))
            current = []
    if current:
        spans.append(" ".join(current))
    return spans


def _is_prose(tokens: list[str]) -> bool:
    """Reject CoNLL's table rows (sports standings, financial tables).

    A large fraction of the CoNLL-2003 test split is tabular, e.g.
    "Strasbourg 20 11 1 8 27 27 34". Extraction from those is degenerate --
    every model trivially copies the one capitalised token -- so they add
    no signal about which model is worth routing to.
    """
    if len(tokens) < 8:
        return False
    wordlike = sum(1 for t in tokens if any(c.isalpha() for c in t))
    return wordlike / len(tokens) >= 0.6


def load_conll(spec: DatasetSpec) -> list[dict]:
    from datasets import load_dataset

    # tomaarsen/conll2003 is the parquet mirror: the original `conll2003`
    # repos are loading scripts, which `datasets` >=4 refuses to execute.
    ds = load_dataset(spec.hf_id, split="test")
    names = getattr(ds.features["ner_tags"].feature, "names", None) or BIO_TAGS

    def build(row, _i):
        tokens = row["tokens"]
        if not _is_prose(tokens):
            return None
        entities = _entities_from_bio(tokens, row["ner_tags"], names)
        if not entities:
            return None
        text = " ".join(tokens)
        return {
            "prompt": EXTRACT_TMPL.format(text=text),
            "gold": "|".join(entities),
            "meta": {"n_entities": len(entities), "input_chars": len(text)},
        }

    # build() filters rows, so pre-filter before splitting to keep counts exact.
    keep = [r for r in ds if _is_prose(r["tokens"])
            and _entities_from_bio(r["tokens"], r["ner_tags"], names)]
    return _rows(spec, keep, build)


def load_arc_easy(spec: DatasetSpec) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(spec.hf_id, spec.hf_config, split="test")

    def build(row, _i):
        texts = row["choices"]["text"]
        labels = row["choices"]["label"]
        if len(texts) < 2 or row["answerKey"] not in labels:
            return None
        # ARC labels are sometimes 1-4 instead of A-D; normalize to letters.
        gold = LETTERS[labels.index(row["answerKey"])]
        options = "\n".join(f"{LETTERS[i]}) {t}" for i, t in enumerate(texts))
        return {
            "prompt": REASON_TMPL.format(question=row["question"].strip(),
                                         options=options),
            "gold": gold,
            "meta": {"options": texts, "n_options": len(texts),
                     "input_chars": len(row["question"])},
        }

    keep = [r for r in ds if len(r["choices"]["text"]) >= 2
            and r["answerKey"] in r["choices"]["label"]]
    return _rows(spec, keep, build)


LOADERS: dict[str, Callable[[DatasetSpec], list[dict]]] = {
    "ag_news": load_ag_news,
    "conll2003": load_conll,
    "arc_easy": load_arc_easy,
}


def load(spec: DatasetSpec) -> list[dict]:
    return LOADERS[spec.name](spec)
