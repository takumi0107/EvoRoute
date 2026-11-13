"""Central configuration: model registry, objective weights, paths."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
# Overridable so a snapshot round-trip can be verified against a scratch
# database without touching the measured one.
DB_PATH = Path(os.environ.get("EVOROUTE_DB", RESULTS_DIR / "evoroute.db"))

OLLAMA_HOST = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class ModelSpec:
    """A local model under evaluation.

    `tier` is the symbolic name routing strategies refer to, so the genome
    stays valid even if the concrete ollama tag is swapped out later.
    """

    tier: str
    ollama_tag: str
    params_b: float
    quant: str


# 16GB RAM: these are loaded one at a time, never concurrently.
#
# All three are Qwen2.5 at Q4_K_M so parameter count is the only variable.
# An earlier medium (phi3:3.8b at Q8_0) was dropped after profiling showed it
# strictly dominated by the 7B Q4 on both latency and footprint -- inference
# here is memory-bandwidth bound, so 8-bit weights cost more per token than a
# larger model's 4-bit ones. Holding quantisation fixed removes that
# confound; varying it is left to future work.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("small", "qwen2.5:1.5b-instruct-q4_K_M", 1.5, "Q4_K_M"),
    ModelSpec("medium", "qwen2.5:3b-instruct-q4_K_M", 3.0, "Q4_K_M"),
    ModelSpec("large", "qwen2.5:7b-instruct-q4_K_M", 7.0, "Q4_K_M"),
)

TIERS: tuple[str, ...] = tuple(m.tier for m in MODELS)


@dataclass(frozen=True)
class ObjectiveWeights:
    """J = accuracy - w_time*(inference + load)/time_ref - w_mem*memory/mem_ref

    `time` covers inference and model loading together: they are the same
    physical quantity, and splitting them across two normalisers made J
    insensitive to load cost (see objective.py).
    """

    time: float = 0.40
    memory: float = 0.10


DEFAULT_WEIGHTS = ObjectiveWeights()

# Model load time has two regimes on a 16GB machine, and both occur in
# practice, so results are reported under each rather than one being declared
# correct. Values are filled in by `evoroute profile` and read from the DB;
# these are only the names of the two columns.
LOAD_PROFILES: tuple[str, ...] = ("warm", "cold")


@dataclass(frozen=True)
class GenerationParams:
    """Decoding settings. Greedy + fixed seed so the response matrix is
    reproducible; any remaining variance comes from the runtime, not sampling.
    """

    temperature: float = 0.0
    seed: int = 0
    num_predict: int = 128
    num_ctx: int = 2048


DEFAULT_GEN = GenerationParams()


@dataclass
class DatasetSpec:
    name: str
    task_type: str
    metric: str
    n_train: int = 100
    n_test: int = 100
    hf_id: str = ""
    hf_config: str | None = None
    extra: dict = field(default_factory=dict)


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("ag_news", "classification", "f1_macro", hf_id="fancyzhx/ag_news"),
    DatasetSpec("conll2003", "extraction", "exact_match", hf_id="tomaarsen/conll2003"),
    DatasetSpec("arc_easy", "reasoning", "accuracy",
                hf_id="allenai/ai2_arc", hf_config="ARC-Easy"),
)
