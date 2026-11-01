"""Thin Ollama client with the timing breakdown this project needs.

Ollama reports `load_duration` separately from `eval_duration`, which maps
directly onto the two cost terms in the objective: a warm inference cost that
the response matrix caches, and a cold load cost that only the *sequence* of
routing decisions can trigger. Keeping them apart is what lets the switch
penalty be computed analytically in Weeks 3-4 instead of re-measured.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from ..config import OLLAMA_HOST, GenerationParams, DEFAULT_GEN

NS = 1e6  # nanoseconds -> milliseconds


@dataclass
class Completion:
    text: str
    total_ms: float
    load_ms: float
    prompt_eval_ms: float
    eval_ms: float
    prompt_count: int
    eval_count: int
    wall_ms: float

    @property
    def inference_ms(self) -> float:
        """Latency excluding model load -- the number cached in `runs`."""
        return self.prompt_eval_ms + self.eval_ms


class OllamaClient:
    def __init__(self, host: str = OLLAMA_HOST, timeout: float = 300.0) -> None:
        self._client = httpx.Client(base_url=host, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def health(self) -> bool:
        try:
            return self._client.get("/api/tags").status_code == 200
        except httpx.HTTPError:
            return False

    def available(self) -> list[str]:
        r = self._client.get("/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def loaded(self) -> list[dict]:
        """Models currently resident, with their RAM footprint."""
        r = self._client.get("/api/ps")
        r.raise_for_status()
        return r.json().get("models", [])

    def resident_mb(self, tag: str) -> float | None:
        for m in self.loaded():
            if m["name"] == tag or m.get("model") == tag:
                return m.get("size", 0) / 1024 / 1024
        return None

    def generate(self, tag: str, prompt: str, gen: GenerationParams = DEFAULT_GEN,
                 keep_alive: str | int = "10m") -> Completion:
        started = time.perf_counter()
        r = self._client.post("/api/generate", json={
            "model": tag,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {
                "temperature": gen.temperature,
                "seed": gen.seed,
                "num_predict": gen.num_predict,
                "num_ctx": gen.num_ctx,
            },
        })
        wall = (time.perf_counter() - started) * 1000
        r.raise_for_status()
        d = r.json()
        return Completion(
            text=d.get("response", ""),
            total_ms=d.get("total_duration", 0) / NS,
            load_ms=d.get("load_duration", 0) / NS,
            prompt_eval_ms=d.get("prompt_eval_duration", 0) / NS,
            eval_ms=d.get("eval_duration", 0) / NS,
            prompt_count=d.get("prompt_eval_count", 0),
            eval_count=d.get("eval_count", 0),
            wall_ms=wall,
        )

    def unload(self, tag: str) -> None:
        """Evict a model so the next call measures a genuine cold load."""
        self._client.post("/api/generate",
                          json={"model": tag, "prompt": "", "keep_alive": 0})

    def unload_all(self) -> None:
        for m in self.loaded():
            self.unload(m.get("model") or m["name"])
