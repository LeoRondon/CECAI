"""Vector search over the pre-computed CEC rule embeddings."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cec import ollama
from cec.config import settings


@dataclass(frozen=True)
class Rule:
    rule: str
    title: str
    text: str


@dataclass(frozen=True)
class Match:
    rule: Rule
    score: float


class RuleIndex:
    """Rules plus a row-normalised embedding matrix for fast cosine search."""

    def __init__(self, rules: Sequence[Rule], embeddings: np.ndarray) -> None:
        if len(rules) != len(embeddings):
            raise ValueError("rules and embeddings must have the same length")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.rules = list(rules)
        self._matrix = (embeddings / norms).astype(np.float32)

    @classmethod
    def load(cls, path: Path | None = None) -> RuleIndex:
        path = path or settings.embeddings_file
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        rules = [Rule(r["rule"], r["title"], r["text"]) for r in records]
        embeddings = np.array([r["embedding"] for r in records], dtype=np.float32)
        return cls(rules, embeddings)

    def __len__(self) -> int:
        return len(self.rules)

    def search_vector(self, vector: Sequence[float], k: int | None = None) -> list[Match]:
        """Return the ``k`` rules most similar to ``vector`` (cosine similarity)."""
        k = min(k or settings.top_k, len(self.rules))
        query = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm:
            query = query / norm
        scores = self._matrix @ query
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [Match(self.rules[i], float(scores[i])) for i in top]

    def search(self, question: str, k: int | None = None) -> list[Match]:
        """Embed ``question`` with Ollama and return the closest rules."""
        return self.search_vector(ollama.embed_one(question), k)
