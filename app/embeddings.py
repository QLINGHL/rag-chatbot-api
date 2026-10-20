"""Embedding providers.

Two interchangeable implementations behind one small interface:

* ``OpenAIEmbedder``  - calls any OpenAI-compatible /embeddings endpoint.
* ``HashingEmbedder`` - pure Python/NumPy bag-of-words hashing embedder.
  Deterministic and fully offline, so the whole demo runs with no API keys.
  Retrieval quality is keyword-based (think TF-IDF), which is plenty for
  document Q&A demos and tests.
"""

import hashlib
import math
import re
from typing import List, Sequence

import httpx
import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Local fallback embedder: hashed bag-of-words with log term frequency."""

    name = "local-hashing"

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> List[np.ndarray]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        counts: dict = {}
        for token in _TOKEN_RE.findall(text.lower()):
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[digest % self._dim] += 1.0 + math.log(count)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec


class OpenAIEmbedder:
    """Embeddings via an OpenAI-compatible HTTP API (OpenAI, Azure, local
    gateways such as Ollama/LM Studio, etc.)."""

    name = "openai-compatible"

    def __init__(self, api_key: str, base_url: str, model: str):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed(["dimension probe"])[0].shape[0])
        return self._dim

    def embed(self, texts: Sequence[str]) -> List[np.ndarray]:
        resp = httpx.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": list(texts)},
            timeout=60.0,
        )
        resp.raise_for_status()
        items = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [np.asarray(item["embedding"], dtype=np.float32) for item in items]


def get_embedder(settings) -> object:
    """Pick the embedder based on configuration."""
    if settings.openai_api_key:
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.embedding_model,
        )
    return HashingEmbedder()
