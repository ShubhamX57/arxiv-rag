"""Embedder: wraps a sentence-transformers model with auto-device selection.

Default model is BAAI/bge-m3, top of MTEB for its size. On Apple Silicon it
runs on the MPS device — meaningfully faster than CPU, no CUDA needed.

Embeddings are L2-normalized, so cosine similarity = dot product. This
matters for LanceDB which can use either; sticking to one keeps the math
unambiguous downstream.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import cached_property

import numpy as np

from arxiv_rag.config import settings

log = logging.getLogger(__name__)


def _resolve_device(requested: str) -> str:
    """Pick a working device. 'mps' falls back to 'cpu' if torch isn't built with MPS."""
    if requested == "auto":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    if requested == "mps":
        try:
            import torch

            if not torch.backends.mps.is_available():
                log.warning("MPS requested but unavailable; falling back to CPU.")
                return "cpu"
        except ImportError:
            return "cpu"
    return requested


class Embedder:
    """Stateful wrapper. Loads the model lazily on first .embed() call.

    The lazy load matters because model load is ~5s and we don't want it
    happening at import time in tests. cached_property ensures one load.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = _resolve_device(device or settings.embedding_device)
        self.batch_size = batch_size or settings.embedding_batch_size
        self.normalize = normalize

    @cached_property
    def _model(self) -> object:
        """Load the model once on first use."""
        log.info(
            "Loading embedding model %s on %s (this can take a moment)...",
            self.model_name,
            self.device,
        )
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name, device=self.device)

    @cached_property
    def dim(self) -> int:
        """Embedding dimensionality. Triggers model load."""
        return int(self._model.get_sentence_embedding_dimension())  # type: ignore[attr-defined]

    def embed(
        self,
        texts: Iterable[str],
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """Embed a batch of texts. Returns shape (N, dim) float32 array."""
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(  # type: ignore[attr-defined]
            text_list,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


__all__ = ["Embedder"]
