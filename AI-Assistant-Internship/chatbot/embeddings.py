"""Sentence Transformer embeddings compatible with LangChain."""

from __future__ import annotations

from functools import cached_property

import numpy as np
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingFunction(Embeddings):
    """LangChain embedding adapter for all-MiniLM-L6-v2."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name

    @cached_property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model."""

        return SentenceTransformer(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""

        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string."""

        vector = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vector.astype(np.float32).tolist()
