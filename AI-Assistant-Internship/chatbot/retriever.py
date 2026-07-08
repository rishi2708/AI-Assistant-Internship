"""FAISS vector store management and retrieval helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from chatbot.embeddings import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage a persisted FAISS vector store."""

    def __init__(
        self,
        index_dir: Path,
        embeddings: SentenceTransformerEmbeddingFunction,
    ) -> None:
        self.index_dir = index_dir
        self.embeddings = embeddings
        self.vector_store: FAISS | None = None

    @property
    def exists(self) -> bool:
        """Return whether the FAISS index exists on disk."""

        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    def load(self) -> bool:
        """Load the vector store if present."""

        if not self.exists:
            return False
        self.vector_store = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded FAISS index from %s", self.index_dir)
        return True

    def build(self, documents: list[Document]) -> None:
        """Build and persist a FAISS vector store from documents."""

        if not documents:
            raise ValueError("Cannot build vector store without documents")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.save()

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the existing index, creating one if needed."""

        if not documents:
            return
        if self.vector_store is None and not self.load():
            self.build(documents)
            return
        assert self.vector_store is not None
        self.vector_store.add_documents(documents)
        self.save()

    def save(self) -> None:
        """Persist the vector store to disk."""

        if self.vector_store is None:
            raise ValueError("Vector store has not been initialized")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.index_dir))
        logger.info("Saved FAISS index to %s", self.index_dir)

    def similarity_search(self, query: str, top_k: int = 4) -> list[Document]:
        """Retrieve top-k relevant documents."""

        if self.vector_store is None and not self.load():
            return []
        assert self.vector_store is not None
        return self.vector_store.similarity_search(query, k=top_k)

    def as_retriever(self, top_k: int = 4):
        """Return a LangChain retriever object."""

        if self.vector_store is None and not self.load():
            raise ValueError(f"No vector store found at {self.index_dir}")
        assert self.vector_store is not None
        return self.vector_store.as_retriever(search_kwargs={"k": top_k})
