"""Complete Retrieval Augmented Generation pipeline."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from chatbot.base_chatbot import GeminiChatbot
from chatbot.embeddings import SentenceTransformerEmbeddingFunction
from chatbot.retriever import VectorStoreManager
from config import AppConfig
from utils.text_processing import ChunkingConfig, chunk_documents


class RAGPipeline:
    """Document Loader -> Chunking -> Embeddings -> FAISS -> Retriever -> Gemini."""

    def __init__(
        self,
        config: AppConfig,
        chatbot: GeminiChatbot,
        index_dir: Path,
        index_name: str,
    ) -> None:
        self.config = config
        self.chatbot = chatbot
        self.index_dir = index_dir / index_name
        self.embeddings = SentenceTransformerEmbeddingFunction(config.embedding_model)
        self.store = VectorStoreManager(self.index_dir, self.embeddings)

    def ingest_documents(self, documents: list[Document]) -> int:
        """Chunk, embed, and persist documents in FAISS."""

        chunks = chunk_documents(
            documents,
            ChunkingConfig(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            ),
        )
        self.store.add_documents(chunks)
        return len(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Retrieve relevant chunks."""

        return self.store.similarity_search(query, top_k or self.config.top_k)

    def answer(self, query: str, mode: str, top_k: int | None = None) -> dict[str, object]:
        """Generate a grounded answer with citations."""

        documents = self.retrieve(query, top_k)
        context = self._format_context(documents)
        prompt = (
            "Use the retrieved context to answer. If the answer is not in the "
            "context, say what is missing and provide a cautious general answer.\n\n"
            f"Question:\n{query}\n\nRetrieved context:\n{context}\n\n"
            "Answer with concise reasoning and a References section."
        )
        answer = self.chatbot.generate_response(prompt, mode=mode)
        return {
            "answer": answer,
            "documents": documents,
            "references": self._references(documents),
        }

    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        """Format retrieved documents for Gemini."""

        blocks: list[str] = []
        for index, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "unknown")
            title = doc.metadata.get("title") or doc.metadata.get("question") or ""
            blocks.append(f"[{index}] Source: {source} {title}\n{doc.page_content}")
        return "\n\n".join(blocks) or "No context retrieved."

    @staticmethod
    def _references(documents: list[Document]) -> list[dict[str, str]]:
        """Build citation metadata."""

        references: list[dict[str, str]] = []
        for index, doc in enumerate(documents, start=1):
            metadata = {key: str(value) for key, value in doc.metadata.items()}
            metadata["citation_id"] = str(index)
            references.append(metadata)
        return references
