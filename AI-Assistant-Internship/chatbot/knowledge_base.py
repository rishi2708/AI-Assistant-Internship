"""Dynamic knowledge base for user-uploaded documents."""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_core.documents import Document

from chatbot.base_chatbot import GeminiChatbot
from chatbot.rag import RAGPipeline
from config import UPLOAD_DIR, VECTOR_DB_DIR, AppConfig
from utils.document_loader import load_uploaded_document


class DynamicKnowledgeBase:
    """Upload, extract, chunk, embed, persist, and reload custom documents."""

    def __init__(self, config: AppConfig, chatbot: GeminiChatbot) -> None:
        self.pipeline = RAGPipeline(config, chatbot, VECTOR_DB_DIR, "dynamic_knowledge_base")
        self.pipeline.store.load()

    def add_file(self, source_path: Path) -> int:
        """Copy a file into uploads, parse it, and add chunks to FAISS."""

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        destination = UPLOAD_DIR / source_path.name
        if source_path.resolve() != destination.resolve():
            shutil.copy2(source_path, destination)
        document = load_uploaded_document(destination)
        return self.add_documents([document])

    def add_documents(self, documents: list[Document]) -> int:
        """Add LangChain documents to the dynamic index."""

        return self.pipeline.ingest_documents(documents)

    def query(self, query: str, top_k: int = 4) -> dict[str, object]:
        """Query the dynamic knowledge base."""

        return self.pipeline.answer(query, mode="Knowledge Base", top_k=top_k)
