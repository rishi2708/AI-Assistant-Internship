"""MedQuAD-powered medical chatbot."""

from __future__ import annotations

from chatbot.base_chatbot import GeminiChatbot
from chatbot.medical_entities import MedicalEntityRecognizer
from chatbot.rag import RAGPipeline
from config import MEDICAL_DATA_DIR, MEDICAL_DISCLAIMER, VECTOR_DB_DIR, AppConfig
from utils.document_loader import load_medquad_documents


class MedicalChatbot:
    """Medical RAG assistant backed by MedQuAD."""

    def __init__(self, config: AppConfig, chatbot: GeminiChatbot) -> None:
        self.pipeline = RAGPipeline(config, chatbot, VECTOR_DB_DIR, "medical_medquad")
        self.entity_recognizer = MedicalEntityRecognizer()

    def bootstrap(self) -> int:
        """Load MedQuAD records and build the FAISS index when missing."""

        if self.pipeline.store.load():
            return 0
        documents = load_medquad_documents(MEDICAL_DATA_DIR)
        if not documents:
            return 0
        return self.pipeline.ingest_documents(documents)

    def answer(self, query: str, top_k: int = 4) -> dict[str, object]:
        """Answer a medical question with citations and disclaimer."""

        entities = self.entity_recognizer.extract(query)
        entity_text = self.entity_recognizer.format_entities(entities)
        enriched_query = (
            f"{query}\n\nDetected medical entities: {entity_text}\n"
            "Use the entities to improve retrieval and explain them if relevant."
        )
        result = self.pipeline.answer(enriched_query, mode="Medical", top_k=top_k)
        result["entities"] = [{"text": entity.text, "label": entity.label} for entity in entities]
        result["answer"] = f"{result['answer']}\n\n{MEDICAL_DISCLAIMER}"
        return result
