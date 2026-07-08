"""AI Assistant modules.

Submodules are imported directly to avoid eager loading of provider SDKs,
embedding models, and vector-store dependencies during lightweight tests.
"""

__all__ = [
    "base_chatbot",
    "embeddings",
    "knowledge_base",
    "medical",
    "memory",
    "multilingual",
    "rag",
    "retriever",
    "sentiment",
    "vision",
]
