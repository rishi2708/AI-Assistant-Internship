"""Run a one-shot dynamic knowledge-base update from configured sources."""

from __future__ import annotations

import json

from chatbot.base_chatbot import GeminiChatbot
from chatbot.knowledge_base import DynamicKnowledgeBase
from chatbot.knowledge_updater import KnowledgeBaseUpdater
from chatbot.memory import ConversationMemory
from config import AppConfig, BASE_DIR, VECTOR_DB_DIR, ensure_directories


def main() -> None:
    """Update the dynamic knowledge base from `knowledge_sources.json`."""

    ensure_directories()
    config = AppConfig.from_env()
    memory = ConversationMemory(VECTOR_DB_DIR / "conversation_memory.json")
    chatbot = GeminiChatbot(config, memory)
    knowledge_base = DynamicKnowledgeBase(config, chatbot)
    updater = KnowledgeBaseUpdater(
        knowledge_base=knowledge_base,
        sources_file=BASE_DIR / "knowledge_sources.json",
        state_file=VECTOR_DB_DIR / "knowledge_update_state.json",
    )
    result = updater.update()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
