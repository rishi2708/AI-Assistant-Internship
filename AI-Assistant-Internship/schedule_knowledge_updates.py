"""Run scheduled incremental knowledge-base expansion.

Examples:
    One run for CI or manual verification:
        python schedule_knowledge_updates.py --once

    Run every 30 minutes:
        python schedule_knowledge_updates.py --interval-minutes 30
"""

from __future__ import annotations

import argparse
import json

from chatbot.base_chatbot import GeminiChatbot
from chatbot.knowledge_base import DynamicKnowledgeBase
from chatbot.knowledge_updater import KnowledgeBaseUpdater
from chatbot.memory import ConversationMemory
from config import BASE_DIR, VECTOR_DB_DIR, AppConfig, ensure_directories


def build_updater() -> KnowledgeBaseUpdater:
    """Create the production updater with persisted state."""

    ensure_directories()
    config = AppConfig.from_env()
    memory = ConversationMemory(VECTOR_DB_DIR / "scheduled_update_memory.json")
    chatbot = GeminiChatbot(config, memory)
    kb = DynamicKnowledgeBase(config, chatbot)
    return KnowledgeBaseUpdater(
        kb,
        sources_file=BASE_DIR / "knowledge_sources.json",
        state_file=VECTOR_DB_DIR / "knowledge_update_manifest.json",
    )


def main() -> None:
    """CLI entry point for scheduled updates."""

    parser = argparse.ArgumentParser(description="Scheduled incremental FAISS updater")
    parser.add_argument("--interval-minutes", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    updater = build_updater()
    if args.once:
        print(json.dumps(updater.update(), indent=2))
        return
    updater.run_periodically(interval_minutes=args.interval_minutes, on_result=lambda item: print(json.dumps(item)))


if __name__ == "__main__":
    main()
