"""Conversation memory implementations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str
    content: str
    timestamp: str
    mode: str = "General"


class ConversationMemory:
    """Windowed and persistent chat memory."""

    def __init__(self, storage_path: Path, window_size: int = 12) -> None:
        self.storage_path = storage_path
        self.window_size = window_size
        self.messages: list[ChatMessage] = []
        self.load()

    def add(self, role: str, content: str, mode: str = "General") -> None:
        """Add a message to memory and persist it."""

        self.messages.append(
            ChatMessage(
                role=role,
                content=content,
                timestamp=datetime.now(timezone.utc).isoformat(),
                mode=mode,
            )
        )
        self.save()

    def recent(self, limit: int | None = None) -> list[ChatMessage]:
        """Return recent messages using window memory."""

        size = limit or self.window_size
        return self.messages[-size:]

    def as_gemini_history(self) -> list[dict[str, object]]:
        """Convert recent messages to Gemini REST chat history format."""

        history = []
        for message in self.recent():
            role = "model" if message.role == "assistant" else "user"
            history.append({"role": role, "parts": [{"text": message.content}]})
        return history

    def remember_previous_questions(self, limit: int = 5) -> list[str]:
        """Return previous user questions."""

        return [msg.content for msg in self.messages if msg.role == "user"][-limit:]

    def clear(self) -> None:
        """Clear memory and persistence file."""

        self.messages = []
        self.save()

    def save(self) -> None:
        """Persist messages as JSON."""

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(message) for message in self.messages]
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> None:
        """Load messages from disk if available."""

        if not self.storage_path.exists():
            self.messages = []
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self.messages = [ChatMessage(**item) for item in payload]
        except (json.JSONDecodeError, TypeError):
            self.messages = []

    def export_markdown(self) -> str:
        """Export chat history to Markdown."""

        lines = ["# AI Assistant Chat Export", ""]
        for message in self.messages:
            lines.append(f"## {message.role.title()} - {message.mode} - {message.timestamp}")
            lines.append(message.content)
            lines.append("")
        return "\n".join(lines)
