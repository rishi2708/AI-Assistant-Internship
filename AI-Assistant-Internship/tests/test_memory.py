from chatbot.memory import ConversationMemory


def test_memory_persists_and_exports(tmp_path):
    path = tmp_path / "memory.json"
    memory = ConversationMemory(path, window_size=2)
    memory.add("user", "What is RAG?", "General")
    memory.add("assistant", "Retrieval augmented generation.", "General")

    loaded = ConversationMemory(path, window_size=2)

    assert len(loaded.messages) == 2
    assert loaded.remember_previous_questions() == ["What is RAG?"]
    assert "Retrieval augmented generation" in loaded.export_markdown()


def test_memory_window():
    memory = ConversationMemory.__new__(ConversationMemory)
    memory.storage_path = None
    memory.window_size = 1
    memory.messages = []
    memory.add = lambda role, content, mode="General": memory.messages.append(  # noqa: E731
        type("Message", (), {"role": role, "content": content, "timestamp": "", "mode": mode})()
    )

    memory.add("user", "first")
    memory.add("user", "second")

    assert [message.content for message in ConversationMemory.recent(memory)] == ["second"]
