"""Context-preserving multilingual conversation manager."""

from __future__ import annotations

from dataclasses import dataclass, field

from chatbot.multilingual import MultilingualAssistant, MultilingualResult


@dataclass
class LanguageTurn:
    """One multilingual conversation turn."""

    original_text: str
    detected_language: str
    english_text: str
    answer: str


@dataclass
class MultilingualConversationState:
    """Conversation state that survives language switching."""

    turns: list[LanguageTurn] = field(default_factory=list)
    active_language: str = "en"

    def context_summary(self, max_turns: int = 4) -> str:
        """Return English context for cross-lingual reasoning."""

        recent = self.turns[-max_turns:]
        return "\n".join(f"User({turn.detected_language}): {turn.english_text}" for turn in recent)


class ContextPreservingMultilingualAssistant:
    """Detect languages, translate, preserve context, and translate answers back."""

    def __init__(self, translator: MultilingualAssistant | None = None) -> None:
        self.translator = translator or MultilingualAssistant()
        self.state = MultilingualConversationState()

    def answer(self, query: str, answer_fn) -> MultilingualResult:
        """Answer multilingual input while preserving English retrieval context."""

        language = self.translator.detect_language(query)
        segments = self.translator.detect_language_segments(query)
        english_query = self.translator.to_english(query, language)
        context = self.state.context_summary()
        contextual_query = (
            f"Previous cross-lingual context:\n{context or 'No previous context.'}\n\n"
            f"Current user query translated to English:\n{english_query}\n\n"
            "Resolve references using the previous context even if the user changed language."
        )
        english_answer = answer_fn(contextual_query)
        translated = self.translator.from_english(english_answer, language)
        self.state.active_language = language
        self.state.turns.append(LanguageTurn(query, language, english_query, translated))
        return MultilingualResult(language, english_query, translated, segments)
