"""Language detection and translation workflow."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from deep_translator import GoogleTranslator
from langdetect import LangDetectException, detect

from config import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultilingualResult:
    """Result of multilingual processing."""

    detected_language: str
    english_query: str
    answer: str
    language_segments: list[dict[str, str]]


class MultilingualAssistant:
    """Detect language, translate to English for retrieval, translate answer back."""

    def detect_language(self, text: str) -> str:
        """Detect the input language and normalize unsupported languages to English."""

        try:
            language = detect(text)
        except LangDetectException:
            language = "en"
        return language if language in SUPPORTED_LANGUAGES else "en"

    def detect_language_segments(self, text: str) -> list[dict[str, str]]:
        """Detect language for mixed-language chunks in one message."""

        chunks = [chunk.strip() for chunk in re.split(r"([.!?।]+)", text) if chunk.strip()]
        segments: list[dict[str, str]] = []
        buffer = ""
        for chunk in chunks:
            buffer = f"{buffer}{chunk}".strip()
            if chunk in {".", "!", "?", "।"}:
                if buffer:
                    segments.append({"text": buffer, "language": self.detect_language(buffer)})
                buffer = ""
        if buffer:
            segments.append({"text": buffer, "language": self.detect_language(buffer)})
        return segments or [{"text": text, "language": self.detect_language(text)}]

    def to_english(self, text: str, source_language: str) -> str:
        """Translate text to English if necessary."""

        if source_language == "en":
            return text
        try:
            return GoogleTranslator(source=source_language, target="en").translate(text)
        except Exception as exc:  # pragma: no cover - external service
            logger.warning("Translation to English failed: %s", exc)
            return text

    def from_english(self, text: str, target_language: str) -> str:
        """Translate English text back to the user's language if necessary."""

        if target_language == "en":
            return text
        try:
            return GoogleTranslator(source="en", target=target_language).translate(text)
        except Exception as exc:  # pragma: no cover - external service
            logger.warning("Translation from English failed: %s", exc)
            return text

    def answer(self, query: str, answer_fn) -> MultilingualResult:
        """Run a multilingual retrieval/generation function."""

        language = self.detect_language(query)
        segments = self.detect_language_segments(query)
        english_query = self.to_english(query, language)
        answer = answer_fn(english_query)
        translated = self.from_english(answer, language)
        return MultilingualResult(language, english_query, translated, segments)
