"""Central configuration for the AI Assistant project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

load_dotenv()

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
MEDICAL_DATA_DIR: Final[Path] = DATA_DIR / "medical"
ARXIV_DATA_DIR: Final[Path] = DATA_DIR / "arxiv"
UPLOAD_DIR: Final[Path] = BASE_DIR / "uploads"
VECTOR_DB_DIR: Final[Path] = BASE_DIR / "vector_db"
REPORTS_DIR: Final[Path] = BASE_DIR / "reports"
IMAGES_DIR: Final[Path] = BASE_DIR / "images"

SUPPORTED_LANGUAGES: Final[dict[str, str]] = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "es": "Spanish",
}

DEFAULT_GEMINI_MODEL: Final[str] = "gemini-2.5-flash"
GEMINI_MODEL_FALLBACKS: Final[tuple[str, ...]] = (
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash",
)
GEMINI_KEY_PLACEHOLDERS: Final[tuple[str, ...]] = (
    "",
    "your_google_gemini_api_key_here",
    "your_gemini_api_key_here",
    "paste_your_real_key_here",
)

MEDICAL_DISCLAIMER: Final[str] = (
    "Medical disclaimer: This assistant provides educational information only. "
    "It is not a substitute for diagnosis, treatment, or professional medical "
    "advice. Consult a qualified clinician for personal medical decisions."
)

DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "You are a careful, helpful AI assistant. Answer clearly, acknowledge "
    "uncertainty, cite retrieved context when provided, and avoid inventing facts."
)

SCIENTIFIC_SYSTEM_PROMPT: Final[str] = (
    "You are a scientific research assistant. Explain concepts precisely, "
    "summarize papers, compare methods, and include references from retrieved "
    "arXiv metadata whenever available."
)

MEDICAL_SYSTEM_PROMPT: Final[str] = (
    "You are a medical information assistant. Use retrieved MedQuAD context, "
    "include citations, state limitations, and always include the medical "
    "disclaimer."
)


@dataclass(frozen=True)
class AppConfig:
    """Typed settings loaded from environment variables."""

    gemini_api_key: str
    gemini_text_model: str = DEFAULT_GEMINI_MODEL
    gemini_vision_model: str = DEFAULT_GEMINI_MODEL
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 4
    max_history_messages: int = 12
    app_env: str = "development"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create settings from environment variables."""

        load_dotenv(BASE_DIR / ".env", override=True)
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_text_model=os.getenv("GEMINI_TEXT_MODEL", DEFAULT_GEMINI_MODEL),
            gemini_vision_model=os.getenv("GEMINI_VISION_MODEL", DEFAULT_GEMINI_MODEL),
            app_env=os.getenv("APP_ENV", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            top_k=int(os.getenv("TOP_K", "4")),
            max_history_messages=int(os.getenv("MAX_HISTORY_MESSAGES", "12")),
        )

    @property
    def has_gemini_key(self) -> bool:
        """Return whether a Gemini API key is configured."""

        return self.gemini_api_key.strip() not in GEMINI_KEY_PLACEHOLDERS


def ensure_directories() -> None:
    """Create runtime directories required by the application."""

    for path in (
        DATA_DIR,
        MEDICAL_DATA_DIR,
        ARXIV_DATA_DIR,
        UPLOAD_DIR,
        VECTOR_DB_DIR,
        REPORTS_DIR,
        IMAGES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
