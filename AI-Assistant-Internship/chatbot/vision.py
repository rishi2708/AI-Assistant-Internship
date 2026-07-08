"""Gemini Vision and OCR utilities."""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path

import pytesseract
import requests
from PIL import Image

from config import AppConfig, GEMINI_MODEL_FALLBACKS

logger = logging.getLogger(__name__)


class GeminiVisionAssistant:
    """Image understanding, OCR, and mixed image-text reasoning."""

    def __init__(self, config: AppConfig, temperature: float = 0.2) -> None:
        self.config = config
        self.temperature = temperature

    def extract_ocr(self, image_path: Path) -> str:
        """Extract OCR text from an image with Tesseract when available."""

        try:
            image = Image.open(image_path)
            return pytesseract.image_to_string(image).strip()
        except Exception as exc:  # pragma: no cover - local binary dependent
            logger.warning("OCR failed: %s", exc)
            return ""

    def analyze_image(self, image_path: Path, prompt: str) -> str:
        """Ask Gemini Vision to reason about an uploaded image."""

        if not self.config.has_gemini_key:
            return "Gemini API key is not configured. Add it to `.env` as GEMINI_API_KEY."

        image = Image.open(image_path)
        ocr_text = self.extract_ocr(image_path)
        full_prompt = (
            f"{prompt}\n\n"
            f"OCR text extracted from the image, if any:\n{ocr_text or 'No OCR text found.'}\n\n"
            "Explain visible evidence and avoid unsupported claims."
        )
        last_error: Exception | None = None
        for model_name in self._model_candidates():
            try:
                return self._generate_with_model(model_name, image_path, full_prompt)
            except Exception as exc:  # pragma: no cover - provider dependent
                last_error = exc
                logger.warning("Gemini Vision model %s failed: %s", model_name, exc)
                if "not found" not in str(exc).lower() and "404" not in str(exc):
                    break

        logger.exception("Gemini Vision failed")
        return f"Image analysis failed: {last_error}"

    def _model_candidates(self) -> list[str]:
        """Return configured and available Gemini vision-capable models."""

        candidates = [self.config.gemini_vision_model, *GEMINI_MODEL_FALLBACKS]
        try:
            response = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": self.config.gemini_api_key},
                timeout=20,
            )
            self._raise_clean_error(response)
            available = []
            for model in response.json().get("models", []):
                methods = model.get("supportedGenerationMethods", [])
                if "generateContent" in methods:
                    available.append(model.get("name", "").replace("models/", ""))
            candidates.extend(available)
        except Exception as exc:  # pragma: no cover - provider dependent
            logger.info("Could not list Gemini models: %s", exc)

        unique: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in unique:
                unique.append(candidate)
        return unique

    def _generate_with_model(self, model_name: str, image_path: Path, prompt: str) -> str:
        """Analyze an image with Gemini through the public REST API."""

        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        headers = {"x-goog-api-key": self.config.gemini_api_key}
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": mime_type, "data": image_data}},
                    ],
                }
            ],
            "generationConfig": {"temperature": self.temperature},
        }
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        self._raise_clean_error(response)
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "I could not analyze the image."
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts).strip() or "I could not analyze the image."

    @staticmethod
    def _raise_clean_error(response: requests.Response) -> None:
        """Raise an HTTP error without leaking request URLs or API keys."""

        if response.ok:
            return
        try:
            payload = response.json()
            message = payload.get("error", {}).get("message") or response.text
        except ValueError:
            message = response.text
        raise RuntimeError(f"{response.status_code} {response.reason}: {message}")
