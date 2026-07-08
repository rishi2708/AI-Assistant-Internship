"""Gemini base chatbot with history, streaming, and error handling."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import requests

from chatbot.memory import ConversationMemory
from config import AppConfig, DEFAULT_SYSTEM_PROMPT, GEMINI_MODEL_FALLBACKS

logger = logging.getLogger(__name__)


class GeminiChatbot:
    """Production-style Gemini chatbot wrapper."""

    def __init__(
        self,
        config: AppConfig,
        memory: ConversationMemory,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.4,
    ) -> None:
        self.config = config
        self.memory = memory
        self.system_prompt = system_prompt
        self.temperature = temperature

    def _generate_with_model(self, model_name: str, prompt: str) -> str:
        """Generate text with Gemini through the public REST API."""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        headers = {"x-goog-api-key": self.config.gemini_api_key}
        contents = self.memory.as_gemini_history()
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        payload = {
            "systemInstruction": {"parts": [{"text": self.system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": self.temperature},
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        self._raise_clean_error(response)
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "I could not generate a response."
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts).strip() or "I could not generate a response."

    def _model_candidates(self) -> list[str]:
        """Return configured and available Gemini text models."""

        candidates = [self.config.gemini_text_model, *GEMINI_MODEL_FALLBACKS]
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

    def generate_response(self, prompt: str, mode: str = "General") -> str:
        """Generate a non-streaming response and update memory."""

        if not self.config.has_gemini_key:
            return "Gemini API key is not configured. Add it to `.env` as GEMINI_API_KEY."
        last_error: Exception | None = None
        for model_name in self._model_candidates():
            try:
                text = self._generate_with_model(model_name, prompt)
                self.memory.add("user", prompt, mode)
                self.memory.add("assistant", text, mode)
                return text
            except Exception as exc:  # pragma: no cover - provider dependent
                last_error = exc
                clean_error = self._clean_error_message(exc)
                logger.warning("Gemini model %s failed: %s", model_name, clean_error)
                if "not found" not in clean_error.lower() and "404" not in clean_error:
                    break

        logger.error("Gemini response failed: %s", self._clean_error_message(last_error))
        return f"Sorry, I could not complete the request: {self._clean_error_message(last_error)}"

    def stream_response(self, prompt: str, mode: str = "General") -> Iterator[str]:
        """Stream a response from Gemini and persist the final answer."""

        if not self.config.has_gemini_key:
            yield "Gemini API key is not configured. Add it to `.env` as GEMINI_API_KEY."
            return
        collected: list[str] = []
        last_error: Exception | None = None
        for model_name in self._model_candidates():
            try:
                text = self._generate_with_model(model_name, prompt)
                collected.append(text)
                yield text
                final_text = "".join(collected).strip()
                self.memory.add("user", prompt, mode)
                self.memory.add("assistant", final_text, mode)
                return
            except Exception as exc:  # pragma: no cover - provider dependent
                last_error = exc
                clean_error = self._clean_error_message(exc)
                logger.warning("Gemini streaming model %s failed: %s", model_name, clean_error)
                if "not found" not in clean_error.lower() and "404" not in clean_error:
                    break
        logger.error("Gemini streaming failed: %s", self._clean_error_message(last_error))
        yield f"Sorry, I could not complete the request: {self._clean_error_message(last_error)}"

    def clear_history(self) -> None:
        """Clear conversation memory."""

        self.memory.clear()

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

    def _clean_error_message(self, error: Exception | None) -> str:
        """Return a user-safe error message."""

        if error is None:
            return "Unknown Gemini API error."
        message = str(error).replace(self.config.gemini_api_key, "[API_KEY]")
        if "API key not valid" in message or "API_KEY_INVALID" in message:
            return "The Gemini API key is invalid. Create a new Google AI Studio key that starts with AIza and update `.env`."
        return message
