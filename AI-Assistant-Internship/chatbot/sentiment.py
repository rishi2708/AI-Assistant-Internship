"""Sentiment analysis and tone adaptation."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SentimentResult:
    """Sentiment label and confidence."""

    label: str
    score: float
    tone_instruction: str


class SentimentAnalyzer:
    """Detect positive, negative, and neutral sentiment."""

    def __init__(self) -> None:
        self._vader = None

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment and return tone guidance."""

        if self._vader:
            compound = self._vader.polarity_scores(text)["compound"]
        else:
            compound = self._fallback_score(text)

        if compound >= 0.2:
            label = "Positive"
            tone = "The user sounds positive; keep the tone energetic and concise."
        elif compound <= -0.2:
            label = "Negative"
            tone = "The user may be frustrated or worried; be empathetic, calm, and reassuring."
        else:
            label = "Neutral"
            tone = "The user sounds neutral; use a balanced professional tone."

        return SentimentResult(label=label, score=float(compound), tone_instruction=tone)

    @staticmethod
    def _fallback_score(text: str) -> float:
        """Small lexical fallback when VADER data is unavailable."""

        positive = {"good", "great", "excellent", "happy", "love", "helpful", "thanks"}
        negative = {"bad", "terrible", "sad", "angry", "hate", "worried", "problem"}
        tokens = set(re.findall(r"[a-zA-Z']+", text.lower()))
        return (len(tokens & positive) - len(tokens & negative)) / max(len(tokens), 1)
