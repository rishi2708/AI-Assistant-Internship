"""Complete sentiment-aware conversational response pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from chatbot.sentiment import SentimentAnalyzer, SentimentResult


@dataclass(frozen=True)
class SentimentResponse:
    """Sentiment-aware response with evaluation metadata."""

    sentiment: SentimentResult
    adapted_prompt: str
    response: str
    evaluation: str


class SentimentAwareResponder:
    """Detect emotion, adapt prompt/response, and evaluate tone alignment."""

    def __init__(self, analyzer: SentimentAnalyzer | None = None) -> None:
        self.analyzer = analyzer or SentimentAnalyzer()

    def adapt_prompt(self, user_message: str, sentiment: SentimentResult) -> str:
        """Inject explicit tone policy for generation."""

        return (
            f"{user_message}\n\n"
            "Customer emotion analysis:\n"
            f"- Sentiment: {sentiment.label}\n"
            f"- Confidence score: {sentiment.score:.2f}\n"
            f"- Required tone: {sentiment.tone_instruction}\n"
            "Generate the answer in a way that matches the required tone."
        )

    def adapt_response(self, raw_response: str, sentiment: SentimentResult) -> str:
        """Post-process response with emotion-specific conversational framing."""

        if sentiment.label == "Negative":
            prefix = "I understand this may feel frustrating. "
        elif sentiment.label == "Positive":
            prefix = "Great, glad to help. "
        else:
            prefix = ""
        return f"{prefix}{raw_response}".strip()

    def evaluate_alignment(self, response: str, sentiment: SentimentResult) -> str:
        """Evaluate whether the answer tone matches the detected emotion."""

        lowered = response.lower()
        if sentiment.label == "Negative" and any(term in lowered for term in ["understand", "sorry", "frustrating"]):
            return "aligned: empathetic response for negative emotion"
        if sentiment.label == "Positive" and any(term in lowered for term in ["great", "glad", "happy"]):
            return "aligned: positive response preserves user enthusiasm"
        if sentiment.label == "Neutral":
            return "aligned: neutral response uses professional tone"
        return "needs_review: response tone may not match detected sentiment"

    def generate(self, user_message: str, generation_fn) -> SentimentResponse:
        """Run the full sentiment-aware generation pipeline."""

        sentiment = self.analyzer.analyze(user_message)
        adapted_prompt = self.adapt_prompt(user_message, sentiment)
        raw_response = generation_fn(adapted_prompt)
        response = self.adapt_response(raw_response, sentiment)
        evaluation = self.evaluate_alignment(response, sentiment)
        return SentimentResponse(sentiment, adapted_prompt, response, evaluation)
