from chatbot.sentiment import SentimentAnalyzer


def test_sentiment_positive_or_neutral_for_thanks():
    result = SentimentAnalyzer().analyze("Thanks, this is helpful and good.")

    assert result.label in {"Positive", "Neutral"}
    assert result.tone_instruction


def test_sentiment_negative_for_problem():
    analyzer = SentimentAnalyzer()
    analyzer._vader = None
    result = analyzer.analyze("This is a bad problem and I am worried.")

    assert result.label == "Negative"
