import pytest
from unittest.mock import patch, MagicMock

from agentic_crypto_trading_system.sentiment.analyzer import SentimentAnalyzer, SentimentResult


@pytest.fixture
def mock_analyzer():
    """Create analyzer with mocked pipeline."""
    analyzer = SentimentAnalyzer()
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"label": "POSITIVE", "score": 0.92}]
    analyzer._pipeline = mock_pipe
    return analyzer


def test_analyze_positive(mock_analyzer):
    """Test positive sentiment analysis."""
    result = mock_analyzer.analyze_text("Bitcoin surges to new all-time high", source="test")
    assert result.score > 0
    assert result.label == "positive"
    assert result.confidence > 0


def test_analyze_negative():
    """Test negative sentiment analysis."""
    analyzer = SentimentAnalyzer()
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"label": "NEGATIVE", "score": 0.88}]
    analyzer._pipeline = mock_pipe

    result = analyzer.analyze_text("Crypto market crashes amid regulatory fears", source="test")
    assert result.score < 0
    assert result.label == "negative"


def test_analyze_batch(mock_analyzer):
    """Test batch analysis."""
    articles = [
        {"text": "BTC up 10%", "source": "news1"},
        {"text": "ETH rallies", "source": "news2"},
    ]
    results = mock_analyzer.analyze_batch(articles)
    assert len(results) == 2


def test_get_current_sentiment(mock_analyzer):
    """Test aggregated sentiment."""
    mock_analyzer.analyze_text("Good news", source="test")
    mock_analyzer.analyze_text("More good news", source="test")
    sentiment = mock_analyzer.get_current_sentiment()
    assert sentiment > 0


def test_high_impact_detection():
    """Test high-impact news detection."""
    analyzer = SentimentAnalyzer()
    high = SentimentResult(score=0.95, confidence=0.95, magnitude=0.95, label="positive", source="test")
    low = SentimentResult(score=0.3, confidence=0.6, magnitude=0.3, label="positive", source="test")
    assert analyzer.is_high_impact(high) is True
    assert analyzer.is_high_impact(low) is False


def test_sentiment_trend(mock_analyzer):
    """Test sentiment trend tracking."""
    for _ in range(5):
        mock_analyzer.analyze_text("Bullish news", source="test")
    trend = mock_analyzer.get_sentiment_trend()
    assert len(trend) == 5
    assert all(s > 0 for s in trend)
