import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Result of sentiment analysis on a news article."""
    score: float          # -1.0 (bearish) to 1.0 (bullish)
    confidence: float     # 0.0 to 1.0
    magnitude: float      # 0.0 to 1.0 (impact strength)
    label: str            # "positive", "negative", "neutral"
    source: str
    article_id: Optional[str] = None
    keywords: Optional[List[str]] = None


class SentimentAnalyzer:
    """Analyzes news sentiment for crypto assets."""

    HIGH_IMPACT_THRESHOLD = 0.8

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.model_name = model_name
        self._pipeline = None
        self.sentiment_history: List[Dict] = []
        self.subscribers: List[Callable] = []

    def _get_pipeline(self):
        """Lazy-load the sentiment pipeline."""
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline("sentiment-analysis", model=self.model_name)
        return self._pipeline

    def analyze_text(self, text: str, source: str = "unknown", article_id: str = None) -> SentimentResult:
        """Analyze sentiment of a text."""
        pipe = self._get_pipeline()
        result = pipe(text[:512])[0]  # Truncate to model max

        label = result["label"].lower()
        raw_score = result["score"]

        # Map to -1 to 1 scale
        if label == "positive":
            score = raw_score
        elif label == "negative":
            score = -raw_score
        else:
            score = 0.0

        # Estimate magnitude from score strength
        magnitude = abs(score)
        confidence = raw_score

        sentiment_label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"

        result = SentimentResult(
            score=score,
            confidence=confidence,
            magnitude=magnitude,
            label=sentiment_label,
            source=source,
            article_id=article_id,
        )

        self._record(result)
        return result

    def analyze_batch(self, texts: List[Dict]) -> List[SentimentResult]:
        """Analyze sentiment for multiple articles.
        
        Each item should have 'text', 'source', and optionally 'article_id'.
        """
        results = []
        for item in texts:
            result = self.analyze_text(
                text=item["text"],
                source=item.get("source", "unknown"),
                article_id=item.get("article_id"),
            )
            results.append(result)
        return results

    def get_current_sentiment(self, symbol: str = None, window: int = 10) -> float:
        """Get aggregated current sentiment score."""
        recent = self.sentiment_history[-window:]
        if not recent:
            return 0.0
        scores = [r["score"] for r in recent]
        return sum(scores) / len(scores)

    def get_sentiment_trend(self, window: int = 20) -> List[float]:
        """Get sentiment scores over time."""
        return [r["score"] for r in self.sentiment_history[-window:]]

    def is_high_impact(self, result: SentimentResult) -> bool:
        """Check if a sentiment result is high-impact."""
        return result.magnitude > self.HIGH_IMPACT_THRESHOLD

    def subscribe_high_impact(self, callback: Callable) -> None:
        """Subscribe to high-impact news notifications."""
        self.subscribers.append(callback)

    def _record(self, result: SentimentResult) -> None:
        """Record sentiment result to history."""
        self.sentiment_history.append({
            "score": result.score,
            "confidence": result.confidence,
            "magnitude": result.magnitude,
            "label": result.label,
            "source": result.source,
            "timestamp": datetime.utcnow().isoformat(),
        })
