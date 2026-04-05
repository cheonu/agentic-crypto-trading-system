"""News signal provider for the day trading strategy.

Fetches and scores crypto news headlines to produce a sentiment signal.
Uses HuggingFace Transformers for sentiment scoring and CryptoPanic API
for headline fetching. Falls back gracefully on any component failure.
"""

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

from .models import NewsSignal

logger = logging.getLogger(__name__)

# Event patterns: (search term, event flag)
EVENT_PATTERNS: List[Tuple[str, str]] = [
    ("fed meeting", "fed_meeting"),
    ("federal reserve", "fed_meeting"),
    ("fomc", "fed_meeting"),
    ("etf decision", "etf_decision"),
    ("etf approval", "etf_decision"),
    ("etf rejection", "etf_decision"),
    ("etf filing", "etf_decision"),
    ("exchange hack", "exchange_hack"),
    ("exchange breach", "exchange_hack"),
    ("funds stolen", "exchange_hack"),
    ("regulation", "regulation"),
    ("regulatory", "regulation"),
    ("sec ", "regulation"),
    ("halving", "halving"),
    ("halvening", "halving"),
]

LABEL_SCORE_MAP: Dict[str, float] = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
    "bullish": 1.0,
    "bearish": -1.0,
}


class HeadlineFetcher:
    """Fetches crypto news headlines from RSS feeds.

    Uses public RSS feeds from major crypto news outlets (CoinDesk,
    CoinTelegraph, Decrypt). No API key required, no rate limits.
    On any failure: logs warning, returns [].
    Max 50 headlines, sorted newest-first.
    """

    RSS_FEEDS = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
    ]

    def __init__(self, api_key: str = "", source: str = "rss") -> None:
        self._api_key = api_key  # kept for future paid API support
        self._source = source

    def fetch(self, symbol: str) -> List[Dict]:
        """Fetch headlines from RSS feeds, filtered by symbol.

        Returns list of {"title": str}. Max 50, newest-first.
        If no symbol-specific headlines found, returns general crypto headlines.
        Returns empty list on any failure. Never raises.
        """
        try:
            import xml.etree.ElementTree as ET

            symbol_headlines = []
            general_headlines = []

            for feed_url in self.RSS_FEEDS:
                try:
                    resp = requests.get(feed_url, timeout=10)
                    resp.raise_for_status()
                    root = ET.fromstring(resp.content)

                    # Standard RSS 2.0: channel/item/title
                    for item in root.findall(".//item"):
                        title_el = item.find("title")
                        if title_el is not None and title_el.text:
                            title = title_el.text.strip()
                            headline = {"title": title}
                            if symbol.upper() in title.upper() or self._symbol_aliases(symbol, title):
                                symbol_headlines.append(headline)
                            else:
                                general_headlines.append(headline)
                except Exception as exc:
                    logger.warning("Failed to fetch RSS feed %s: %s", feed_url, exc)
                    continue

            # Prefer symbol-specific headlines, fall back to general crypto news
            all_headlines = symbol_headlines if symbol_headlines else general_headlines

            # Deduplicate by title, keep first 50
            seen = set()
            unique = []
            for h in all_headlines:
                if h["title"] not in seen:
                    seen.add(h["title"])
                    unique.append(h)

            result = unique[:50]
            logger.info(
                "Fetched %d headlines for %s (%d symbol-specific) from %d RSS feeds",
                len(result), symbol, len(symbol_headlines), len(self.RSS_FEEDS),
            )
            return result

        except Exception as exc:
            logger.warning("HeadlineFetcher failed for %s: %s", symbol, exc)
            return []

    @staticmethod
    def _symbol_aliases(symbol: str, title: str) -> bool:
        """Check if title mentions common aliases for the symbol."""
        aliases = {
            "BTC": ["bitcoin"],
            "ETH": ["ethereum"],
            "SOL": ["solana"],
            "XRP": ["ripple"],
            "ADA": ["cardano"],
            "DOGE": ["dogecoin"],
            "DOT": ["polkadot"],
            "AVAX": ["avalanche"],
        }
        title_lower = title.lower()
        for alias in aliases.get(symbol.upper(), []):
            if alias in title_lower:
                return True
        return False


class ModelManager:
    """Manages HuggingFace model lifecycle: lazy load, cache, single instance.

    Docs: https://huggingface.co/docs/transformers/main_classes/pipelines
    """

    def __init__(self, model_name: str = "ElKulako/cryptobert") -> None:
        self._model_name = model_name
        self._pipeline = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_pipeline(self):
        """Return the loaded pipeline, loading on first call.

        Uses transformers.pipeline() which wraps:
        1. AutoTokenizer.from_pretrained(model_name) — converts text to token IDs
        2. AutoModelForSequenceClassification.from_pretrained(model_name) — the neural net
        3. Post-processing — softmax on logits, returns {"label": str, "score": float}

        device="cpu" forces CPU inference (no CUDA needed).
        HF_HOME env var controls where model files are cached on disk.

        Docs: https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
        """
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self._model_name,
                device="cpu",
            )
            logger.info("Loaded HF model '%s' on CPU", self._model_name)
            return self._pipeline
        except Exception as exc:
            logger.error("Failed to load HF model '%s': %s", self._model_name, exc)
            raise RuntimeError(
                f"Failed to load HuggingFace model '{self._model_name}': {exc}"
            ) from exc

    def is_loaded(self) -> bool:
        """Return True if the model is currently loaded in memory."""
        return self._pipeline is not None

class SentimentAnalyzer:
    """Scores headlines using one or two HuggingFace sentiment pipelines.

    When an ensemble_manager is provided, scores are weighted:
    70% primary (CryptoBERT — crypto-native) + 30% ensemble (FinBERT — better negatives).
    """

    PRIMARY_WEIGHT = 0.7
    ENSEMBLE_WEIGHT = 0.3

    def __init__(self, model_manager: ModelManager, ensemble_manager: ModelManager = None):
        self._model_manager = model_manager
        self._ensemble_manager = ensemble_manager

    def score_single(self, text: str) -> float:
        """Score a single headline. Returns numeric score or 0.0 on failure."""
        score1 = self._score_with(self._model_manager, text)

        if self._ensemble_manager:
            score2 = self._score_with(self._ensemble_manager, text)
            return (self.PRIMARY_WEIGHT * score1) + (self.ENSEMBLE_WEIGHT * score2)

        return score1

    def score_headlines(self, headlines: List[Dict]) -> float:
        """Score headlines and return aggregate sentiment in [-1.0, 1.0]."""
        if not headlines:
            return 0.0

        scores = []
        for headline in headlines:
            title = headline.get("title", "")
            if not title:
                continue
            score = self.score_single(title)
            scores.append(score)

        if not scores:
            return 0.0

        mean_score = sum(scores) / len(scores)
        return max(-1.0, min(1.0, mean_score))

    def _score_with(self, manager: ModelManager, text: str) -> float:
        """Score text with a single model. Returns 0.0 on failure."""
        try:
            pipe = manager.get_pipeline()
            result = pipe(text, truncation=True, max_length=512)
            if not result:
                return 0.0
            label = result[0]["label"].lower()
            confidence = result[0]["score"]
            direction = LABEL_SCORE_MAP.get(label, 0.0)
            if direction == 0.0 and label not in LABEL_SCORE_MAP:
                logger.warning("Unknown label '%s' from model '%s'", label, manager.model_name)
            return direction * confidence
        except Exception as exc:
            logger.warning("Inference failed for model '%s': %s", manager.model_name, exc)
            return 0.0


class NewsSignalProvider:
    """Fetches and scores crypto news headlines to produce a sentiment signal.

    Uses HeadlineFetcher for real headline retrieval and SentimentAnalyzer
    (backed by a HuggingFace model) for scoring. Falls back to neutral
    signal on any component failure.

    Args:
        api_key: API key for headline source (CryptoPanic).
        cache_ttl_minutes: Minutes to cache results before re-fetching.
        sources: Optional list of news source identifiers.
        model_name: HuggingFace model identifier for sentiment analysis.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl_minutes: int = 15,
        sources: Optional[List[str]] = None,
        model_name: str = "ElKulako/cryptobert",
    ) -> None:
        self._cache_ttl_minutes = cache_ttl_minutes
        self._cache: Dict[str, Tuple[NewsSignal, datetime]] = {}

        # New components
        self._model_manager = ModelManager(model_name=model_name)
        ensemble_model = os.environ.get("ENSEMBLE_MODEL_NAME")
        ensemble_mgr = ModelManager(model_name=ensemble_model) if ensemble_model else None
        self._sentiment_analyzer = SentimentAnalyzer(self._model_manager, ensemble_mgr)
        self._headline_fetcher = HeadlineFetcher(
            api_key=api_key or "",
            source=(sources[0] if sources else "cryptopanic"),
        )

    def get_news_signal(self, symbol: str) -> NewsSignal:
        """Get a news sentiment signal for the given symbol.

        Returns a cached result if the cache TTL has not expired,
        otherwise fetches fresh headlines and scores them.
        Falls back to neutral signal on any failure.
        """
        now = datetime.now(UTC)

        # Check cache
        if symbol in self._cache:
            cached_signal, cached_time = self._cache[symbol]
            if now - cached_time < timedelta(minutes=self._cache_ttl_minutes):
                logger.debug(
                    "Returning cached news signal for %s (age: %s)",
                    symbol,
                    now - cached_time,
                )
                return cached_signal

        try:
            # Fetch headlines
            headlines = self._headline_fetcher.fetch(symbol)

            if not headlines:
                signal = NewsSignal(
                    score=0.0,
                    headline_count=0,
                    top_headlines=[],
                    event_flags=[],
                    timestamp=now,
                )
                self._cache[symbol] = (signal, now)
                return signal

            # Score with HF model
            score = self._sentiment_analyzer.score_headlines(headlines)
            event_flags = self._detect_events(headlines)

            # Clamp score to [-1.0, 1.0]
            clamped_score = max(-1.0, min(1.0, score))

            # Extract top headline titles (up to 5)
            top_headlines = [
                h.get("title", "") for h in headlines[:5] if h.get("title")
            ]

            # Fine-tuning data logging: per-headline scores with model name
            for headline in headlines:
                title = headline.get("title", "")
                if title:
                    individual_score = self._sentiment_analyzer.score_single(title)
                    logger.debug(
                        "Headline score [model=%s]: %.3f — %s",
                        self._model_manager.model_name,
                        individual_score,
                        title,
                    )

            signal = NewsSignal(
                score=clamped_score,
                headline_count=len(headlines),
                top_headlines=top_headlines,
                event_flags=event_flags,
                timestamp=now,
            )

            self._cache[symbol] = (signal, now)

            logger.info(
                "News signal for %s: score=%.2f, headlines=%d, events=%s, model=%s",
                symbol,
                clamped_score,
                len(headlines),
                event_flags,
                self._model_manager.model_name,
            )

            return signal

        except Exception as exc:
            logger.warning(
                "NewsSignalProvider failed for %s, returning neutral: %s",
                symbol,
                exc,
            )
            signal = NewsSignal(
                score=0.0,
                headline_count=0,
                top_headlines=[],
                event_flags=[],
                timestamp=now,
            )
            self._cache[symbol] = (signal, now)
            return signal

    def _detect_events(self, headlines: List[Dict]) -> List[str]:
        """Detect known market-moving event types in headlines.

        Checks headline titles against known event patterns and returns
        a deduplicated list of event flags.
        """
        detected: set[str] = set()

        for headline in headlines:
            title = headline.get("title", "").lower()
            if not title:
                continue

            for pattern, flag in EVENT_PATTERNS:
                if pattern in title:
                    detected.add(flag)

        return sorted(detected)
