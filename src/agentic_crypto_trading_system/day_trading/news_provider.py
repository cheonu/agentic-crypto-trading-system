"""News signal provider for the day trading strategy.

Fetches and scores crypto news headlines to produce a sentiment signal.
Uses keyword-based scoring and event detection to generate a NewsSignal
that feeds into DayTradingStrategy.evaluate().
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .models import NewsSignal

logger = logging.getLogger(__name__)

# Positive keywords and their score contributions
POSITIVE_KEYWORDS: List[Tuple[str, float]] = [
    ("rally", 0.3),
    ("approval", 0.4),
    ("adoption", 0.3),
    ("surge", 0.3),
    ("bullish", 0.25),
    ("breakout", 0.25),
    ("partnership", 0.2),
    ("upgrade", 0.2),
]

# Negative keywords and their score contributions
NEGATIVE_KEYWORDS: List[Tuple[str, float]] = [
    ("crash", -0.4),
    ("hack", -0.4),
    ("ban", -0.35),
    ("fraud", -0.35),
    ("lawsuit", -0.3),
    ("exploit", -0.3),
    ("dump", -0.25),
    ("bearish", -0.25),
]

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


class NewsSignalProvider:
    """Fetches and scores crypto news headlines to produce a sentiment signal.

    Scores headlines using keyword-based sentiment analysis and detects
    known market-moving event types. Results are cached per symbol with
    a configurable TTL to avoid redundant API calls.

    Args:
        api_key: Optional API key for news source.
        cache_ttl_minutes: Minutes to cache results before re-fetching.
        sources: Optional list of news source identifiers.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl_minutes: int = 15,
        sources: Optional[List[str]] = None,
    ) -> None:
        self._api_key = api_key
        self._cache_ttl_minutes = cache_ttl_minutes
        self._sources = sources or []
        # Cache: symbol -> (NewsSignal, fetch_timestamp)
        self._cache: Dict[str, Tuple[NewsSignal, datetime]] = {}

    def get_news_signal(self, symbol: str) -> NewsSignal:
        """Get a news sentiment signal for the given symbol.

        Returns a cached result if the cache TTL has not expired,
        otherwise fetches fresh headlines and scores them.

        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH").

        Returns:
            NewsSignal with score clamped to [-1.0, 1.0].
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

        # Fetch and score
        headlines = self._fetch_headlines(symbol)
        score = self._score_headlines(headlines)
        event_flags = self._detect_events(headlines)

        # Clamp score to [-1.0, 1.0]
        clamped_score = max(-1.0, min(1.0, score))

        # Extract top headline titles (up to 5)
        top_headlines = [
            h.get("title", "") for h in headlines[:5] if h.get("title")
        ]

        signal = NewsSignal(
            score=clamped_score,
            headline_count=len(headlines),
            top_headlines=top_headlines,
            event_flags=event_flags,
            timestamp=now,
        )

        # Update cache
        self._cache[symbol] = (signal, now)

        logger.info(
            "News signal for %s: score=%.2f, headlines=%d, events=%s",
            symbol,
            clamped_score,
            len(headlines),
            event_flags,
        )

        return signal

    def _fetch_headlines(self, symbol: str) -> List[Dict]:
        """Fetch news headlines for a symbol.

        This is a stub implementation that returns an empty list.
        Override this method or replace it with actual API integration
        (e.g., CryptoPanic, RSS feeds) for production use.

        Args:
            symbol: Trading symbol to fetch headlines for.

        Returns:
            List of headline dicts, each with at least a "title" key.
        """
        logger.debug(
            "Stub _fetch_headlines called for %s — returning empty list",
            symbol,
        )
        return []

    def _score_headlines(self, headlines: List[Dict]) -> float:
        """Score headlines using keyword-based sentiment analysis.

        Iterates through each headline title, checking for positive and
        negative keywords. Sums individual scores and normalizes by the
        number of headlines to produce a per-headline average score.

        Args:
            headlines: List of headline dicts with "title" keys.

        Returns:
            Raw sentiment score (not yet clamped).
        """
        if not headlines:
            return 0.0

        total_score = 0.0

        for headline in headlines:
            title = headline.get("title", "").lower()
            if not title:
                continue

            for keyword, weight in POSITIVE_KEYWORDS:
                if keyword in title:
                    total_score += weight

            for keyword, weight in NEGATIVE_KEYWORDS:
                if keyword in title:
                    total_score += weight  # weight is already negative

        # Normalize by headline count
        return total_score / len(headlines)

    def _detect_events(self, headlines: List[Dict]) -> List[str]:
        """Detect known market-moving event types in headlines.

        Checks headline titles against known event patterns and returns
        a deduplicated list of event flags.

        Args:
            headlines: List of headline dicts with "title" keys.

        Returns:
            List of unique event flag strings (e.g., ["fed_meeting", "etf_decision"]).
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
