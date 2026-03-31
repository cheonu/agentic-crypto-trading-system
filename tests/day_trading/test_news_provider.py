"""Unit tests for NewsSignalProvider."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from agentic_crypto_trading_system.day_trading.news_provider import (
    NewsSignalProvider,
)


class TestScoreHeadlines:
    """Tests for _score_headlines keyword-based scoring."""

    def test_positive_keywords_produce_positive_score(self):
        provider = NewsSignalProvider()
        headlines = [{"title": "Bitcoin rally continues as adoption grows"}]
        score = provider._score_headlines(headlines)
        assert score > 0

    def test_negative_keywords_produce_negative_score(self):
        provider = NewsSignalProvider()
        headlines = [{"title": "Major exchange hack leads to crash"}]
        score = provider._score_headlines(headlines)
        assert score < 0

    def test_empty_headlines_return_zero(self):
        provider = NewsSignalProvider()
        assert provider._score_headlines([]) == 0.0

    def test_no_keywords_return_zero(self):
        provider = NewsSignalProvider()
        headlines = [{"title": "Weather is nice today"}]
        assert provider._score_headlines(headlines) == 0.0

    def test_mixed_keywords_net_out(self):
        provider = NewsSignalProvider()
        headlines = [
            {"title": "Bitcoin rally amid crash fears"},
        ]
        # rally (+0.3) + crash (-0.4) = -0.1 / 1 headline
        score = provider._score_headlines(headlines)
        assert score == pytest.approx(-0.1, abs=0.01)

    def test_normalizes_by_headline_count(self):
        provider = NewsSignalProvider()
        headlines = [
            {"title": "Bitcoin rally"},
            {"title": "No keywords here"},
        ]
        # rally (+0.3) / 2 headlines = 0.15
        score = provider._score_headlines(headlines)
        assert score == pytest.approx(0.15, abs=0.01)

    def test_missing_title_key_skipped(self):
        provider = NewsSignalProvider()
        headlines = [{"body": "no title"}, {"title": "Bitcoin rally"}]
        score = provider._score_headlines(headlines)
        # rally (+0.3) / 2 headlines = 0.15
        assert score == pytest.approx(0.15, abs=0.01)


class TestDetectEvents:
    """Tests for _detect_events event flag detection."""

    def test_fed_meeting_detected(self):
        provider = NewsSignalProvider()
        headlines = [{"title": "Fed meeting could impact crypto markets"}]
        events = provider._detect_events(headlines)
        assert "fed_meeting" in events

    def test_etf_decision_detected(self):
        provider = NewsSignalProvider()
        headlines = [{"title": "SEC delays ETF decision again"}]
        events = provider._detect_events(headlines)
        assert "etf_decision" in events

    def test_exchange_hack_detected(self):
        provider = NewsSignalProvider()
        headlines = [{"title": "Major exchange hack reported"}]
        events = provider._detect_events(headlines)
        assert "exchange_hack" in events

    def test_regulation_detected(self):
        provider = NewsSignalProvider()
        headlines = [{"title": "New regulatory framework proposed"}]
        events = provider._detect_events(headlines)
        assert "regulation" in events

    def test_halving_detected(self):
        provider = NewsSignalProvider()
        headlines = [{"title": "Bitcoin halving countdown begins"}]
        events = provider._detect_events(headlines)
        assert "halving" in events

    def test_no_events_returns_empty(self):
        provider = NewsSignalProvider()
        headlines = [{"title": "Weather is nice today"}]
        assert provider._detect_events(headlines) == []

    def test_deduplicates_events(self):
        provider = NewsSignalProvider()
        headlines = [
            {"title": "Fed meeting scheduled"},
            {"title": "FOMC announces rate decision"},
        ]
        events = provider._detect_events(headlines)
        assert events.count("fed_meeting") == 1


class TestGetNewsSignal:
    """Tests for get_news_signal including caching."""

    def test_returns_news_signal_with_stub(self):
        provider = NewsSignalProvider()
        signal = provider.get_news_signal("BTC")
        assert signal.score == 0.0
        assert signal.headline_count == 0
        assert signal.top_headlines == []
        assert signal.event_flags == []

    def test_score_clamped_to_max(self):
        provider = NewsSignalProvider()
        # Override _fetch_headlines to return many positive headlines
        headlines = [{"title": "rally approval adoption surge bullish breakout partnership upgrade"}]
        provider._fetch_headlines = lambda s: headlines
        signal = provider.get_news_signal("BTC")
        assert signal.score <= 1.0

    def test_score_clamped_to_min(self):
        provider = NewsSignalProvider()
        headlines = [{"title": "crash hack ban fraud lawsuit exploit dump bearish"}]
        provider._fetch_headlines = lambda s: headlines
        signal = provider.get_news_signal("BTC")
        assert signal.score >= -1.0

    def test_cache_returns_same_signal_within_ttl(self):
        provider = NewsSignalProvider(cache_ttl_minutes=15)
        call_count = 0
        original_fetch = provider._fetch_headlines

        def counting_fetch(symbol):
            nonlocal call_count
            call_count += 1
            return original_fetch(symbol)

        provider._fetch_headlines = counting_fetch

        signal1 = provider.get_news_signal("BTC")
        signal2 = provider.get_news_signal("BTC")

        assert call_count == 1  # Only fetched once
        assert signal1.score == signal2.score

    def test_cache_expires_after_ttl(self):
        provider = NewsSignalProvider(cache_ttl_minutes=1)
        call_count = 0

        def counting_fetch(symbol):
            nonlocal call_count
            call_count += 1
            return []

        provider._fetch_headlines = counting_fetch

        provider.get_news_signal("BTC")
        assert call_count == 1

        # Manually expire the cache
        symbol = "BTC"
        cached_signal, _ = provider._cache[symbol]
        provider._cache[symbol] = (
            cached_signal,
            datetime.now(UTC) - timedelta(minutes=2),
        )

        provider.get_news_signal("BTC")
        assert call_count == 2

    def test_different_symbols_cached_separately(self):
        provider = NewsSignalProvider()
        provider.get_news_signal("BTC")
        provider.get_news_signal("ETH")
        assert "BTC" in provider._cache
        assert "ETH" in provider._cache

    def test_top_headlines_limited_to_five(self):
        provider = NewsSignalProvider()
        headlines = [{"title": f"Headline {i}"} for i in range(10)]
        provider._fetch_headlines = lambda s: headlines
        signal = provider.get_news_signal("BTC")
        assert len(signal.top_headlines) <= 5
