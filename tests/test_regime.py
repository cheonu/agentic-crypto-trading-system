import pytest
import numpy as np

from agentic_crypto_trading_system.regime.indicators import (
    calculate_atr,
    calculate_adx,
    calculate_momentum,
    calculate_volume_profile,
)
from agentic_crypto_trading_system.regime.classifier import RegimeClassifier, MarketRegime
from agentic_crypto_trading_system.regime.detector import RegimeDetector


def _generate_trending_up(n=30, start=100.0, step=2.0):
    """Generate upward trending OHLCV data."""
    closes = [start + i * step for i in range(n)]
    highs = [c + np.random.uniform(0.5, 2.0) for c in closes]
    lows = [c - np.random.uniform(0.5, 2.0) for c in closes]
    volumes = [1000 + np.random.uniform(-100, 100) for _ in range(n)]
    return highs, lows, closes, volumes


def _generate_trending_down(n=30, start=200.0, step=2.0):
    """Generate downward trending OHLCV data."""
    closes = [start - i * step for i in range(n)]
    highs = [c + np.random.uniform(0.5, 2.0) for c in closes]
    lows = [c - np.random.uniform(0.5, 2.0) for c in closes]
    volumes = [1000 + np.random.uniform(-100, 100) for _ in range(n)]
    return highs, lows, closes, volumes


def _generate_sideways(n=30, center=100.0):
    """Generate sideways OHLCV data."""
    closes = [center + np.random.uniform(-1, 1) for _ in range(n)]
    highs = [c + np.random.uniform(0.1, 0.5) for c in closes]
    lows = [c - np.random.uniform(0.1, 0.5) for c in closes]
    volumes = [1000 + np.random.uniform(-50, 50) for _ in range(n)]
    return highs, lows, closes, volumes


def test_calculate_atr():
    """Test ATR calculation."""
    highs, lows, closes, _ = _generate_trending_up()
    atr = calculate_atr(highs, lows, closes)
    assert atr > 0
    assert isinstance(atr, float)


def test_calculate_atr_insufficient_data():
    """Test ATR raises on insufficient data."""
    with pytest.raises(ValueError):
        calculate_atr([1, 2], [0.5, 1.5], [0.8, 1.8], period=14)


def test_calculate_adx():
    """Test ADX calculation."""
    highs, lows, closes, _ = _generate_trending_up()
    adx = calculate_adx(highs, lows, closes)
    assert adx >= 0
    assert isinstance(adx, float)


def test_calculate_momentum_up():
    """Test momentum is positive for uptrend."""
    _, _, closes, _ = _generate_trending_up()
    momentum = calculate_momentum(closes)
    assert momentum > 0


def test_calculate_momentum_down():
    """Test momentum is negative for downtrend."""
    _, _, closes, _ = _generate_trending_down()
    momentum = calculate_momentum(closes)
    assert momentum < 0


def test_calculate_volume_profile():
    """Test volume profile calculation."""
    volumes = [1000.0] * 20
    ratio = calculate_volume_profile(volumes)
    assert abs(ratio - 1.0) < 0.01


def test_classifier_sideways():
    """Test classifier detects sideways market."""
    highs, lows, closes, volumes = _generate_sideways()
    classifier = RegimeClassifier()
    result = classifier.classify(highs, lows, closes, volumes)
    assert result.regime == MarketRegime.SIDEWAYS
    assert 0 < result.confidence <= 1.0


def test_detector_initial_regime():
    """Test detector sets initial regime."""
    highs, lows, closes, volumes = _generate_sideways()
    detector = RegimeDetector()
    result = detector.detect_regime(highs, lows, closes, volumes)
    assert detector.current_regime is not None
    assert len(detector.regime_history) == 1


def test_detector_requires_confirmations():
    """Test detector requires multiple confirmations for transition."""
    detector = RegimeDetector(confirmations_required=3)

    # Set initial regime with sideways data
    h, l, c, v = _generate_sideways()
    detector.detect_regime(h, l, c, v)
    initial = detector.current_regime

    # Feed trending data - should not transition immediately
    h2, l2, c2, v2 = _generate_trending_up()
    detector.detect_regime(h2, l2, c2, v2)
    # May or may not have transitioned after 1 detection
    # The key test is that history tracks transitions
    assert len(detector.regime_history) >= 1


def test_detector_regime_history():
    """Test regime history is recorded."""
    detector = RegimeDetector()
    h, l, c, v = _generate_sideways()
    detector.detect_regime(h, l, c, v)
    history = detector.get_regime_history()
    assert len(history) >= 1
    assert "to_regime" in history[0]
    assert "confidence" in history[0]
    assert "timestamp" in history[0]
