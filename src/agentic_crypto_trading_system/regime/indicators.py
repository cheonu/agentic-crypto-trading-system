from typing import List

import numpy as np


def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """Calculate Average True Range (ATR) for volatility measurement.
    
    ATR = SMA of True Range over the period.
    True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    """
    if len(closes) < period + 1:
        raise ValueError(f"Need at least {period + 1} data points, got {len(closes)}")

    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)

    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))

    return float(np.mean(true_range[-period:]))


def calculate_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """Calculate Average Directional Index (ADX) for trend strength.
    
    ADX > 25 = strong trend, ADX < 20 = weak/no trend.
    """
    if len(closes) < period * 2:
        raise ValueError(f"Need at least {period * 2} data points, got {len(closes)}")

    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)

    # Directional movement
    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # True range
    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))

    # Smoothed averages (simple moving average for simplicity)
    atr = np.mean(true_range[-period:])
    if atr == 0:
        return 0.0

    plus_di = 100 * np.mean(plus_dm[-period:]) / atr
    minus_di = 100 * np.mean(minus_dm[-period:]) / atr

    di_sum = plus_di + minus_di
    if di_sum == 0:
        return 0.0

    dx = 100 * abs(plus_di - minus_di) / di_sum
    return float(dx)


def calculate_momentum(closes: List[float], periods: List[int] = None) -> float:
    """Calculate price momentum across multiple timeframes.
    
    Returns average rate of change across specified periods.
    Positive = bullish momentum, Negative = bearish momentum.
    """
    if periods is None:
        periods = [5, 10, 20]

    closes = np.array(closes, dtype=float)
    min_needed = max(periods) + 1
    if len(closes) < min_needed:
        raise ValueError(f"Need at least {min_needed} data points, got {len(closes)}")

    rocs = []
    for p in periods:
        if closes[-p - 1] != 0:
            roc = (closes[-1] - closes[-p - 1]) / closes[-p - 1] * 100
            rocs.append(roc)

    return float(np.mean(rocs)) if rocs else 0.0


def calculate_volume_profile(volumes: List[float], period: int = 20) -> float:
    """Calculate volume profile ratio (recent vs average).
    
    > 1.0 = above average volume, < 1.0 = below average.
    """
    if len(volumes) < period:
        raise ValueError(f"Need at least {period} data points, got {len(volumes)}")

    volumes = np.array(volumes, dtype=float)
    avg_volume = np.mean(volumes[-period:])
    if avg_volume == 0:
        return 0.0

    recent_avg = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
    return float(recent_avg / avg_volume)
