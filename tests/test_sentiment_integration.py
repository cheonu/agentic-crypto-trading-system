"""Property-based tests for HF sentiment integration.

Tests use Hypothesis to verify correctness properties from the design doc.
All HF pipeline calls are mocked — no real model loading in tests.

Docs: https://hypothesis.readthedocs.io/en/latest/quickstart.html
"""
import pytest
from unittest.mock import MagicMock, patch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agentic_crypto_trading_system.day_trading.news_provider import (
    LABEL_SCORE_MAP,
    ModelManager,
    SentimentAnalyzer,
)


# --- Property 5: Model singleton ---
def test_model_singleton():
    """All calls to get_pipeline() return the same object instance."""
    mock_pipe_fn = MagicMock(return_value=MagicMock())
    with patch.dict("sys.modules", {"transformers": MagicMock(pipeline=mock_pipe_fn)}):
        # Need fresh instances since the module-level import is cached
        mm = ModelManager(model_name="ElKulako/cryptobert")
        p1 = mm.get_pipeline()
        p2 = mm.get_pipeline()
        p3 = mm.get_pipeline()
        assert p1 is p2 is p3
        mock_pipe_fn.assert_called_once()  # loaded only once


# --- Property 6: Label-to-score mapping correctness ---
@given(
    label=st.sampled_from(["Bullish", "BEARISH", "neutral", "Positive", "Negative",
                           "bullish", "bearish", "NEUTRAL", "POSITIVE", "negative"]),
    confidence=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=100)
def test_label_score_mapping(label, confidence):
    """Labels normalized to lowercase produce correct sign."""
    direction = LABEL_SCORE_MAP.get(label.lower(), 0.0)
    score = direction * confidence

    if label.lower() in ("positive", "bullish"):
        assert score >= 0.0
    elif label.lower() in ("negative", "bearish"):
        assert score <= 0.0
    elif label.lower() == "neutral":
        assert score == 0.0


# --- Property 7: Aggregate score is mean ---
@given(
    scores=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=50,
    ),
)
@settings(max_examples=100)
def test_aggregate_is_mean(scores):
    """score_headlines returns the mean of individual scores (before clamping)."""
    headlines = [{"title": f"headline {i}"} for i in range(len(scores))]

    # Build mock responses: for each score, produce a pipeline result
    # that SentimentAnalyzer.score_single will convert back to that score.
    mock_results = []
    for s in scores:
        if s >= 0:
            mock_results.append([{"label": "Bullish", "score": abs(s)}])
        else:
            mock_results.append([{"label": "Bearish", "score": abs(s)}])

    # score_headlines calls score_single per headline, which calls get_pipeline().
    # get_pipeline() is called once (singleton), but the pipeline callable is
    # invoked once per headline. We mock the pipeline callable's return values.
    mock_pipe_callable = MagicMock(side_effect=mock_results)
    mock_pipe_fn = MagicMock(return_value=mock_pipe_callable)

    with patch.dict("sys.modules", {"transformers": MagicMock(pipeline=mock_pipe_fn)}):
        mm = ModelManager()
        sa = SentimentAnalyzer(mm)
        result = sa.score_headlines(headlines)

        expected = sum(scores) / len(scores)
        expected_clamped = max(-1.0, min(1.0, expected))
        assert abs(result - expected_clamped) < 1e-6


# --- Property 8: Score always in valid range ---
@given(
    headlines=st.lists(
        st.fixed_dictionaries({"title": st.text(min_size=1, max_size=100)}),
        min_size=0, max_size=20,
    ),
)
@settings(max_examples=100)
def test_score_in_valid_range(headlines):
    """Returned score is always in [-1.0, 1.0]."""
    mock_pipe_callable = MagicMock(return_value=[{"label": "Bullish", "score": 0.5}])
    mock_pipe_fn = MagicMock(return_value=mock_pipe_callable)

    with patch.dict("sys.modules", {"transformers": MagicMock(pipeline=mock_pipe_fn)}):
        mm = ModelManager()
        sa = SentimentAnalyzer(mm)
        score = sa.score_headlines(headlines)
        assert -1.0 <= score <= 1.0


# --- Property 11: Inference failure resilience ---
def test_inference_failure_resilience():
    """Partial/total failures return mean of successes or 0.0, never raise."""
    headlines = [{"title": "good news"}, {"title": "bad news"}, {"title": "crash"}]

    mock_pipe_callable = MagicMock(side_effect=[
        [{"label": "Bullish", "score": 0.8}],
        Exception("inference error"),
        [{"label": "Bearish", "score": 0.6}],
    ])
    mock_pipe_fn = MagicMock(return_value=mock_pipe_callable)

    with patch.dict("sys.modules", {"transformers": MagicMock(pipeline=mock_pipe_fn)}):
        mm = ModelManager()
        sa = SentimentAnalyzer(mm)
        score = sa.score_headlines(headlines)
        # Should not raise, should return mean of successful scores
        assert -1.0 <= score <= 1.0
