"""Metrics collection for system observability.

Tracks trade latency, agent decisions, risk rejections, and system resources.
Exposes metrics in a format compatible with Prometheus scraping.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects and exposes system metrics."""

    def __init__(self):
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)

    def increment(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        self.counters[key] += value

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric to a specific value."""
        key = self._make_key(name, labels)
        self.gauges[key] = value

    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record an observation for a histogram metric."""
        key = self._make_key(name, labels)
        self.histograms[key].append(value)

    def record_latency(self, operation: str, duration_ms: float) -> None:
        """Record operation latency."""
        self.observe("operation_latency_ms", duration_ms, {"operation": operation})

    def record_trade(self, symbol: str, side: str, success: bool) -> None:
        """Record a trade execution."""
        self.increment("trades_total", labels={"symbol": symbol, "side": side, "success": str(success)})

    def record_agent_decision(self, agent_role: str, decision: str) -> None:
        """Record an agent decision."""
        self.increment("agent_decisions_total", labels={"role": agent_role, "decision": decision})

    def record_risk_rejection(self, reason: str) -> None:
        """Record a risk manager rejection."""
        self.increment("risk_rejections_total", labels={"reason": reason})

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dict."""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {k: self._summarize_histogram(v) for k, v in self.histograms.items()},
        }

    def _summarize_histogram(self, values: List[float]) -> Dict[str, float]:
        """Summarize histogram values."""
        if not values:
            return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
