"""
Prometheus Metrics for CellType-Agent.

Implements comprehensive metrics collection for monitoring.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("ct.monitoring.metrics")


@dataclass
class MetricValue:
    """A single metric value."""
    name: str
    value: float
    labels: dict
    timestamp: float


class MetricsCollector:
    """
    Prometheus-style metrics collector.

    Supports:
    - Counters (monotonically increasing)
    - Gauges (can go up or down)
    - Histograms (distribution of values)
    - Summaries (quantiles)

    Usage:
        metrics = MetricsCollector()
        metrics.counter("requests_total", 1, {"method": "GET"})
        metrics.gauge("active_connections", 10)
        metrics.histogram("request_duration", 0.5)
    """

    def __init__(self, namespace: str = "celltype_agent"):
        """
        Initialize metrics collector.

        Args:
            namespace: Metrics namespace prefix
        """
        self.namespace = namespace
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
        self._labels: dict[str, dict] = {}

    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[dict] = None,
    ) -> float:
        """
        Increment a counter.

        Args:
            name: Metric name
            value: Value to add
            labels: Optional labels

        Returns:
            New counter value
        """
        key = self._make_key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value
        self._labels[key] = labels or {}

        return self._counters[key]

    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[dict] = None,
    ) -> float:
        """
        Set a gauge value.

        Args:
            name: Metric name
            value: Value to set
            labels: Optional labels

        Returns:
            New gauge value
        """
        key = self._make_key(name, labels)
        self._gauges[key] = value
        self._labels[key] = labels or {}

        return self._gauges[key]

    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[dict] = None,
    ) -> list[float]:
        """
        Record a histogram value.

        Args:
            name: Metric name
            value: Value to record
            labels: Optional labels

        Returns:
            All recorded values
        """
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
        self._labels[key] = labels or {}

        return self._histograms[key]

    def increment(self, name: str, labels: Optional[dict] = None) -> float:
        """Increment a counter by 1."""
        return self.counter(name, 1.0, labels)

    def decrement(self, name: str, labels: Optional[dict] = None) -> float:
        """Decrement a gauge by 1."""
        key = self._make_key(name, labels)
        self._gauges[key] = self._gauges.get(key, 0) - 1
        return self._gauges[key]

    @contextmanager
    def track_time(self, name: str, labels: Optional[dict] = None):
        """
        Context manager to track timing.

        Args:
            name: Metric name
            labels: Optional labels
        """
        start = time.time()
        yield
        duration = time.time() - start
        self.histogram(name, duration, labels)

    def _make_key(self, name: str, labels: Optional[dict]) -> str:
        """Create a unique key from name and labels."""
        if not labels:
            return f"{self.namespace}_{name}"

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{self.namespace}_{name}{{{label_str}}}"

    def get_counter(self, name: str, labels: Optional[dict] = None) -> float:
        """Get counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0)

    def get_gauge(self, name: str, labels: Optional[dict] = None) -> float:
        """Get gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key, 0)

    def get_histogram_stats(
        self,
        name: str,
        labels: Optional[dict] = None,
    ) -> dict:
        """Get histogram statistics."""
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])

        if not values:
            return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}

        sorted_values = sorted(values)
        count = len(values)

        return {
            "count": count,
            "sum": sum(values),
            "avg": sum(values) / count,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "p50": sorted_values[int(count * 0.5)],
            "p90": sorted_values[int(count * 0.9)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)],
        }

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        # Export counters
        for key, value in self._counters.items():
            name = key.split("{")[0]
            labels = self._labels.get(key, {})

            # Add HELP and TYPE
            lines.append(f"# HELP {name} Counter metric")
            lines.append(f"# TYPE {name} counter")

            if labels:
                label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                lines.append(f"{name}{{{label_str}}} {value}")
            else:
                lines.append(f"{name} {value}")

        # Export gauges
        for key, value in self._gauges.items():
            name = key.split("{")[0]
            labels = self._labels.get(key, {})

            lines.append(f"# HELP {name} Gauge metric")
            lines.append(f"# TYPE {name} gauge")

            if labels:
                label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                lines.append(f"{name}{{{label_str}}} {value}")
            else:
                lines.append(f"{name} {value}")

        # Export histograms
        for key, values in self._histograms.items():
            if not values:
                continue

            name = key.split("{")[0]
            stats = self.get_histogram_stats(name, self._labels.get(key))

            lines.append(f"# HELP {name} Histogram metric")
            lines.append(f"# TYPE {name} histogram")

            # Buckets
            buckets = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
            for bucket in buckets:
                count = sum(1 for v in values if v <= bucket)
                lines.append(f'{name}_bucket{{le="{bucket}"}} {count}')

            lines.append(f'{name}_bucket{{le="+Inf"}} {len(values)}')
            lines.append(f"{name}_sum {stats['sum']}")
            lines.append(f"{name}_count {stats['count']}")

        return "\n".join(lines)

    def reset(self):
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._labels.clear()


# Global metrics instance
_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


def track_request(method: str, endpoint: str, status: int, duration: float):
    """Track an API request."""
    metrics = get_metrics()
    labels = {"method": method, "endpoint": endpoint, "status": str(status)}

    metrics.counter("http_requests_total", 1, labels)
    metrics.histogram("http_request_duration_seconds", duration, labels)

    if status >= 400:
        metrics.counter("http_errors_total", 1, labels)


def track_tool_call(tool: str, success: bool, duration: float):
    """Track a tool call."""
    metrics = get_metrics()
    labels = {"tool": tool, "success": str(success).lower()}

    metrics.counter("tool_calls_total", 1, labels)
    metrics.histogram("tool_call_duration_seconds", duration, labels)


def track_llm_call(provider: str, model: str, tokens: int, duration: float):
    """Track an LLM API call."""
    metrics = get_metrics()
    labels = {"provider": provider, "model": model}

    metrics.counter("llm_calls_total", 1, labels)
    metrics.counter("llm_tokens_total", tokens, labels)
    metrics.histogram("llm_call_duration_seconds", duration, labels)