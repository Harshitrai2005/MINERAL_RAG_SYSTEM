"""
Prometheus Metrics — Mineral Exploration Intelligence System
─────────────────────────────────────────────────────────────────────────────
Exposes /api/metrics endpoint (text/plain Prometheus scrape format).
All metrics use the "meis_" prefix so Grafana dashboards can filter cleanly.

Metric catalogue:
  meis_http_requests_total          — counter, labels: method, path, status
  meis_http_request_duration_seconds — histogram, labels: method, path
  meis_rag_queries_total            — counter, labels: query_type
  meis_rag_query_duration_seconds   — histogram, labels: query_type
  meis_rag_chunks_retrieved         — histogram (chunk count per query)
  meis_ingest_documents_total       — counter, labels: doc_type, status
  meis_ingest_duration_seconds      — histogram
  meis_evaluation_scores            — histogram, labels: criterion
  meis_vector_store_documents       — gauge, labels: collection
  meis_errors_total                 — counter, labels: endpoint, error_type
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

# ── Lightweight pure-Python Prometheus client ─────────────────────────────────
# We avoid prometheus_client library to keep the Docker image lean on free tier.
# This implementation is wire-compatible with Prometheus scraping.

class _Counter:
    def __init__(self, name: str, help_text: str, label_names: list[str]):
        self.name = name
        self.help = help_text
        self.label_names = label_names
        self._data: Dict[tuple, float] = defaultdict(float)

    def labels(self, **kwargs) -> "_CounterChild":
        key = tuple(kwargs.get(l, "") for l in self.label_names)
        return _CounterChild(self, key)

    def inc(self, labels: tuple, amount: float = 1.0):
        self._data[labels] += amount

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        for labels, val in self._data.items():
            label_str = ",".join(f'{n}="{v}"' for n, v in zip(self.label_names, labels))
            suffix = f"{{{label_str}}}" if label_str else ""
            lines.append(f"{self.name}{suffix} {val}")
        return "\n".join(lines)


class _CounterChild:
    def __init__(self, parent: _Counter, key: tuple):
        self._parent = parent
        self._key = key

    def inc(self, amount: float = 1.0):
        self._parent.inc(self._key, amount)


class _Gauge:
    def __init__(self, name: str, help_text: str, label_names: list[str]):
        self.name = name
        self.help = help_text
        self.label_names = label_names
        self._data: Dict[tuple, float] = {}

    def labels(self, **kwargs) -> "_GaugeChild":
        key = tuple(kwargs.get(l, "") for l in self.label_names)
        return _GaugeChild(self, key)

    def set(self, labels: tuple, value: float):
        self._data[labels] = value

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        for labels, val in self._data.items():
            label_str = ",".join(f'{n}="{v}"' for n, v in zip(self.label_names, labels))
            suffix = f"{{{label_str}}}" if label_str else ""
            lines.append(f"{self.name}{suffix} {val}")
        return "\n".join(lines)


class _GaugeChild:
    def __init__(self, parent: _Gauge, key: tuple):
        self._parent = parent
        self._key = key

    def set(self, value: float):
        self._parent.set(self._key, value)

    def inc(self, amount: float = 1.0):
        current = self._parent._data.get(self._key, 0.0)
        self._parent._data[self._key] = current + amount

    def dec(self, amount: float = 1.0):
        current = self._parent._data.get(self._key, 0.0)
        self._parent._data[self._key] = current - amount


class _Histogram:
    BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")]

    def __init__(self, name: str, help_text: str, label_names: list[str], buckets: list[float] | None = None):
        self.name = name
        self.help = help_text
        self.label_names = label_names
        self.buckets = buckets or self.BUCKETS
        self._data: Dict[tuple, dict] = defaultdict(lambda: {
            "sum": 0.0,
            "count": 0,
            "buckets": defaultdict(int),
        })

    def labels(self, **kwargs) -> "_HistogramChild":
        key = tuple(kwargs.get(l, "") for l in self.label_names)
        return _HistogramChild(self, key)

    def observe(self, labels: tuple, value: float):
        d = self._data[labels]
        d["sum"] += value
        d["count"] += 1
        for b in self.buckets:
            if value <= b:
                d["buckets"][b] += 1

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for labels, d in self._data.items():
            base = ",".join(f'{n}="{v}"' for n, v in zip(self.label_names, labels))
            for b in self.buckets:
                le = "+Inf" if b == float("inf") else str(b)
                label_str = f'{base},le="{le}"' if base else f'le="{le}"'
                lines.append(f"{self.name}_bucket{{{label_str}}} {d['buckets'][b]}")
            suffix = f"{{{base}}}" if base else ""
            lines.append(f"{self.name}_sum{suffix} {d['sum']}")
            lines.append(f"{self.name}_count{suffix} {d['count']}")
        return "\n".join(lines)


class _HistogramChild:
    def __init__(self, parent: _Histogram, key: tuple):
        self._parent = parent
        self._key = key

    def observe(self, value: float):
        self._parent.observe(self._key, value)


# ── Registry ─────────────────────────────────────────────────────────────────

class _Registry:
    def __init__(self):
        self._collectors = []

    def register(self, collector):
        self._collectors.append(collector)
        return collector

    def render_all(self) -> str:
        return "\n\n".join(c.render() for c in self._collectors)


registry = _Registry()

def _counter(name, help_text, labels=None):
    return registry.register(_Counter(name, help_text, labels or []))

def _gauge(name, help_text, labels=None):
    return registry.register(_Gauge(name, help_text, labels or []))

def _histogram(name, help_text, labels=None, buckets=None):
    return registry.register(_Histogram(name, help_text, labels or [], buckets))


# ── Metric definitions ────────────────────────────────────────────────────────

http_requests_total = _counter(
    "meis_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

http_request_duration_seconds = _histogram(
    "meis_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

rag_queries_total = _counter(
    "meis_rag_queries_total",
    "Total RAG queries executed",
    ["query_type"],
)

rag_query_duration_seconds = _histogram(
    "meis_rag_query_duration_seconds",
    "RAG pipeline end-to-end duration in seconds",
    ["query_type"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, float("inf")],
)

rag_chunks_retrieved = _histogram(
    "meis_rag_chunks_retrieved",
    "Number of chunks retrieved per RAG query",
    ["query_type"],
    buckets=[0, 1, 2, 3, 5, 8, 13, 20, float("inf")],
)

ingest_documents_total = _counter(
    "meis_ingest_documents_total",
    "Total documents ingested",
    ["doc_type", "status"],
)

ingest_duration_seconds = _histogram(
    "meis_ingest_duration_seconds",
    "Document ingestion duration in seconds",
    ["doc_type"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 15.0, 30.0, 60.0, float("inf")],
)

evaluation_scores = _histogram(
    "meis_evaluation_scores",
    "RAG evaluation criterion scores (0.0–1.0)",
    ["criterion"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float("inf")],
)

vector_store_documents = _gauge(
    "meis_vector_store_documents",
    "Number of documents currently in each vector collection",
    ["collection"],
)

errors_total = _counter(
    "meis_errors_total",
    "Total errors by endpoint and error type",
    ["endpoint", "error_type"],
)


# ── Context manager helpers ───────────────────────────────────────────────────

class timer:
    """
    Usage:
        with timer(rag_query_duration_seconds.labels(query_type="all")) as t:
            result = rag.query(...)
    """
    def __init__(self, histogram_child: _HistogramChild):
        self._child = histogram_child
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
        self._child.observe(self.elapsed)


def generate_metrics_text() -> str:
    """Return the full Prometheus text exposition format."""
    return registry.render_all() + "\n"
